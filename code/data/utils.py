import subprocess

import os
import json
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split

import re
import ast

import numpy as np
import pandas as pd

import tensorflow as tf
from transformers import PreTrainedTokenizer, AutoTokenizer

from collections import namedtuple
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple, Union, Optional

Triple = Tuple[int, int, int]
Triples = List[Triple]
# Named Tuple for DF SPlit
SplitTuple = namedtuple("SplitTuple", ["train", "dev", "test"])

@dataclass
class DFSplit:
    train: pd.DataFrame
    dev: pd.DataFrame
    test: pd.DataFrame

def set_seeds(seed):
    import numpy as np
    import random
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def get_git_root() -> Optional[str]:
    # NOTE: This will break if we remove the .git folder (could very well happen).
    # IN that case:
    # TODO: simply return the path until the `MultiHopKG` folder is found. 

    try:
        # Run the git command to get the top-level directory of the repository
        result = subprocess.run(['git', 'rev-parse', '--show-toplevel'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        # The output will be the path to the root of the repository
        git_root = result.stdout.strip()
        return git_root
    except subprocess.CalledProcessError:
        # Handle the case where the command fails (e.g., not in a git repository)
        return None

def load_json(file_path: str) -> Dict[str, any]:
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def extract_literals(column: Union[str, pd.Series], flatten: bool = False) -> Union[pd.Series, List[str]]:
    """
    Extracts the list of string literals from each entry in the provided column (Pandas Series or string)
    using ast.literal_eval. Optionally flattens the extracted lists into a single list if 'flatten' is set to True.

    Args:
        column (Union[str, pd.Series]): The column containing string representations of lists. Can be a
                                        Pandas Series or a string representation of a list.
        flatten (bool): If True, flattens the lists into a single list. Default is False.

    Returns:
        Union[pd.Series, List[str]]: A Pandas Series of lists if flatten is False, otherwise a single flattened list of strings.
    """

    # Convert the input to a Pandas Series if it's a string
    if isinstance(column, str):
        column = pd.Series([column])

    # Convert string representations of lists into actual Python lists
    column = column.apply(ast.literal_eval)

    # Flatten the lists if the flatten argument is True
    if flatten: column = [item for sublist in column for item in sublist]
    return column

def process_and_cache_triviaqa_data(
    raw_QAData_path: str,
    cached_toked_qatriples_metadata_path: str,
    question_tokenizer: PreTrainedTokenizer,
    answer_tokenizer: PreTrainedTokenizer,
    entity2id: Dict[str, int],
    relation2id: Dict[str, int],
    override_split: bool = True,
    logger: Optional[logging.Logger] = None,
) -> Tuple[DFSplit, Dict] :
    """
    Args:
        raw_triples_loc (str) : Place where the unprocessed triples are
        cached_toked_qatriples_path (str) : Place where processed triples are meante to go. You must format them.
        idx_2_graphEnc (Dict[str, np.array]) : The encoding of the tripleshttps://www.youtube.com/watch?v=f-sRcVkZ9yg
        text_tokenizer (AutoTokenizer) : The tokenizer for the text
    Returns:

    Data Assumptions:
        - csv file consists of 1..N columns.
          N-1 is Question, N is Answer
        - 1..N-2 represent the path
          These columns are organized as Entity, Relation, Entity,...
    LG: We might change this assumption to a whole graph later on:
    """

    ## NOTE:  ---
    ## Old Data Loading has been moved elsewhere
    ## ----------
    ## Processing
    csv_df = pd.read_csv(raw_QAData_path)
    assert (
        len(csv_df.columns) > 2
    ), "The CSV file should have at least 2 columns. One triplet and one QA pair"
    
    # FIX: The harcoding of things like "Question" and "Answer" is not good.
    # !TODO: Make this more flexible and relavant entities and relations be optional features
    questions = csv_df["Question"]
    answers = csv_df["Answer"]
    query_ent = csv_df["Query-Entity"]
    query_rel = csv_df["Query-Relation"]
    answer_ent = csv_df["Answer-Entity"]
    paths = extract_literals(csv_df["Paths"]) if 'Paths' in csv_df.columns else None
    splitLabel = csv_df["SplitLabel"] if 'SplitLabel' in csv_df.columns else None
    hops = csv_df["Hops"] if 'Hops' in csv_df.columns else None

    # Ensure directory exists
    dir_name = os.path.dirname(cached_toked_qatriples_metadata_path)
    os.makedirs(dir_name, exist_ok=True)

    ## Prepare the language data
    questions = questions.map(lambda x: question_tokenizer.encode(x, add_special_tokens=False))
    answers = answers.map(
        lambda x: [answer_tokenizer.bos_token_id]
        + answer_tokenizer.encode(x, add_special_tokens=False)
        + [answer_tokenizer.eos_token_id]
    )

    # Preparing the KG data by converting text to indices
    query_ent = query_ent.map(lambda ent: entity2id[ent])
    query_rel = query_rel.map(lambda rel: relation2id[rel])
    answer_ent = answer_ent.map(lambda ent: entity2id[ent])
    if paths is not None:
        paths = paths.map(lambda path: [[entity2id[head], relation2id[rel], entity2id[tail]] for head, rel, tail in path])

    # timestamp without nanoseconds
    timestamp = str(int(datetime.now().timestamp()))
    cached_split_locations: Dict[str, str] = {
        name: cached_toked_qatriples_metadata_path.replace(".json", "") + f"_Split-{name}" + f"_date-{timestamp}" + ".parquet"
        for name in ["train", "dev", "test"]
    }

    repo_root = get_git_root()
    if repo_root is None:
        raise ValueError("Cannot get the git root path. Please make sure you are running a clone of the repo")

    cached_split_locations = {key : val.replace(repo_root + "/", "") for key,val in cached_split_locations.items()}

    # Start amalgamating the data into its final form
    # TODO: test set
    new_df = pd.concat([questions, answers, query_ent, query_rel, answer_ent, paths, hops, splitLabel], axis=1)
    new_df = new_df.sample(frac=1).reset_index(drop=True) # Shuffle before splitting by label

    # Check if splitLabel column has meaningful values to guide the split
    if override_split and 'SplitLabel' in new_df.columns and new_df['SplitLabel'].notna().any() and not new_df['SplitLabel'].eq('').all():
        train_df = new_df[new_df['SplitLabel'] == 'train'].reset_index(drop=True)
        test_df = new_df[new_df['SplitLabel'] != 'train'].reset_index(drop=True)
        if logger: logger.info(f"Using splitLabel column to split the data into train and test sets.")
    else:
        new_df = new_df.sample(frac=1).reset_index(drop=True)
        train_df, test_df = train_test_split(new_df, test_size=0.2, random_state=42)

     # If the test set is too small, use it as dev
    if len(test_df) < 100:
        dev_df = test_df
        if logger: logger.warning("Test set is too small, using it as dev set!!")
    else:
        dev_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42)

    if not isinstance(train_df, pd.DataFrame) or not isinstance(dev_df, pd.DataFrame) or not isinstance(test_df, pd.DataFrame):
        raise RuntimeError("The data was not loaded properly. Please check the data loading code.")

    for name,df in {"train": train_df, "dev": dev_df, "test": test_df}.items():
        df.to_parquet(cached_split_locations[name], index=False)

    ## Prepare metadata for export
    # Tokenize the text by applying a pandas map function
    # Store the metadata
    metadata = {
        "question_tokenizer": question_tokenizer.name_or_path,
        "answer_tokenizer": answer_tokenizer.name_or_path,
        "question_column": "Question",
        "answer_column": "Answer",
        "query_entities_column": "Query-Entity",
        "query_relations_column": "Query-Relation",
        "answer_entity_column": "Answer-Entity",
        "paths_column": "Paths",
        "hops_column": "Hops",
        "splitLabel_column": "SplitLabel",
        "0-index_column": True,
        "date_processed": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "saved_paths": cached_split_locations,
        "timestamp": timestamp,
    }

    with open(cached_toked_qatriples_metadata_path, "w") as f:
        json.dump(metadata, f)

    return DFSplit(train=train_df, dev=dev_df, test=test_df), metadata


def load_qa_data(
    cached_metadata_path: str,
    raw_QAData_path,
    question_tokenizer_name: str,
    answer_tokenizer_name: str,
    entity2id: Dict[str, int],
    relation2id: Dict[str, int], 
    logger: logging.Logger = None,
    force_recompute: bool = False,
    override_split: bool = True,
):

    if os.path.exists(cached_metadata_path) and not force_recompute:
        print(
            f"\033[93m Found cache for the QA data {cached_metadata_path} will load it instead of working on {raw_QAData_path}. \033[0m"
        )
        # Read the first line of the raw csv to count the number of columns
        train_metadata = json.load(open(cached_metadata_path.format(question_tokenizer_name, answer_tokenizer_name)))
        saved_paths: Dict[str, str] = train_metadata["saved_paths"]

        train_df = pd.read_parquet(saved_paths["train"])
        # TODO: Eventually use this to avoid data leakage
        dev_df = pd.read_parquet(saved_paths["dev"])
        test_df = pd.read_parquet(saved_paths["test"])

        # Ensure that we are not reading them integers as strings, but also not as floats
        print(
            f"Loaded cached data from \033[93m\033[4m{json.dumps(cached_metadata_path,indent=4)} \033[0m"
        )
    else:
        ########################################
        # Actually compute the data.
        ########################################
        print(
            f"\033[93m Did not find cache for the QA data {cached_metadata_path}. Will now process it from {raw_QAData_path} \033[0m"
        )
        question_tokenizer = AutoTokenizer.from_pretrained(question_tokenizer_name)
        answer_tokenzier   = AutoTokenizer.from_pretrained(answer_tokenizer_name)
        df_split, train_metadata = ( # Includes shuffling
            process_and_cache_triviaqa_data(  # TOREM: Same here, might want to remove if not really used
                raw_QAData_path,
                cached_metadata_path,
                question_tokenizer,
                answer_tokenzier,
                entity2id,
                relation2id,
                override_split=override_split,
                logger=logger,
            )
        )
        train_df, dev_df, test_df = df_split.train, df_split.dev, df_split.test
        print(
            f"Done. Result dumped at : \n\033[93m\033[4m{train_metadata['saved_paths']}\033[0m"
        )

    return train_df, dev_df, test_df, train_metadata

def load_dictionary(data_dir):
    ent2id = load_json(os.path.join(data_dir, "vocab/entity_vocab.json"))  # Assuming this function loads the dictionaries correctly
    rel2id = load_json(os.path.join(data_dir, "vocab/relation_vocab.json"))  # Assuming this function loads the dictionaries correctly

    id2ent = {v: k for k, v in ent2id.items()}
    id2rel = {v: k for k, v in rel2id.items()}

    return ent2id, rel2id, id2ent, id2rel

def ids_to_embeddings_tf(
    token_id_batches: List[List[int]],
    model,                              # e.g., TFAutoModel.from_pretrained(..., from_pt=False)
    pad_id: int = 0,                    # BERT pad is usually 0
    add_special_tokens: bool = True,    # add [CLS]=101 and [SEP]=102 if not present
    cls_id: int = 101,
    sep_id: int = 102,
    max_length: Optional[int] = 128,
) -> tf.Tensor:
    """
    Args:
      token_id_batches: list of sequences of token IDs (variable length)
      model: TF encoder returning .last_hidden_state
    Returns:
      [batch, hidden] masked-mean pooled embeddings
    """

    # Optionally wrap with special tokens
    if add_special_tokens:
        token_id_batches = [
            ([cls_id] + seq + [sep_id]) if (len(seq) == 0 or seq[0] != cls_id) else seq
            for seq in token_id_batches
        ]

    # Truncate if needed
    if max_length is not None:
        token_id_batches = [seq[:max_length] for seq in token_id_batches]

    # Ragged -> padded int32 [B, Lmax]
    rag = tf.ragged.constant(token_id_batches, dtype=tf.int32)
    lengths = rag.row_lengths()
    Lmax = tf.reduce_max(lengths) if max_length is None else tf.constant(max_length, dtype=tf.int64)
    input_ids = rag.to_tensor(shape=[rag.shape[0], Lmax], default_value=pad_id)  # [B, L]

    # Attention mask: 1 for tokens, 0 for padding
    attn = tf.sequence_mask(lengths, maxlen=Lmax)          # [B, L] bool
    attn = tf.cast(attn, tf.int32)

    # Encode
    outputs = model(input_ids=input_ids, attention_mask=attn)
    last_hidden = outputs.last_hidden_state                 # [B, L, H]

    # Masked mean pooling
    mask = tf.cast(attn, tf.float32)[:, :, None]            # [B, L, 1]
    summed = tf.reduce_sum(last_hidden * mask, axis=1)      # [B, H]
    denom = tf.maximum(tf.reduce_sum(mask, axis=1), 1e-9)   # [B, 1]
    return summed / denom                                   # [B, H]