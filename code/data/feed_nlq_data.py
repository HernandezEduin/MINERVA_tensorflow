import os

import numpy as np
import tensorflow as tf
from transformers import TFAutoModel, AutoTokenizer

from code.data.utils import load_dictionary, load_qa_data, ids_to_embeddings_tf

from typing import Generator, Dict, Any, Tuple

class QuestionBatcher():
    """
    Batches Natural Language Questions (NLQ) for training and evaluation.
    Includes tokenization and embedding of questions, source entity (starting position),
    and answer entity.
    """
    def __init__(
            self, 
            input_dir: str,
            batch_size: int, 
            question_tokenizer_name: str,
            cached_QAMetaData_path: str,
            raw_QAData_path: str,
            mode: str = "train",
            force_data_prepro: bool = False,
        ) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.batch_size = batch_size

        ent2id, rel2id, id2ent, id2rel = load_dictionary(input_dir)
        self.entity_vocab = ent2id
        self.relation_vocab = rel2id
        self.id2entity = id2ent
        self.id2relation = id2rel

        self.train_df, self.dev_df, self.test_df, self.train_metadata = load_qa_data(
            cached_metadata_path=cached_QAMetaData_path,
            raw_QAData_path=raw_QAData_path,
            question_tokenizer_name=question_tokenizer_name,
            answer_tokenizer_name=question_tokenizer_name, 
            entity2id=ent2id,
            relation2id=rel2id,
            logger=None,
            force_recompute=force_data_prepro,
        )
        self.set_mode(mode)

        # TODO: See if we can reduce this to 1 model
        self.question_embedder = TFAutoModel.from_pretrained(question_tokenizer_name, from_pt=False) # Text to Embeddings
        self.question_tokenizer = AutoTokenizer.from_pretrained(question_tokenizer_name) # Text to Tokens and vice versa

    def set_mode(self, mode: str) -> None:
        """
        Change the mode of the batcher and assigns the respective evaluation dataframe.
        """
        assert mode in ['train', 'dev', 'test'], "Mode must be one of ['train', 'dev', 'test']"
        self.mode = mode
        if mode == 'train':
            self.eval_df = self.train_df
        elif mode == 'dev':
            self.eval_df = self.dev_df
        else:
            self.eval_df = self.test_df

    def get_mode(self) -> str:
        """
        Get the current mode of the batcher.
        """
        return self.mode

    def get_question_num(self) -> int:
        """
        Get the number of questions in the current batch.
        """
        return len(self.eval_df)

    def yield_next_batch_train(self) -> Generator[Tuple[list, tf.Tensor, np.ndarray, np.ndarray], None, None]:
        assert self.mode == 'train', "Batcher is not in training mode"
        while True:
            batch_idx = np.random.randint(0, len(self.eval_df), size=self.batch_size)  # randomly samples batch indices
            batch = self.eval_df.iloc[batch_idx]                                       # extract dataframes with the sampled indices
            questions = batch['Question'].tolist()                                      # extract the list of questions
            source_ent = batch["Query-Entity"].to_numpy(dtype=int)                      # extract the starting node, already in id format
            answers = batch['Answer-Entity'].to_numpy(dtype=int)                        # already in id format

            # NOTE: Addiontal keys include {"Hops", "Paths"}

            # convert question tokens into embeddings of sizes [batch, hidden]
            question_embeddings = ids_to_embeddings_tf(
                token_id_batches=questions,
                model=self.question_embedder,
                pad_id=0,                  # adjust if your PAD differs
                add_special_tokens=True,   # set False if your IDs already include [CLS]/[SEP]
                cls_id=101,
                sep_id=102,
                max_length=128,
            )

            yield questions, question_embeddings, source_ent, answers

    def yield_next_batch_test(self) -> Generator[Tuple[list, tf.Tensor, np.ndarray, np.ndarray], None, None]:
        remaining_questions = len(self.eval_df)                                         # remaining number of questions for evaluation (starting value)
        current_idx = 0
        while True:
            if remaining_questions - self.batch_size > 0:                                 # so long as the number of remaining questions is bigger than the batch size
                batch_idx = np.arange(current_idx, current_idx+self.batch_size)         # extract the batch indices
                current_idx += self.batch_size                                          # increment the current index counter
                remaining_questions -= self.batch_size                                    # reduce the number of remaining questions
            else:
                batch_idx = np.arange(current_idx, len(self.eval_df))                 # extract the batch indices from current position to maximum
                remaining_questions = 0                                                   # set remaining question value to 0

            batch = self.eval_df.iloc[batch_idx]                                       # extract dataframes with the sampled indices
            questions = batch['Question'].tolist()                                      # extract the list of questions
            source_ent = batch["Query-Entity"].to_numpy(dtype=int)                      # extract the starting node, already in id format
            answers = batch['Answer-Entity'].to_numpy(dtype=int)                        # already in id format

            # convert question tokens into embeddings of sizes [batch, hidden]
            question_embeddings = ids_to_embeddings_tf(
                token_id_batches=questions,
                model=self.question_embedder,
                pad_id=0,                  # adjust if your PAD differs
                add_special_tokens=True,   # set False if your IDs already include [CLS]/[SEP]
                cls_id=101,
                sep_id=102,
                max_length=128,
            )

            yield questions, question_embeddings, source_ent, answers

    def translate_entities(self, entity_ids: np.ndarray) -> list:
        """
        Translate entity IDs into their corresponding entity names.
        """
        return [self.id2entity.get(eid, "Unknown") for eid in entity_ids]

    def translate_relations(self, relation_ids: np.ndarray) -> list:
        """
        Translate relation IDs into their corresponding relation names.
        """
        return [self.id2relation.get(rid, "Unknown") for rid in relation_ids]

    def translate_questions(self, questions: list) -> list:
        """
        Translate question IDs into their corresponding question texts.
        """
        return [self.question_tokenizer.decode(question) for question in questions]