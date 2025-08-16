import os

import numpy as np
import tensorflow as tf
from transformers import TFAutoModel, AutoTokenizer

from code.data.utils import load_dictionary, load_qa_data, ids_to_embeddings_tf
from code.data.embedding_server import EmbeddingServer

from typing import Generator, Dict, Any, Tuple
from types import TracebackType

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
            embedding_server: EmbeddingServer = None,
        ) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.batch_size = batch_size

        ent2id, rel2id, id2ent, id2rel = load_dictionary(input_dir)
        self.entity_vocab = ent2id
        self.relation_vocab = rel2id
        self.rev_entity_vocab = id2ent
        self.rev_relation_vocab = id2rel

        self.train_df, self.dev_df, self.test_df, self.train_metadata = load_qa_data(
            cached_metadata_path=cached_QAMetaData_path,
            raw_QAData_path=raw_QAData_path,
            question_tokenizer_name=question_tokenizer_name,
            entity2id=ent2id,
            relation2id=rel2id,
            logger=None,
            force_recompute=force_data_prepro,
        )

        self.set_mode(mode)

        # self.question_embedder = TFAutoModel.from_pretrained(question_tokenizer_name, from_pt=False) # Text to Embeddings
        self.embedding_server = embedding_server or EmbeddingServer(question_tokenizer_name) # On-the-fly embedding via separate eager process

        self.question_tokenizer = AutoTokenizer.from_pretrained(question_tokenizer_name) # Text to Tokens and vice versa
        self.pad_id = self.question_tokenizer.pad_token_id or 0
        self.cls_id = self.question_tokenizer.cls_token_id or 101
        self.sep_id = self.question_tokenizer.sep_token_id or 102


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

    def set_batch_size(self, batch_size: int) -> None:
        """
        Set the batch size for the batcher.
        """
        self.batch_size = batch_size

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

    def yield_next_batch_train(self) -> Generator[Tuple[list, np.ndarray, np.ndarray, np.ndarray], None, None]:
        assert self.mode == 'train', "Batcher is not in training mode"
        while True:
            batch_idx = np.random.randint(0, len(self.eval_df), size=self.batch_size)  # randomly samples batch indices
            batch = self.eval_df.iloc[batch_idx]                                       # extract dataframes with the sampled indices
            questions = batch['Question'].tolist()                                      # extract the list of questions
            source_ent = batch["Query-Entity"].to_numpy(dtype=int)                      # extract the starting node, already in id format
            answers = batch['Answer-Entity'].to_numpy(dtype=int)                        # already in id format

            # NOTE: Addiontal keys include {"Hops", "Paths"}

            # On-the-fly embeddings from the server (eager in child process)
            question_embeddings = self.embedding_server.embed(
                token_id_batches=questions,
                pad_id=self.pad_id,
                cls_id=self.cls_id,
                sep_id=self.sep_id,
                max_length=128,
            )

            yield questions, question_embeddings, source_ent, answers

    def yield_next_batch_test(self) -> Generator[Tuple[list, np.ndarray, np.ndarray, np.ndarray], None, None]:
        remaining_questions = len(self.eval_df)                                         # remaining number of questions for evaluation (starting value)
        current_idx = 0
        while True:
            if remaining_questions==0:
                return
            
            if remaining_questions - self.batch_size > 0:                                 # so long as the number of remaining questions is bigger than the batch size
                batch_idx = np.arange(current_idx, current_idx+self.batch_size)         # extract the batch indices
                current_idx += self.batch_size                                          # increment the current index counter
                remaining_questions -= self.batch_size                                    # reduce the number of remaining questions
            else:
                batch_idx = np.arange(current_idx, len(self.eval_df))                 # extract the batch indices from current position to maximum
                remaining_questions = 0                                                   # set remaining question value to 0

            batch = self.eval_df.iloc[batch_idx]                                        # extract dataframes with the sampled indices
            questions = batch['Question'].tolist()                                      # extract the list of questions
            source_ent = batch["Query-Entity"].to_numpy(dtype=int)                      # extract the starting node, already in id format
            answers = batch['Answer-Entity'].to_numpy(dtype=int)                        # already in id format

            # On-the-fly embeddings from the server (eager in child process)
            question_embeddings = self.embedding_server.embed(
                token_id_batches=questions,
                pad_id=self.pad_id,
                cls_id=self.cls_id,
                sep_id=self.sep_id,
                max_length=128,
            )

            yield questions, question_embeddings, source_ent, answers

    def translate_entities(self, entity_ids: np.ndarray) -> list:
        """
        Translate entity IDs into their corresponding entity names.
        """
        return [self.rev_entity_vocab.get(eid, "Unknown") for eid in entity_ids]

    def translate_relations(self, relation_ids: np.ndarray) -> list:
        """
        Translate relation IDs into their corresponding relation names.
        """
        return [self.rev_relation_vocab.get(rid, "Unknown") for rid in relation_ids]

    def translate_questions(self, questions: list) -> list:
        """
        Translate question IDs into their corresponding question texts.
        """
        return [self.question_tokenizer.decode(question) for question in questions]

    def __enter__(self):
        """
        Enter the runtime context related to this object.
        """
        return self

    def __exit__(self, exc_type: type, exc_value: Exception, traceback: TracebackType):
        """
        Exit the runtime context related to this object.
        """
        # Clean up the embedding server when exiting the batcher.
        if getattr(self, "embedding_server", None):
            self.embedding_server.close()

    def __del__(self):
        """
        Destructor for the class.
        """
        # safety net
        try:
            if getattr(self, "embedding_server", None):
                self.embedding_server.close()
        except Exception:
            pass