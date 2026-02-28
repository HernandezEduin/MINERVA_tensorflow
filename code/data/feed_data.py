import os
from typing import Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from code.data.data_utils import load_dictionary, load_qa_data
from code.data.embedding_server import EmbeddingServer

class QuestionBatcher:
    """
    A data batcher for Natural Language Question answering over Knowledge Graphs.
    
    This class handles the loading, preprocessing, and batching of question-answer pairs
    for training and evaluation. It provides functionality to:
    
    1. Load and preprocess QA datasets with entity/relation vocabularies
    2. Generate embeddings for questions using transformer models via EmbeddingServer
    3. Batch data for training (random sampling) and testing (sequential)
    4. Translate between entity/relation IDs and human-readable names
    
    The batcher supports both training and evaluation modes, automatically managing
    data splits and providing appropriate batching strategies for each phase.
    
    Attributes:
        batch_size (int): Number of samples per batch
        mode (str): Current mode ('train', 'dev', or 'test')
        entity_vocab (Dict[str, int]): Entity name to ID mapping
        relation_vocab (Dict[str, int]): Relation name to ID mapping
        rev_entity_vocab (Dict[int, str]): Entity ID to name mapping
        rev_relation_vocab (Dict[int, str]): Relation ID to name mapping
        ent2name (Dict[str, str]): Entity name to human-readable title mapping
        rel2name (Dict[str, str]): Relation name to human-readable title mapping
        train_df (pd.DataFrame): Training dataset
        dev_df (pd.DataFrame): Development/validation dataset
        test_df (pd.DataFrame): Test dataset
        eval_df (pd.DataFrame): Current evaluation dataset based on mode
        embedding_server (EmbeddingServer): Server for generating question embeddings
        question_tokenizer (AutoTokenizer): Tokenizer for question text
        pad_id (int): Padding token ID
        cls_id (int): CLS token ID for BERT-style models
        sep_id (int): SEP token ID for BERT-style models
    """
    def __init__(
        self, 
        input_dir: str,
        batch_size: int,
        test_batch_size: int, 
        question_tokenizer_name: str,
        question_format: str,
        cached_QAMetaData_path: str,
        raw_QAData_path: str,
        mode: str = "train",
        multi_answers: bool = False,
        seed: Optional[int] = None,
        force_data_prepro: bool = False,
        embedding_server: Optional[EmbeddingServer] = None,
    ) -> None:
        """
        Initialize the QuestionBatcher with data loading and preprocessing.
        
        Args:
            input_dir: Directory containing entity and relation vocabularies
            batch_size: Number of samples per batch
            test_batch_size: Number of samples per batch during evaluation
            question_tokenizer_name: HuggingFace model name for question tokenization
            question_format: Format of the question input ('full_text', 'relation_only', 'graph_only')
            cached_QAMetaData_path: Path to cached preprocessed QA metadata JSON
            raw_QAData_path: Path to raw QA dataset CSV file
            mode: Initial mode ('train', 'dev', or 'test')
            multi_answers: Whether to handle multiple answers per question
            seed: Optional seed for random number generation
            force_data_prepro: Whether to force reprocessing of cached data
            embedding_server: Optional pre-initialized embedding server
        """
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.batch_size: int = batch_size
        self.test_batch_size: int = test_batch_size

        # Load knowledge graph vocabularies
        ent2id, rel2id, id2ent, id2rel, ent2name, rel2name = load_dictionary(input_dir)
        self.entity_vocab: Dict[str, int] = ent2id
        self.relation_vocab: Dict[str, int] = rel2id
        self.rev_entity_vocab: Dict[int, str] = id2ent
        self.rev_relation_vocab: Dict[int, str] = id2rel
        self.ent2name: Dict[str, str] = ent2name
        self.rel2name: Dict[str, str] = rel2name

        # Load and preprocess QA datasets
        self.train_df: pd.DataFrame
        self.dev_df: pd.DataFrame
        self.test_df: pd.DataFrame
        self.train_metadata: Dict
        self.question_format: str = question_format
        self.multi_answers: bool = multi_answers
        
        self.train_df, self.dev_df, self.test_df, self.train_metadata = load_qa_data(
            cached_metadata_path=cached_QAMetaData_path,
            raw_QAData_path=raw_QAData_path,
            multi_answers=multi_answers,
            question_tokenizer_name=question_tokenizer_name,
            entity2id=ent2id,
            relation2id=rel2id,
            seed=seed,
            logger=None,
            force_recompute=force_data_prepro,
        )

        # check if paths exist in dev/test sets
        self.path_exists: bool = True
        for df in [self.dev_df, self.test_df]:
            if 'Paths' not in df.columns:
                self.path_exists = False
                break

        # Set initial mode and evaluation dataset
        self.mode: str = mode
        self.eval_df: pd.DataFrame
        self.set_mode(mode)

        # Initialize embedding server for question processing
        self.embedding_server: EmbeddingServer = embedding_server or EmbeddingServer(question_tokenizer_name)

        # Initialize tokenizer and special token IDs
        self.question_tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(question_tokenizer_name)
        self.pad_id: int = self.question_tokenizer.pad_token_id or 0
        self.cls_id: int = self.question_tokenizer.cls_token_id or 101
        self.sep_id: int = self.question_tokenizer.sep_token_id or 102
        
        # Cache embedding dimensions for easy access
        self._embedding_dim: Optional[int] = None
        self.calculate_embedding_dimensions()  # Pre-fetch dimensions

    def calculate_embedding_dimensions(self) -> Tuple[int, int]:
        """
        Get the embedding dimensions by testing the embedding server.

        Returns:
            Tuple of (batch_size, embedding_dim) where batch_size is 1 for the test
        """
        # Test with a simple question to get dimensions
        test_question = [[0, 0, 0]]
        test_embedding = self.embedding_server.embed(
            token_id_batches=test_question,
            pad_id=self.pad_id,
            cls_id=self.cls_id,
            sep_id=self.sep_id,
        )
        # Cache the embedding dimension for future use
        if self._embedding_dim is None:
            self._embedding_dim = test_embedding.shape[1]
        return test_embedding.shape  # Returns (1, embedding_dim)

    def get_embedding_dim(self) -> int:
        """
        Get just the embedding dimension (not the batch size).

        Returns:
            The embedding dimension as an integer
        """
        if self._embedding_dim is None:
            self.calculate_embedding_dimensions()
        return self._embedding_dim


    def set_mode(self, mode: str) -> None:
        """
        Change the batcher mode and set the corresponding evaluation dataset.
        
        Args:
            mode: New mode ('train', 'dev', or 'test')
            
        Raises:
            AssertionError: If mode is not one of the valid options
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
        Update the batch size for subsequent batching operations.
        
        Args:
            batch_size: New batch size
        """
        self.batch_size = batch_size

    def set_test_batch_size(self, test_batch_size: int) -> None:
        """
        Update the test batch size for subsequent evaluation batching operations.
        
        Args:
            test_batch_size: New test batch size
        """
        self.test_batch_size = test_batch_size

    def get_mode(self) -> str:
        """
        Get the current batcher mode.
        
        Returns:
            Current mode string ('train', 'dev', or 'test')
        """
        return self.mode

    def get_question_num(self) -> int:
        """
        Get the total number of questions in the current evaluation dataset.
        
        Returns:
            Number of questions in current mode's dataset
        """
        return len(self.eval_df)

    def yield_next_batch_train(self) -> Generator[Tuple[List[str], Union[np.ndarray, List[List[int]]], np.ndarray, np.ndarray], None, None]:
        """
        Generate infinite training batches with random sampling.
        
        Yields batches by randomly sampling questions from the training dataset.
        Each batch contains question texts, embeddings, source entities, and answer entities.
        
        Yields:
            Tuple containing:
                - questions (List[str]): Raw question text strings
                - question_embeddings (np.ndarray): Question embeddings [batch_size, embedding_dim]
                - source_ent (np.ndarray): Source entity IDs [batch_size]
                - answers (Union[np.ndarray, List[List[int]]]): Answer(s) entity IDs [batch_size]
                
        Raises:
            AssertionError: If batcher is not in training mode
        """
        assert self.mode == 'train', "Batcher is not in training mode"
        while True:
            # Randomly sample batch indices
            batch_idx = np.random.randint(0, len(self.eval_df), size=self.batch_size)
            batch = self.eval_df.iloc[batch_idx]
            
            source_ent: np.ndarray = batch["Source-Entity"].to_numpy(dtype=int)
            answers: Union[np.ndarray, List[List[int]]] = batch['Answer-Entity'].to_numpy(dtype=int) if not self.multi_answers else batch['Answer-Entity'].tolist()
            paths: List[List[List[str, str, str]]] = batch['Paths'].tolist() if self.path_exists else None

            # Extract questions based on the specified format
            if self.question_format == 'full_text':
                questions: List[List[int]] = batch['Question'].tolist() # already tokenized
            elif self.question_format == 'relation_only' and self.path_exists:
                # extract relation sequences
                relations_only: List[str] = []
                for path in paths:
                    rel_seq = [triple[1] for triple in path]
                    relations_only.append(" ".join([f"[{rel}]" for rel in self.translate_relations(rel_seq)]))
                questions: List[List[int]] = self.tokenize_questions(relations_only)
            else:  # 'graph_only'
                # add empty questions and 0 vector embeddings
                questions: List[List[int]] = self.tokenize_questions([""] * len(batch)) 
                question_embeddings = np.zeros((len(questions), self.get_embedding_dim()), dtype=np.float32)

                yield questions, question_embeddings, source_ent, answers, paths
                continue # skip embedding generation

            # Generate embeddings via the embedding server
            question_embeddings: np.ndarray = self.embedding_server.embed(
                token_id_batches=questions,
                pad_id=self.pad_id,
                cls_id=self.cls_id,
                sep_id=self.sep_id,
                max_length=128,
            )

            yield questions, question_embeddings, source_ent, answers, paths

    def yield_next_batch_test(self) -> Generator[Tuple[List[str], Union[np.ndarray, List[List[int]]], np.ndarray, np.ndarray], None, None]:
        """
        Generate sequential test/evaluation batches without repetition.
        
        Iterates through the evaluation dataset sequentially, yielding batches until
        all questions have been processed. Handles partial batches at the end.
        
        Yields:
            Tuple containing:
                - questions (List[str]): Raw question text strings
                - question_embeddings (np.ndarray): Question embeddings [test_batch_size, embedding_dim]
                - source_ent (np.ndarray): Source entity IDs [test_batch_size]
                - answers (Union[np.ndarray, List[List[int]]]): Answer entity IDs [test_batch_size]
        """
        remaining_questions: int = len(self.eval_df)
        current_idx: int = 0
        
        while True:
            if remaining_questions == 0:
                return
            
            # Determine batch indices for current iteration
            if remaining_questions - self.test_batch_size > 0:
                batch_idx = np.arange(current_idx, current_idx + self.test_batch_size)
                current_idx += self.test_batch_size
                remaining_questions -= self.test_batch_size
            else:
                # Handle final partial batch
                batch_idx = np.arange(current_idx, len(self.eval_df))
                remaining_questions = 0

            # Extract batch data
            batch = self.eval_df.iloc[batch_idx]
            source_ent: np.ndarray = batch["Source-Entity"].to_numpy(dtype=int)
            answers: Union[np.ndarray, List[List[int]]] = batch['Answer-Entity'].to_numpy(dtype=int) if not self.multi_answers else batch['Answer-Entity'].tolist()
            paths: List[List[List[str, str, str]]] = batch['Paths'].tolist() if self.path_exists else None

            # Extract questions based on the specified format
            if self.question_format == 'full_text':
                questions: List[List[int]] = batch['Question'].tolist() # already tokenized
            elif self.question_format == 'relation_only':
                # extract relation sequences
                relations_only: List[str] = []
                for path in paths:
                    rel_seq = [triple[1] for triple in path]
                    relations_only.append(" ".join([f"[{rel}]" for rel in self.translate_relations(rel_seq)]))
                questions: List[List[int]] = self.tokenize_questions(relations_only)
            else:  # 'graph_only'
                # add empty questions and 0 vector embeddings
                questions: List[List[int]] = self.tokenize_questions([""] * len(batch)) 
                question_embeddings = np.zeros((len(questions), self.get_embedding_dim()), dtype=np.float32)

                yield questions, question_embeddings, source_ent, answers, paths
                continue # skip embedding generation

            # Generate embeddings via the embedding server
            question_embeddings: np.ndarray = self.embedding_server.embed(
                token_id_batches=questions,
                pad_id=self.pad_id,
                cls_id=self.cls_id,
                sep_id=self.sep_id,
                max_length=128,
            )

            yield questions, question_embeddings, source_ent, answers, paths

    def translate_entities(self, entity_ids: np.ndarray, dynamic_list: bool = False) -> List[str]:
        """
        Convert entity IDs to their human-readable names.
        
        Args:
            entity_ids: Array of entity IDs to translate
            
        Returns:
            List of entity names corresponding to the input IDs
        """
        if dynamic_list: # assume List[List[int]] (multi-answers)
            result = []
            for sublist in entity_ids:
                if self.ent2name:
                    result.append([self.ent2name.get(self.rev_entity_vocab.get(eid, "Unknown"), "Unknown") for eid in sublist])
                else:
                    result.append([self.rev_entity_vocab.get(eid, "Unknown") for eid in sublist])
            return result
        else: # assuming np.ndarray
            if self.ent2name:
                return [self.ent2name.get(self.rev_entity_vocab.get(eid, "Unknown"), "Unknown") for eid in entity_ids]
            else:
                return [self.rev_entity_vocab.get(eid, "Unknown") for eid in entity_ids]

    def translate_relations(self, relation_ids: np.ndarray) -> List[str]:
        """
        Convert relation IDs to their human-readable names.
        
        Args:
            relation_ids: Array of relation IDs to translate
            
        Returns:
            List of relation names corresponding to the input IDs
        """
        if self.rel2name:
            return [self.rel2name.get(self.rev_relation_vocab.get(rid, "Unknown"), "Unknown") for rid in relation_ids]
        else:
            return [self.rev_relation_vocab.get(rid, "Unknown") for rid in relation_ids]

    def translate_questions(self, questions: Union[List[List[int]], List[str]]) -> List[str]:
        """
        Convert tokenized questions back to human-readable text.
        
        Args:
            questions: List of tokenized questions (as token ID lists) or text strings
            
        Returns:
            List of decoded question text strings
        """
        if isinstance(questions[0], str):
            return questions  # Already decoded
        return [self.question_tokenizer.decode(question) for question in questions]
    
    def tokenize_questions(self, questions: List[str]) -> List[List[int]]:
        """
        Tokenize raw question text into token ID sequences.
        
        Args:
            questions: List of raw question text strings
            
        Returns:
            List of token ID sequences corresponding to the input questions
        """
        if isinstance(questions[0], list):
            return questions  # Already tokenized
        return [self.question_tokenizer.encode(q, add_special_tokens=False) for q in questions]
    