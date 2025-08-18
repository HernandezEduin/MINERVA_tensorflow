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
        question_tokenizer_name: str,
        cached_QAMetaData_path: str,
        raw_QAData_path: str,
        mode: str = "train",
        force_data_prepro: bool = False,
        embedding_server: Optional[EmbeddingServer] = None,
    ) -> None:
        """
        Initialize the QuestionBatcher with data loading and preprocessing.
        
        Args:
            input_dir: Directory containing entity and relation vocabularies
            batch_size: Number of samples per batch
            question_tokenizer_name: HuggingFace model name for question tokenization
            cached_QAMetaData_path: Path to cached preprocessed QA metadata JSON
            raw_QAData_path: Path to raw QA dataset CSV file
            mode: Initial mode ('train', 'dev', or 'test')
            force_data_prepro: Whether to force reprocessing of cached data
            embedding_server: Optional pre-initialized embedding server
        """
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.batch_size: int = batch_size

        # Load knowledge graph vocabularies
        ent2id, rel2id, id2ent, id2rel = load_dictionary(input_dir)
        self.entity_vocab: Dict[str, int] = ent2id
        self.relation_vocab: Dict[str, int] = rel2id
        self.rev_entity_vocab: Dict[int, str] = id2ent
        self.rev_relation_vocab: Dict[int, str] = id2rel

        # Load and preprocess QA datasets
        self.train_df: pd.DataFrame
        self.dev_df: pd.DataFrame
        self.test_df: pd.DataFrame
        self.train_metadata: Dict
        
        self.train_df, self.dev_df, self.test_df, self.train_metadata = load_qa_data(
            cached_metadata_path=cached_QAMetaData_path,
            raw_QAData_path=raw_QAData_path,
            question_tokenizer_name=question_tokenizer_name,
            entity2id=ent2id,
            relation2id=rel2id,
            logger=None,
            force_recompute=force_data_prepro,
        )

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

    def yield_next_batch_train(self) -> Generator[Tuple[List[str], np.ndarray, np.ndarray, np.ndarray], None, None]:
        """
        Generate infinite training batches with random sampling.
        
        Yields batches by randomly sampling questions from the training dataset.
        Each batch contains question texts, embeddings, source entities, and answer entities.
        
        Yields:
            Tuple containing:
                - questions (List[str]): Raw question text strings
                - question_embeddings (np.ndarray): Question embeddings [batch_size, embedding_dim]
                - source_ent (np.ndarray): Source entity IDs [batch_size]
                - answers (np.ndarray): Answer entity IDs [batch_size]
                
        Raises:
            AssertionError: If batcher is not in training mode
        """
        assert self.mode == 'train', "Batcher is not in training mode"
        while True:
            # Randomly sample batch indices
            batch_idx = np.random.randint(0, len(self.eval_df), size=self.batch_size)
            batch = self.eval_df.iloc[batch_idx]
            
            # Extract data fields
            questions: List[str] = batch['Question'].tolist()
            source_ent: np.ndarray = batch["Query-Entity"].to_numpy(dtype=int)
            answers: np.ndarray = batch['Answer-Entity'].to_numpy(dtype=int)

            # Generate embeddings via the embedding server
            question_embeddings: np.ndarray = self.embedding_server.embed(
                token_id_batches=questions,
                pad_id=self.pad_id,
                cls_id=self.cls_id,
                sep_id=self.sep_id,
                max_length=128,
            )

            yield questions, question_embeddings, source_ent, answers

    def yield_next_batch_test(self) -> Generator[Tuple[List[str], np.ndarray, np.ndarray, np.ndarray], None, None]:
        """
        Generate sequential test/evaluation batches without repetition.
        
        Iterates through the evaluation dataset sequentially, yielding batches until
        all questions have been processed. Handles partial batches at the end.
        
        Yields:
            Tuple containing:
                - questions (List[str]): Raw question text strings
                - question_embeddings (np.ndarray): Question embeddings [batch_size, embedding_dim]
                - source_ent (np.ndarray): Source entity IDs [batch_size]
                - answers (np.ndarray): Answer entity IDs [batch_size]
        """
        remaining_questions: int = len(self.eval_df)
        current_idx: int = 0
        
        while True:
            if remaining_questions == 0:
                return
            
            # Determine batch indices for current iteration
            if remaining_questions - self.batch_size > 0:
                batch_idx = np.arange(current_idx, current_idx + self.batch_size)
                current_idx += self.batch_size
                remaining_questions -= self.batch_size
            else:
                # Handle final partial batch
                batch_idx = np.arange(current_idx, len(self.eval_df))
                remaining_questions = 0

            # Extract batch data
            batch = self.eval_df.iloc[batch_idx]
            questions: List[str] = batch['Question'].tolist()
            source_ent: np.ndarray = batch["Query-Entity"].to_numpy(dtype=int)
            answers: np.ndarray = batch['Answer-Entity'].to_numpy(dtype=int)

            # Generate embeddings via the embedding server
            question_embeddings: np.ndarray = self.embedding_server.embed(
                token_id_batches=questions,
                pad_id=self.pad_id,
                cls_id=self.cls_id,
                sep_id=self.sep_id,
                max_length=128,
            )

            yield questions, question_embeddings, source_ent, answers

    def translate_entities(self, entity_ids: np.ndarray) -> List[str]:
        """
        Convert entity IDs to their human-readable names.
        
        Args:
            entity_ids: Array of entity IDs to translate
            
        Returns:
            List of entity names corresponding to the input IDs
        """
        return [self.rev_entity_vocab.get(eid, "Unknown") for eid in entity_ids]

    def translate_relations(self, relation_ids: np.ndarray) -> List[str]:
        """
        Convert relation IDs to their human-readable names.
        
        Args:
            relation_ids: Array of relation IDs to translate
            
        Returns:
            List of relation names corresponding to the input IDs
        """
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
    