"""
Knowledge graph construction and navigation utilities for MINERVA.

This module provides efficient graph construction and action space generation for
knowledge graph reasoning tasks. It builds an adjacency representation optimized
for reinforcement learning agents to navigate knowledge graphs and find reasoning
paths between entities.

Key features:
- Efficient knowledge graph construction from triple files
- Pre-computed action spaces for fast agent navigation
- Intelligent action masking to prevent trivial solutions
- Support for multi-hop reasoning with padding and masking
- Integration with MINERVA's reinforcement learning framework

The module creates a compact array-based representation of the knowledge graph
that enables fast lookup of available actions (relation, target_entity pairs)
for any given entity, with built-in support for masking invalid or unwanted
actions during training and evaluation.

Classes:
    RelationEntityGrapher: Main class for knowledge graph construction and navigation
"""

import csv
import logging
from collections import defaultdict

import numpy as np

from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

class RelationEntityGrapher:
    """
    Knowledge graph constructor and navigator for reinforcement learning agents.
    
    Builds an efficient array-based representation of a knowledge graph from triple
    files and provides methods for agent navigation with intelligent action masking.
    The graph structure is optimized for fast lookup of available actions during
    multi-hop reasoning tasks.
    
    The class constructs two main data structures:
    1. A temporary store for building the graph during initialization
    2. An array-based store for efficient action lookup during reasoning
    
    Action masking prevents agents from taking trivial shortcuts and ensures
    proper evaluation by blocking direct paths to answers and alternative
    correct answers in multi-choice scenarios.
    
    Attributes:
        ePAD (int): Padding token ID for entities (masks invalid entity actions)
        rPAD (int): Padding token ID for relations (masks invalid relation actions)
        triple_store (str): Path to the knowledge graph triple file
        relation_vocab (Dict[str, int]): Relation name to ID mapping
        entity_vocab (Dict[str, int]): Entity name to ID mapping
        array_store (np.ndarray): Pre-computed action space array of shape
                                 (num_entities, max_actions, 2) where last dim
                                 contains (target_entity_id, relation_id) pairs
        rev_relation_vocab (Dict[int, str]): Reverse relation ID to name mapping
        rev_entity_vocab (Dict[int, str]): Reverse entity ID to name mapping
        
    Example:
        >>> grapher = RelationEntityGrapher(
        ...     "data/triples.txt", 
        ...     relation_vocab, 
        ...     entity_vocab, 
        ...     max_num_actions=200
        ... )
        >>> actions = grapher.return_next_actions(current_entities, ...)
        >>> print(actions.shape)  # (batch_size, max_actions, 2)
    """
    def __init__(
        self, 
        triple_store: str, 
        relation_vocab: Dict[str, int], 
        entity_vocab: Dict[str, int], 
        max_num_actions: int
    ) -> None:
        """
        Initialize the knowledge graph from a triple file.
        
        Loads knowledge graph triples and constructs an efficient array-based
        representation for fast action lookup during reasoning. Each entity
        gets a pre-computed action space containing all available (relation, target)
        pairs, padded to a uniform size for batch processing.
        
        Args:
            triple_store: Path to TSV file containing knowledge graph triples
                         Expected format: entity1 \t relation \t entity2
            relation_vocab: Mapping from relation names to integer IDs
                           Must contain 'PAD' and 'NO_OP' special tokens
            entity_vocab: Mapping from entity names to integer IDs
                         Must contain 'PAD' special token
            max_num_actions: Maximum number of actions per entity (for padding)
                            Determines the size of the action space array
                            
        Raises:
            KeyError: If required special tokens ('PAD', 'NO_OP') are missing
            FileNotFoundError: If triple_store file doesn't exist
            ValueError: If triple file format is invalid
            
        Note:
            - Triples should be in tab-separated format: head \t relation \t tail
            - The first action for each entity is always a self-loop (NO_OP)
            - Action space is padded with PAD tokens for entities with few neighbors
            - Memory usage is O(num_entities * max_num_actions)
        """

        # Initialize special token IDs and core attributes
        self.ePAD: int = entity_vocab['PAD']    # Padding for invalid entities
        self.rPAD: int = relation_vocab['PAD']  # Padding for invalid relations
        self.triple_store: str = triple_store
        self.relation_vocab: Dict[str, int] = relation_vocab
        self.entity_vocab: Dict[str, int] = entity_vocab
        
        # Temporary storage for graph construction
        self.store: defaultdict = defaultdict(list)
        
        # Pre-computed action space: (num_entities, max_actions, 2)
        # Last dimension: [target_entity_id, relation_id]
        self.array_store: np.ndarray = np.ones(
            (len(entity_vocab), max_num_actions, 2), 
            dtype=np.int32
        )
        self.array_store[:, :, 0] *= self.ePAD  # Initialize with entity padding
        self.array_store[:, :, 1] *= self.rPAD  # Initialize with relation padding
        self.masked_array_store: Optional[np.ndarray] = None

        # Create reverse vocabularies for debugging/translation
        self.rev_relation_vocab: Dict[int, str] = {v: k for k, v in relation_vocab.items()}
        self.rev_entity_vocab: Dict[int, str] = {v: k for k, v in entity_vocab.items()}
        
        # Build the graph and action spaces
        self.create_graph()
        logging.info("Knowledge graph constructed successfully")

    def create_graph(self) -> None:
        """
        Build the knowledge graph from triples and create action spaces.
        
        Reads the triple file, builds entity neighborhoods, and pre-computes
        the action space for each entity. The action space contains all available
        (relation, target_entity) pairs that an agent can take from each entity.
        
        Process:
        1. Load triples from TSV file and build adjacency lists
        2. For each entity, create action space with NO_OP self-loop as first action
        3. Add all neighboring (relation, entity) pairs to action space
        4. Pad action spaces to uniform size for batch processing
        5. Clean up temporary storage to free memory
        
        File format:
            Tab-separated values: head_entity \t relation \t tail_entity
            
        Action space format:
            array_store[entity_id, action_idx, :] = [target_entity_id, relation_id]
            Action 0 is always a self-loop: [entity_id, NO_OP_relation_id]
            
        Note:
            - Processes the entire knowledge graph into memory
            - NO_OP relation must exist in relation_vocab
            - Entities with more neighbors than max_actions are truncated
            - Temporary store is deleted after completion to save memory
        """
        # Build adjacency lists from triple file
        with open(self.triple_store, 'r') as triple_file_raw:
            triple_file = csv.reader(triple_file_raw, delimiter='\t')
            for line in triple_file:
                head_entity = self.entity_vocab[line[0]]
                relation = self.relation_vocab[line[1]]
                tail_entity = self.entity_vocab[line[2]]
                # Add to adjacency list: head -> [(relation, tail), ...]
                self.store[head_entity].append((relation, tail_entity))

        # Build action spaces for each entity
        for entity_id in self.store:
            action_count = 1
            
            # Action 0: NO_OP self-loop (stay at current entity)
            self.array_store[entity_id, 0, 1] = self.relation_vocab['NO_OP']
            self.array_store[entity_id, 0, 0] = entity_id
            
            # Add all neighboring actions
            for relation, target_entity in self.store[entity_id]:
                if action_count >= self.array_store.shape[1]:
                    # Maximum actions reached, truncate remaining neighbors
                    break
                self.array_store[entity_id, action_count, 0] = target_entity
                self.array_store[entity_id, action_count, 1] = relation
                action_count += 1
                
        # Clean up temporary storage to free memory
        del self.store
        self.store = None

    def return_next_actions(
        self, 
        current_entities: np.ndarray, 
        start_entities: np.ndarray, 
        query_relations: np.ndarray, 
        answers: np.ndarray, 
        all_correct_answers: np.ndarray, 
        last_step: bool, 
        rollouts: int
    ) -> np.ndarray:
        """
        Get available actions for entities with intelligent masking.
        
        Returns the action space for the given entities while applying masking
        to prevent trivial solutions and ensure proper evaluation. Masks out
        direct paths to answers and alternative correct answers to force the
        agent to learn meaningful multi-hop reasoning.
        
        Args:
            current_entities: Array of current entity positions [batch_size]
            start_entities: Array of starting entity positions [batch_size]
            query_relations: Array of query relation IDs [batch_size]
            answers: Array of target answer entity IDs [batch_size]
            all_correct_answers: Array of all possible correct answers for each query
                                [num_unique_queries, num_possible_answers]
            last_step: Whether this is the final reasoning step
            rollouts: Number of rollouts per unique query (for indexing all_correct_answers)
            
        Returns:
            Action array of shape [batch_size, max_actions, 2] where last dimension
            contains [target_entity_id, relation_id]. Invalid actions are masked
            with PAD tokens.
            
        Masking Rules:
            1. At start position: Block direct (query_relation, answer) paths
            2. At last step: Block paths to alternative correct answers
            3. Preserves the correct answer path while blocking shortcuts
            
        Example:
            >>> current = np.array([1, 5, 3])  # Current positions
            >>> actions = grapher.return_next_actions(
            ...     current, start_entities, query_relations, 
            ...     answers, all_correct_answers, last_step=True, rollouts=5
            ... )
            >>> print(actions.shape)  # (3, max_actions, 2)
        """

        # Get base action spaces for current entities
        ret = self.array_store[current_entities, :, :].copy()
        
        # Apply masking for each sample in the batch
        for i in range(current_entities.shape[0]):
            
            # Mask 1: Prevent direct shortcuts from start position
            if current_entities[i] == start_entities[i]:
                # Block direct (query_relation -> answer) paths to force multi-hop reasoning
                relations = ret[i, :, 1]
                entities = ret[i, :, 0]
                direct_path_mask = np.logical_and(
                    relations == query_relations[i], 
                    entities == answers[i]
                )
                ret[i, :, 0][direct_path_mask] = self.ePAD
                ret[i, :, 1][direct_path_mask] = self.rPAD
            
            # Mask 2: Prevent alternative (correct) answers in final step
            if last_step:
                entities = ret[i, :, 0]
                relations = ret[i, :, 1]
                correct_answer = answers[i]
                
                # Find unique query index for this sample
                query_idx = i // rollouts
                
                # Mask all alternative correct answers except the target
                for j in range(entities.shape[0]):
                    if (entities[j] in all_correct_answers[query_idx] and 
                        entities[j] != correct_answer):
                        entities[j] = self.ePAD
                        relations[j] = self.rPAD

        return ret

    def return_next_raw_actions(
        self, 
        current_entities: np.ndarray
    ) -> np.ndarray:
        """
        Get all available actions for entities without any masking.
        
        Returns the complete action space for the given entities without
        applying any masking or filtering. Useful for exploration, debugging,
        or when masking is not desired.
        
        Args:
            current_entities: Array of entity IDs to get actions for
                            Shape: [batch_size] or [num_entities]
                            
        Returns:
            Action array of shape [batch_size, max_actions, 2] where last dimension
            contains [target_entity_id, relation_id]. Padded with PAD tokens for
            entities with fewer than max_actions neighbors.
            
        Example:
            >>> entities = np.array([1, 5, 10])
            >>> raw_actions = grapher.return_next_raw_actions(entities)
            >>> print(raw_actions.shape)  # (3, max_actions, 2)
            >>> print(raw_actions[0, 0])   # [1, NO_OP_id] - self loop
            
        Note:
            - First action is always a NO_OP self-loop
            - No masking is applied - all valid actions are returned
            - Useful for analyzing the full action space or debugging
        """
        return self.array_store[current_entities, :, :].copy()