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

from typing import Dict, List, Optional, Tuple, Union, Set, Sequence

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
        max_num_actions: int,
        use_stop_signal: bool = False,
        use_restart_signal: bool = False,
        use_directed_graph: bool = False
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
            use_stop_signal: Whether to include a STOP action in the action space
            use_restart_signal: Whether to include a RESTART action in the action space
            use_directed_graph: Whether to treat the graph as directed (no inverse relations) or undirected (include inverse relations).
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
        self.rNO_OP: int = relation_vocab['NO_OP']  # No-op relation ID for self-loop
        self.rSTOP: int = relation_vocab['STOP']  # Stop action ID
        self.rRESTART: int = relation_vocab['RESTART']  # Restart action ID
        self.eUNKNOWN: int = entity_vocab['UNK']  # UNKNOWN entity ID
        self.rUNKNOWN: int = relation_vocab['UNK']  # UNKNOWN relation ID
        self.rDUMMY: int = relation_vocab['DUMMY_START_RELATION']  # Dummy start relation ID
        self.triple_store: str = triple_store
        self.relation_vocab: Dict[str, int] = relation_vocab
        self.entity_vocab: Dict[str, int] = entity_vocab
        self.use_stop_signal: bool = use_stop_signal
        self.use_restart_signal: bool = use_restart_signal
        self.use_directed_graph: bool = use_directed_graph
        # Temporary storage used only while constructing a graph representation.
        # It is released after construction and recreated lazily only when needed.
        self.store: Optional[defaultdict] = None

        # Untruncated relation-indexed adjacency used by semantic relation-chain
        # traversal. Keep this lazy: train/dev and normal test navigation never
        # allocate it unless relation-chain traversal is explicitly requested.
        self._relation_chain_adjacency: Optional[Dict[int, Dict[int, Tuple[int, ...]]]] = None
        
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
        
        # inverse tokens for evaluation purposes (e.g. _relation)
        self.inverse_tokens = set()
        self.inverse_mapping = {}        
        for r, r_id in relation_vocab.items():
            if r.startswith('_'):
                self.inverse_tokens.add(r_id)
                self.inverse_mapping[r_id] = relation_vocab[r[1:]]  # map _relation to relation for evaluation purposes

        # Build the graph and action spaces
        self.create_graph()
        logging.info("Knowledge graph constructed successfully")

    def _load_store(self) -> None:
        """
        Load the untruncated graph triples into the temporary ``self.store``.

        ``self.store`` is intentionally transient. The normal navigation action array
        is built from it during initialization and then the store is released. If a
        later test-time analysis needs an untruncated graph representation, the store
        is recreated lazily from ``triple_store`` and released again after use.
        """
        if self.store is not None:
            return

        store = defaultdict(list)
        with open(self.triple_store, 'r') as triple_file_raw:
            triple_file = csv.reader(triple_file_raw, delimiter='\t')
            for line in triple_file:
                head_entity = self.entity_vocab[line[0]]
                relation = self.relation_vocab[line[1]]
                tail_entity = self.entity_vocab[line[2]]
                # Add to adjacency list: head -> [(relation, tail), ...]
                store[head_entity].append((relation, tail_entity))

        self.store = store

    def _free_store(self) -> None:
        """Release the temporary untruncated triple store."""
        self.store = None

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
        self._load_store()  # Load triples into temporary store
        assert self.store is not None, "Temporary store should be initialized"

        # Build action spaces for each entity
        for entity_id in self.store:
            action_count = 1
            
            # Action 0: NO_OP self-loop (stay at current entity)
            self.array_store[entity_id, 0, 1] = self.rNO_OP
            self.array_store[entity_id, 0, 0] = entity_id

            # Action 1: STOP action (optional, can be used to indicate stopping)
            if self.use_stop_signal:
                self.array_store[entity_id, action_count, 1] = self.rSTOP
                self.array_store[entity_id, action_count, 0] = entity_id
                action_count += 1

            # Action 2: RESTART action (optional, can be used to indicate restarting)
            if self.use_restart_signal:
                self.array_store[entity_id, action_count, 1] = self.rRESTART
                self.array_store[entity_id, action_count, 0] = self.ePAD # must change entity to first action to avoid confusion with NO_OP self loop
                action_count += 1
            
            # Add all neighboring actions
            for relation, target_entity in self.store[entity_id]:
                if action_count >= self.array_store.shape[1]:
                    # Maximum actions reached, truncate remaining neighbors
                    break
                if self.use_directed_graph and relation in self.inverse_tokens:
                    # Skip inverse relations if only directed relations are used
                    continue
                self.array_store[entity_id, action_count, 0] = target_entity
                self.array_store[entity_id, action_count, 1] = relation
                action_count += 1
                
        # The normal agent never needs the untruncated store after array_store is built.
        self._free_store()

    def get_relation_chain_adjacency(self) -> Dict[int, Dict[int, Tuple[int, ...]]]:
        """
        Lazily build and cache an untruncated relation-indexed adjacency.

        This representation is intended for semantic relation-chain traversal during
        test-time path-fidelity evaluation. It is deliberately *not* created during
        grapher initialization, training, dev evaluation, or ordinary test navigation.
        The first caller triggers construction; subsequent callers reuse the cache.

        Unlike ``array_store``, this adjacency is not limited by ``max_num_actions``.
        It respects ``use_directed_graph`` by excluding inverse-relation tokens when
        directed navigation is requested.
        """
        if self._relation_chain_adjacency is not None:
            return self._relation_chain_adjacency

        # Recreate the temporary untruncated store only when this lazy test-time
        # representation is actually requested. Do not retain the store afterward.
        created_store = self.store is None
        if created_store:
            self._load_store()
        assert self.store is not None

        adjacency_sets: Dict[int, Dict[int, Set[int]]] = {}
        for head_entity, actions in self.store.items():
            for relation, target_entity in actions:
                if self.use_directed_graph and relation in self.inverse_tokens:
                    continue
                adjacency_sets.setdefault(head_entity, {}).setdefault(relation, set()).add(target_entity)

        # We no longer need the original list-based graph.
        if created_store:
            self._free_store()

        self._relation_chain_adjacency = {
            head: {
                relation: tuple(sorted(targets))
                for relation, targets in rel_targets.items()
            }
            for head, rel_targets in adjacency_sets.items()
        }

        assert self._relation_chain_adjacency is not None
        return self._relation_chain_adjacency

    def clear_relation_chain_adjacency(self) -> None:
        """Release the lazily-created untruncated relation adjacency cache."""
        self._relation_chain_adjacency = None

    def find_paths_by_relation_chain(
        self,
        start_entity: int,
        relation_chain: Sequence[int],
        target_entities: Optional[Set[int]] = None,
    ) -> List[List[Tuple[int, int, int]]]:
        """
        Enumerate untruncated graph paths that follow an exact relation sequence.

        Args:
            start_entity: Entity from which every path starts.
            relation_chain: Ordered relation IDs that each returned path must follow.
            target_entities: Optional endpoint filter. When provided, only paths whose
                final entity belongs to this set are returned.

        Returns:
            Entity-level paths represented as ``(head, relation, tail)`` triples.

        Note:
            Calling this method is what lazily creates the untruncated relation
            adjacency. Therefore normal train/dev execution incurs no extra memory.
        """
        relation_chain = [int(r) for r in relation_chain]
        if not relation_chain:
            return []

        start_entity = int(start_entity)
        targets = None if target_entities is None else set(int(e) for e in target_entities)
        adjacency = self.get_relation_chain_adjacency()

        invalid_entities = {self.ePAD, self.eUNKNOWN}
        invalid_relations = {self.rPAD, self.rUNKNOWN, self.rDUMMY}
        frontier: List[Tuple[int, List[Tuple[int, int, int]]]] = [(start_entity, [])]

        for relation in relation_chain:
            next_frontier: List[Tuple[int, List[Tuple[int, int, int]]]] = []
            seen_paths = set()
            for current_entity, path in frontier:
                for target_entity in adjacency.get(current_entity, {}).get(relation, ()):
                    target_entity = int(target_entity)
                    if target_entity in invalid_entities or relation in invalid_relations:
                        continue

                    next_path = path + [(current_entity, relation, target_entity)]
                    path_key = tuple(next_path)
                    if path_key in seen_paths:
                        continue
                    seen_paths.add(path_key)
                    next_frontier.append((target_entity, next_path))

            frontier = next_frontier
            if not frontier:
                break

        if targets is None:
            return [path for _, path in frontier]
        return [path for entity, path in frontier if entity in targets]

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