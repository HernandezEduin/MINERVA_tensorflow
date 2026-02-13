"""
Natural Language Question (NLQ) environment for knowledge graph reasoning.

This module provides the reinforcement learning environment for MINERVA's natural
language question answering over knowledge graphs. It manages the interaction between
RL agents and knowledge graphs, handling multi-hop reasoning episodes where agents
navigate from query entities to answer entities.

Key components:
- Episode management with support for multiple rollouts per question
- Integration with question batching and embedding generation  
- State management for multi-step reasoning paths
- Reward computation based on reaching correct answer entities
- Mode switching between training and evaluation

The environment supports batch processing and multiple simultaneous exploration
paths (rollouts) per question to improve sample efficiency and exploration during
training. It integrates with the knowledge graph structure and question processing
pipeline to provide a complete framework for NLQ reasoning.

Classes:
    EpisodeNLQ: Manages a single reasoning episode with explicit data parameters
    EnvNLQ: Main environment class coordinating episodes and data flow
"""

from __future__ import absolute_import
from __future__ import division

import os
import sys
import logging

import numpy as np

from code.data.embedding_server import EmbeddingServer
from code.data.feed_nlq_data import QuestionBatcher
from code.data.grapher import RelationEntityGrapher

from typing import Any, Dict, Generator, List, Optional, Set, Union, Tuple

logger = logging.getLogger()
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

class EpisodeNLQ(object):
    """
    Single episode manager for natural language question reasoning.
    
    Manages the state and dynamics of a single multi-hop reasoning episode where
    an RL agent navigates through a knowledge graph to answer a natural language
    question. Supports multiple simultaneous rollouts per question to improve
    sample efficiency and exploration.
    
    The episode tracks the agent's current position, available actions, and provides
    rewards based on whether the agent reaches the correct answer entity. State
    includes current entities, possible next actions (relations and target entities),
    and question context.
    
    Attributes:
        grapher (RelationEntityGrapher): Knowledge graph navigator
        batch_size (int): Number of questions in the batch
        path_len (int): Maximum number of reasoning steps allowed
        num_rollouts (int): Number of simultaneous paths per question
        mode (str): Current mode ('train', 'dev', or 'test')
        current_hop (int): Current step number in the reasoning path
        no_examples (int): Number of unique questions in the batch
        positive_reward (float): Reward for reaching correct answer
        negative_reward (float): Reward for incorrect or no answer
        start_entities (np.ndarray): Starting entity IDs for each rollout
        end_entities (np.ndarray): Target answer entity IDs for each rollout
        current_entities (np.ndarray): Current agent positions
        question_embeddings (np.ndarray): Question embeddings for each rollout
        question_tokens (List[str]): Original question text tokens
        state (Dict[str, np.ndarray]): Current environment state
        
    Example:
        >>> episode = EpisodeNLQ(
        ...     grapher, question_tokens, question_embeddings, start_entities, end_entities,
        ...     batch_size=128, path_len=3, num_rollouts=20, test_rollouts=100,
        ...     positive_reward=1.0, negative_reward=-1.0, mode='train'
        ... )
        >>> state = episode.get_state()
        >>> new_state = episode(action_indices)
        >>> rewards = episode.get_reward()
    """

    def __init__(
        self, 
        graph: RelationEntityGrapher, 
        question_tokens: List[str],
        question_embeddings: np.ndarray,
        start_entities: np.ndarray,
        end_entities: Union[np.ndarray, List[List[int]]],
        batch_size: int,
        path_len: int,
        num_rollouts: int,
        positive_reward: float,
        negative_reward: float,
        mode: str,
        multi_answers: bool = False,
        paths: Optional[List[List[List[str]]]] = None,
    ) -> None:
        """
        Initialize a reinforcement learning episode for knowledge graph reasoning.
        
        Sets up the environment state for multi-path exploration in knowledge graphs,
        where the agent learns to navigate from start entities to target entities
        by following relation edges. Supports batch processing with multiple rollouts
        for improved training efficiency.
        
        Args:
            graph: Knowledge graph navigator providing action spaces and transitions
            question_tokens: List of question token ints
            question_embeddings: Question embeddings [batch_size, embedding_dim]
            start_entities: Starting entity IDs [batch_size]
            end_entities: Target answer entity IDs [batch_size]
            batch_size: Number of questions in batch
            path_len: Maximum reasoning steps allowed
            num_rollouts: Number of training rollouts per question
            positive_reward: Reward for correct answers
            negative_reward: Reward for incorrect answers
            mode: Current mode ('train', 'dev', or 'test')
            multi_answers: Whether to handle multiple answers per question
            paths: Optional list of paths for each question
        Note:
            - Creates multiple rollouts by repeating each question/entity
            - Initializes state with available actions from starting positions
            - Supports different rollout counts for training vs evaluation
        """
        self.grapher = graph
        self.batch_size = batch_size
        self.path_len = path_len
        self.mode = mode
        self.num_rollouts = num_rollouts
        self.multi_answers = multi_answers
        self.paths = paths
        self.paths_exists = paths is not None
        self.current_hop = 0
        self.no_examples = start_entities.shape[0]
        self.positive_reward = positive_reward
        self.negative_reward = negative_reward
        self.use_stop_signal = self.grapher.use_stop_signal
        self.use_restart_signal = self.grapher.use_restart_signal
        self.cycle_tokens = set([self.grapher.rNO_OP, self.grapher.rSTOP, self.grapher.rRESTART])  # if using stop/restart signals, we want to ignore them in path faithfulness evaluation since they are not part of the original graph

        # Repeat entities/embeddings for multiple rollouts per question [batch_size,] -> [batch_size * num_rollouts]
        start_entities = np.repeat(start_entities, self.num_rollouts)
        # either [batch_size * num_rollouts] or [batch_size, variable_num_answers]
        end_entities  = np.repeat(end_entities, self.num_rollouts) if not self.multi_answers else [set(sublist) for sublist in end_entities] # faster lookup with set
        self.start_entities = start_entities
        self.end_entities = end_entities
        self.current_entities = np.array(start_entities)
        self.question_embeddings = np.repeat(question_embeddings, self.num_rollouts, axis=0) # [batch_size * num_rollouts, embedding_dim]
        self.question_tokens = question_tokens

        # Track which rollouts have stopped (if using stop signal) to prevent further transitions, but still allow reward calculation at the end of episode
        self.stopped_mask = np.zeros(self.current_entities.shape[0], dtype=bool)
        self.stop_steps = np.full(self.current_entities.shape[0], fill_value=self.path_len, dtype=int)  # track at which step each rollout stopped, initialized to max path length (i.e. not stopped)

        # Initialize state with available actions from starting positions
        next_actions = self.grapher.return_next_raw_actions(self.current_entities)

        if self.use_restart_signal: next_actions = self.process_restart_actions(next_actions)

        self.state = {}                                                         # RL states (next_relations, next_entities, current_entities)
        self.state['next_relations'] = next_actions[:, :, 1]
        self.state['next_entities'] = next_actions[:, :, 0]
        self.state['current_entities'] = self.current_entities

    def get_state(self) -> Dict[str, np.ndarray]:
        """
        Get the current state of the reinforcement learning environment.
        
        Returns all necessary information for the agent to make informed decisions
        during knowledge graph reasoning. The state includes current entity positions
        and available next actions (entities and relations). Question embeddings are
        provided separately as they remain constant throughout the episode.
        
        Returns:
            Dictionary containing exactly three keys:
                - 'current_entities': Current entity positions [total_rollouts]
                - 'next_entities': Available next entities [total_rollouts, max_actions]
                - 'next_relations': Available relations [total_rollouts, max_actions]  
                
        Note:
            - State is updated after each environment step
            - Action spaces are dynamically computed based on current positions
            - No history tracking - only current position and immediate options
            - Used by agents to select optimal reasoning actions
        """
        return self.state

    def get_question_embedding(self) -> np.ndarray:
        """
        Get the question embeddings for the current episode batch.
        
        Returns the pre-computed embeddings for all questions in the current
        episode batch. These embeddings encode the natural language questions
        into dense vector representations that guide the reasoning process.
        The embeddings remain constant throughout the episode as the questions
        do not change during reasoning.
        
        Returns:
            Question embeddings array [batch_size * num_rollouts, embedding_dim]
            where each question is repeated for multiple rollouts to enable
            diverse exploration paths during training
            
        Note:
            - Embeddings are generated once during episode initialization
            - Same question embedding is shared across all rollouts
            - Used by agents to condition their reasoning decisions
        """
        return self.question_embeddings

    def get_reward(self) -> np.ndarray:
        """
        Calculate reward signal for the current state of all batches and rollouts.
        
        Computes binary rewards based on whether agents have reached their
        target entities. Used to train the reinforcement learning policy
        to navigate toward correct answers in the knowledge graph.
        
        Returns:
            Reward array [batch_size*total_rollouts] containing:
                - positive_reward: For agents at target entities
                - negative_reward: For agents not at target entities
                
        Note:
            - Rewards are computed by comparing current positions to target entities
            - Positive rewards encourage successful reasoning paths
            - Negative rewards discourage incorrect reasoning directions
            - Reward values are configured during environment initialization
        """
        if self.multi_answers:
            # if any of the answers in the list match current entity, give positive reward (following literature convention)
            reward = np.array([
                self.positive_reward if self.current_entities[i] in self.end_entities[i // self.num_rollouts]  # use this if repeating end_entities per rollout
                else self.negative_reward 
                for i in range(self.current_entities.shape[0])
            ])
        else:
            reward = (self.current_entities == self.end_entities)
            condlist = [reward == True, reward == False]
            choicelist = [self.positive_reward, self.negative_reward]
            reward = np.select(condlist, choicelist)
        return reward
    
    def _has_hit_answer(self) -> np.ndarray:
        """
        Helper function to check if current entities have hit the target answer entities.
        Returns a boolean array indicating which rollouts have reached an answer.
        Used internally for reward adjustment when using STOP signals.
        """
        if self.multi_answers:
            return np.array([
                self.current_entities[i] in self.end_entities[i // self.num_rollouts]
                for i in range(self.current_entities.shape[0])
            ])
        else:
            return self.current_entities == self.end_entities
    
    def _reward_to_hit_answer(self, reward: np.ndarray) -> np.ndarray:
        """
        Helper function to determine which rollouts have hit the answer based on the reward array.
        Returns a boolean array indicating which rollouts have received a positive reward (i.e., hit an answer).
        Used internally for reward adjustment when using STOP signals.
        """
        return reward == self.positive_reward

    def adjust_rewards(
            self,
            reward: np.ndarray, 
            stop_bonus: float, 
            stop_penalty: float, 
            length_penalty: float,
            hit_mask: Optional[np.ndarray] = None
        ) -> np.ndarray:
        """
        If using a STOP signal, we want to provide a small bonus for correctly stopping at an answer and a small penalty for incorrectly stopping at a non-answer.
        This function adjusts the reward array to include these bonuses/penalties based on the stopped_mask and whether the current entities are correct answers.
        
        Args:
            reward: Original reward array [batch_size*total_rollouts] before adjustment
            stop_bonus: Bonus to apply for correctly stopping at an answer
            stop_penalty: Penalty to apply for incorrectly stopping at a non-answer
            length_penalty: Penalty to apply based on the length of the episode
            hit_mask: Optional boolean array indicating which rollouts have hit an answer (if not provided, it will be computed from the reward array)
        Returns:
            Adjusted reward array [batch_size*total_rollouts] after applying STOP signal bonuses/penalties
        """
        if self.use_stop_signal:
            if hit_mask is None: hit_mask = self._reward_to_hit_answer(reward)  # which rollouts have hit an answer based on original reward
            correct_stop_mask = hit_mask & self.stopped_mask
            incorrect_stop_mask = (~hit_mask) & self.stopped_mask

            # ---- length cost in [0,1], 0 = earliest stop, 1 = latest / no stop ----
            denom = max(1, int(self.current_hop) - 1)
            step_cost = (self.stop_steps - 1) / denom  # shape [N], float in [0,1]

            # apply length penalty to all rollouts (no-stop should have cost ~1)
            if length_penalty > 0:
                reward -= length_penalty * step_cost
                
            # STOP bonuses/penalties
            reward[correct_stop_mask] += stop_bonus
            reward[incorrect_stop_mask] -= stop_penalty

        return reward

    def get_multi_answer_coverage(self) -> Tuple[float, float, float]:
        """
        Calculate multi-answer coverage metrics (recall, precision, F1) for the last nodes reached by all rollouts in the current batch.
        Used to evaluate how well the agent's final positions cover the set of correct answers when multiple answers are possible.
        Returns:
            precision: Average precision of predicted answers vs gold answers across the batch
            recall: Average recall of predicted answers vs gold answers across the batch
            f1_score: Average F1 score of predicted answers vs gold answers across the batch
        Note:
            - Only applicable if multi_answers is True and end_entities are provided as sets of answers
            - Computes metrics by comparing the unique set of final entities reached by all rollouts for each question against the set of correct answer entities
            - Provides insight into how well the agent is covering the answer space when multiple correct answers exist
        """
        recall = np.zeros(self.no_examples, dtype=np.float32)
        precision = np.zeros(self.no_examples, dtype=np.float32)
        f1_score = np.zeros(self.no_examples, dtype=np.float32)

        if self.multi_answers:
            for i0 in range(self.no_examples):
                rollout_ends = self.current_entities[i0*self.num_rollouts:(i0+1)*self.num_rollouts]
                current_answers = set(rollout_ends)          # unique predicted endpoints
                correct_answers = self.end_entities[i0]      # set of gold endpoints

                tp = len(current_answers & correct_answers)

                recall[i0] = tp / (len(correct_answers) + 1e-8)  # unique answer coverage (recall)
                precision[i0] = tp / (len(current_answers) + 1e-8)
                f1_score[i0] = 2 * tp / (len(current_answers) + len(correct_answers) + 1e-8)

        return precision, recall, f1_score

    def canon_edge(self, h: int, r: int, t: int) -> Tuple[int, int, int]:
        """
        Convert an edge to its canonical form (head, relation, tail).
        If the relation is an inverse token, swap head and tail and map relation back to original.
        Returns:
            (h, r, t): Canonical edge representation
        """
        if r in self.grapher.inverse_tokens:
            r = self.grapher.inverse_mapping[r]
            h, t = t, h
        return (h, r, t)
    
    def is_inverse_rel(self, r_prev: int, r_cur: int) -> bool:
        # True if r_cur is the inverse token of r_prev OR r_prev is inverse token of r_cur
        # (assumes inverse_mapping is inverse_token -> base_relation)
        if r_cur in self.grapher.inverse_tokens and self.grapher.inverse_mapping.get(r_cur) == r_prev:
            return True
        if r_prev in self.grapher.inverse_tokens and self.grapher.inverse_mapping.get(r_prev) == r_cur:
            return True
        return False

    def get_subgraph_overlap(self, pred_path: List[List[int]], idx) -> Tuple[float, float, float]:
        """
        Calculate the subgraph overlap between the predicted path and the ground-truth path for a given question index.
        DO NOT USE AS A REWARD SIGNAL.

        - Edges are compared in a permutation-invariant way, so the order of edges in the path does not affect the score. 
            This allows for more flexible evaluation of reasoning paths that may take different orders but still cover 
            the same underlying subgraph.
        - No-op, restarts, and stop signals are ignored in the evaluation since they are not part of 
            the original graph and do not represent meaningful reasoning steps.
        - Both paths are compared as sets of edges, so order and multiplicity
            of edges do not affect the score.
        Returns:
            precision: Proportion of predicted edges that are in the ground-truth path
            recall: Proportion of ground-truth edges that are in the predicted path
            f1_score: Harmonic mean of precision and recall
        """
        assert self.paths_exists, "No ground-truth paths available for faithfulness evaluation!"
        gt_path = self.paths[idx]

        # convert to a set of edges for easier comparison, edge-based
        pred_edges = set(
            self.canon_edge(h, r, t)  # map inverse tokens back to their original relation for evaluation purposes (e.g. _relation -> relation)
            for h, r, t in pred_path 
            if r not in self.cycle_tokens   #   remove cycles and stop/restart signals
        )
        gt_edges = set((h, r, t) for h, r, t in gt_path)

        tp = len(pred_edges & gt_edges)
        fp = len(pred_edges - gt_edges)
        fn = len(gt_edges - pred_edges)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1_score = 2 * precision * recall / (precision + recall + 1e-8)
        return precision, recall, f1_score

    def get_path_edit_distance(self, pred_path: List[List[int]], idx) -> int:
        """
        Calculate the edit distance between the predicted path and the ground-truth path for a given question index.
        Edit distance is defined as the minimum number of edge insertions, deletions, or substitutions required to transform the predicted path into the ground-truth path.
        Returns:
            edit_distance: Integer representing the edit distance between the predicted and ground-truth paths
        Note:
            - This is a more strict metric than path faithfulness, as it considers the order and multiplicity of edges.
            - No-op, restarts, and stop signals are ignored in the evaluation since they are not part of the original graph and do not represent meaningful reasoning steps.
            - Useful for evaluating how closely the agent's reasoning path matches the exact ground-truth path, but should not be used as a reward signal due to its strictness and potential sparsity.
        """
        assert self.paths_exists, "No ground-truth paths available for edit distance evaluation!"
        gt_path = self.paths[idx]

        # Filter out no-op, restart, and stop signals from both paths
        pred_path = [self.canon_edge(h, r, t) for h, r, t in pred_path if r not in self.cycle_tokens]
        gt_path = [(h, r, t) for h, r, t in gt_path]

        # Create a matrix to compute edit distance using dynamic programming
        m = len(pred_path)
        n = len(gt_path)
        dp = np.zeros((m + 1, n + 1), dtype=int)

        for i0 in range(m + 1):
            dp[i0][0] = i0  # Deletion cost
        for j0 in range(n + 1):
            dp[0][j0] = j0  # Insertion cost

        for i0 in range(1, m + 1):
            for j0 in range(1, n + 1):
                if pred_path[i0 - 1] == gt_path[j0 - 1]:
                    dp[i0][j0] = dp[i0 - 1][j0 - 1]  # No cost if edges match
                else:
                    dp[i0][j0] = min(
                        dp[i0 - 1][j0] + 1,    # Deletion
                        dp[i0][j0 - 1] + 1,    # Insertion
                        dp[i0 - 1][j0 - 1] + 1 # Substitution
                    )
        edit_distance = dp[m][n]
        return edit_distance/(max(m, n) + 1e-8)  # normalize by path length to get a score between 0 and 1

    def get_node_coverage(self, pred_entities: List[int], idx) -> Tuple[float, float, float]:
        """
        Calculate permutation-invariant node-based Path Faithfulness between
        predicted and ground-truth path for a given question index.
        """

        assert self.paths_exists, "No ground-truth paths available for faithfulness evaluation!"
        gt_path = self.paths[idx]

        pred_nodes = set(pred_entities)
        gt_nodes = set(t for _, _, t in gt_path) | set(h for h, _, _ in gt_path)  # include start entity from gt path

        tp = len(pred_nodes & gt_nodes)
        fp = len(pred_nodes - gt_nodes)
        fn = len(gt_nodes - pred_nodes)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1_score = 2 * precision * recall / (precision + recall + 1e-8)
        return precision, recall, f1_score

    def get_relation_coverage(self, pred_relations: List[int], idx) -> Tuple[float, float, float]:
        """
        Calculate permutation-invariant relation-based Path Faithfulness between
        predicted and ground-truth path for a given question index.
        Returns:
            precision: Proportion of predicted edges that are in the ground-truth path
            recall: Proportion of ground-truth edges that are in the predicted path
            f1_score: Harmonic mean of precision and recall
        """
        assert self.paths_exists, "No ground-truth paths available for faithfulness evaluation!"
        gt_path = self.paths[idx]

        pred_rels = set(
            self.grapher.inverse_mapping.get(r, r)      # map inverse tokens back to their original relation for evaluation purposes (e.g. _relation -> relation)
            for r in pred_relations
            if r not in self.cycle_tokens               #   remove cycles and stop/restart signals
        )
        gt_rels = set(r for _, r, _ in gt_path)

        tp = len(pred_rels & gt_rels)
        fp = len(pred_rels - gt_rels)
        fn = len(gt_rels - pred_rels)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1_score = 2 * precision * recall / (precision + recall + 1e-8)
        return precision, recall, f1_score

    def get_reasoning_diagnostic(self, pred_path: List[List[int]]) -> Tuple[float, float, float, float, float]:
        """
        Compute diagnostic metrics for a predicted reasoning path.

        This function evaluates the quality of a predicted reasoning path by calculating the following metrics:
        
        - **Invalid Step Rate**: Fraction of steps that are "special" (e.g., restart, stop, no-op) or involve actions that do not correspond to a valid knowledge graph edge.
        - **Cycle Ratio**: Fraction of steps that revisit an entity already visited in the same rollout.
        - **Backtrack Ratio**: Fraction of steps that reverse the immediately previous relation (i.e., backtracking).
        - **Effective Path Length**: Number of unique knowledge graph edges traversed, excluding special tokens.
        - **Redundancy**: Measures the proportion of redundant edges in the path, calculated as \(1 - \frac{\text{unique\_edges}}{H}\), where \(H\) is the total number of non-invalid steps.

        Args:
            pred_path (List[List[int]]): The predicted reasoning path, where each step is represented as a tuple (head, relation, tail).

        Returns:
            Tuple[float, float, float, float, float]: A tuple containing the following diagnostic metrics:
                - invalid_step_rate (float): Fraction of invalid steps in the path.
                - cycle_rate (float): Fraction of steps revisiting previously visited entities.
                - backtrack_rate (float): Fraction of steps that backtrack to the previous entity.
                - unique_edges (float): Number of unique edges traversed in the path.
                - redundancy (float): Measure of redundant edges in the path.

        Note:
            - Special tokens (e.g., NO_OP, STOP, RESTART) are ignored in the calculation of effective path length and redundancy.
            - These metrics are intended for diagnostic purposes and should not be used as reward signals during training.
        """
        invalid_steps = 0
        cycle_steps = 0
        backtrack_steps = 0
        non_invalid_steps = 0

        visited_nodes: Set[int] = {pred_path[0][0]}
        unique_edge_set: Set[Tuple[int, int, int]] = set()

        for i0, edge in enumerate(pred_path):
            h, r, t = edge

            # special cycle token (e.g. NO_OP, STOP, RESTART) do not represent meaningful reasoning steps.
            invalid = r in self.cycle_tokens
            if invalid:
                invalid_steps += 1
                continue

            non_invalid_steps += 1

            # cycle: next node already visited
            if t in visited_nodes:
                cycle_steps += 1

            # backtrack: go back to entity_{i-1} via inverse of previous relation
            if i0 >= 1:
                if (t == pred_path[i0 - 1][0]) and self.is_inverse_rel(pred_path[i0 - 1][1], r):
                    backtrack_steps += 1

            visited_nodes.add(t)
            unique_edge_set.add(self.canon_edge(h, r, t))
        
        invalid_step_rate = invalid_steps / (len(pred_path) + 1e-8)
        cycle_rate = cycle_steps / (non_invalid_steps + 1e-8)
        backtrack_rate = backtrack_steps / (non_invalid_steps + 1e-8)

        unique_edges = float(len(unique_edge_set))
        redundancy = 1.0 - (unique_edges / (non_invalid_steps + 1e-8))
        return invalid_step_rate, cycle_rate, backtrack_rate, unique_edges, redundancy

    def process_restart_actions(self, next_actions: np.ndarray) -> np.ndarray:
        """
        If using a RESTART signal, we want to ensure that when an agent selects the RESTART action, it transitions back to the start entity in the next step.
        This function modifies the next_actions array to enforce this constraint based on the restart_mask. If stop signal is used, the restart action will disappear for stopped agents,
        so we only apply the restart logic to non-stopped agents.

        Args:
            next_actions: The original next actions array [total_rollouts, max_actions, 2] containing entity and relation IDs for each possible action
        Returns:
            Modified next_actions array where the RESTART action leads back to the start entity for that agent.
        """
        restart_mask = (next_actions[:, :, 1] == self.grapher.rRESTART) # Mask to identify which actions are RESTART actions
        
        if not np.any(restart_mask):
            return next_actions  # No RESTART actions, return original next_actions
        
        rollout_idx, action_idx = np.where(restart_mask) # Get indices of RESTART actions (not all agents may have a RESTART action, so we check if any exist first)

        next_actions[rollout_idx, action_idx, 0] = self.start_entities[rollout_idx] # Set the next entity for RESTART actions to the start entity for that rollout
        return next_actions
    
    def process_stop_actions(self, next_actions: np.ndarray) -> np.ndarray:
        """
        If using a STOP signal, we want to ensure that once an agent selects the STOP action,
        it cannot transition to any new entity in subsequent steps.
            - We can achieve this by masking out all other actions except the STOP action for that agent in future steps.
        
        This function modifies the next_actions array to enforce this constraint based on the stopped_mask.
        
        Args:
            next_actions: The original next actions array [total_rollouts, max_actions, 2] containing entity and relation IDs for each possible action
        
        Returns:
            Modified next_actions array where agents that have selected STOP can only select the STOP action (which keeps them at the same entity) and all other actions are masked out.
        """
        for i0 in range(len(self.stopped_mask)):
            if self.stopped_mask[i0]:
                next_actions[i0, :, 0] = self.current_entities[i0]      # Stay at current entity
                next_actions[i0, :, 1] = self.grapher.rPAD              # Mask everything
                next_actions[i0, 0, 0] = self.current_entities[i0]      # Except STOP
                next_actions[i0, 0, 1] = self.grapher.rSTOP
        return next_actions

    def __call__(self, action: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Execute an action step in the knowledge graph reasoning environment.
        
        Takes the agent's selected actions and transitions all rollouts to new
        entity positions. Updates the environment state with new current positions
        and computes the next available actions from those positions.
        
        Args:
            action: Selected action indices [batch_size*total_rollouts] indicating which
                   next_entity/next_relation pair to follow
                   
        Returns:
            Updated state dictionary containing:
                - 'current_entities': New current entity positions
                - 'next_entities': Available next entities from new positions
                - 'next_relations': Available relations from new positions
                
        Note:
            - Increments the current reasoning step counter
            - Uses action indices to select from available next_entities
            - Dynamically computes new action spaces using the grapher (no masking)
            - State is automatically updated for the next reasoning step
        """
        self.current_hop += 1
        self.current_entities = self.state['next_entities'][np.arange(self.no_examples*self.num_rollouts), action]

        if self.use_stop_signal: 
            selected_relations = self.state['next_relations'][np.arange(self.no_examples*self.num_rollouts), action]

            stop_action_mask = (selected_relations == self.grapher.rSTOP)  # Identify which agents have selected the STOP action
            newly_stopped = stop_action_mask & (~self.stopped_mask)        # Identify which agents are newly stopped in this step (i.e., selected STOP now but were not previously stopped)
            
            self.stop_steps[newly_stopped] = self.current_hop              # Update stop_steps for newly stopped agents

            self.stopped_mask = np.logical_or(self.stopped_mask, stop_action_mask) # Update stopped_mask to include newly stopped agents (once an agent is stopped, it remains stopped)

        # Update state with new actions from new positions
        next_actions = self.grapher.return_next_raw_actions(self.current_entities)

        if self.use_stop_signal: next_actions = self.process_stop_actions(next_actions)            
        if self.use_restart_signal: next_actions = self.process_restart_actions(next_actions)

        self.state['next_relations'] = next_actions[:, :, 1]
        self.state['next_entities'] = next_actions[:, :, 0]
        self.state['current_entities'] = self.current_entities
        return self.state

    def get_effective_path_length(self) -> np.ndarray:
        """
        Get the effective path length for each rollout in the current batch.
        
        Effective path length is defined as the number of steps taken until the first STOP action is selected (if using STOP signal), or the total number of steps taken if not using STOP signal.
        This metric provides insight into how long the agent's reasoning paths are, and can be used for analysis or diagnostic purposes.
        
        Returns:
            effective_path_lengths: Array [batch_size*total_rollouts] containing the effective path length for each rollout
        """
        return np.clip(self.stop_steps, a_min=0, a_max=self.path_len)
        # return self.stop_steps


class EnvNLQ(object):
    """
    Natural Language Query (NLQ) environment for reinforcement learning agents.
    
    Manages the complete reinforcement learning setup for knowledge graph reasoning
    with natural language questions. Combines knowledge graph navigation with 
    question understanding to train agents that can answer complex queries by
    traversing multi-hop reasoning paths.
    
    This environment serves as the main interface between RL agents and the
    knowledge graph reasoning task, providing batch processing capabilities
    and episodic interaction patterns suitable for policy gradient training.
    
    Attributes:
        grapher: Knowledge graph navigator for action space management
        batch_loader: Data generator for question/answer batches
        mode: Current operation mode ('train', 'dev', or 'test')
        embedding_server: Service for generating question embeddings
        
    Example:
        >>> env = EnvNLQ(
        ...     batch_size=128, num_rollouts=20, positive_reward=1.0, negative_reward=-1.0,
        ...     path_length=3, test_rollouts=100, data_input_dir="./data", 
        ...     question_tokenizer_name="bert-base", cached_QAMetaData_path="./cache",
        ...     raw_QAData_path="./raw", max_num_actions=200,
        ...     entity_vocab=entity_vocab, relation_vocab=relation_vocab, mode='train'
        ... )
        >>> episode = env.get_episodes()
        >>> state = episode.get_state()
        >>> new_state = episode(action)
        >>> reward = episode.get_reward()
    """
    def __init__(
        self, 
        batch_size: int,
        test_batch_size: int,
        num_rollouts: int,
        test_rollouts: int,
        positive_reward: float,
        negative_reward: float,
        path_length: int,
        data_input_dir: str,
        question_tokenizer_name: str,
        question_format: str,
        cached_QAMetaData_path: str,
        raw_QAData_path: str,
        max_num_actions: int,
        entity_vocab: Dict[str, int], 
        relation_vocab: Dict[str, int], 
        mode: str = 'train', 
        multi_answers: bool = False,
        use_full_graph: bool = False,
        use_stop_signal: bool = False,
        use_restart_signal: bool = False,
        seed: Optional[int] = None,
        embedding_server: Optional[EmbeddingServer] = None
    ) -> None:
        """
        Initialize the NLQ environment with knowledge graph and data processing components.
        
        Sets up the complete environment infrastructure including knowledge graph
        navigation, question batch processing, and embedding generation. Creates
        the foundation for episodic reinforcement learning interactions.
        
        Args:
            batch_size: Number of questions per training batch
            num_rollouts: Number of training rollouts per question
            positive_reward: Reward value for reaching correct answer entities
            negative_reward: Reward value for incorrect or no answer
            path_length: Maximum number of reasoning steps allowed per episode
            test_rollouts: Number of evaluation rollouts per question
            data_input_dir: Directory containing knowledge graph and question data
            question_tokenizer_name: Name/path of tokenizer for question processing
            question_format: Format of the question input ('full_text', 'relation_only', 'graph_only')
            cached_QAMetaData_path: Path to cached question-answer metadata
            raw_QAData_path: Path to raw question-answer data files
            max_num_actions: Maximum number of actions/relations per entity
            entity_vocab: Mapping from entity names to unique integer IDs
            relation_vocab: Mapping from relation names to unique integer IDs  
            mode: Operation mode - 'train' for training, 'dev'/'test' for evaluation
            multi_answers: Whether to handle questions with multiple correct answers
            use_full_graph: Whether to use the full graph (including test/dev triples) or only training graph
            use_stop_signal: Whether to include a STOP action in the action space
            use_restart_signal: Whether to include a RESTART action in the action space
            seed: Optional seed for random number generation
            embedding_server: Optional service for generating question embeddings
                             from natural language text
                             
        Note:
            - Creates RelationEntityGrapher for knowledge graph navigation
            - Initializes QuestionBatcher for efficient batch processing
            - Stores vocabularies for entity/relation ID conversion
            - Embedding server enables dynamic question encoding
        """
        self.batch_size = batch_size
        self.num_rollouts = num_rollouts
        self.positive_reward = positive_reward
        self.negative_reward = negative_reward
        self.mode = mode
        self.path_len = path_length
        self.test_batch_size = test_batch_size
        self.test_rollouts = test_rollouts
        self.multi_answers = multi_answers
        self.no_op_id = relation_vocab['NO_OP']
        input_dir = data_input_dir

        self.batcher = QuestionBatcher(
            input_dir=input_dir,
            batch_size=self.batch_size,
            test_batch_size=self.test_batch_size,
            question_tokenizer_name=question_tokenizer_name,
            cached_QAMetaData_path=cached_QAMetaData_path,
            question_format=question_format,
            multi_answers=multi_answers,
            raw_QAData_path=raw_QAData_path,
            force_data_prepro=False,
            mode=self.mode,
            seed=seed,
            embedding_server=embedding_server,
        )
        self.paths_exists = self.batcher.path_exists

        self.total_no_examples = self.batcher.get_question_num()
        self.token_embedding_dim = self.batcher.get_embedding_dim()

        graph_path = os.path.join(input_dir, 'full_graph.txt') if use_full_graph else os.path.join(input_dir, 'graph.txt')

        # Initialize the knowledge graph
        self.grapher = RelationEntityGrapher(
            triple_store=graph_path,
            max_num_actions=max_num_actions,
            entity_vocab=entity_vocab,
            relation_vocab=relation_vocab,
            use_stop_signal=use_stop_signal,
            use_restart_signal=use_restart_signal
        )

    def get_episodes(self) -> Generator[EpisodeNLQ, None, None]:
        """
        Generate episodes for reinforcement learning training or evaluation.
        
        Creates a generator that yields EpisodeNLQ instances for each batch
        of questions. Each episode encapsulates the complete state and interaction
        interface needed for RL agents to perform knowledge graph reasoning.
        
        Yields:
            EpisodeNLQ instances containing:
                - Initialized environment state for the batch
                - Question embeddings and target entities  
                - Knowledge graph navigation interface
                - Reward computation capabilities
                
        Note:
            - Training mode yields batches continuously for epoch-based training
            - Evaluation modes (dev/test) yield finite batches then terminate
            - Each episode supports multiple rollouts for variance reduction
            - Episodes are automatically configured with current environment parameters
        """
        if self.mode == 'train':
            for data in self.batcher.yield_next_batch_train():
                question_tokens, question_embeddings, start_entities, end_entities, paths = data
                yield EpisodeNLQ(
                    self.grapher, 
                    question_tokens,
                    question_embeddings,
                    start_entities,
                    end_entities,
                    batch_size=self.batch_size,
                    path_len=self.path_len,
                    num_rollouts=self.num_rollouts,
                    positive_reward=self.positive_reward,
                    negative_reward=self.negative_reward,
                    mode=self.mode,
                    multi_answers=self.multi_answers,
                    paths=paths,
                )
        else:
            for data in self.batcher.yield_next_batch_test():
                if data == None:
                    return
                question_tokens, question_embeddings, start_entities, end_entities, paths = data
                yield EpisodeNLQ(
                    self.grapher, 
                    question_tokens,
                    question_embeddings,
                    start_entities,
                    end_entities,
                    batch_size=self.test_batch_size,
                    path_len=self.path_len,
                    num_rollouts=self.test_rollouts,
                    positive_reward=self.positive_reward,
                    negative_reward=self.negative_reward,
                    mode=self.mode,
                    multi_answers=self.multi_answers,
                    paths=paths,
                )

    def change_mode(self, mode: str) -> None:
        """
        Switch the environment between training and evaluation modes.
        
        Changes the operational mode of both the environment and its batch
        generator, affecting data sampling patterns and rollout behavior.
        Training mode enables continuous batch generation while evaluation
        modes process finite datasets.
        
        Args:
            mode: Target operation mode - 'train', 'dev', or 'test'
            
        Raises:
            AssertionError: If mode is not one of the valid options
            
        Note:
            - Training mode uses different rollout counts than evaluation
            - Mode change propagates to the underlying batch generator
            - Affects episode generation behavior in get_episodes()
        """
        assert mode in ['train', 'dev', 'test'], f"Error! Invalid mode: {mode}"
        self.mode = mode
        self.batcher.set_mode(mode)
        self.total_no_examples = self.batcher.get_question_num()
    
    def change_test_rollouts(self, test_rollouts: int) -> None:
        """
        Update the number of rollouts used during evaluation/testing.
        
        Modifies the rollout count for evaluation modes (dev/test) to control
        the amount of exploration during inference. Higher rollout counts
        improve answer accuracy through increased exploration but require
        more computational resources.
        
        Args:
            test_rollouts: Number of rollouts to use per question during evaluation.
                          Typical values range from 1 (fast inference) to 100+
                          (thorough exploration for maximum accuracy)
                          
        Note:
            - Only affects evaluation modes, training rollouts remain unchanged
            - Higher values improve accuracy but increase inference time
            - Used for ablation studies and performance tuning
            - Takes effect for subsequent episodes generated after this call
        """
        self.test_rollouts = test_rollouts

    def check_paths_exist(self) -> bool:
        """
        Check if ground-truth paths are available for faithfulness evaluation.
        
        Returns whether the current batch of questions includes ground-truth
        reasoning paths. This information is used to determine if path-based
        evaluation metrics can be computed.
        
        Returns:
            True if ground-truth paths are available, False otherwise
        """
        return self.paths_exists