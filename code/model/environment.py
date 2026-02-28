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
from code.data.feed_data import QuestionBatcher
from code.data.grapher import RelationEntityGrapher

from typing import Any, Dict, Generator, List, Optional, Set, Union, Tuple, Literal, Sequence

logger = logging.getLogger()
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

SegmentPolicy = Literal["raw", "truncate_at_stop", "final_segment", "final_segment_truncate"]

class EpisodeNLQ(object):
    """
    Single episode manager for natural language question (NLQ) multi-hop reasoning.

    Manages the state and dynamics of a multi-hop reasoning episode where an RL agent
    navigates a knowledge graph to answer a question. Supports multiple rollouts per
    question (by repeating the batch) to improve exploration and sample efficiency.

    State includes current entities and the next-step action space (next entities and
    relations). Rewards are based on whether the rollout ends at a correct answer.

    Attributes:
        grapher (RelationEntityGrapher): Knowledge graph navigator
        batch_size (int): Number of questions in the batch (unique questions)
        path_len (int): Maximum number of reasoning steps allowed
        num_rollouts (int): Number of rollouts per question
        mode (str): Current mode ('train', 'dev', or 'test')
        no_examples (int): Number of unique questions (same as batch_size)
        start_entities (np.ndarray): Starting entity IDs repeated per rollout
        end_entities:
            - If multi_answers=False: np.ndarray of target entity IDs repeated per rollout
            - If multi_answers=True: List[Set[int]] of gold answer sets (one set per question)
        question_embeddings (np.ndarray): Question embeddings repeated per rollout
        question_tokens (List[str]): Question tokens/text for analysis/logging (not used for transitions)
        paths (Optional[List[List[Tuple[int,int,int]]]]): Optional ground-truth edge paths (one per question)
        state (Dict[str, np.ndarray]): Current environment state

    Example:
        >>> episode = EpisodeNLQ(
        ...     grapher, question_tokens, question_embeddings, start_entities, end_entities,
        ...     batch_size=128, path_len=3, num_rollouts=20,
        ...     positive_reward=1.0, negative_reward=0.0, mode='train',
        ...     multi_answers=False, paths=paths
        ... )
        >>> state = episode.get_state()
        >>> state = episode(action_indices)
        >>> reward = episode.get_reward()
    """

    # 1) Lifecycle & main interface
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
        paths: Optional[List[List[Tuple[int, int, int]]]] = None,
    ) -> None:
        """
        Initialize a reinforcement learning episode for knowledge graph reasoning.

        Sets up the environment state for multi-path exploration in knowledge graphs,
        where the agent learns to navigate from start entities to target entities by
        following relation edges. Supports batch processing with multiple rollouts.

        Args:
            graph: Knowledge graph navigator providing action spaces and transitions.
            question_tokens: Question tokens/text (kept for analysis/logging).
            question_embeddings: Question embeddings [batch_size, embedding_dim].
            start_entities: Starting entity IDs [batch_size].
            end_entities:
                - If multi_answers=False: target answer entity IDs [batch_size].
                - If multi_answers=True: list of answer-ID lists (one list per question).
            batch_size: Number of questions in batch.
            path_len: Maximum reasoning steps allowed.
            num_rollouts: Number of rollouts per question.
            positive_reward: Reward for correct answers.
            negative_reward: Reward for incorrect answers.
            mode: Current mode ('train', 'dev', or 'test').
            multi_answers: Whether each question can have multiple correct answers.
            paths: Optional ground-truth paths (one per question), where each path is a list
                of edges (head, relation, tail) using integer IDs.

        Note:
            - Internally repeats start_entities/question_embeddings for multiple rollouts.
            - For multi_answers=True, end_entities are converted to sets for fast membership tests.
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
        self.special_tokens = set([self.grapher.rNO_OP, self.grapher.rSTOP, self.grapher.rRESTART])  # if using stop/restart signals, we want to ignore them in path faithfulness evaluation since they are not part of the original graph

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
        self.restarted_mask = np.zeros(self.current_entities.shape[0], dtype=bool)
        self.stop_steps = np.full(self.current_entities.shape[0], fill_value=self.path_len, dtype=int)  # track at which step each rollout stopped, initialized to max path length (i.e. not stopped)

        # Initialize state with available actions from starting positions
        next_actions = self.grapher.return_next_raw_actions(self.current_entities)

        if self.use_restart_signal: next_actions = self.process_restart_actions(next_actions)

        self.state = {}                                                         # RL states (next_relations, next_entities, current_entities)
        self.state['next_relations'] = next_actions[:, :, 1]
        self.state['next_entities'] = next_actions[:, :, 0]
        self.state['current_entities'] = self.current_entities

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
            
            self.stop_steps[newly_stopped] = self.current_hop - 1           # Update stop_steps for newly stopped agents

            self.stopped_mask = np.logical_or(self.stopped_mask, stop_action_mask) # Update stopped_mask to include newly stopped agents (once an agent is stopped, it remains stopped)
        if self.use_restart_signal:            
            selected_relations = self.state['next_relations'][np.arange(self.no_examples*self.num_rollouts), action]
            restart_action_mask = (selected_relations == self.grapher.rRESTART)
            self.restarted_mask = np.logical_or(self.restarted_mask, restart_action_mask)

        # Update state with new actions from new positions
        next_actions = self.grapher.return_next_raw_actions(self.current_entities)

        if self.use_stop_signal: next_actions = self.process_stop_actions(next_actions)            
        if self.use_restart_signal: next_actions = self.process_restart_actions(next_actions)

        self.state['next_relations'] = next_actions[:, :, 1]
        self.state['next_entities'] = next_actions[:, :, 0]
        self.state['current_entities'] = self.current_entities
        return self.state

    # 2) State & observations
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

    def get_termination_step(self, clipped: bool = True) -> np.ndarray:
        """
        Get the termination step index for each rollout.

        If STOP is used, we track the (0-indexed) step at which STOP was first selected.
        Rollouts that never STOP are initialized to path_len (i.e. max path length) and 
        can be optionally clipped to path_len-1 to indicate "no stop" while keeping the 
        metric in [0, path_len-1].

        Returns:
            Array [batch_size * num_rollouts] containing a value in [0, path_len-1] per rollout:
                - 0 means STOP immediately (at the first step)
                - path_len-1 means STOP very late or never STOP (default / clipped)
        """
        if clipped: return np.clip(self.stop_steps, a_min=0, a_max=self.path_len - 1)
        return self.stop_steps

    # 3) Question encoding
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

    # 4) Reward computation & shaping
    # 4-a) Core reward
    def get_reward(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate hits and reward signal for the current state of all batches and rollouts.
        
        Computes binary rewards based on whether agents have reached their
        target entities. Used to train the reinforcement learning policy
        to navigate toward correct answers in the knowledge graph.
        
        Returns:
            Reward array [batch_size*total_rollouts] containing:
                - positive_reward: For agents at target entities
                - negative_reward: For agents not at target entities
            Hit mask array [batch_size*total_rollouts] indicating which rollouts have hit an answer
                
        Note:
            - Rewards are computed by comparing current positions to target entities
            - Positive rewards encourage successful reasoning paths
            - Negative rewards discourage incorrect reasoning directions
            - Reward values are configured during environment initialization
        """
        if self.multi_answers:
            hit_mask = self._has_hit_answer()  # boolean array indicating which rollouts have hit an answer
        else:
            hit_mask = (self.current_entities == self.end_entities)

        reward = np.where(hit_mask, self.positive_reward, self.negative_reward)  # assign rewards based on hit_mask
        return reward, hit_mask
    
    # 4-b) Answer-hit logic (helpers)
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

    # 4-c) Post-processing / shaping
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
            denom = max(1, int(self.path_len))
            step_cost = (self.stop_steps) / denom  # shape [N], float in [0,1]

            # convert to float for reward adjustment calculations
            reward = reward.astype(np.float32)

            # apply length penalty to all rollouts (no-stop should have cost ~1)
            if length_penalty > 0:
                reward -= length_penalty * step_cost
                
            # STOP bonuses/penalties
            reward[correct_stop_mask] += stop_bonus
            reward[incorrect_stop_mask] -= stop_penalty

        return reward

    # 5) Special action handling
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

    # 6) Path utilities & normalization
    # 6-a) Path access
    def get_path(self, idx: int) -> Optional[List[Tuple[int, int, int]]]:
        """
        Get the ground-truth path for a given question index.

        Returns:
            A list of edges for the question, where each edge is a tuple (head, relation, tail)
            using integer IDs; or None if ground-truth paths are not available.
        """
        if self.paths_exists:
            return self.paths[idx]
        else:
            return None

    def get_path_length(self, idx: int) -> int:
        """
        Get the length of the ground-truth path for a given question index.

        Returns:
            The number of edges in the ground-truth path for the question; or 0 if paths are not available.
        """
        if self.paths_exists:
            return len(self.paths[idx])
        else:
            return 0
    
    # 6-b) Edge / relation helpers
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
        """
        Check if the current relation is the inverse of the previous relation (i.e., backtracking).
        Returns:
            True if r_cur is the inverse token of r_prev OR r_prev is the inverse token of r_cur; False otherwise.
        """
        # True if r_cur is the inverse token of r_prev OR r_prev is inverse token of r_cur
        # (assumes inverse_mapping is inverse_token -> base_relation)
        if r_cur in self.grapher.inverse_tokens and self.grapher.inverse_mapping.get(r_cur) == r_prev:
            return True
        if r_prev in self.grapher.inverse_tokens and self.grapher.inverse_mapping.get(r_prev) == r_cur:
            return True
        return False

    def clean_pred_path_for_eval(
        self,
        pred_path: Sequence[Tuple[int, int, int]],
        policy: SegmentPolicy = "final_segment_truncate",
    ) -> List[Tuple[int, int, int]]:
        """
        Normalize a logged rollout path for evaluation.

        Policies:
        - raw:                 return as-is (no filtering)
        - truncate_at_stop:    cut at first STOP (exclusive of STOP edge), keep earlier attempts
        - final_segment:       keep only edges after last RESTART, keep possible STOP trailing edges
        - final_segment_truncate: keep only edges after last RESTART and cut at first STOP

        Returns:
        A list of (h, r, t) edges, still containing KG edges + possibly special edges depending on policy.
        """
        if not pred_path:
            return []
        
        if policy == "raw":
            return list(pred_path)  # no filtering, return as-is

        # --- remove any NO_OP edges (if present) ---
        pred_path = [edge for edge in pred_path if edge[1] != self.grapher.rNO_OP]

        # --- find first STOP index (if any) ---
        stop_idx = None
        if self.use_stop_signal:
            for i0, (_, r, _) in enumerate(pred_path):
                if r == self.grapher.rSTOP:
                    stop_idx = i0
                    break

        # --- find last RESTART index (if any) ---
        last_restart_idx = -1
        if self.use_restart_signal:
            for i0, (_, r, _) in enumerate(pred_path):
                if r == self.grapher.rRESTART:
                    last_restart_idx = i0

        out = list(pred_path)

        if policy in ("truncate_at_stop", "final_segment_truncate") and stop_idx is not None:
            out = out[:stop_idx]  # exclude the STOP edge and anything after

        if policy in ("final_segment", "final_segment_truncate") and last_restart_idx >= 0:
            out = out[last_restart_idx + 1 :]  # keep only after last RESTART

        return out

    # 7) Metrics & diagnostics
    # 7-a) Answer-set metric
    def get_multi_answer_coverage(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate multi-answer coverage metrics (precision, recall, F1) per question.

        For each question i, we take the set of final entities reached by its rollouts and compare
        it against the set of gold answers (end_entities[i]). Metrics are computed per question.
        
        Used to evaluate how well the agent's final positions cover the set of correct answers when 
        multiple answers are possible.
        
        Returns:
            precision: np.ndarray [no_examples] precision per question
            recall: np.ndarray [no_examples] recall per question
            f1_score: np.ndarray [no_examples] F1 per question

        Note:
            - Only meaningful when multi_answers=True.
            - If multi_answers=False, returns zero arrays (no multi-answer set to compare against).
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
        else:
            for i0 in range(self.no_examples):
                rollout_ends = self.current_entities[i0*self.num_rollouts:(i0+1)*self.num_rollouts]
                current_answers = set(rollout_ends)                 # unique predicted endpoints
                correct_answer = {self.end_entities[i0*self.num_rollouts]}      # single gold endpoint

                tp = len(current_answers & correct_answer)  # assuming single correct answer in list

                recall[i0] = tp / 1  # either we covered the single correct answer or not
                precision[i0] = tp / (len(current_answers) + 1e-8)
                f1_score[i0] = 2 * tp / (len(current_answers) + 1 + 1e-8)

        return precision, recall, f1_score

    # 7-b) Path similarity metrics
    def get_subgraph_overlap(self, pred_path: List[List[int]], idx: int) -> Tuple[float, float, float]:
        """
        Calculate permutation-invariant edge-set overlap between predicted and ground-truth paths.

        Edges are compared as sets (order and multiplicity do not matter). Special tokens
        (e.g., NO_OP, STOP, RESTART) are ignored.

        Args:
            pred_path: Sequence of edges (h, r, t) using integer IDs (may include special tokens).
            idx: Question index into the ground-truth path list.

        Returns:
            precision: Fraction of predicted edges that appear in the ground-truth path.
            recall: Fraction of ground-truth edges recovered by the predicted path.
            f1_score: Harmonic mean of precision and recall.

        Note:
            DO NOT USE AS A REWARD SIGNAL.
        """
        assert self.paths_exists, "No ground-truth paths available for faithfulness evaluation!"
        gt_path = self.paths[idx]

        # convert to a set of edges for easier comparison, edge-based
        pred_edges = set(
            self.canon_edge(h, r, t)  # map inverse tokens back to their original relation for evaluation purposes (e.g. _relation -> relation)
            for h, r, t in pred_path 
            if r not in self.special_tokens   #   remove cycles and stop/restart signals
        )
        gt_edges = set((h, r, t) for h, r, t in gt_path)

        tp = len(pred_edges & gt_edges)
        fp = len(pred_edges - gt_edges)
        fn = len(gt_edges - pred_edges)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1_score = 2 * precision * recall / (precision + recall + 1e-8)
        return precision, recall, f1_score

    def get_path_edit_distance(self, pred_path: List[List[int]], idx: int) -> float:
        """
        Compute normalized edit distance between predicted and ground-truth paths.

        Edit distance is computed via dynamic programming over the edge sequences after filtering
        special tokens (NO_OP/STOP/RESTART). The returned value is normalized by max(m, n), so it
        lies in [0, 1], where 0 indicates an exact match.

        Args:
            pred_path: Sequence of edges (h, r, t) using integer IDs (may include special tokens).
            idx: Question index into the ground-truth path list.

        Returns:
            normalized_edit_distance (float): Edit distance / max(len(pred), len(gt)) in [0, 1].

        Note:
            - Stricter than set-based overlap because order matters.
            - Intended for analysis; typically too strict/sparse for reward shaping.
        """
        assert self.paths_exists, "No ground-truth paths available for edit distance evaluation!"
        gt_path = self.paths[idx]

        # Filter out no-op, restart, and stop signals from both paths
        pred_path = [self.canon_edge(h, r, t) for h, r, t in pred_path if r not in self.special_tokens]
        gt_path = [(h, r, t) for h, r, t in gt_path]

        # Create a matrix to compute edit distance using dynamic programming
        m = len(pred_path)
        n = len(gt_path)

        if m == 0 and n == 0:
            return 0.0
        if m == 0 or n == 0:
            return 1.0  # normalized by max(m,n)

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

    # 7-c) Coverage metrics
    def get_node_coverage(self, pred_entities: List[int], idx: int) -> Tuple[float, float, float]:
        """
        Compute permutation-invariant node coverage between predicted nodes and ground-truth path nodes.

        Args:
            pred_entities: List of visited entity IDs (duplicates allowed; evaluated as a set).
            idx: Question index into the ground-truth path list.

        Returns:
            precision: Fraction of predicted nodes that are in the ground-truth node set.
            recall: Fraction of ground-truth nodes that are present in the predicted node set.
            f1_score: Harmonic mean of precision and recall.

        Note:
            - Ground-truth node set includes both heads and tails from the ground-truth path.
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

    def get_relation_coverage(self, pred_relations: List[int], idx: int) -> Tuple[float, float, float]:
        """
        Compute permutation-invariant relation coverage between predicted and ground-truth relations.

        Predicted relations are evaluated as a set. Inverse relation tokens are mapped back to their
        original relation IDs for evaluation, and special tokens (NO_OP/STOP/RESTART) are ignored.

        Args:
            pred_relations: List of relation IDs used in the predicted rollout (duplicates allowed).
            idx: Question index into the ground-truth path list.

        Returns:
            precision: Fraction of predicted relations that appear in the ground-truth relation set.
            recall: Fraction of ground-truth relations recovered by the predicted relation set.
            f1_score: Harmonic mean of precision and recall.
        """
        assert self.paths_exists, "No ground-truth paths available for faithfulness evaluation!"
        gt_path = self.paths[idx]

        pred_rels = set(
            self.grapher.inverse_mapping.get(r, r)      # map inverse tokens back to their original relation for evaluation purposes (e.g. _relation -> relation)
            for r in pred_relations
            if r not in self.special_tokens               #   remove cycles and stop/restart signals
        )
        gt_rels = set(r for _, r, _ in gt_path)

        tp = len(pred_rels & gt_rels)
        fp = len(pred_rels - gt_rels)
        fn = len(gt_rels - pred_rels)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1_score = 2 * precision * recall / (precision + recall + 1e-8)
        return precision, recall, f1_score

    # 7-d) High-level analysis
    def get_reasoning_diagnostic(self, pred_path: List[List[int]]) -> Tuple[float, float, float, float, float, float, float]:
        """
        Compute diagnostic metrics for a predicted reasoning path.

        This function evaluates the quality of a predicted reasoning path by calculating the following metrics:
        
        - **Special Action Rate**: Fraction of steps that are "special" (e.g., restart, stop, no-op) or involve actions that do not correspond to a valid knowledge graph edge.
        - **Cycle Ratio**: Fraction of steps that revisit an entity already visited in the same rollout.
        - **Backtrack Ratio**: Fraction of steps that reverse the immediately previous relation (i.e., backtracking).
        - **Effective Path Length**: Number of unique knowledge graph edges traversed, excluding special tokens.
        - **Redundancy**: Measures the proportion of redundant edges in the path, calculated as \(1 - \frac{\text{unique\_edges}}{H}\), where \(H\) is the total number of non-special steps.
        - **Restart Rate**: Fraction of restart actions in the path.
        - **No-Op Rate**: Fraction of no-op actions in the path.
        Args:
            pred_path (List[List[int]]): The predicted reasoning path, where each step is represented as a tuple (head, relation, tail).

        Returns:
            Tuple[float, float, float, float, float, float, float]: A tuple containing the following diagnostic metrics:
                - special_action_rate (float): Fraction of special actions in the path.
                - cycle_rate (float): Fraction of steps revisiting previously visited entities.
                - backtrack_rate (float): Fraction of steps that backtrack to the previous entity.
                - unique_edges (float): Number of unique edges traversed in the path.
                - redundancy (float): Measure of redundant edges in the path.
                - restart_rate (float): Fraction of restart actions in the path.
                - no_op_rate (float): Fraction of no-op actions in the path.

        Note:
            - Special tokens (e.g., NO_OP, STOP, RESTART) are ignored in the calculation of effective path length and redundancy.
            - These metrics are intended for diagnostic purposes and should not be used as reward signals during training.
        """
        if not pred_path:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        special_steps = 0
        cycle_steps = 0
        backtrack_steps = 0
        non_special_steps = 0
        restart_steps = 0
        no_op_steps = 0

        visited_nodes: Set[int] = {pred_path[0][0]}
        unique_edge_set: Set[Tuple[int, int, int]] = set()

        for i0, (h, r, t) in enumerate(pred_path):
            # special cycle token (e.g. NO_OP, STOP, RESTART) do not represent meaningful reasoning steps.
            is_special = r in self.special_tokens
            if is_special:
                special_steps += 1
                if r == self.grapher.rRESTART:
                    restart_steps += 1
                elif r == self.grapher.rNO_OP:
                    no_op_steps += 1
                continue

            non_special_steps += 1

            # cycle: next node already visited
            if t in visited_nodes:
                cycle_steps += 1

            # backtrack: go back to entity_{i-1} via inverse of previous relation
            if i0 >= 1:
                if (t == pred_path[i0 - 1][0]) and self.is_inverse_rel(pred_path[i0 - 1][1], r):
                    backtrack_steps += 1

            visited_nodes.add(t)
            unique_edge_set.add(self.canon_edge(h, r, t))
        
        special_action_rate = special_steps / (len(pred_path) + 1e-8)
        restart_rate = restart_steps / (len(pred_path) + 1e-8)
        no_op_rate = no_op_steps / (len(pred_path) + 1e-8)

        cycle_rate = cycle_steps / (non_special_steps + 1e-8)
        backtrack_rate = backtrack_steps / (non_special_steps + 1e-8)

        unique_edges = float(len(unique_edge_set))
        redundancy = 1.0 - (unique_edges / (non_special_steps + 1e-8))
        
        return special_action_rate, cycle_rate, backtrack_rate, unique_edges, redundancy, restart_rate, no_op_rate

    def get_stop_quality(self, hit_mask: np.ndarray) -> Tuple[float, float, float, float]:
        """
        STOP quality diagnostics (rollout-level).

        Returns:
            stop_rate (float): P(stopped)
            correct_stop_rate (float): P(stopped and hit)
            incorrect_stop_rate (float): P(stopped and miss)
            hit_without_stop_rate (float): P(hit and not stopped)

        Note:
            - Only meaningful if use_stop_signal=True.
            - All rates are w.r.t. all rollouts (not conditioned), so they sum sensibly:
                stop_rate = correct_stop_rate + incorrect_stop_rate
        """
        if not self.use_stop_signal:
            return 0.0, 0.0, 0.0, 0.0

        hit_mask = hit_mask.astype(bool)
        stopped = self.stopped_mask.astype(bool)

        n = float(hit_mask.shape[0]) + 1e-8

        stop_rate = float(np.sum(stopped) / n)
        correct_stop_rate = float(np.sum(stopped & hit_mask) / n)
        incorrect_stop_rate = float(np.sum(stopped & ~hit_mask) / n)
        hit_without_stop_rate = float(np.sum(~stopped & hit_mask) / n)

        return stop_rate, correct_stop_rate, incorrect_stop_rate, hit_without_stop_rate

    def get_restart_quality(
        self,
        hit_mask: np.ndarray,
    ) -> Tuple[float, float, float]:
        """
        RESTART quality diagnostics (rollout-level).

        Args:
            hit_mask: boolean array [N] final success per rollout.

        Returns:
            restart_any_rate (float): P(ever_restarted)
            post_restart_success_rate (float): P(hit | ever_restarted)  (0 if none restarted)
            restart_and_hit_rate (float): P(ever_restarted and hit)

        Note:
            - Only meaningful if use_restart_signal=True.
        """
        if not self.use_restart_signal:
            return 0.0, 0.0, 0.0

        hit_mask = hit_mask.astype(bool)
        ever_restarted = self.restarted_mask.astype(bool)

        n = float(hit_mask.shape[0]) + 1e-8
        restart_any_rate = float(np.sum(ever_restarted) / n)
        restart_and_hit_rate = float(np.sum(ever_restarted & hit_mask) / n)

        denom = float(np.sum(ever_restarted)) + 1e-8
        post_restart_success_rate = float(np.sum(ever_restarted & hit_mask) / denom) if denom > 1e-7 else 0.0

        return restart_any_rate, post_restart_success_rate, restart_and_hit_rate

class EnvNLQ(object):
    """
    Natural Language Question (NLQ) environment for reinforcement learning agents.

    Manages the full NLQ KG-reasoning setup: batching questions/answers, building the
    knowledge-graph navigator, and yielding EpisodeNLQ objects for training/evaluation.

    Attributes:
        grapher (RelationEntityGrapher): Knowledge graph navigator / action-space provider
        batcher (QuestionBatcher): Batch generator for question/answer data
        mode (str): Current mode ('train', 'dev', or 'test')
        embedding_server (Optional[EmbeddingServer]): Optional embedding service used by the batcher

    Example:
        >>> env = EnvNLQ(...)
        >>> episode = next(env.get_episodes())
        >>> state = episode.get_state()
        >>> state = episode(action)
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
        use_directed_graph: bool = True,
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
            test_batch_size: Number of questions per evaluation batch (dev/test)
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
            use_directed_graph: Whether to treat the graph as directed (no inverse relations) or undirected (include inverse relations)
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
            use_restart_signal=use_restart_signal,
            use_directed_graph=use_directed_graph,
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
                question_tokens, question_embeddings, start_entities, end_entities, _, _ = data
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
                    paths=None,
                )
        else:
            for data in self.batcher.yield_next_batch_test():
                if data == None:
                    return
                question_tokens, question_embeddings, start_entities, end_entities, paths, ques_ids = data
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