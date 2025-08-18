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
    EpisodeNLQ: Manages a single reasoning episode with state tracking
    EnvNLQ: Main environment class coordinating episodes and data flow
"""

from __future__ import absolute_import
from __future__ import division

import os
import logging

import numpy as np

from code.data.embedding_server import EmbeddingServer
from code.data.feed_nlq_data import QuestionBatcher
from code.data.grapher import RelationEntityGrapher

from typing import Any, Dict, Generator, List, Optional, Tuple

logger = logging.getLogger()

class EpisodeNLQ:
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
        >>> episode = EpisodeNLQ(grapher, data_batch, episode_params)
        >>> state = episode.get_state()
        >>> new_state = episode(action_indices)
        >>> rewards = episode.get_reward()
    """

    def __init__(
        self, 
        graph: RelationEntityGrapher, 
        data: Tuple[List[str], np.ndarray, np.ndarray, np.ndarray], 
        params: Tuple[int, int, int, int, float, float, str]
    ) -> None:
        """
        Initialize a reinforcement learning episode for knowledge graph reasoning.
        
        Sets up the environment state for multi-path exploration in knowledge graphs,
        where the agent learns to navigate from start entities to target entities
        by following relation edges. Supports batch processing with multiple rollouts
        for improved training efficiency.
        
        Args:
            graph: Knowledge graph navigator providing action spaces and transitions
            data: Batch data tuple containing:
                - question_tokens: List of question token ints
                - question_embeddings: Question embeddings [batch_size, embedding_dim]
                - start_entities: Starting entity IDs [batch_size]
                - end_entities: Target answer entity IDs [batch_size]
            params: Episode configuration tuple containing:
                - batch_size: Number of questions in batch
                - path_len: Maximum reasoning steps allowed
                - num_rollouts: Number of training rollouts per question
                - test_rollouts: Number of evaluation rollouts per question
                - positive_reward: Reward for correct answers
                - negative_reward: Reward for incorrect answers
                - mode: Current mode ('train', 'dev', or 'test')
                
        Note:
            - Creates multiple rollouts by repeating each question/entity
            - Initializes state with available actions from starting positions
            - Supports different rollout counts for training vs evaluation
        """
        self.grapher = graph
        self.batch_size, self.path_len, num_rollouts, test_rollouts, positive_reward, negative_reward, mode = params
        self.mode = mode
        if self.mode == 'train':
            self.num_rollouts = num_rollouts
        else:
            self.num_rollouts = test_rollouts

        self.current_hop = 0
        q_tokens, question_embeddings, start_entities, end_entities = data
        self.no_examples = start_entities.shape[0]
        self.positive_reward = positive_reward
        self.negative_reward = negative_reward

        # Repeat entities/embeddings for multiple rollouts per question [batch_size,] -> [batch_size * num_rollouts]
        start_entities = np.repeat(start_entities, self.num_rollouts)
        end_entities = np.repeat(end_entities, self.num_rollouts)
        self.start_entities = start_entities
        self.end_entities = end_entities
        self.current_entities = np.array(start_entities)
        self.question_embeddings = np.repeat(question_embeddings, self.num_rollouts, axis=0) # [batch_size * num_rollouts, embedding_dim]
        self.question_tokens = q_tokens

        # Initialize state with available actions from starting positions
        next_actions = self.grapher.return_next_raw_actions(self.current_entities)

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
        reward = (self.current_entities == self.end_entities)
        condlist = [reward == True, reward == False]
        choicelist = [self.positive_reward, self.negative_reward]
        reward = np.select(condlist, choicelist)
        return reward

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

        # Update state with new actions from new positions
        next_actions = self.grapher.return_next_raw_actions(self.current_entities)
        self.state['next_relations'] = next_actions[:, :, 1]
        self.state['next_entities'] = next_actions[:, :, 0]
        self.state['current_entities'] = self.current_entities
        return self.state


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
        >>> env = EnvNLQ(params, entity_vocab, relation_vocab, mode='train')
        >>> episode = env.get_episodes()
        >>> state = episode.get_state()
        >>> new_state = episode(action)
        >>> reward = episode.get_reward()
    """
    def __init__(
        self, 
        params: Any, 
        entity_vocab: Dict[str, int], 
        relation_vocab: Dict[str, int], 
        mode: str = 'train', 
        embedding_server: Optional[EmbeddingServer] = None
    ) -> None:
        """
        Initialize the NLQ environment with knowledge graph and data processing components.
        
        Sets up the complete environment infrastructure including knowledge graph
        navigation, question batch processing, and embedding generation. Creates
        the foundation for episodic reinforcement learning interactions.
        
        Args:
            params: Configuration object containing environment parameters like
                   batch_size, path_length, rollout counts, and reward values
            entity_vocab: Mapping from entity names to unique integer IDs
            relation_vocab: Mapping from relation names to unique integer IDs  
            mode: Operation mode - 'train' for training, 'dev'/'test' for evaluation
            embedding_server: Optional service for generating question embeddings
                             from natural language text
                             
        Note:
            - Creates RelationEntityGrapher for knowledge graph navigation
            - Initializes QuestionBatcher for efficient batch processing
            - Stores vocabularies for entity/relation ID conversion
            - Embedding server enables dynamic question encoding
        """
        self.batch_size = params['batch_size']
        self.num_rollouts = params['num_rollouts']
        self.positive_reward = params['positive_reward']
        self.negative_reward = params['negative_reward']
        self.mode = mode
        self.path_len = params['path_length']
        self.test_rollouts = params['test_rollouts']
        input_dir = params['data_input_dir']

        # TODO: Improve this so it is shared, might be too heavy having multiple instances
        self.batcher = QuestionBatcher(
            input_dir=input_dir,
            batch_size=self.batch_size,
            question_tokenizer_name=params['question_tokenizer_name'],
            cached_QAMetaData_path=params['cached_QAMetaData_path'],
            raw_QAData_path=params['raw_QAData_path'],
            force_data_prepro=False,
            mode=self.mode,
            embedding_server=embedding_server,
        )

        self.total_no_examples = self.batcher.get_question_num()

        # Initialize the knowledge graph
        self.grapher = RelationEntityGrapher(triple_store=os.path.join(input_dir, 'graph.txt'),
                                             max_num_actions=params['max_num_actions'],
                                             entity_vocab=entity_vocab,
                                             relation_vocab=relation_vocab)

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
        params = self.batch_size, self.path_len, self.num_rollouts, self.test_rollouts, self.positive_reward, self.negative_reward, self.mode
        if self.mode == 'train':
            for data in self.batcher.yield_next_batch_train():

                yield EpisodeNLQ(self.grapher, data, params)
        else:
            for data in self.batcher.yield_next_batch_test():
                if data == None:
                    return
                yield EpisodeNLQ(self.grapher, data, params)

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