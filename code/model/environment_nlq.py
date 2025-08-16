from __future__ import absolute_import
from __future__ import division

import os

import numpy as np
import tensorflow as tf

from code.data.feed_nlq_data import QuestionBatcher
from code.data.grapher import RelationEntityGrapher
from code.data.embedding_server import EmbeddingServer
import logging

from typing import Dict, Any, Tuple, Generator

logger = logging.getLogger()

class EpisodeNLQ(object):
    """
    Class representing a single episode of interaction with the environment.
    """

    def __init__(
            self, 
            graph: RelationEntityGrapher, 
            data: Tuple[list, np.ndarray, np.ndarray, np.ndarray], 
            params: Dict[str, Any]
        ):
        """
        Initialize a reinforcement learning episode for knowledge graph reasoning.
        
        Sets up the environment state for multi-path exploration in knowledge graphs,
        where the agent learns to navigate from start entities to target entities
        by following relation edges. Supports batch processing with multiple rollouts
        for improved training efficiency.
        """
        self.grapher = graph  # environment graph containing the neighborhood of the current position + possible action
        self.batch_size, self.path_len, num_rollouts, test_rollouts, positive_reward, negative_reward, mode = params # parameters for episode
        self.mode = mode                                                        # evaluation mode
        if self.mode == 'train':
            self.num_rollouts = num_rollouts                                    # number of rollouts (simultaneous paths taken) during training
        else:
            self.num_rollouts = test_rollouts                                   # number of rollouts (simultaneous paths taken) during testing
        self.current_hop = 0                                                    # number of hops taken
        q_tokens, question_embeddings, start_entities, end_entities = data      # question_tokens, question_embeddings, starting node, answer node
        self.no_examples = start_entities.shape[0]                              # number of examples in the batch
        self.positive_reward = positive_reward                                  # reward for arriving at the answer (sparse, probably 1)
        self.negative_reward = negative_reward                                  # reward for not arriving at the answer
        start_entities = np.repeat(start_entities, self.num_rollouts)           # repeat start entities for each rollout (batch_size*num_rollouts,), repeats the element after itself
        end_entities = np.repeat(end_entities, self.num_rollouts)               # repeat end entities for each rollout (batch_size*num_rollouts,), repeats the element after itself
        self.start_entities = start_entities
        self.end_entities = end_entities
        self.current_entities = np.array(start_entities)                        # make a copy of start entities (non-addressable)
        self.question_embeddings = np.repeat(question_embeddings, self.num_rollouts, axis=0)     # repeat question embeddings for each rollout (batch_size*num_rollouts, hidden dim), repeats the element after itself
        self.question_tokens = q_tokens                                         # store the question tokens

        # extract the next possible actions
        next_actions = self.grapher.return_next_raw_actions(self.current_entities)

        self.state = {}                                                         # RL states (next_relations, next_entities, current_entities)
        self.state['next_relations'] = next_actions[:, :, 1]
        self.state['next_entities'] = next_actions[:, :, 0]
        self.state['current_entities'] = self.current_entities

    def get_state(self) -> Dict[str, np.ndarray]:
        """
        Return the current state of the environment.
        """
        return self.state

    def get_question_embedding(self) -> tf.Tensor:
        return self.question_embeddings

    def get_reward(self) -> np.ndarray:
        """
        Return the reward signal for the current state.
        """
        reward = (self.current_entities == self.end_entities)       # if the current position matches the answer entities

        # set the True and False values to the values of positive and negative rewards.
        condlist = [reward == True, reward == False]                # condition list for rewards
        choicelist = [self.positive_reward, self.negative_reward]   # choice list for rewards

        reward = np.select(condlist, choicelist)  # [B*num_rollouts,] assigns the reward accordingly to whichever statement is true
        return reward

    def __call__(self, action) -> Dict[str, np.ndarray]:
        """
        Take an action in the environment.
        """
        self.current_hop += 1   # increment the current step
        self.current_entities = self.state['next_entities'][np.arange(self.no_examples*self.num_rollouts), action] # for each sample, take the action to get new location

        # extract the next possible actions
        next_actions = self.grapher.return_next_raw_actions(self.current_entities)

        # store the new states
        self.state['next_relations'] = next_actions[:, :, 1]
        self.state['next_entities'] = next_actions[:, :, 0]
        self.state['current_entities'] = self.current_entities
        return self.state


class EnvNLQ(object):
    """
    Environment for the RL agent, contains the knowledge graph and batch generator. Calls upon the episode for interaction
    """
    def __init__(
            self, 
            params, 
            entity_vocab: Dict[str, int], 
            relation_vocab: Dict[str, int], 
            mode: str = 'train', 
            embedding_server: EmbeddingServer = None
        ):
        """
        Initialize the environment by storing initial parameters, setting up the knowledge graph, and initializing the batch generator.
        """
        self.batch_size = params['batch_size']               # batch size
        self.num_rollouts = params['num_rollouts']           # number of rollouts (simultaneous paths taken)
        self.positive_reward = params['positive_reward']     # positive reward value
        self.negative_reward = params['negative_reward']     # negative reward value
        self.mode = mode                                     # mode (train/dev/test)
        self.path_len = params['path_length']                # max number of steps agent is allowed to take
        self.test_rollouts = params['test_rollouts']         # number of rollouts (simultaneous paths taken) during testing
        input_dir = params['data_input_dir']                 # input directory for data

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

        self.total_no_examples = self.batcher.get_question_num()    # total number of examples in the dataset

        # Initialize the knowledge graph
        self.grapher = RelationEntityGrapher(triple_store=os.path.join(input_dir, 'graph.txt'),
                                             max_num_actions=params['max_num_actions'],
                                             entity_vocab=entity_vocab,
                                             relation_vocab=relation_vocab)

    def get_episodes(self) -> Generator[EpisodeNLQ, None, None]:
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
        Change the mode of the environment (train/dev/test).
        """
        assert mode in ['train', 'dev', 'test'], f"Error! Invalid mode: {mode}"
        self.mode = mode
        self.batcher.set_mode(mode)
        self.total_no_examples = self.batcher.get_question_num()  # update the total number of examples in the dataset
    
    def change_test_rollouts(self, test_rollouts: int) -> None:
        """
        Changes the number of test rollouts.
        """
        self.test_rollouts = test_rollouts