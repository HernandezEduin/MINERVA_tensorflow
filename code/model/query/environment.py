from __future__ import absolute_import
from __future__ import division
import numpy as np
from code.data.feed_data import RelationEntityBatcher
from code.data.grapher import RelationEntityGrapher
import logging

from typing import Dict, Any, Tuple, Generator

logger = logging.getLogger()

class Episode(object):
    """
    Class representing a single episode of interaction with the environment.
    """

    def __init__(
            self, 
            graph: RelationEntityGrapher, 
            data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], 
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
        self.batch_size, self.path_len, num_rollouts, test_rollouts, positive_reward, negative_reward, mode, batcher = params # parameters for episode
        self.mode = mode                                                        # evaluation mode
        if self.mode == 'train':
            self.num_rollouts = num_rollouts                                    # number of rollouts (simultaneous paths taken) during training
        else:
            self.num_rollouts = test_rollouts                                   # number of rollouts (simultaneous paths taken) during testing
        self.current_hop = 0                                                    # number of hops taken
        start_entities, query_relation,  end_entities, all_answers = data       # starting node, query question, answer node, alternative valid answers
        self.no_examples = start_entities.shape[0]                              # number of examples in the batch
        self.positive_reward = positive_reward                                  # reward for arriving at the answer (sparse, probably 1)
        self.negative_reward = negative_reward                                  # reward for not arriving at the answer
        start_entities = np.repeat(start_entities, self.num_rollouts)           # repeat start entities for each rollout (batch_size*num_rollouts,), repeats the element after itself
        batch_query_relation = np.repeat(query_relation, self.num_rollouts)     # repeat query relation for each rollout (batch_size*num_rollouts,), repeats the element after itself
        end_entities = np.repeat(end_entities, self.num_rollouts)               # repeat end entities for each rollout (batch_size*num_rollouts,), repeats the element after itself
        self.start_entities = start_entities
        self.end_entities = end_entities
        self.current_entities = np.array(start_entities)                        # make a copy of start entities (non-addressable)
        self.query_relation = batch_query_relation
        self.all_answers = all_answers

        # extract the next possible actions
        next_actions = self.grapher.return_next_actions(self.current_entities, self.start_entities, self.query_relation,
                                                        self.end_entities, self.all_answers, self.current_hop == self.path_len - 1,
                                                        self.num_rollouts)
        self.state = {}                                                         # RL states (next_relations, next_entities, current_entities)
        self.state['next_relations'] = next_actions[:, :, 1]
        self.state['next_entities'] = next_actions[:, :, 0]
        self.state['current_entities'] = self.current_entities

    def get_state(self) -> Dict[str, np.ndarray]:
        """
        Return the current state of the environment.
        """
        return self.state

    def get_query_relation(self) -> np.ndarray:
        """
        Return the query relation of the environment.
        """
        return self.query_relation

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
        next_actions = self.grapher.return_next_actions(self.current_entities, self.start_entities, self.query_relation,
                                                        self.end_entities, self.all_answers, self.current_hop == self.path_len - 1,
                                                        self.num_rollouts )

        # store the new states
        self.state['next_relations'] = next_actions[:, :, 1]
        self.state['next_entities'] = next_actions[:, :, 0]
        self.state['current_entities'] = self.current_entities
        return self.state


class env(object):
    """
    Environment for the RL agent, contains the knowledge graph and batch generator. Calls upon the episode for interaction
    """
    def __init__(self, params, mode='train'):
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
        if mode == 'train':                                  # Initialize the batch generator
            self.batcher = RelationEntityBatcher(input_dir=input_dir,
                                                 batch_size=params['batch_size'],
                                                 entity_vocab=params['entity_vocab'],
                                                 relation_vocab=params['relation_vocab']
                                                 )
        else:
            self.batcher = RelationEntityBatcher(input_dir=input_dir,
                                                 mode =mode,
                                                 batch_size=params['batch_size'],
                                                 entity_vocab=params['entity_vocab'],
                                                 relation_vocab=params['relation_vocab'])

            self.total_no_examples = self.batcher.store.shape[0]    # total number of examples in the dataset

        # Initialize the knowledge graph
        self.grapher = RelationEntityGrapher(triple_store=params['data_input_dir'] + '/' + 'graph.txt',
                                             max_num_actions=params['max_num_actions'],
                                             entity_vocab=params['entity_vocab'],
                                             relation_vocab=params['relation_vocab'])

    def get_episodes(self) -> Generator[Episode, None, None]:
        params = self.batch_size, self.path_len, self.num_rollouts, self.test_rollouts, self.positive_reward, self.negative_reward, self.mode, self.batcher
        if self.mode == 'train':
            for data in self.batcher.yield_next_batch_train():

                yield Episode(self.grapher, data, params)
        else:
            for data in self.batcher.yield_next_batch_test():
                if data == None:
                    return
                yield Episode(self.grapher, data, params)
