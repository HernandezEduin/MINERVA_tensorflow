from __future__ import absolute_import
from __future__ import division
from tqdm import tqdm
import json
import time
import os
import logging
import numpy as np
import tensorflow as tf
from typing import Dict, Any, List, Tuple, Optional, Union
tf.compat.v1.disable_eager_execution()

from code.model.agent import Agent
from code.options import read_options
from code.model.environment import env
from code.data.utils import set_seeds
import codecs
from collections import defaultdict
import gc
import resource
import sys
from code.model.baseline import ReactiveBaseline
from code.model.nell_eval import nell_eval
from scipy.special import logsumexp as lse

logger = logging.getLogger()
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


class Trainer(object):
    """
    MINERVA trainer for reinforcement learning-based knowledge graph reasoning.
    
    This class orchestrates the training and evaluation of the MINERVA agent, which
    learns to navigate knowledge graphs through multi-hop reasoning. The trainer
    handles episode generation, reward computation, policy gradient updates, and
    performance evaluation using various metrics (Hits@K, MRR).
    
    Key Components:
    - Policy gradient training with REINFORCE algorithm
    - Baseline variance reduction using reactive baseline
    - Multi-environment support (train/dev/test)
    - Comprehensive evaluation with beam search
    - Model checkpointing and restoration
    """
    
    def __init__(self, params: Dict[str, Any]) -> None:
        """
        Initialize the MINERVA trainer with configuration parameters.
        
        Sets up the agent, environments, baseline, optimizer, and all necessary
        components for training and evaluation.
        
        Args:
            params (Dict[str, Any]): Configuration dictionary containing:
                - Agent parameters (embedding sizes, LSTM layers, etc.)
                - Training parameters (learning rate, batch size, etc.) 
                - Environment parameters (path length, rollouts, etc.)
                - Evaluation parameters (beam size, metrics, etc.)
                - Data paths and vocabulary mappings
        """

        # transfer parameters to self
        for key, val in params.items(): setattr(self, key, val);

        self.agent = Agent(params)
        self.save_path = None
        self.train_environment = env(params, 'train')
        self.dev_test_environment = env(params, 'dev')
        self.test_test_environment = env(params, 'test')
        self.test_environment = self.dev_test_environment
        self.rev_relation_vocab = self.train_environment.grapher.rev_relation_vocab
        self.rev_entity_vocab = self.train_environment.grapher.rev_entity_vocab
        self.max_hits_at_10 = 0
        self.ePAD = self.entity_vocab['PAD']
        self.rPAD = self.relation_vocab['PAD']
        # optimize
        self.baseline = ReactiveBaseline(l=self.Lambda)
        self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate)


    def calc_reinforce_loss(self) -> tf.Tensor:
        """
        Calculate the REINFORCE policy gradient loss with baseline variance reduction.
        
        Implements the REINFORCE algorithm by:
        1. Computing per-example losses from agent policy
        2. Subtracting baseline value for variance reduction
        3. Normalizing advantages by standard deviation
        4. Weighting losses by normalized advantages
        5. Adding entropy regularization for exploration
        
        The loss encourages actions that lead to higher-than-expected rewards
        while penalizing those that lead to lower rewards.
        
        Returns:
            tf.Tensor: Scalar loss value combining policy gradient loss and 
                entropy regularization, ready for gradient descent optimization.
        """
        loss = tf.stack(self.per_example_loss, axis=1)  # [B, T]

        self.tf_baseline = self.baseline.get_baseline_value()
        # self.pp = tf.Print(self.tf_baseline)
        # multiply with rewards
        final_reward = self.cum_discounted_reward - self.tf_baseline
        # reward_std = tf.sqrt(tf.reduce_mean(tf.square(final_reward))) + 1e-5 # constant addded for numerical stability
        reward_mean, reward_var = tf.nn.moments(final_reward, axes=[0, 1])
        # Constant added for numerical stability
        reward_std = tf.sqrt(reward_var) + 1e-6
        final_reward =  tf.math.divide(final_reward - reward_mean, reward_std)

        loss = tf.multiply(loss, final_reward)  # [B, T]
        self.loss_before_reg = loss

        total_loss = tf.reduce_mean(loss) - self.decaying_beta * self.entropy_reg_loss(self.per_example_logits)  # scalar

        return total_loss

    def entropy_reg_loss(self, all_logits: List[tf.Tensor]) -> tf.Tensor:
        """
        Calculate entropy regularization loss to encourage exploration.
        
        Computes the negative entropy of the policy to add to the main loss.
        Higher entropy (more uniform action probabilities) gets lower penalty,
        encouraging the agent to explore different actions rather than being
        too deterministic early in training.
        
        Args:
            all_logits (List[tf.Tensor]): Log probabilities over actions at each
                time step. Each tensor has shape [batch_size, max_actions].
                
        Returns:
            tf.Tensor: Scalar entropy regularization loss. Negative entropy
                means higher entropy (exploration) reduces the total loss.
        """
        all_logits = tf.stack(all_logits, axis=2)  # [B, MAX_NUM_ACTIONS, T]
        entropy_policy = - tf.reduce_mean(tf.reduce_sum(tf.multiply(tf.exp(all_logits), all_logits), axis=1))  # scalar
        return entropy_policy

    def initialize(self, restore: Optional[str] = None, sess: Optional[tf.compat.v1.Session] = None) -> None:
        """
        Initialize the TensorFlow computational graph and training components.
        
        Sets up all placeholders, variables, and operations needed for training:
        - Input placeholders for candidate actions and queries
        - Agent policy network and loss computation
        - Training operations with gradient clipping
        - Model saving and restoration capabilities
        - Optional pretrained embedding initialization
        
        Args:
            restore (Optional[str]): Path to checkpoint file for model restoration.
                If None, initializes with random weights.
            sess (Optional[tf.Session]): TensorFlow session for initialization.
                If None, uses current default session.
        """

        logger.info("Creating TF graph...")
        self.candidate_relation_sequence = []
        self.candidate_entity_sequence = []
        self.input_path = []
        self.first_state_of_test = tf.compat.v1.placeholder(tf.bool, name="is_first_state_of_test")
        self.query_relation = tf.compat.v1.placeholder(tf.int32, [None], name="query_relation")
        self.range_arr = tf.compat.v1.placeholder(tf.int32, shape=[None, ])
        self.global_step = tf.Variable(0, trainable=False)
        self.decaying_beta = tf.compat.v1.train.exponential_decay(self.beta, self.global_step,
                                                   200, 0.90, staircase=False)
        self.entity_sequence = []

        # to feed in the discounted reward tensor
        self.cum_discounted_reward = tf.compat.v1.placeholder(tf.float32, [None, self.path_length],
                                                    name="cumulative_discounted_reward")



        for t in range(self.path_length):
            next_possible_relations = tf.compat.v1.placeholder(tf.int32, [None, self.max_num_actions],
                                                   name="next_relations_{}".format(t))
            next_possible_entities = tf.compat.v1.placeholder(tf.int32, [None, self.max_num_actions],
                                                     name="next_entities_{}".format(t))
            input_label_relation = tf.compat.v1.placeholder(tf.int32, [None], name="input_label_relation_{}".format(t))
            start_entities = tf.compat.v1.placeholder(tf.int32, [None, ])
            self.input_path.append(input_label_relation)
            self.candidate_relation_sequence.append(next_possible_relations)
            self.candidate_entity_sequence.append(next_possible_entities)
            self.entity_sequence.append(start_entities)
        self.loss_before_reg = tf.constant(0.0)
        self.per_example_loss, self.per_example_logits, self.action_idx = self.agent(
            self.candidate_relation_sequence,
            self.candidate_entity_sequence, self.entity_sequence,
            self.input_path,
            self.query_relation, self.range_arr, self.first_state_of_test, self.path_length)


        self.loss_op = self.calc_reinforce_loss()

        # backprop
        self.train_op = self.bp(self.loss_op)

        # Building the test graph
        self.prev_state = tf.compat.v1.placeholder(tf.float32, self.agent.get_mem_shape(), name="memory_of_agent")
        self.prev_relation = tf.compat.v1.placeholder(tf.int32, [None, ], name="previous_relation")
        self.query_embedding = tf.nn.embedding_lookup(self.agent.relation_lookup_table, self.query_relation)  # [B, 2D]
        layer_state = tf.unstack(self.prev_state, self.LSTM_layers)
        formated_state = [tf.unstack(s, 2) for s in layer_state]
        self.next_relations = tf.compat.v1.placeholder(tf.int32, shape=[None, self.max_num_actions])
        self.next_entities = tf.compat.v1.placeholder(tf.int32, shape=[None, self.max_num_actions])

        self.current_entities = tf.compat.v1.placeholder(tf.int32, shape=[None,])



        with tf.compat.v1.variable_scope("policy_steps_unroll") as scope:
            scope.reuse_variables()
            self.test_loss, test_state, self.test_logits, self.test_action_idx, self.chosen_relation = self.agent.step(
                self.next_relations, self.next_entities, formated_state, self.prev_relation, self.query_embedding,
                self.current_entities, self.input_path[0], self.range_arr, self.first_state_of_test)
            self.test_state = tf.stack(test_state)

        logger.info('TF Graph creation done..')
        self.model_saver = tf.compat.v1.train.Saver(max_to_keep=2)

        # return the variable initializer Op.
        if not restore:
            return tf.compat.v1.global_variables_initializer()
        else:
            return  self.model_saver.restore(sess, restore)



    def initialize_pretrained_embeddings(self, sess: tf.compat.v1.Session) -> None:
        """
        Load and initialize pretrained embeddings for entities and relations.
        
        If pretrained embedding files are specified in the configuration,
        loads them and initializes the corresponding embedding lookup tables.
        This can significantly improve training speed and final performance.
        
        Args:
            sess (tf.Session): TensorFlow session for running initialization ops.
        """
        if self.pretrained_embeddings_action != '':
            embeddings = np.loadtxt(open(self.pretrained_embeddings_action))
            _ = sess.run((self.agent.relation_embedding_init),
                         feed_dict={self.agent.action_embedding_placeholder: embeddings})
        if self.pretrained_embeddings_entity != '':
            embeddings = np.loadtxt(open(self.pretrained_embeddings_entity))
            _ = sess.run((self.agent.entity_embedding_init),
                         feed_dict={self.agent.entity_embedding_placeholder: embeddings})

    def bp(self, cost: tf.Tensor) -> tf.Operation:
        """
        Set up backpropagation with baseline update and gradient clipping.
        
        Creates the training operation that:
        1. Updates the baseline with current reward
        2. Computes gradients of cost w.r.t. trainable variables  
        3. Clips gradients by global norm to prevent exploding gradients
        4. Applies gradients using Adam optimizer
        
        Args:
            cost (tf.Tensor): Scalar loss tensor to minimize.
            
        Returns:
            tf.Operation: Training operation that updates model parameters.
        """
        self.baseline.update(tf.reduce_mean(self.cum_discounted_reward))
        tvars = tf.compat.v1.trainable_variables()
        grads = tf.compat.v1.gradients(cost, tvars)
        grads, _ = tf.clip_by_global_norm(grads, self.grad_clip_norm)
        train_op = self.optimizer.apply_gradients(zip(grads, tvars))
        with tf.control_dependencies([train_op]):  # see https://github.com/tensorflow/tensorflow/issues/1899
            self.dummy = tf.constant(0)
        return train_op


    def calc_cum_discounted_reward(self, rewards: np.ndarray) -> np.ndarray:
        """
        Calculate cumulative discounted rewards for policy gradient training.
        
        Computes the discounted return from each time step using the formula:
        G_t = R_t + γ*R_{t+1} + γ²*R_{t+2} + ... + γ^{T-t}*R_T
        
        This provides the expected long-term reward from each state, which
        serves as the target for the baseline and the weight for policy gradients.
        
        Args:
            rewards (np.ndarray): Final rewards received at episode end.
                Shape: [batch_size]
                
        Returns:
            np.ndarray: Cumulative discounted rewards for each time step.
                Shape: [batch_size, path_length]. Entry [i,t] is the discounted
                return from time step t for episode i.
        """
        running_add = np.zeros([rewards.shape[0]])  # [B]
        cum_disc_reward = np.zeros([rewards.shape[0], self.path_length])  # [B, T]
        cum_disc_reward[:,
        self.path_length - 1] = rewards  # set the last time step to the reward received at the last state
        for t in reversed(range(self.path_length)):
            running_add = self.gamma * running_add + cum_disc_reward[:, t]
            cum_disc_reward[:, t] = running_add
        return cum_disc_reward

    def gpu_io_setup(self) -> Tuple[List[tf.Tensor], List[tf.Tensor], List[Dict[tf.Tensor, Any]]]:
        """
        Set up TensorFlow partial_run configuration for efficient episode processing.
        
        Creates the fetches, feeds, and feed_dict structures needed for 
        TensorFlow's partial_run functionality, which allows dynamic unrolling
        of episodes while maintaining computational efficiency.
        
        Returns:
            Tuple containing:
                - fetches: List of tensors to fetch during partial_run
                - feeds: List of placeholder tensors for feeding data
                - feed_dict: List of feed dictionaries for each time step
        """
        # create fetches for partial_run_setup
        fetches = self.per_example_loss  + self.action_idx + [self.loss_op] + self.per_example_logits + [self.dummy]
        feeds =  [self.first_state_of_test] + self.candidate_relation_sequence+ self.candidate_entity_sequence + self.input_path + \
                [self.query_relation] + [self.cum_discounted_reward] + [self.range_arr] + self.entity_sequence


        feed_dict = [{} for _ in range(self.path_length)]

        feed_dict[0][self.first_state_of_test] = False
        feed_dict[0][self.query_relation] = None
        feed_dict[0][self.range_arr] = np.arange(self.batch_size*self.num_rollouts)
        for i in range(self.path_length):
            feed_dict[i][self.input_path[i]] = np.zeros(self.batch_size * self.num_rollouts)  # placebo
            feed_dict[i][self.candidate_relation_sequence[i]] = None
            feed_dict[i][self.candidate_entity_sequence[i]] = None
            feed_dict[i][self.entity_sequence[i]] = None

        return fetches, feeds, feed_dict

    def train(self, sess: tf.compat.v1.Session) -> None:
        """
        Execute one epoch of MINERVA training using policy gradients.
        
        Performs the complete training loop:
        1. Iterates through all training episodes
        2. For each episode, unrolls the policy for path_length steps
        3. Collects actions, logits, and losses at each step
        4. Computes final rewards and discounted returns
        5. Updates policy parameters using REINFORCE algorithm
        6. Updates baseline for variance reduction
        7. Logs training statistics and progress
        
        Uses TensorFlow's partial_run for efficient episode processing,
        allowing dynamic unrolling while maintaining computational efficiency.
        
        Args:
            sess (tf.Session): Active TensorFlow session for training operations.
        """
        # import pdb
        # pdb.set_trace()
        fetches, feeds, feed_dict = self.gpu_io_setup()

        train_loss = 0.0
        start_time = time.time()
        self.batch_counter = 0
        for episode in self.train_environment.get_episodes():

            self.batch_counter += 1
            h = sess.partial_run_setup(fetches=fetches, feeds=feeds)
            feed_dict[0][self.query_relation] = episode.get_query_relation()

            # get initial state
            state = episode.get_state()
            # for each time step
            loss_before_regularization = []
            logits = []
            for i in range(self.path_length):
                feed_dict[i][self.candidate_relation_sequence[i]] = state['next_relations']
                feed_dict[i][self.candidate_entity_sequence[i]] = state['next_entities']
                feed_dict[i][self.entity_sequence[i]] = state['current_entities']
                per_example_loss, per_example_logits, idx = sess.partial_run(h, [self.per_example_loss[i], self.per_example_logits[i], self.action_idx[i]],
                                                  feed_dict=feed_dict[i])
                loss_before_regularization.append(per_example_loss)
                logits.append(per_example_logits)
                # action = np.squeeze(action, axis=1)  # [B,]
                state = episode(idx)
            loss_before_regularization = np.stack(loss_before_regularization, axis=1)

            # get the final reward from the environment
            rewards = episode.get_reward()

            # computed cumulative discounted reward
            cum_discounted_reward = self.calc_cum_discounted_reward(rewards)  # [B, T]


            # backprop
            batch_total_loss, _ = sess.partial_run(h, [self.loss_op, self.dummy],
                                                   feed_dict={self.cum_discounted_reward: cum_discounted_reward})

            # print statistics
            train_loss = 0.98 * train_loss + 0.02 * batch_total_loss
            avg_reward = np.mean(rewards)
            # now reshape the reward to [orig_batch_size, num_rollouts], I want to calculate for how many of the
            # entity pair, atleast one of the path get to the right answer
            reward_reshape = np.reshape(rewards, (self.batch_size, self.num_rollouts))  # [orig_batch, num_rollouts]
            reward_reshape = np.sum(reward_reshape, axis=1)  # [orig_batch]
            reward_reshape = (reward_reshape > 0)
            num_ep_correct = np.sum(reward_reshape)
            if np.isnan(train_loss):
                raise ArithmeticError("Error in computing loss")

            logger.info("batch_counter: {0:4d}, num_hits: {1:7.4f}, avg. reward per batch {2:7.4f}, "
                        "num_ep_correct {3:4d}, avg_ep_correct {4:7.4f}, train loss {5:7.4f}".
                        format(self.batch_counter, np.sum(rewards), avg_reward, num_ep_correct,
                               (num_ep_correct / self.batch_size),
                               train_loss))

            if self.batch_counter%self.eval_every == 0:
                with open(self.output_dir + '/scores.txt', 'a') as score_file:
                    score_file.write("Score for iteration " + str(self.batch_counter) + "\n")
                os.makedirs(self.path_logger_file + "/" + str(self.batch_counter), exist_ok=True)
                self.path_logger_file_ = self.path_logger_file + "/" + str(self.batch_counter) + "/paths"



                self.test(sess, beam=True, print_paths=False)

            logger.info('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

            gc.collect()
            if self.batch_counter >= self.total_iterations:
                break

    def test(self, sess: tf.compat.v1.Session, beam: bool = False, print_paths: bool = False, 
             save_model: bool = True) -> Tuple[float, float, float, float, float]:
        """
        Evaluate the trained MINERVA agent on test data with multiple metrics.
        
        Performs comprehensive evaluation including:
        - Hits@K metrics (K=1,3,5,10,20) for answer prediction accuracy
        - Mean Reciprocal Rank (MRR) for ranking quality assessment  
        - Optional beam search for improved performance
        - Path visualization and analysis
        - Model checkpointing based on performance
        
        The evaluation uses multiple rollouts per query and aggregates results
        to provide robust performance estimates across different reasoning paths.
        
        Args:
            sess (tf.Session): Active TensorFlow session for inference.
            beam (bool, optional): Whether to use beam search decoding.
                Defaults to False (greedy decoding).
            print_paths (bool, optional): Whether to print reasoning paths.
                Defaults to False.
            save_model (bool, optional): Whether to save model if performance improves.
                Defaults to True.
                
        Returns:
            Tuple[float, float, float, float, float]: Performance metrics:
                - Hits@1: Fraction of queries with correct answer in top 1
                - Hits@3: Fraction of queries with correct answer in top 3  
                - Hits@5: Fraction of queries with correct answer in top 5
                - Hits@10: Fraction of queries with correct answer in top 10
                - Hits@20: Fraction of queries with correct answer in top 20
        """
        batch_counter = 0
        paths = defaultdict(list)
        answers = []
        feed_dict = {}
        all_final_reward_1 = 0
        all_final_reward_3 = 0
        all_final_reward_5 = 0
        all_final_reward_10 = 0
        all_final_reward_20 = 0
        auc = 0

        total_examples = self.test_environment.total_no_examples
        for episode in tqdm(self.test_environment.get_episodes()):
            batch_counter += 1

            temp_batch_size = episode.no_examples

            self.qr = episode.get_query_relation()
            feed_dict[self.query_relation] = self.qr
            # set initial beam probs
            beam_probs = np.zeros((temp_batch_size * self.test_rollouts, 1))
            # get initial state
            state = episode.get_state()
            mem = self.agent.get_mem_shape()
            agent_mem = np.zeros((mem[0], mem[1], temp_batch_size*self.test_rollouts, mem[3]) ).astype('float32')
            previous_relation = np.ones((temp_batch_size * self.test_rollouts, ), dtype='int64') * self.relation_vocab[
                'DUMMY_START_RELATION']
            feed_dict[self.range_arr] = np.arange(temp_batch_size * self.test_rollouts)
            feed_dict[self.input_path[0]] = np.zeros(temp_batch_size * self.test_rollouts)

            ####logger code####
            if print_paths:
                self.entity_trajectory = []
                self.relation_trajectory = []
            ####################

            self.log_probs = np.zeros((temp_batch_size*self.test_rollouts,)) * 1.0

            # for each time step
            for i in range(self.path_length):
                if i == 0:
                    feed_dict[self.first_state_of_test] = True
                feed_dict[self.next_relations] = state['next_relations']
                feed_dict[self.next_entities] = state['next_entities']
                feed_dict[self.current_entities] = state['current_entities']
                feed_dict[self.prev_state] = agent_mem
                feed_dict[self.prev_relation] = previous_relation

                loss, agent_mem, test_scores, test_action_idx, chosen_relation = sess.run(
                    [ self.test_loss, self.test_state, self.test_logits, self.test_action_idx, self.chosen_relation],
                    feed_dict=feed_dict)


                if beam:
                    k = self.test_rollouts
                    new_scores = test_scores + beam_probs
                    if i == 0:
                        idx = np.argsort(new_scores)
                        idx = idx[:, -k:]
                        ranged_idx = np.tile([b for b in range(k)], temp_batch_size)
                        idx = idx[np.arange(k*temp_batch_size), ranged_idx]
                    else:
                        idx = self.top_k(new_scores, k)

                    y = idx//self.max_num_actions
                    x = idx%self.max_num_actions

                    y += np.repeat([b*k for b in range(temp_batch_size)], k)
                    state['current_entities'] = state['current_entities'][y]
                    state['next_relations'] = state['next_relations'][y,:]
                    state['next_entities'] = state['next_entities'][y, :]
                    agent_mem = agent_mem[:, :, y, :]
                    test_action_idx = x
                    chosen_relation = state['next_relations'][np.arange(temp_batch_size*k), x]
                    beam_probs = new_scores[y, x]
                    beam_probs = beam_probs.reshape((-1, 1))
                    if print_paths:
                        for j in range(i):
                            self.entity_trajectory[j] = self.entity_trajectory[j][y]
                            self.relation_trajectory[j] = self.relation_trajectory[j][y]
                previous_relation = chosen_relation

                ####logger code####
                if print_paths:
                    self.entity_trajectory.append(state['current_entities'])
                    self.relation_trajectory.append(chosen_relation)
                ####################
                state = episode(test_action_idx)
                self.log_probs += test_scores[np.arange(self.log_probs.shape[0]), test_action_idx]
            if beam:
                self.log_probs = beam_probs

            ####Logger code####

            if print_paths:
                self.entity_trajectory.append(
                    state['current_entities'])


            # ask environment for final reward
            rewards = episode.get_reward()  # [B*test_rollouts]
            reward_reshape = np.reshape(rewards, (temp_batch_size, self.test_rollouts))  # [orig_batch, test_rollouts]
            self.log_probs = np.reshape(self.log_probs, (temp_batch_size, self.test_rollouts))
            sorted_indx = np.argsort(-self.log_probs)
            final_reward_1 = 0
            final_reward_3 = 0
            final_reward_5 = 0
            final_reward_10 = 0
            final_reward_20 = 0
            AP = 0
            ce = episode.state['current_entities'].reshape((temp_batch_size, self.test_rollouts))
            se = episode.start_entities.reshape((temp_batch_size, self.test_rollouts))
            for b in range(temp_batch_size):
                answer_pos = None
                seen = set()
                pos=0
                if self.pool == 'max':
                    for r in sorted_indx[b]:
                        if reward_reshape[b,r] == self.positive_reward:
                            answer_pos = pos
                            break
                        if ce[b, r] not in seen:
                            seen.add(ce[b, r])
                            pos += 1
                if self.pool == 'sum':
                    scores = defaultdict(list)
                    answer = ''
                    for r in sorted_indx[b]:
                        scores[ce[b,r]].append(self.log_probs[b,r])
                        if reward_reshape[b,r] == self.positive_reward:
                            answer = ce[b,r]
                    final_scores = defaultdict(float)
                    for e in scores:
                        final_scores[e] = lse(scores[e])
                    sorted_answers = sorted(final_scores, key=final_scores.get, reverse=True)
                    if answer in  sorted_answers:
                        answer_pos = sorted_answers.index(answer)
                    else:
                        answer_pos = None


                if answer_pos != None:
                    if answer_pos < 20:
                        final_reward_20 += 1
                        if answer_pos < 10:
                            final_reward_10 += 1
                            if answer_pos < 5:
                                final_reward_5 += 1
                                if answer_pos < 3:
                                    final_reward_3 += 1
                                    if answer_pos < 1:
                                        final_reward_1 += 1
                if answer_pos == None:
                    AP += 0
                else:
                    AP += 1.0/((answer_pos+1))
                if print_paths:
                    qr = self.train_environment.grapher.rev_relation_vocab[self.qr[b * self.test_rollouts]]
                    start_e = self.rev_entity_vocab[episode.start_entities[b * self.test_rollouts]]
                    end_e = self.rev_entity_vocab[episode.end_entities[b * self.test_rollouts]]
                    paths[str(qr)].append(str(start_e) + "\t" + str(end_e) + "\n")
                    paths[str(qr)].append("Reward:" + str(1 if answer_pos != None and answer_pos < 10 else 0) + "\n")
                    for r in sorted_indx[b]:
                        indx = b * self.test_rollouts + r
                        if rewards[indx] == self.positive_reward:
                            rev = 1
                        else:
                            rev = -1
                        answers.append(self.rev_entity_vocab[se[b,r]]+'\t'+ self.rev_entity_vocab[ce[b,r]]+'\t'+ str(self.log_probs[b,r])+'\n')
                        paths[str(qr)].append(
                            '\t'.join([str(self.rev_entity_vocab[e[indx]]) for e in
                                       self.entity_trajectory]) + '\n' + '\t'.join(
                                [str(self.rev_relation_vocab[re[indx]]) for re in self.relation_trajectory]) + '\n' + str(
                                rev) + '\n' + str(
                                self.log_probs[b, r]) + '\n___' + '\n')
                    paths[str(qr)].append("#####################\n")

            all_final_reward_1 += final_reward_1
            all_final_reward_3 += final_reward_3
            all_final_reward_5 += final_reward_5
            all_final_reward_10 += final_reward_10
            all_final_reward_20 += final_reward_20
            auc += AP

        all_final_reward_1 /= total_examples
        all_final_reward_3 /= total_examples
        all_final_reward_5 /= total_examples
        all_final_reward_10 /= total_examples
        all_final_reward_20 /= total_examples
        auc /= total_examples
        if save_model:
            if all_final_reward_10 >= self.max_hits_at_10:
                self.max_hits_at_10 = all_final_reward_10
                self.save_path = self.model_saver.save(sess, self.model_dir + "model" + '.ckpt')

        if print_paths:
            logger.info("[ printing paths at {} ]".format(self.output_dir+'/test_beam/'))
            for q in paths:
                j = q.replace('/', '-')
                with codecs.open(self.path_logger_file_ + '_' + j, 'a', 'utf-8') as pos_file:
                    for p in paths[q]:
                        pos_file.write(p)
            with open(self.path_logger_file_ + 'answers', 'w') as answer_file:
                for a in answers:
                    answer_file.write(a)

        with open(self.output_dir + '/scores.txt', 'a') as score_file:
            score_file.write("Hits@1: {0:7.4f}".format(all_final_reward_1))
            score_file.write("\n")
            score_file.write("Hits@3: {0:7.4f}".format(all_final_reward_3))
            score_file.write("\n")
            score_file.write("Hits@5: {0:7.4f}".format(all_final_reward_5))
            score_file.write("\n")
            score_file.write("Hits@10: {0:7.4f}".format(all_final_reward_10))
            score_file.write("\n")
            score_file.write("Hits@20: {0:7.4f}".format(all_final_reward_20))
            score_file.write("\n")
            score_file.write("auc: {0:7.4f}".format(auc))
            score_file.write("\n")
            score_file.write("\n")

        logger.info("Hits@1: {0:7.4f}".format(all_final_reward_1))
        logger.info("Hits@3: {0:7.4f}".format(all_final_reward_3))
        logger.info("Hits@5: {0:7.4f}".format(all_final_reward_5))
        logger.info("Hits@10: {0:7.4f}".format(all_final_reward_10))
        logger.info("Hits@20: {0:7.4f}".format(all_final_reward_20))
        logger.info("auc: {0:7.4f}".format(auc))

    def top_k(self, scores: np.ndarray, k: int) -> np.ndarray:
        """
        Extract top-k indices from beam search scores for each batch element.
        
        Used in beam search decoding to select the k highest-scoring paths
        from the expanded search space at each time step.
        
        Args:
            scores (np.ndarray): Beam search scores for each batch element.
                Shape: [batch_size, k * max_num_actions] where k is beam size.
            k (int): Number of top scoring paths to keep (beam size).
            
        Returns:
            np.ndarray: Flattened indices of top-k scoring actions.
                Shape: [batch_size * k]. Can be reshaped to [batch_size, k]
                to get top-k indices for each batch element.
        """
        scores = scores.reshape(-1, k * self.max_num_actions)  # [B, (k*max_num_actions)]
        idx = np.argsort(scores, axis=1)
        idx = idx[:, -k:]  # take the last k highest indices # [B , k]
        return idx.reshape((-1))

if __name__ == '__main__':

    # read command line options
    options = read_options()
    # Set logging
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%Y/%m/%d %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    logfile = logging.FileHandler(options['log_file_name'], 'w')
    logfile.setFormatter(fmt)
    logger.addHandler(logfile)
    # read the vocab files, it will be used by many classes hence global scope
    logger.info('reading vocab files...')
    options['relation_vocab'] = json.load(open(options['vocab_dir'] + '/relation_vocab.json'))
    options['entity_vocab'] = json.load(open(options['vocab_dir'] + '/entity_vocab.json'))
    logger.info('Reading mid to name map')
    mid_to_word = {}
    # with open('/iesl/canvas/rajarshi/data/RL-Path-RNN/FB15k-237/fb15k_names', 'r') as f:
    #     mid_to_word = json.load(f)
    logger.info('Done..')
    logger.info('Total number of entities {}'.format(len(options['entity_vocab'])))
    logger.info('Total number of relations {}'.format(len(options['relation_vocab'])))
    save_path = ''
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = False
    config.log_device_placement = False

    # Set seed for reproducibility
    set_seeds(options['seed'])

    #Training
    if not options['load_model']:
        trainer = Trainer(options)
        with tf.compat.v1.Session(config=config) as sess:
            sess.run(trainer.initialize())
            trainer.initialize_pretrained_embeddings(sess=sess)

            trainer.train(sess)
            save_path = trainer.save_path
            path_logger_file = trainer.path_logger_file
            output_dir = trainer.output_dir

        tf.compat.v1.reset_default_graph()
    #Testing on test with best model
    else:
        logger.info("Skipping training")
        logger.info("Loading model from {}".format(options["model_load_dir"]))

    trainer = Trainer(options)
    if options['load_model']:
        save_path = options['model_load_dir']
        path_logger_file = trainer.path_logger_file
        output_dir = trainer.output_dir
    with tf.compat.v1.Session(config=config) as sess:
        trainer.initialize(restore=save_path, sess=sess)

        trainer.test_rollouts = 100

        os.makedirs(path_logger_file + "/" + "test_beam", exist_ok=True)
        trainer.path_logger_file_ = path_logger_file + "/" + "test_beam" + "/paths"
        with open(output_dir + '/scores.txt', 'a') as score_file:
            score_file.write("Test (beam) scores with best model from " + save_path + "\n")
        trainer.test_environment = trainer.test_test_environment
        trainer.test_environment.test_rollouts = 100

        trainer.test(sess, beam=True, print_paths=True, save_model=False)


        print(options['nell_evaluation'])
        if options['nell_evaluation'] == 1:
            nell_eval(path_logger_file + "/" + "test_beam/" + "pathsanswers", trainer.data_input_dir+'/sort_test.pairs' )

