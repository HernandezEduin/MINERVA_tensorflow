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

from code.model.agent_nlq import AgentNLQ
from code.model.environment_nlq import EnvNLQ
from code.data.embedding_server import EmbeddingServer
from code.options import read_options_nlq
from code.data.utils import set_seeds
import codecs
from collections import defaultdict
import gc
import resource
import sys
from code.model.baseline import ReactiveBaseline
from scipy.special import logsumexp as lse
from collections import defaultdict

logger = logging.getLogger()
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

class TrainerNLQ(object):
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

    def __init__(
            self,
            params: Dict[str, Any],
            entity_vocab: Dict[str, int],
            relation_vocab: Dict[str, int],
            embedding_server: EmbeddingServer = None
        ) -> None:
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
            entity_vocab (Dict[str, int]): Vocabulary mapping for entities
            relation_vocab (Dict[str, int]): Vocabulary mapping for relations
        """

        # transfer parameters to self
        for key, val in params.items(): setattr(self, key, val)

        self.environment = EnvNLQ(
            params, 
            entity_vocab=entity_vocab, 
            relation_vocab=relation_vocab, 
            mode='train',
            embedding_server=embedding_server
        ) # shared environment accross modes, save space with graph builder and textual embeddings
        
        # Disable Eager Execution for the rest of the code
        tf.compat.v1.disable_eager_execution()

        # Agent and Environment
        self.agent = AgentNLQ(
            params, 
            entity_vocab=entity_vocab, 
            relation_vocab=relation_vocab
        )

        self.save_path = None

        # Provide the vocab2id and id2vocab mappings
        self.entity_vocab = entity_vocab
        self.relation_vocab = relation_vocab
        self.rev_relation_vocab = self.environment.grapher.rev_relation_vocab
        self.rev_entity_vocab = self.environment.grapher.rev_entity_vocab
        self.ePAD = self.entity_vocab['PAD']
        self.rPAD = self.relation_vocab['PAD']

        # Optimization Algorithms
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

        # multiply with rewards
        final_reward = self.cum_discounted_reward - self.tf_baseline

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
        - Input placeholders for candidate actions and questions
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

        # Variables List
        self.candidate_relation_sequence = []
        self.candidate_entity_sequence = []
        self.input_path = []                                                                                    # TODO: Remove if unused
        self.entity_sequence = []

        # Tensorflow Placeholders
        # New: external question embedding (e.g., BERT). Dim can be anything; we let dense learn to use it.
        # TODO: Find a better way to pass the token embedding size
        self.question_embedding = tf.compat.v1.placeholder(tf.float32, [None, 768], name="question_embedding") # [B*num_rollouts, token_embedding_dim]
        
        self.first_state_of_test = tf.compat.v1.placeholder(tf.bool, name="is_first_state_of_test")             # TODO: Remove this if unused
        self.range_arr = tf.compat.v1.placeholder(tf.int32, shape=[None, ])                                     # Range array for indexing operations.
        self.global_step = tf.Variable(0, trainable=False)                                                      # Global training step counter
        self.decaying_beta = tf.compat.v1.train.exponential_decay(
            self.beta, 
            self.global_step,
            200, 
            0.90, 
            staircase=False
        )                                                                                                       # Decaying beta for exploration

        # Cumulative Discounted Reward Tensor
        self.cum_discounted_reward = tf.compat.v1.placeholder(tf.float32, [None, self.path_length],
                                                    name="cumulative_discounted_reward")

        for t in range(self.path_length):
            next_rel = tf.compat.v1.placeholder(tf.int32, [None, self.max_num_actions],
                                                   name=f"next_relations_{t}")                                  # candidate relations from current entity  [B*num_rollouts,]
            next_ent = tf.compat.v1.placeholder(tf.int32, [None, self.max_num_actions],
                                                     name=f"next_entities_{t}")                                 # candidate entities from current entity [B*num_rollouts,]
            label_t = tf.compat.v1.placeholder(tf.int32, [None], name=f"input_label_relation_{t}")              # Ground truth action labels for training [B*num_rollouts,]
            cur_ent = tf.compat.v1.placeholder(tf.int32, [None, ], name=f"current_entities_{t}")                # current locations [B*num_rollouts,]

            self.input_path.append(label_t)                                                                     # TODO: Remove if unused
            self.candidate_relation_sequence.append(next_rel)                                                   # list of candidate relations at each step
            self.candidate_entity_sequence.append(next_ent)                                                     # list of candidate entities at each step
            self.entity_sequence.append(cur_ent)                                                                # list of current entities at each step

        self.loss_before_reg = tf.constant(0.0)

        # Building the computation graph for the agent, calls the forward method and within it, step
        # Graph for calculating the per-example loss, per-example logits and the action to take
        self.per_example_loss, self.per_example_logits, self.action_idx = self.agent(
            self.candidate_relation_sequence,
            self.candidate_entity_sequence,
            self.entity_sequence,
            self.question_embedding, 
            self.range_arr, 
            self.path_length
        )

        # Graph for calculating the Loss
        self.loss_op = self.calc_reinforce_loss()

        # Graph for performing Backpropagation (RL-style)
        self.train_op = self.bp(self.loss_op)

        # Building the test graph
        # TODO: Check if shape is correct for prev_shape
        self.prev_state = tf.compat.v1.placeholder(tf.float32, self.agent.get_mem_shape(), name="memory_of_agent")  # LSTM Memory Shape (num lstm layers, 2, batch size, memory size)
        self.prev_relation = tf.compat.v1.placeholder(tf.int32, [None, ], name="previous_relation")
        
        # Format the state properly for MultiRNNCell
        layer_state = tf.unstack(self.prev_state, self.LSTM_layers)
        formated_state = [tf.unstack(s, 2) for s in layer_state]
        
        self.next_relations = tf.compat.v1.placeholder(tf.int32, shape=[None, self.max_num_actions])
        self.next_entities = tf.compat.v1.placeholder(tf.int32, shape=[None, self.max_num_actions])
        self.current_entities = tf.compat.v1.placeholder(tf.int32, shape=[None,])

        with tf.compat.v1.variable_scope("policy_steps_unroll") as scope:
            scope.reuse_variables()
            self.test_loss, test_state, self.test_logits, self.test_action_idx, self.chosen_relation = self.agent.step(
                self.next_relations, 
                self.next_entities, 
                formated_state,  # Use properly formatted state
                self.prev_relation, 
                self.question_embedding,                        # TODO: verify if we don't need to initialize a new tf question embedding for testing
                self.current_entities, 
                self.range_arr                                  # Note: this tf variable is reused
            )
            self.test_state = tf.stack(test_state)

        logger.info('TF Graph ready (NLQ).')
        self.model_saver = tf.compat.v1.train.Saver(max_to_keep=2)  # save the model checkpoints (best 2)

        # return the variable initializer Op.
        if not restore:
            return tf.compat.v1.global_variables_initializer()      # initialize all variables
        else:
            return  self.model_saver.restore(sess, restore)         # restore checkpoint weights



    def initialize_pretrained_embeddings(self, sess: tf.compat.v1.Session) -> None:
        """
        Load and initialize pretrained embeddings for entities, relations, 
        and question projector.
        
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
        if self.pretrained_question_projector != '':
            embeddings = np.loadtxt(open(self.pretrained_question_projector))
            # Make sure the agent's question_proj has been called at least once to create variables
            if self.agent.question_proj_init is not None:
                _ = sess.run(self.agent.question_proj_init,
                            feed_dict={self.agent.question_embedding_placeholder: embeddings})
            else:
                logger.warning("Question projector variables not yet created. Skipping pretrained initialization.")

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
        cum_disc_reward[:, self.path_length - 1] = rewards  # set the last time step to the reward received at the last state
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
                - feed_dict: List of feed dictionaries for each hop/step
        """
        # create fetches for partial_run_setup
        fetches = self.per_example_loss  + self.action_idx + [self.loss_op] + self.per_example_logits + [self.dummy]
        feeds =  [self.first_state_of_test] + self.candidate_relation_sequence + self.candidate_entity_sequence + self.input_path + \
                [self.question_embedding] + [self.cum_discounted_reward] + [self.range_arr] + self.entity_sequence


        feed_dict = [{} for _ in range(self.path_length)]

        # Pass the memory address of the placeholder to the feed_dict
        # The following placeholders that stay constant through the hops/steps:
        feed_dict[0][self.first_state_of_test] = False # TODO: Remove this if unused
        feed_dict[0][self.question_embedding] = None
        feed_dict[0][self.range_arr] = np.arange(self.batch_size*self.num_rollouts)
        # The following placeholders vary across the hops/steps:
        for i in range(self.path_length):
            feed_dict[i][self.input_path[i]] = np.zeros(self.batch_size * self.num_rollouts)  # TODO: Remove this if unused
            feed_dict[i][self.candidate_relation_sequence[i]] = None
            feed_dict[i][self.candidate_entity_sequence[i]] = None
            feed_dict[i][self.entity_sequence[i]] = None

        return fetches, feeds, feed_dict

    def train(self, sess: tf.compat.v1.Session) -> None:
        """
        Execute multiple episodes of MINERVA training using policy gradients
        until a max number of episodes have been completed.

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
        logger.info("Starting training...")
        fetches, feeds, feed_dict = self.gpu_io_setup()

        train_loss = 0.0
        self.batch_counter = 0
        self.environment.change_mode('train')                           # Change environment mode to training
        for episode in self.environment.get_episodes():                 # Provide the current episode, can be repeated

            self.batch_counter += 1                                     # Increment batch count by 1 to eventually break the loop
            h = sess.partial_run_setup(fetches=fetches, feeds=feeds)    # Set up graph from fetches and feeds
            batch_qemb = episode.get_question_embedding()               # [B*num_rollouts, Q]
            feed_dict[0][self.question_embedding] = batch_qemb          # Provide question embeddings for this batch

            # Get Initial State
            state = episode.get_state()                                 # Provide the initial State (current_entities, next_entities, next_relations)

            # For each hop/step (tf)
            loss_before_regularization = []
            logits = []
            for i in range(self.path_length):
                feed_dict[i][self.candidate_relation_sequence[i]] = state['next_relations'] # Copy candidate relations
                feed_dict[i][self.candidate_entity_sequence[i]] = state['next_entities']    # Copy candidate entities
                feed_dict[i][self.entity_sequence[i]] = state['current_entities']           # Copy current position/ entity
                
                # Actual Execution of the TF Graph (Agent Call at hop i)
                per_example_loss, per_example_logits, idx = sess.partial_run(
                    h, 
                    [self.per_example_loss[i], self.per_example_logits[i], self.action_idx[i]],
                    feed_dict=feed_dict[i]
                )

                # Store the results
                loss_before_regularization.append(per_example_loss)
                logits.append(per_example_logits)

                # Interact with the environment by giving the action and receiving the next state
                state = episode(idx)

            # Process the results (numpy)
            loss_before_regularization = np.stack(loss_before_regularization, axis=1)
            rewards = episode.get_reward()  # get environment reward by checking the current position and the answer's position
            cum_discounted_reward = self.calc_cum_discounted_reward(rewards)  # computed cumulative discounted reward [B, T]

            # Backpropagate the results
            batch_total_loss, _ = sess.partial_run(
                h,
                [self.loss_op, self.dummy],
                feed_dict={self.cum_discounted_reward: cum_discounted_reward}
            )

            # Calculate the Loss and Average Reward
            train_loss = 0.98 * train_loss + 0.02 * batch_total_loss
            avg_reward = np.mean(rewards)
            if np.isnan(train_loss):
                raise ArithmeticError("NaN loss")

            # Reshape the reward to [orig_batch_size, num_rollouts], to calculate for how many of the
            # entity pair, at least one of the paths arrive at the correct answer
            reward_reshape = np.reshape(rewards, (self.batch_size, self.num_rollouts))  # [orig_batch, num_rollouts]
            reward_reshape = np.sum(reward_reshape, axis=1)                             # [orig_batch]
            reward_reshape = (reward_reshape > 0)
            num_ep_correct = np.sum(reward_reshape)

            # Log the Statistics
            logger.info(
                f"batch_counter: {self.batch_counter:<4d}, num_hits: {np.sum(rewards):<7.4f}, "
                f"avg. reward per batch: {avg_reward:<7.4f}, num_ep_correct: {num_ep_correct:<4d}, "
                f"avg_ep_correct: {num_ep_correct / self.batch_size:<7.4f}, train_loss: {train_loss:<7.4f}"
            )

            # Store current performance, logger path, and evaluate with dev
            if self.batch_counter % self.eval_every == 0:
                with open(os.path.join(self.output_dir, 'scores.txt'), 'a') as score_file:
                    score_file.write("Score for iteration " + str(self.batch_counter) + "\n")
                
                os.makedirs(os.path.join(self.path_logger_file, str(self.batch_counter)), exist_ok=True)
                self.path_logger_file_ = os.path.join(self.path_logger_file, str(self.batch_counter), "paths")

                self.test(sess, beam=True, print_paths=False, mode='dev')

            logger.info('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

            # Clean up (garbage collector to free space)
            gc.collect()

            if self.batch_counter >= self.total_iterations: # if enough iterations have been completed, break out of training
                break

    def test(self, sess: tf.compat.v1.Session, beam: bool = False, print_paths: bool = False, 
             save_model: bool = True, mode='dev') -> Tuple[float, float, float, float, float]:
        """
        Evaluate the trained MINERVA agent on test data with multiple metrics.
        
        Performs comprehensive evaluation including:
        - Hits@K metrics (K=1,3,5,10,20) for answer prediction accuracy
        - Mean Reciprocal Rank (MRR) for ranking quality assessment  
        - Optional beam search for improved performance
        - Path visualization and analysis
        - Model checkpointing based on performance
        
        The evaluation uses multiple rollouts per question and aggregates results
        to provide robust performance estimates across different reasoning paths.
        
        Args:
            sess (tf.Session): Active TensorFlow session for inference.
            beam (bool, optional): Whether to use beam search decoding.
                Defaults to False (greedy decoding).
            print_paths (bool, optional): Whether to print reasoning paths.
                Defaults to False.
            save_model (bool, optional): Whether to save model if performance improves.
                Defaults to True.
            auc (bool, optional): Whether to compute AUC metrics. Defaults to False.
                
        Returns:
            Tuple[float, float, float, float, float]: Performance metrics:
                - Hits@1: Fraction of queries with correct answer in top 1
                - Hits@3: Fraction of queries with correct answer in top 3  
                - Hits@5: Fraction of queries with correct answer in top 5
                - Hits@10: Fraction of queries with correct answer in top 10
                - Hits@20: Fraction of queries with correct answer in top 20
        """
        # NOTE: Hits@N are based on the num of rollouts, each respective rollout's scores, 
        # and how many arrived at the correct answer, this is not Entity Ranking as in KGE
        # Additionally assumes that there are at least 20 rollouts per question, 
        # otherwise Hits@N is capped at the max number of rollouts, 
        # i.e., rollout = 5, then Hits@5 = Hits@10 = Hits@20

        paths = defaultdict(list)       # Store paths for each question if print_paths is True
        answers = []                    # Store answers entity for each question if print_paths is True
        feed_dict = {}                  # Feed dictionaries, gets updated each hop during evaluation
        all_final_reward_1 = 0          # Overall results for hits@1
        all_final_reward_3 = 0          # Overall results for hits@3
        all_final_reward_5 = 0          # Overall results for hits@5
        all_final_reward_10 = 0         # Overall results for hits@10
        all_final_reward_20 = 0         # Overall results for hits@20
        mrr = 0                         # Overall results for MRR
        max_hits_at_10 = 0              # Condition for Saving Model


        # Changing the environment to test/dev data and resetting values
        self.environment.change_mode(mode)
        self.environment.change_test_rollouts(self.test_rollouts)   # modifying the number of rollouts for evaluation
        total_examples = self.environment.total_no_examples         # total number of questions
        test_batch_counter = 0
        logger.info(f"Testing with mode: {mode} on {total_examples} samples...")
        for episode in tqdm(self.environment.get_episodes(), desc="Evaluating"):
            temp_batch_size = episode.no_examples                   # batch size, can vary in test due to the last batch
            test_batch_counter += temp_batch_size
            logger.info(f"Evaluating samples {test_batch_counter}/{total_examples} with {self.test_rollouts} rollouts...")

            # Set Initial Beams Probs
            beam_probs = np.zeros((temp_batch_size * self.test_rollouts, 1)) # Cumulative scores from previous steps [batch_size*k, 1]

            # Provide Initial Variables
            state = episode.get_state()                             # Initial State (current_entities, next_entities, next_relations)
            mem = self.agent.get_mem_shape()                        # LSTM Memory Shape (num lstm layers, 2, batch size, memory size)
            agent_mem = np.zeros((mem[0], mem[1], temp_batch_size*self.test_rollouts, mem[3]), dtype='float32')
            previous_relation = np.ones((temp_batch_size * self.test_rollouts, ), dtype='int64') * self.relation_vocab['DUMMY_START_RELATION']
            
            feed_dict = {
                self.range_arr: np.arange(temp_batch_size * self.test_rollouts),
                self.input_path[0]: np.zeros(temp_batch_size * self.test_rollouts),     # TODO: Remove this if unused
                self.first_state_of_test: True,                                         # TODO: Remove this if unused
                self.question_embedding: episode.get_question_embedding(),              # question embeddings
            }

            # TODO: Remove this if it goes unused
            ####logger code####
            if print_paths:
                self.entity_trajectory = []
                self.relation_trajectory = []
            ####################

            self.log_probs = np.zeros((temp_batch_size*self.test_rollouts,)) * 1.0

            # For each hop/step
            for i in range(self.path_length):
                # Update the feed_dict with the current info
                feed_dict.update({
                    self.next_relations: state['next_relations'],
                    self.next_entities: state['next_entities'],
                    self.current_entities: state['current_entities'],
                    self.prev_state: agent_mem,
                    self.prev_relation: previous_relation
                })

                # Full execution of the TF graph (Agent Step)
                # ? I have no idea how they decided with parts to execute
                loss, agent_mem, test_scores, test_action_idx, chosen_relation = sess.run(
                    [self.test_loss, self.test_state, self.test_logits, self.test_action_idx, self.chosen_relation],
                    feed_dict=feed_dict
                )

                # Perform beam search
                # If beam is on, this will override the agent's actions based on agent's logits scores
                # hence, the agent only calculates the action probability while beam predicts the best actions
                if beam:
                    # Instead of greedily selecting the single best action at each step, 
                    # beam search maintains multiple promising paths simultaneously 
                    # to find better reasoning chains.
                    k = self.test_rollouts                  # Beam size (number of paths to maintain)
                    new_scores = test_scores + beam_probs   # Combine current action scores with cumulative beam scores [batch_size*k, max_actions]
                    if i == 0:                              # At step 0, all beams start from the same state, so we need to select diverse starting paths.
                        idx = np.argsort(new_scores)        # Sort all scores
                        idx = idx[:, -k:]                   # Take top-k indices
                        ranged_idx = np.tile([b for b in range(k)], temp_batch_size)
                        idx = idx[np.arange(k*temp_batch_size), ranged_idx]
                    else:
                        idx = self.top_k(new_scores, k)     # Use general top-k selection to select best paths from the expanded search space.

                    y = idx//self.max_num_actions           # Which beam/path each selected action comes from
                    x = idx%self.max_num_actions            # Which action within that beam

                    y += np.repeat([b*k for b in range(temp_batch_size)], k) # beam index adjustment for each question
                    
                    # Reorders all state information to match the selected beams
                    state['current_entities'] = state['current_entities'][y]
                    state['next_relations'] = state['next_relations'][y,:]
                    state['next_entities'] = state['next_entities'][y,:]
                    agent_mem = agent_mem[:, :, y, :]
                    
                    # Override Action Selection
                    test_action_idx = x # Selected actions
                    chosen_relation = state['next_relations'][np.arange(temp_batch_size*k), x]

                    # Score Tracking
                    beam_probs = new_scores[y, x]
                    beam_probs = beam_probs.reshape((-1, 1))

                    # Path History Maintenance
                    if print_paths:
                        for j in range(i):
                            self.entity_trajectory[j] = self.entity_trajectory[j][y]
                            self.relation_trajectory[j] = self.relation_trajectory[j][y]
                
                ####logger code####
                if print_paths: # Store the current path before the environment update
                    self.entity_trajectory.append(state['current_entities'])
                    self.relation_trajectory.append(chosen_relation)
                ####################

                # Update the states for the next hop
                previous_relation = chosen_relation
                state = episode(test_action_idx)

                # Aggregate Results
                self.log_probs += test_scores[np.arange(self.log_probs.shape[0]), test_action_idx]
            
            # After the last hop
            # If beam search was used, override the probabilities
            if beam:
                self.log_probs = beam_probs

            ####Logger code####
            if print_paths: # Store the current paths (entity only)
                self.entity_trajectory.append(state['current_entities'])

            # Calculate the final reward
            rewards = episode.get_reward()  # [B*test_rollouts]

            # Reshape the reward to [orig_batch_size, num_rollouts], to calculate for how many of the
            # entity pair, at least one of the paths arrive at the correct answer
            reward_reshape = np.reshape(rewards, (temp_batch_size, self.test_rollouts))  # [orig_batch, test_rollouts]
            self.log_probs = np.reshape(self.log_probs, (temp_batch_size, self.test_rollouts))
            sorted_indx = np.argsort(-self.log_probs)
            
            # Calculate the episode's metrics based on the sorted indices
            final_reward_1 = 0
            final_reward_3 = 0
            final_reward_5 = 0
            final_reward_10 = 0
            final_reward_20 = 0
            final_mrr = 0

            # Get current and start entities
            ce = episode.state['current_entities'].reshape((temp_batch_size, self.test_rollouts))
            se = episode.start_entities.reshape((temp_batch_size, self.test_rollouts))
            
            # Evaluate each sample/question's performance
            for b in range(temp_batch_size):
                answer_pos = None
                seen = set()
                pos = 0

                if self.pool == 'max':          # Evaluation done based on best performing rollout
                    for r in sorted_indx[b]:    # Go through paths sorted by score (highest first)
                        if reward_reshape[b,r] == self.positive_reward:  # Found correct answer
                            answer_pos = pos      # answer position is the current rank
                            break
                        if ce[b, r] not in seen:  # Only count unique entities
                            seen.add(ce[b, r])
                            pos += 1              # increment rank as penalty
                elif self.pool == 'sum':        # Evaluation done based on all rollouts
                    scores = defaultdict(list)
                    answer = ''
                    for r in sorted_indx[b]:
                        scores[ce[b,r]].append(self.log_probs[b,r])     # Collect all scores for each entity
                        if reward_reshape[b,r] == self.positive_reward:
                            answer = ce[b,r]                            # Remember which entity is correct
                    
                    # Use log-sum-exp to combine scores for each entity
                    final_scores = {e: lse(v) for e,v in scores.items()}
                    sorted_answers = sorted(final_scores, key=final_scores.get, reverse=True)
                    answer_pos = sorted_answers.index(answer) if answer in sorted_answers else None

                # Evaluate the answer position
                if answer_pos is not None:
                    final_mrr += 1.0/((answer_pos+1))
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
                else:
                    final_mrr += 0
                
                # Comprehensive reasoning path report
                if print_paths:
                    # Retrive Sample's context
                    question_txt = self.environment.batcher.translate_questions([episode.question_tokens[b]])[0]    # Convert question back to text
                    start_e = self.rev_entity_vocab[episode.start_entities[b * self.test_rollouts]]                 # Map id to entity for source node
                    end_e = self.rev_entity_vocab[episode.end_entities[b * self.test_rollouts]]                     # Map id to entity for answer node
                    
                    # Question Header Information
                    paths[question_txt].append(str(start_e) + "\t" + str(end_e) + "\n")
                    paths[question_txt].append("Reward:" + str(1 if answer_pos != None and answer_pos < 10 else 0) + "\n") # Answered correctly if top10
                    for r in sorted_indx[b]:                        # Go through paths sorted by score (highest first)
                        indx = b * self.test_rollouts + r           # Convert to global index
                        if rewards[indx] == self.positive_reward:
                            rev = 1                                 # This path succeeded
                        else:
                            rev = -1                                # This path failed

                        # Answer Summary (StartEntity, EndEntity, PathScore)
                        answers.append(self.rev_entity_vocab[se[b,r]]+'\t'+ self.rev_entity_vocab[ce[b,r]]+'\t'+ str(self.log_probs[b,r])+'\n')

                        # Detailed Path Trajectory (entities sequence, relation sequence, success indicator, path score)
                        paths[question_txt].append(
                            '\t'.join([str(self.rev_entity_vocab[e[indx]]) for e in
                                       self.entity_trajectory]) + '\n' + '\t'.join(
                                [str(self.rev_relation_vocab[re[indx]]) for re in self.relation_trajectory]) + '\n' + str(
                                rev) + '\n' + str(
                                self.log_probs[b, r]) + '\n___' + '\n')

                    paths[question_txt].append("#####################\n") # clear distinction for different attempts of same question

            # Update overall rewards (Episode-wise)
            all_final_reward_1 += final_reward_1
            all_final_reward_3 += final_reward_3
            all_final_reward_5 += final_reward_5
            all_final_reward_10 += final_reward_10
            all_final_reward_20 += final_reward_20
            mrr += final_mrr

        # Update total rewards
        all_final_reward_1 /= total_examples
        all_final_reward_3 /= total_examples
        all_final_reward_5 /= total_examples
        all_final_reward_10 /= total_examples
        all_final_reward_20 /= total_examples
        mrr /= total_examples

        # Save best performing model based on hits@10
        if save_model:
            if all_final_reward_10 >= max_hits_at_10:
                max_hits_at_10 = all_final_reward_10 # Update max_hits_at_10
                self.save_path = self.model_saver.save(sess, self.model_dir + "model.ckpt")

        # Store the paths for each question
        if print_paths:
            logger.info(f"[ printing paths at {os.path.join(self.output_dir, 'test_beam')} ]")
            for q in paths:
                j = q.replace('/', '-')
                with codecs.open(self.path_logger_file_ + '_' + j, 'a', 'utf-8') as pos_file:
                    for p in paths[q]:
                        pos_file.write(p)
            with open(self.path_logger_file_ + 'answers', 'w') as answer_file:
                for a in answers:
                    answer_file.write(a)

        with open(os.path.join(self.output_dir, 'scores.txt'), 'a') as score_file:
            score_file.write(f"Hits@1: {all_final_reward_1:7.4f}")
            score_file.write("\n")
            score_file.write(f"Hits@3: {all_final_reward_3:7.4f}")
            score_file.write("\n")
            score_file.write(f"Hits@5: {all_final_reward_5:7.4f}")
            score_file.write("\n")
            score_file.write(f"Hits@10: {all_final_reward_10:7.4f}")
            score_file.write("\n")
            score_file.write(f"Hits@20: {all_final_reward_20:7.4f}")
            score_file.write("\n")
            score_file.write(f"MRR: {mrr:7.4f}")
            score_file.write("\n")
            score_file.write("\n") 

        logger.info(f"Hits@1: {all_final_reward_1:7.4f}")
        logger.info(f"Hits@3: {all_final_reward_3:7.4f}")
        logger.info(f"Hits@5: {all_final_reward_5:7.4f}")
        logger.info(f"Hits@10: {all_final_reward_10:7.4f}")
        logger.info(f"Hits@20: {all_final_reward_20:7.4f}")
        logger.info(f"MRR: {mrr:7.4f}")

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

if __name__ == '__main__':##

    # read command line options
    options = read_options_nlq()

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

    # Reading the vocab files
    logger.info('Reading vocab files (ent & rel to id)...')
    relation_vocab = json.load(open(os.path.join(options['vocab_dir'], 'relation_vocab.json')))
    entity_vocab = json.load(open(os.path.join(options['vocab_dir'], 'entity_vocab.json')))

    logger.info('Total number of entities {}'.format(len(entity_vocab)))
    logger.info('Total number of relations {}'.format(len(relation_vocab)))

    save_path = ''
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = False
    config.log_device_placement = False

    # Set seed for reproducibility
    set_seeds(options['seed'])

    embedding_server = EmbeddingServer(options['question_tokenizer_name'])

    # Training a model from scratch
    if not options['load_model']:
        trainer = TrainerNLQ(
            options, 
            entity_vocab=entity_vocab, 
            relation_vocab=relation_vocab, 
            embedding_server=embedding_server
        )
        with tf.compat.v1.Session(config=config) as sess:
            sess.run(trainer.initialize())
            trainer.initialize_pretrained_embeddings(sess=sess)

            trainer.train(sess)
            save_path = trainer.save_path
            path_logger_file = trainer.path_logger_file
            output_dir = trainer.output_dir

        tf.compat.v1.reset_default_graph()
    # Providing the configurations for best model
    else:
        logger.info("Skipping training")
        logger.info(f"Loading model from {options['model_load_dir']}")

        save_path = options['model_load_dir']
        path_logger_file = options['path_logger_file']
        output_dir = options['output_dir']

    # Evaluating Model, will require new initialization to avoid graph errors
    trainer = TrainerNLQ(
        options, 
        entity_vocab=entity_vocab, 
        relation_vocab=relation_vocab, 
        embedding_server=embedding_server
    )
    
    with tf.compat.v1.Session(config=config) as sess:
        trainer.initialize(restore=save_path, sess=sess) # check if it is fine to initialize an already trained model or if we need to create one before this line

        trainer.test_rollouts = 100                      # set test rollouts to 100 for evaluation

        # create files to store results
        os.makedirs(os.path.join(path_logger_file, "test_beam"), exist_ok=True)
        trainer.path_logger_file_ = os.path.join(path_logger_file, "test_beam", "paths")
        with open(os.path.join(output_dir, 'scores.txt'), 'a') as score_file:
            score_file.write("Test (beam) scores with best model from " + save_path + "\n")

        # Perform Evaluation
        trainer.test(sess, beam=True, print_paths=True, save_model=False, mode='test')
    
    logging.info(f"Evaluation completed. Closing Server")
    embedding_server.close()  # Close the embedding server connection
