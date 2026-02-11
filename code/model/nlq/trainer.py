"""
MINERVA trainer for natural language question answering over knowledge graphs.

This module implements the complete training and evaluation pipeline for the MINERVA
reinforcement learning agent. It orchestrates policy gradient training, baseline
variance reduction, comprehensive evaluation with multiple metrics, and model
checkpointing for knowledge graph reasoning tasks.

Key components:
- REINFORCE policy gradient training with baseline variance reduction
- Multi-environment support for train/dev/test data splits
- Comprehensive evaluation with Hits@K and MRR metrics
- Beam search decoding for improved inference performance
- Model checkpointing and restoration capabilities
- Memory-efficient episode processing with TensorFlow partial_run
- Detailed path logging and reasoning trajectory analysis

Classes:
    TrainerNLQ: Main trainer class for MINERVA NLQ reasoning
"""
# TODO: Inspect best saving for checkpoints
# TODO: Load the model in a separate file to check if everything is working fine

from __future__ import absolute_import
from __future__ import division

import codecs
import gc
import json
import logging
import os
import resource
import sys
import time
from collections import defaultdict, namedtuple

import numpy as np
import tensorflow as tf
from scipy.special import logsumexp as lse
from tqdm import tqdm

import wandb

from code.data.embedding_server import EmbeddingServer
from code.model.baseline import ReactiveBaseline
from code.model.nlq.agent import AgentNLQ
from code.model.nlq.environment import EnvNLQ
from code.data.setup import set_seeds
from code.options import read_options

from typing import Dict, Any, List, Tuple, Optional, Union

logger = logging.getLogger()
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

EvaluationMetrics = namedtuple('EvaluationMetrics', [
    'hits_at_1', 'hits_at_3', 'hits_at_5', 'hits_at_10', 'hits_at_20', 
    'answer_recall', 'answer_precision', 'answer_f1',
    'path_recall', 'path_precision', 'path_f1',
    'node_recall', 'node_precision', 'node_f1',
    'rel_recall', 'rel_precision', 'rel_f1',
    'mrr', 'max_hits_at_1', 'max_mrr'
])

class TrainerNLQ(object):
    """
    MINERVA trainer for reinforcement learning-based knowledge graph reasoning.
    
    Orchestrates the complete training and evaluation pipeline for natural language
    question answering over knowledge graphs. Implements policy gradient training
    with REINFORCE algorithm, baseline variance reduction, and comprehensive
    evaluation using multiple metrics and search strategies.
    
    The trainer manages:
    - Policy gradient training with episode generation and reward computation
    - Baseline variance reduction using reactive baseline for stable training
    - Multi-environment coordination for train/dev/test data splits
    - Comprehensive evaluation with Hits@K, MRR, and optional beam search
    - Model checkpointing based on performance improvements
    - Memory-efficient processing using TensorFlow partial_run
    - Detailed logging and path analysis for reasoning interpretability
    
    Architecture:
    - Agent: LSTM-based policy network for action selection
    - Environment: Knowledge graph navigation with episode management
    - Baseline: Reactive baseline for variance reduction in policy gradients
    - Optimizer: Adam optimizer with gradient clipping for stable training
    
    Attributes:
        agent (AgentNLQ): MINERVA agent for knowledge graph reasoning
        environment (EnvNLQ): Knowledge graph environment for episode generation
        baseline (ReactiveBaseline): Baseline estimator for variance reduction
        optimizer: Adam optimizer for policy gradient updates
        entity_vocab (Dict[str, int]): Entity name to ID vocabulary mapping
        relation_vocab (Dict[str, int]): Relation name to ID vocabulary mapping
        save_path (Optional[str]): Path to saved model checkpoint
        
    Example:
        >>> trainer = TrainerNLQ(
        ...     batch_size=128, num_rollouts=20, positive_reward=1.0, negative_reward=0.0,
        ...     path_length=3, test_rollouts=100, data_input_dir="./data",
        ...     question_tokenizer_name="bert-base-uncased", cached_QAMetaData_path="./cache",
        ...     raw_QAData_path="./raw", max_num_actions=200, embedding_size=50,
        ...     hidden_size=50, use_entity_embeddings=False, train_entity_embeddings=False,
        ...     train_relation_embeddings=True, LSTM_layers=1, learning_rate=1e-3,
        ...     grad_clip_norm=5, gamma=1.0, Lambda=0.0, beta=1e-2, total_iterations=2000,
        ...     eval_every=100, output_dir="./output", model_dir="./models",
        ...     path_logger_file="./logs", 
        ...     pool="max", seed=42, entity_vocab=entity_vocab, relation_vocab=relation_vocab,
        ...     embedding_server=embedding_server
        ... )
        >>> trainer.initialize(sess)
        >>> trainer.train(sess)
        >>> hits1, hits3, hits5, hits10, hits20 = trainer.test(sess, beam=True)
    """

    def __init__(
        self,
        batch_size: int,
        num_rollouts: int,
        positive_reward: float,
        negative_reward: float,
        path_length: int,
        test_rollouts: int,
        data_input_dir: str,
        question_tokenizer_name: str,
        question_format: str,
        cached_QAMetaData_path: str,
        raw_QAData_path: str,
        max_num_actions: int,
        embedding_size: int,
        hidden_size: int,
        use_entity_embeddings: bool,
        train_entity_embeddings: bool,
        train_relation_embeddings: bool,
        LSTM_layers: int,
        projection_adapter: str,
        projection_layers: int,
        projection_hidden: int,
        learning_rate: float,
        grad_clip_norm: int,
        gamma: float,
        Lambda: float,
        beta: float,
        total_iterations: int,
        eval_every: int,
        output_dir: str,
        model_dir: str,
        path_logger_file: str,
        pool: str,
        seed: int,
        entity_vocab: Dict[str, int],
        relation_vocab: Dict[str, int],
        multi_answers: bool = False,
        use_full_graph: bool = False,
        use_stop_signal: bool = False,
        use_restart_signal: bool = False,
        use_beam: Optional[bool] = False,
        embedding_server: Optional[EmbeddingServer] = None,
        use_wandb: bool = False
    ) -> None:
        """
        Initialize the MINERVA trainer with all necessary components for training and evaluation.
        
        Sets up the complete training pipeline including agent, environment, baseline,
        optimizer, and all configuration parameters. Establishes vocabulary mappings
        and prepares the system for policy gradient training on knowledge graph
        reasoning tasks.
        
        Args:
            batch_size: Number of questions processed in each training batch
            num_rollouts: Number of training rollouts per question
            positive_reward: Reward for correct answers
            negative_reward: Reward for incorrect answers
            path_length: Maximum reasoning steps allowed
            test_rollouts: Number of evaluation rollouts per question
            data_input_dir: Directory containing knowledge graph data files
            question_tokenizer_name: Tokenizer name for question embeddings
            question_format: Format of the question input ('full_text', 'relation_only', 'graph_only')
            cached_QAMetaData_path: Path to cached tokenized QA metadata JSON file
            raw_QAData_path: Path to the raw QA CSV dataset
            max_num_actions: Maximum number of relations/actions per entity
            embedding_size: Embedding dimension for entities and relations
            hidden_size: Hidden state size for LSTM layers
            use_entity_embeddings: Whether to use entity embeddings
            train_entity_embeddings: Whether to fine-tune entity embeddings
            train_relation_embeddings: Whether to train relation embeddings
            LSTM_layers: Number of LSTM layers in the agent network
            projection_adapter: Type of question projection adapter ('linear', 'mlp', 'residual')
            projection_layers: Number of layers in the projection adapter (if applicable)
            learning_rate: Learning rate for the optimizer
            grad_clip_norm: Maximum gradient norm for gradient clipping
            gamma: Discount factor for future rewards in RL
            Lambda: Baseline regularization parameter
            beta: Entropy regularization coefficient for exploration
            total_iterations: Total number of training iterations
            eval_every: Frequency of evaluation (every N training iterations)
            output_dir: Base directory for all output files and logs
            model_dir: Directory to save trained model checkpoints
            path_logger_file: Path for logging reasoning trajectories
            pool: Pooling method for evaluation of rollouts ('max', 'sum')
            seed: Random seed for reproducibility
            entity_vocab: Entity name to integer ID mapping for embedding lookup
            relation_vocab: Relation name to integer ID mapping for embedding lookup
            use_beam: Whether to use beam search during decoding
            embedding_server: Optional service for generating question embeddings
                             from natural language text using pre-trained models
                             
        Note:
            - Stores all parameters as instance attributes for easy access
            - Creates shared environment to save memory across train/dev/test modes
            - Disables TensorFlow eager execution for graph-based training
            - Sets up vocabulary mappings and special token IDs (PAD tokens)
        """

        # Store all parameters as instance attributes
        self.batch_size = batch_size
        self.num_rollouts = num_rollouts
        self.positive_reward = positive_reward
        self.negative_reward = negative_reward
        self.path_length = path_length
        self.test_rollouts = test_rollouts
        self.data_input_dir = data_input_dir
        self.question_tokenizer_name = question_tokenizer_name
        self.question_format = question_format
        self.cached_QAMetaData_path = cached_QAMetaData_path
        self.raw_QAData_path = raw_QAData_path
        self.multi_answers = multi_answers
        self.max_num_actions = max_num_actions
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.use_entity_embeddings = use_entity_embeddings
        self.train_entity_embeddings = train_entity_embeddings
        self.train_relation_embeddings = train_relation_embeddings
        self.LSTM_layers = LSTM_layers
        self.projection_adapter = projection_adapter
        self.projection_layers = projection_layers
        self.projection_hidden = projection_hidden
        self.learning_rate = learning_rate
        self.grad_clip_norm = grad_clip_norm
        self.gamma = gamma
        self.Lambda = Lambda
        self.beta = beta
        self.total_iterations = total_iterations
        self.eval_every = eval_every
        self.output_dir = output_dir
        self.model_dir = model_dir
        self.path_logger_file = path_logger_file
        self.pool = pool
        self.use_beam = use_beam
        self.seed = seed
        self.use_wandb = use_wandb

        # Debug logging for WANDB
        logger.info(f"Trainer initialized with use_wandb={self.use_wandb}")

        # shared environment accross modes, save space with graph builder and textual embeddings
        self.environment = EnvNLQ(
            batch_size=batch_size,
            num_rollouts=num_rollouts,
            positive_reward=positive_reward,
            negative_reward=negative_reward,
            path_length=path_length,
            test_rollouts=test_rollouts,
            data_input_dir=data_input_dir,
            question_tokenizer_name=question_tokenizer_name,
            question_format=question_format,
            cached_QAMetaData_path=cached_QAMetaData_path,
            raw_QAData_path=raw_QAData_path,
            multi_answers=multi_answers,
            max_num_actions=max_num_actions,
            entity_vocab=entity_vocab, 
            relation_vocab=relation_vocab, 
            mode='train',
            use_full_graph=use_full_graph,
            use_stop_signal=use_stop_signal,
            use_restart_signal=use_restart_signal,
            seed=seed,
            embedding_server=embedding_server
        )
        
        # Disable Eager Execution for the rest of the code
        # but not before initializing Embedding Server
        tf.compat.v1.disable_eager_execution()

        self.agent = AgentNLQ(
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            use_entity_embeddings=use_entity_embeddings,
            train_entity_embeddings=train_entity_embeddings,
            train_relation_embeddings=train_relation_embeddings,
            num_rollouts=num_rollouts,
            test_rollouts=test_rollouts,
            LSTM_layers=LSTM_layers,
            projection_adapter=projection_adapter,
            projection_layers=projection_layers,
            projection_hidden=projection_hidden,
            batch_size=batch_size,
            entity_vocab=entity_vocab, 
            relation_vocab=relation_vocab
        )

        self.save_path = None

        # Vocabulary mappings for entity and relation conversion
        self.entity_vocab = entity_vocab
        self.relation_vocab = relation_vocab
        self.rev_relation_vocab = self.environment.grapher.rev_relation_vocab
        self.rev_entity_vocab = self.environment.grapher.rev_entity_vocab

        # Training components
        self.baseline = ReactiveBaseline(l=self.Lambda)
        self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate)


    def calc_reinforce_loss(self) -> tf.Tensor:
        """
        Calculate REINFORCE policy gradient loss with baseline variance reduction.
        
        Implements the REINFORCE algorithm for policy gradient training by:
        1. Computing per-example cross-entropy losses from agent policy
        2. Subtracting baseline value to reduce variance in gradient estimates
        3. Normalizing advantages using mean and standard deviation
        4. Weighting policy losses by normalized advantages  
        5. Adding entropy regularization to encourage exploration
        
        The resulting loss encourages actions that lead to higher-than-expected
        rewards while penalizing actions that lead to lower-than-expected rewards.
        Baseline subtraction and advantage normalization significantly reduce
        training variance.
        
        Returns:
            Scalar loss tensor combining weighted policy gradient loss and entropy
            regularization, ready for gradient descent optimization.
            
        Note:
            - Uses reactive baseline for variance reduction
            - Advantage normalization prevents gradient explosion
            - Entropy regularization weight decays during training
            - Final loss is mean over batch and time dimensions
        """
        loss = tf.stack(self.per_example_loss, axis=1)  # [B, T]

        self.tf_baseline = self.baseline.get_baseline_value()

        # Compute advantages and normalize for stable training
        final_reward = self.cum_discounted_reward - self.tf_baseline
        reward_mean, reward_var = tf.nn.moments(final_reward, axes=[0, 1])

        # Constant added for numerical stability
        reward_std = tf.sqrt(reward_var) + 1e-6
        final_reward = tf.math.divide(final_reward - reward_mean, reward_std)

        loss = tf.multiply(loss, final_reward)  # [B, T]
        self.loss_before_reg = loss

        total_loss = tf.reduce_mean(loss) - self.decaying_beta * self.entropy_reg_loss(self.per_example_logits)  # scalar

        return total_loss

    def entropy_reg_loss(self, all_logits: List[tf.Tensor]) -> tf.Tensor:
        """
        Calculate entropy regularization loss to encourage policy exploration.
        
        Computes the negative entropy of action probability distributions to
        encourage exploration during training. Higher entropy (more uniform
        action probabilities) results in lower penalty, preventing the policy
        from becoming too deterministic too early in training.
        
        Args:
            all_logits: Log probabilities over actions at each time step.
                List of length T, each tensor shape: [batch_size, max_actions]
                
        Returns:
            Scalar entropy regularization loss. Negative entropy means higher
            entropy (better exploration) reduces the total training loss.
            
        Note:
            - Stacks logits across time dimension for efficient computation
            - Uses exp(log_probs) to recover probability distributions
            - Computes H(π) = -Σ π(a|s) log π(a|s) for each state
            - Takes mean over batch and action dimensions
        """
        all_logits = tf.stack(all_logits, axis=2)  # [B, MAX_NUM_ACTIONS, T]
        entropy_policy = - tf.reduce_mean(tf.reduce_sum(tf.multiply(tf.exp(all_logits), all_logits), axis=1))  # scalar
        return entropy_policy

    def initialize(self, restore: Optional[str] = None, sess: Optional[tf.compat.v1.Session] = None) -> Union[tf.Operation, None]:
        """
        Initialize TensorFlow computational graph and training infrastructure.
        
        Constructs the complete TensorFlow graph including:
        - Input placeholders for candidate actions, questions, and rewards
        - Agent policy network with forward pass and loss computation
        - Training operations with gradient clipping and optimization
        - Test/inference operations for evaluation and beam search
        - Model saving and restoration capabilities
        - Optional pretrained embedding initialization
        
        Args:
            restore: Path to checkpoint file for model restoration. If None,
                    initializes with random weights according to layer initializers.
            sess: TensorFlow session for restoration operations. Required if
                 restore path is provided.
                 
        Returns:
            Variable initializer operation if training from scratch, or None
            if restoring from checkpoint.
            
        Note:
            - Creates separate graphs for training (full episodes) and testing (single steps)
            - Uses variable scope reuse for parameter sharing between train/test graphs
            - Sets up partial_run compatibility for efficient episode processing
            - Configures model saver for checkpoint management
        """

        logger.info("Creating TF graph...")

        # Initialize placeholder lists for episode sequences
        self.candidate_relation_sequence = []
        self.candidate_entity_sequence = []
        self.entity_sequence = []

        # Tensorflow Placeholders
        # New: external question embedding (e.g., BERT). Dim can be anything; we let dense learn to use it.
        self.question_embedding = tf.compat.v1.placeholder(tf.float32, [None, self.environment.token_embedding_dim], name="question_embedding") # [B*num_rollouts, token_embedding_dim]
        
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

        # Create time-step specific placeholders
        for t in range(self.path_length):
            next_rel = tf.compat.v1.placeholder(tf.int32, [None, self.max_num_actions],
                                                   name=f"next_relations_{t}")                                  # candidate relations from current entity  [B*num_rollouts,]
            next_ent = tf.compat.v1.placeholder(tf.int32, [None, self.max_num_actions],
                                                     name=f"next_entities_{t}")                                 # candidate entities from current entity [B*num_rollouts,]
            cur_ent = tf.compat.v1.placeholder(tf.int32, [None, ], name=f"current_entities_{t}")                # current locations [B*num_rollouts,]

            self.candidate_relation_sequence.append(next_rel)                                                   # list of candidate relations at each step
            self.candidate_entity_sequence.append(next_ent)                                                     # list of candidate entities at each step
            self.entity_sequence.append(cur_ent)                                                                # list of current entities at each step

        self.loss_before_reg = tf.constant(0.0)

        # Build training computation graph
        self.per_example_loss, self.per_example_logits, self.action_idx = self.agent(
            self.candidate_relation_sequence,
            self.candidate_entity_sequence,
            self.entity_sequence,
            self.question_embedding, 
            self.range_arr, 
            self.path_length
        )

        self.loss_op = self.calc_reinforce_loss()
        self.train_op = self.bp(self.loss_op)

        # Building the test graph
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
                formated_state,
                self.prev_relation, 
                self.question_embedding,
                self.current_entities, 
                self.range_arr
            )
            self.test_state = tf.stack(test_state)

        logger.info('TF Graph ready (NLQ).')
        self.model_saver = tf.compat.v1.train.Saver(max_to_keep=2)

        if not restore:
            return tf.compat.v1.global_variables_initializer()
        else:
            return self.model_saver.restore(sess, restore)


    def bp(self, cost: tf.Tensor) -> tf.Operation:
        """
        Set up backpropagation with baseline update and gradient clipping.
        
        Creates the complete training operation that updates both the policy
        parameters and the baseline estimator. Includes gradient clipping to
        prevent exploding gradients, which is crucial for stable training in
        reinforcement learning settings.
        
        Args:
            cost: Scalar loss tensor to minimize through gradient descent.
            
        Returns:
            Training operation that when executed performs:
            - Baseline update with current reward estimates
            - Gradient computation for all trainable variables
            - Gradient clipping by global norm for stability
            - Parameter updates using Adam optimizer
            
        Note:
            - Baseline is updated before gradient computation
            - Uses control dependencies to ensure proper execution order
            - Gradient clipping norm is configurable via grad_clip_norm parameter
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
        
        Computes the discounted return G_t from each time step using the formula:
        G_t = R_t + γ*R_{t+1} + γ²*R_{t+2} + ... + γ^{T-t}*R_T
        
        This provides the expected long-term reward from each state, which serves
        as the target for baseline estimation and the weight for policy gradients.
        The discounting encourages actions that lead to rewards sooner rather
        than later.
        
        Args:
            rewards: Final rewards received at episode termination.
                Shape: [batch_size] with values typically in {-1, +1}
                
        Returns:
            Cumulative discounted rewards for all time steps.
            Shape: [batch_size, path_length] where entry [i,t] represents
            the discounted return from time step t for episode i.
            
        Note:
            - Only final time step gets immediate reward, others get discounted future
            - Uses backward iteration for efficient computation
            - Gamma (discount factor) controls future reward importance
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
        Configure TensorFlow partial_run for efficient episode processing.
        
        Sets up the fetches, feeds, and feed_dict structures required for
        TensorFlow's partial_run functionality. This enables dynamic episode
        unrolling while maintaining computational efficiency by pre-declaring
        all tensors that will be used during training.
        
        Returns:
            Tuple containing:
                - fetches: List of tensors to fetch during partial_run execution
                - feeds: List of placeholder tensors for feeding input data  
                - feed_dict: List of feed dictionaries for each reasoning step,
                  pre-configured with constant values and None placeholders
                  for step-specific data
                  
        Note:
            - Enables memory-efficient processing of variable-length episodes
            - Pre-allocates feed dictionaries to avoid repeated allocation
            - Separates constant (question, range) and step-varying inputs
        """
        # create fetches for partial_run_setup
        fetches = self.per_example_loss  + self.action_idx + [self.loss_op] + self.per_example_logits + [self.dummy]
        feeds =  self.candidate_relation_sequence + self.candidate_entity_sequence + \
                [self.question_embedding] + [self.cum_discounted_reward] + [self.range_arr] + self.entity_sequence


        feed_dict = [{} for _ in range(self.path_length)]

        # Pass the memory address of the placeholder to the feed_dict
        # The following placeholders that stay constant through the hops/steps:
        feed_dict[0][self.question_embedding] = None
        feed_dict[0][self.range_arr] = np.arange(self.batch_size*self.num_rollouts)
        
        # Configure step-varying placeholders
        for i in range(self.path_length):
            feed_dict[i][self.candidate_relation_sequence[i]] = None
            feed_dict[i][self.candidate_entity_sequence[i]] = None
            feed_dict[i][self.entity_sequence[i]] = None

        return fetches, feeds, feed_dict

    def train(self, sess: tf.compat.v1.Session) -> None:
        """
        Execute the complete MINERVA training loop using policy gradient reinforcement learning.

        Performs episodic training where each episode involves multi-hop reasoning
        through the knowledge graph. The training loop:
        1. Iterates through training data until maximum episodes completed
        2. For each episode, unrolls policy for path_length steps
        3. Collects actions, logits, and losses at each reasoning step
        4. Computes final rewards based on answer correctness
        5. Calculates discounted returns for policy gradient weighting
        6. Updates policy parameters using REINFORCE algorithm
        7. Updates baseline estimator for variance reduction
        8. Logs comprehensive training statistics and progress
        9. Periodically evaluates on development data and saves models
        
        Uses TensorFlow's partial_run for memory-efficient episode processing,
        enabling dynamic unrolling while maintaining computational efficiency.
        
        Args:
            sess: Active TensorFlow session for executing training operations.
            
        Note:
            - Training continues until total_iterations batches are processed
            - Evaluation occurs every eval_every batches on development set
            - Model checkpointing based on development set performance
            - Comprehensive logging includes hits, rewards, and loss statistics
            - Memory usage monitoring and garbage collection for stability
        """
        logger.info("Starting training...")
        fetches, feeds, feed_dict = self.gpu_io_setup()

        train_loss = 0.0
        dev_mrr = 0.0
        dev_hits = 0.0
        self.batch_counter = 0
        self.environment.change_mode('train')                           # Change environment mode to training
        for episode in self.environment.get_episodes():                 # Provide the current episode, can be repeated
            assert episode.mode == 'train', "Environment mode must be 'train' for training episodes"

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

            # Update training statistics
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

            # Log training progress
            logger.info(
                f"batch_counter: {self.batch_counter:<4d}, num_hits: {np.sum(rewards):<7.4f}, "
                f"avg. reward per batch: {avg_reward:<7.4f}, num_ep_correct: {num_ep_correct:<4d}, "
                f"avg_ep_correct: {num_ep_correct / self.batch_size:<7.4f}, train_loss: {train_loss:<7.4f}"
            )

            # Log training metrics to WANDB
            if self.use_wandb:
                wandb.log({
                    'train/batch_counter': self.batch_counter,
                    'train/loss': float(train_loss),
                    'train/batch_total_loss': float(batch_total_loss),
                    'train/num_hits': float(np.sum(rewards)),
                    'train/avg_reward': float(avg_reward),
                    'train/num_ep_correct': int(num_ep_correct),
                    'train/avg_ep_correct': float(num_ep_correct / self.batch_size),
                    'train/memory_usage_kb': int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                }, step=self.batch_counter)

            # Periodic evaluation and model saving
            if self.batch_counter % self.eval_every == 0:
                with open(os.path.join(self.output_dir, 'scores.txt'), 'a') as score_file:
                    score_file.write("Score for iteration " + str(self.batch_counter) + "\n")

                dev_metrics = self.test(
                    sess, 
                    beam=self.use_beam, 
                    print_paths=False, 
                    mode='dev',
                    max_hits=dev_hits,
                    max_mrr=dev_mrr,
                )

                dev_hits = dev_metrics.hits_at_1
                dev_mrr = dev_metrics.mrr

                # Important: Change back to training mode to change the data
                self.environment.change_mode('train')

            logger.info('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

            # Clean up (garbage collector to free space)
            gc.collect()

            if self.batch_counter >= self.total_iterations: # if enough iterations have been completed, break out of training
                break

    def test(self, sess: tf.compat.v1.Session, beam: bool = False, print_paths: bool = False, 
             save_model: bool = True, mode: str = 'dev', max_hits: float = 0, max_mrr: float = 0
        ) -> EvaluationMetrics:
        """
        Evaluate the trained MINERVA agent with comprehensive metrics and optional beam search.
        
        Performs thorough evaluation on test/dev data including:
        - Hits@K metrics (K=1,3,5,10,20) measuring answer prediction accuracy
        - Mean Reciprocal Rank (MRR) for ranking quality assessment
        - Optional beam search for improved inference performance  
        - Detailed path visualization and reasoning trajectory analysis
        - Model checkpointing based on performance improvements
        
        The evaluation uses multiple rollouts per question and supports two
        aggregation modes: 'max' (best rollout) and 'sum' (score aggregation).
        
        Args:
            sess: Active TensorFlow session for inference operations.
            beam: Whether to use beam search decoding instead of greedy selection.
                 Beam search typically improves performance but increases computation.
            print_paths: Whether to generate detailed reasoning path logs for analysis.
            save_model: Whether to save model checkpoint if performance improves.
            mode: Data split to evaluate on ('dev', 'test', or 'train').
                
        Returns:
            EvaluationMetrics: Named tuple containing:
                - hits_at_1: Hits@1 score 
                - hits_at_3: Hits@3 score
                - hits_at_5: Hits@5 score
                - hits_at_10: Hits@10 score
                - hits_at_20: Hits@20 score
                - mrr: Mean Reciprocal Rank score
                - max_hits: Maximum Hits@1 observed (for model saving)
                - max_mrr: Maximum MRR observed (for model saving)
                - answer_recall: Answer-level recall (if multi_answers)
                - answer_precision: Answer-level precision (if multi_answers)
                - answer_f1: Answer-level F1 score (if multi_answers)
                - path_recall: Path-level recall (if paths exist)
                - path_precision: Path-level precision (if paths exist)
                - path_f1: Path-level F1 score (if paths exist)
                - node_recall: Node-level recall (if paths exist)
                - node_precision: Node-level precision (if paths exist)
                - node_f1: Node-level F1 score (if paths exist)
                - rel_recall: Relation-level recall (if paths exist)
                - rel_precision: Relation-level precision (if paths exist)
                - rel_f1: Relation-level F1 score (if paths exist)
                
        Note:
            - Hits@N metrics depend on number of rollouts (capped at rollout count)
            - Uses different scoring strategies based on self.pool setting
            - Beam search maintains multiple reasoning paths for better coverage
            - Path logging provides detailed analysis of reasoning trajectories
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
        
        if self.multi_answers:
            all_final_answer_recall = 0
            all_final_answer_precision = 0
            all_final_answer_f1 = 0
        else:
            all_final_answer_recall = None
            all_final_answer_precision = None
            all_final_answer_f1 = None

        if self.environment.check_paths_exist():
            all_final_path_recall = 0
            all_final_path_precision = 0
            all_final_path_f1 = 0
            all_final_node_recall = 0
            all_final_node_precision = 0
            all_final_node_f1 = 0
            all_final_rel_recall = 0
            all_final_rel_precision = 0
            all_final_rel_f1 = 0
        else:
            all_final_path_recall = None
            all_final_path_precision = None
            all_final_path_f1 = None
            all_final_node_recall = None
            all_final_node_precision = None
            all_final_node_f1 = None
            all_final_rel_recall = None
            all_final_rel_precision = None
            all_final_rel_f1 = None
        mrr = 0                         # Overall results for MRR

        # Changing the environment to test/dev data and resetting values
        self.environment.change_mode(mode)
        # For beam search, limit rollouts to max_num_actions to prevent indexing errors
        effective_rollouts = min(self.test_rollouts, self.max_num_actions) if beam else self.test_rollouts
        self.environment.change_test_rollouts(effective_rollouts)   # modifying the number of rollouts for evaluation
        total_examples = self.environment.total_no_examples         # total number of questions
        test_batch_counter = 0
        logger.info(f"Testing with mode: {mode} on {total_examples} samples...")
        if beam:
            logger.info(f"Beam search enabled: using {effective_rollouts} rollouts (limited by max_num_actions={self.max_num_actions})")
        for episode in tqdm(self.environment.get_episodes(), desc="Evaluating"):
            assert episode.mode != 'train', "Environment is in training mode!"

            temp_batch_size = episode.no_examples                   # batch size, can vary in test due to the last batch
            test_batch_counter += temp_batch_size
            logger.info(f"Evaluating samples {test_batch_counter}/{total_examples} with {effective_rollouts} rollouts...")

            # Set Initial Beams Probs
            beam_probs = np.zeros((temp_batch_size * effective_rollouts, 1)) # Cumulative scores from previous steps [batch_size*k, 1]

            # Provide Initial Variables
            state = episode.get_state()                             # Initial State (current_entities, next_entities, next_relations)
            mem = self.agent.get_mem_shape()                        # LSTM Memory Shape (num lstm layers, 2, batch size, memory size)
            agent_mem = np.zeros((mem[0], mem[1], temp_batch_size*effective_rollouts, mem[3]), dtype='float32')
            previous_relation = np.ones((temp_batch_size * effective_rollouts, ), dtype='int64') * self.relation_vocab['DUMMY_START_RELATION']
            
            feed_dict = {
                self.range_arr: np.arange(temp_batch_size * effective_rollouts),
                self.question_embedding: episode.get_question_embedding(),              # question embeddings
            }

            ####logger code####
            if print_paths or self.environment.check_paths_exist():
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
                    k = effective_rollouts  # Use the same effective rollouts calculated earlier
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
                    if print_paths or self.environment.check_paths_exist():
                        for j in range(i):
                            self.entity_trajectory[j] = self.entity_trajectory[j][y]
                            self.relation_trajectory[j] = self.relation_trajectory[j][y]
                
                ####logger code####
                if print_paths or self.environment.check_paths_exist(): # Store the current path before the environment update
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
            if print_paths or self.environment.check_paths_exist(): # Store the current paths (entity only)
                self.entity_trajectory.append(state['current_entities'])

            # Calculate the final reward
            rewards = episode.get_reward()  # [B*test_rollouts]

            # Reshape the reward to [orig_batch_size, num_rollouts], to calculate for how many of the
            # entity pair, at least one of the paths arrive at the correct answer
            reward_reshape = np.reshape(rewards, (temp_batch_size, self.test_rollouts))  # [orig_batch, test_rollouts]
            self.log_probs = np.reshape(self.log_probs, (temp_batch_size, self.test_rollouts))
            sorted_indx = np.argsort(-self.log_probs)

            if self.multi_answers:
                precision, recall, f1_score = episode.get_multi_answer_coverage()
                all_final_answer_recall += recall.sum()
                all_final_answer_precision += precision.sum()
                all_final_answer_f1 += f1_score.sum()
            
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
                
                if self.environment.check_paths_exist():   # If path existence checking is enabled
                    r = sorted_indx[b][0] # highest scoring path
                    indx = b * self.test_rollouts + r           # Convert to global index
                    entities_path = [e[indx] for e in self.entity_trajectory]
                    relations_path = [re[indx] for re in self.relation_trajectory]

                    # pop the first entity which is the source entity
                    entities_path = entities_path[1:]

                    # merge entities and path into a single path
                    merged_path = [[r, e] for r, e in zip(relations_path, entities_path)]
                    precision, recall, f1_score = episode.get_path_faithfulness(merged_path, b)
                    all_final_path_precision += precision
                    all_final_path_recall += recall
                    all_final_path_f1 += f1_score

                    precision, recall, f1_score = episode.get_node_coverage(entities_path, b)
                    all_final_node_precision += precision
                    all_final_node_recall += recall
                    all_final_node_f1 += f1_score

                    precision, recall, f1_score = episode.get_relation_coverage(relations_path, b)
                    all_final_rel_precision += precision
                    all_final_rel_recall += recall
                    all_final_rel_f1 += f1_score

                # Comprehensive reasoning path report
                if print_paths:
                    # Retrive Sample's context
                    question_txt = self.environment.batcher.translate_questions([episode.question_tokens[b]])[0]    # Convert question back to text
                    start_e = self.environment.batcher.translate_entities([episode.start_entities[b * self.test_rollouts]])                 # Map id to entity for source node
                    if self.multi_answers:
                        end_e = self.environment.batcher.translate_entities([episode.end_entities[b]])                     # Map id to entity for answer node
                    else:
                        end_e = self.environment.batcher.translate_entities([episode.end_entities[b * self.test_rollouts]])                     # Map id to entity for answer node

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
                        answers.append(self.environment.batcher.translate_entities([se[b,r]])[0]+'\t'+ self.environment.batcher.translate_entities([ce[b,r]])[0]+'\t'+ str(self.log_probs[b,r])+'\n')

                        # Detailed Path Trajectory (entities sequence, relation sequence, success indicator, path score)
                        paths[question_txt].append(
                            '\t'.join([str(self.environment.batcher.translate_entities([e[indx]])) for e in
                                       self.entity_trajectory]) + '\n' + '\t'.join(
                                [str(self.environment.batcher.translate_relations([re[indx]])) for re in self.relation_trajectory]) + '\n' + str(
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

        if self.multi_answers:
            all_final_answer_recall /= total_examples
            all_final_answer_precision /= total_examples
            all_final_answer_f1 /= total_examples
        
        if self.environment.check_paths_exist():
            all_final_path_recall /= total_examples
            all_final_path_precision /= total_examples
            all_final_path_f1 /= total_examples

            all_final_node_recall /= total_examples
            all_final_node_precision /= total_examples
            all_final_node_f1 /= total_examples

            all_final_rel_recall /= total_examples
            all_final_rel_precision /= total_examples
            all_final_rel_f1 /= total_examples

        # Save best performing model based on hits@1
        if save_model:
            # if a better hits at one is found, save model
            if all_final_reward_1 > max_hits:
                # log this information
                logger.info(f"New best model saved with Hits@1: {all_final_reward_1:7.4f} and MRR: {mrr:7.4f}")
                logger.info(f"Model saved based on improved Hits@1: {max_hits:7.4f} --> {all_final_reward_1:7.4f}")

                max_hits = all_final_reward_1 # Update max hits at 1
                max_mrr = mrr # Update max hits at 3
                self.save_path = self.model_saver.save(sess, self.model_dir + "model.ckpt")

            # if hits at 1 is the same, but a better mrr is found, save model
            elif (all_final_reward_1 == max_hits) and (mrr > max_mrr):
                # log this information
                logger.info(f"New best model saved with Hits@1: {all_final_reward_1:7.4f} and MRR: {mrr:7.4f}")
                logger.info(f"Model saved based on improved MRR: {max_mrr:7.4f} --> {mrr:7.4f}")

                max_mrr = mrr # Update max hits at 3
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
            score_file.write("Answer Metrics\n")
            score_file.write(f"\tHits@1: {all_final_reward_1:7.4f}\n")
            score_file.write(f"\tHits@3: {all_final_reward_3:7.4f}\n")
            score_file.write(f"\tHits@5: {all_final_reward_5:7.4f}\n")
            score_file.write(f"\tHits@10: {all_final_reward_10:7.4f}\n")
            score_file.write(f"\tHits@20: {all_final_reward_20:7.4f}\n")
            score_file.write(f"\tMRR: {mrr:7.4f}\n")
            if self.multi_answers:
                score_file.write(f"Multi-Answer Metrics\n")
                score_file.write(f"\tRecall: {all_final_answer_recall:7.4f}\n")
                score_file.write(f"\tPrecision: {all_final_answer_precision:7.4f}\n")
                score_file.write(f"\tF1 Score: {all_final_answer_f1:7.4f}\n")
            if self.environment.check_paths_exist():
                score_file.write(f"Path Faithfulness Metrics\n")
                score_file.write(f"\tPath Recall: {all_final_path_recall:7.4f}\n")
                score_file.write(f"\tPath Precision: {all_final_path_precision:7.4f}\n")
                score_file.write(f"\tPath F1 Score: {all_final_path_f1:7.4f}\n")

                score_file.write(f"Node Coverage Metrics\n")
                score_file.write(f"\tNode Recall: {all_final_node_recall:7.4f}\n")
                score_file.write(f"\tNode Precision: {all_final_node_precision:7.4f}\n")
                score_file.write(f"\tNode F1 Score: {all_final_node_f1:7.4f}\n")

                score_file.write(f"Relation Coverage Metrics\n")
                score_file.write(f"\tRelation Recall: {all_final_rel_recall:7.4f}\n")
                score_file.write(f"\tRelation Precision: {all_final_rel_precision:7.4f}\n")
                score_file.write(f"\tRelation F1 Score: {all_final_rel_f1:7.4f}\n")

            score_file.write("\n") 

        logger.info("Answer Metrics:")
        logger.info(f"\tHits@1: {all_final_reward_1:7.4f}")
        logger.info(f"\tHits@3: {all_final_reward_3:7.4f}")
        logger.info(f"\tHits@5: {all_final_reward_5:7.4f}")
        logger.info(f"\tHits@10: {all_final_reward_10:7.4f}")
        logger.info(f"\tHits@20: {all_final_reward_20:7.4f}")
        logger.info(f"\tMRR: {mrr:7.4f}")
        if self.multi_answers:
            logger.info("Multi-Answer Metrics:")
            logger.info(f"\tRecall: {all_final_answer_recall:7.4f}")
            logger.info(f"\tPrecision: {all_final_answer_precision:7.4f}")
            logger.info(f"\tF1 Score: {all_final_answer_f1:7.4f}")
        if self.environment.check_paths_exist():
            logger.info("Path Faithfulness Metrics:")
            logger.info(f"\tPath Recall: {all_final_path_recall:7.4f}")
            logger.info(f"\tPath Precision: {all_final_path_precision:7.4f}")
            logger.info(f"\tPath F1 Score: {all_final_path_f1:7.4f}")

            logger.info("Node Coverage Metrics:")
            logger.info(f"\tNode Recall: {all_final_node_recall:7.4f}")
            logger.info(f"\tNode Precision: {all_final_node_precision:7.4f}")
            logger.info(f"\tNode F1 Score: {all_final_node_f1:7.4f}")

            logger.info("Relation Coverage Metrics:")
            logger.info(f"\tRelation Recall: {all_final_rel_recall:7.4f}")
            logger.info(f"\tRelation Precision: {all_final_rel_precision:7.4f}")
            logger.info(f"\tRelation F1 Score: {all_final_rel_f1:7.4f}")

            

        # Log evaluation metrics to WANDB
        if self.use_wandb:
            logger.info(f"Logging {mode} evaluation metrics to WANDB...")
            logger.info(f"WANDB run state: {wandb.run is not None}")
            logger.info(f"WANDB run id: {wandb.run.id if wandb.run else 'None'}")
            try:
                wandb.log({
                    f'{mode}/hits@1': float(all_final_reward_1),
                    f'{mode}/hits@3': float(all_final_reward_3),
                    f'{mode}/hits@5': float(all_final_reward_5),
                    f'{mode}/hits@10': float(all_final_reward_10),
                    f'{mode}/hits@20': float(all_final_reward_20),
                    f'{mode}/mrr': float(mrr),
                    f'{mode}/recall': float(all_final_answer_recall) if all_final_answer_recall is not None else None,
                    f'{mode}/precision': float(all_final_answer_precision) if all_final_answer_precision is not None else None,
                    f'{mode}/f1_score': float(all_final_answer_f1) if all_final_answer_f1 is not None else None,
                    f'{mode}/path_recall': float(all_final_path_recall) if all_final_path_recall is not None else None,
                    f'{mode}/path_precision': float(all_final_path_precision) if all_final_path_precision is not None else None,
                    f'{mode}/path_f1_score': float(all_final_path_f1) if all_final_path_f1 is not None else None,
                    f'{mode}/node_recall': float(all_final_node_recall) if all_final_node_recall is not None else None,
                    f'{mode}/node_precision': float(all_final_node_precision) if all_final_node_precision is not None else None,
                    f'{mode}/node_f1_score': float(all_final_node_f1) if all_final_node_f1 is not None else None,
                    f'{mode}/rel_recall': float(all_final_rel_recall) if all_final_rel_recall is not None else None,
                    f'{mode}/rel_precision': float(all_final_rel_precision) if all_final_rel_precision is not None else None,
                    f'{mode}/rel_f1_score': float(all_final_rel_f1) if all_final_rel_f1 is not None else None,
                    f'{mode}/total_examples': int(total_examples)
                })  # Let WANDB auto-assign step for evaluation metrics
                logger.info(f"Successfully logged {mode} metrics to WANDB")
            except Exception as e:
                logger.error(f"Failed to log {mode} metrics to WANDB: {e}")
                logger.error(f"WANDB run state after error: {wandb.run is not None}")
        else:
            logger.info(f"WANDB logging disabled for {mode} evaluation")

        return EvaluationMetrics(
            hits_at_1=all_final_reward_1,
            hits_at_3=all_final_reward_3,
            hits_at_5=all_final_reward_5,
            hits_at_10=all_final_reward_10,
            hits_at_20=all_final_reward_20,
            mrr=mrr,
            max_hits_at_1=max_hits,
            max_mrr=max_mrr,
            answer_recall=all_final_answer_recall,
            answer_precision=all_final_answer_precision,
            answer_f1=all_final_answer_f1,
            path_recall=all_final_path_recall,
            path_precision=all_final_path_precision,
            path_f1=all_final_path_f1,
            node_recall=all_final_node_recall,
            node_precision=all_final_node_precision,
            node_f1=all_final_node_f1,
            rel_recall=all_final_rel_recall,
            rel_precision=all_final_rel_precision,
            rel_f1=all_final_rel_f1,
        )

    def predict(self, sess: tf.compat.v1.Session, beam: bool = False, mode: str = 'dev'):
        paths = defaultdict(list)       # Store paths for each question if print_paths is True
        feed_dict = {}                  # Feed dictionaries, gets updated each hop during evaluation

        # Changing the environment to test/dev data and resetting values
        self.environment.change_mode(mode)

        # For beam search, limit rollouts to max_num_actions to prevent indexing errors
        effective_rollouts = min(self.test_rollouts, self.max_num_actions) if beam else self.test_rollouts
        self.environment.change_test_rollouts(effective_rollouts)   # modifying the number of rollouts for evaluation
        total_examples = self.environment.total_no_examples         # total number of questions
        test_batch_counter = 0
        logger.info(f"Predicting with mode: {mode} on {total_examples} samples...")
        if beam:
            logger.info(f"Beam search enabled: using {effective_rollouts} rollouts (limited by max_num_actions={self.max_num_actions})")

        for episode in tqdm(self.environment.get_episodes(), desc="Evaluating"):
            assert episode.mode != 'train', "Environment is in training mode!"

            temp_batch_size = episode.no_examples                   # batch size, can vary in test due to the last batch
            test_batch_counter += temp_batch_size
            logger.info(f"Evaluating samples {test_batch_counter}/{total_examples} with {effective_rollouts} rollouts...")

            # Set Initial Beams Probs
            beam_probs = np.zeros((temp_batch_size * effective_rollouts, 1)) # Cumulative scores from previous steps [batch_size*k, 1]

            # Provide Initial Variables
            state = episode.get_state()                             # Initial State (current_entities, next_entities, next_relations)
            mem = self.agent.get_mem_shape()                        # LSTM Memory Shape (num lstm layers, 2, batch size, memory size)
            agent_mem = np.zeros((mem[0], mem[1], temp_batch_size*effective_rollouts, mem[3]), dtype='float32')
            previous_relation = np.ones((temp_batch_size * effective_rollouts, ), dtype='int64') * self.relation_vocab['DUMMY_START_RELATION']
            
            feed_dict = {
                self.range_arr: np.arange(temp_batch_size * effective_rollouts),
                self.question_embedding: episode.get_question_embedding(),              # question embeddings
            }

            ####logger code####
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
                _, agent_mem, test_scores, test_action_idx, chosen_relation = sess.run(
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
                    k = effective_rollouts  # Use the same effective rollouts calculated earlier
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
                    for j in range(i):
                        self.entity_trajectory[j] = self.entity_trajectory[j][y]
                        self.relation_trajectory[j] = self.relation_trajectory[j][y]
                
                ####logger code####
                # Store the current path before the environment update
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
            self.entity_trajectory.append(state['current_entities'])

            # Reshape the reward to [orig_batch_size, num_rollouts], to calculate for how many of the
            # entity pair, at least one of the paths arrive at the correct answer
            self.log_probs = np.reshape(self.log_probs, (temp_batch_size, self.test_rollouts))
            sorted_indx = np.argsort(-self.log_probs)
            
            # Get current and start entities
            ce = episode.state['current_entities'].reshape((temp_batch_size, self.test_rollouts))

            for b in range(temp_batch_size):
                # Retrive Sample's context
                question_txt = self.environment.batcher.translate_questions([episode.question_tokens[b]])[0]    # Convert question back to text
                start_e = self.environment.batcher.translate_entities([episode.start_entities[b * self.test_rollouts]])[0]                 # Map id to entity for source node
                if self.multi_answers:
                    end_e = self.environment.batcher.translate_entities([episode.end_entities[b]], dynamic_list=True)                     # Map id to entity for answer node
                else:
                    end_e = self.environment.batcher.translate_entities([episode.end_entities[b * self.test_rollouts]])                     # Map id to entity for answer node

                # Question Header Information
                paths[question_txt].append(question_txt + "\n")
                paths[question_txt].append(f"KG Start : {start_e}\n")
                paths[question_txt].append(f"KG GT Ans: {end_e}\n")
                
                r = sorted_indx[b][0] # highest scoring path
                indx = b * self.test_rollouts + r           # Convert to global index
                paths[question_txt].append(f"Agent Ans: {str(self.environment.batcher.translate_entities([ce[b, r]])[0])}\n")

                paths[question_txt].append(f"Path Score: {-self.log_probs[b, r]}\n")

                entities_path = [str(self.environment.batcher.translate_entities([e[indx]])[0]) for e in self.entity_trajectory]
                relations_path = [str(self.environment.batcher.translate_relations([re[indx]])[0]) for re in self.relation_trajectory]
                
                question_path = entities_path[0]
                for step in range(self.path_length):
                    question_path += f" --[{relations_path[step]}]--> {entities_path[step+1]}"
                paths[question_txt].append(f"Predicted Path: {question_path}\n")
                paths[question_txt].append("================================\n") # clear distinction for different attempts of same question
        
        # Store the paths for each question
        logger.info(f"[ printing paths at {os.path.join(self.output_dir, 'test_beam')} ]")
        with codecs.open(self.path_logger_file_ + ".txt", 'a', 'utf-8') as pos_file:
            for q in paths:
                for p in paths[q]:
                    pos_file.write(p)
                pos_file.write("\n")

    def finish_wandb(self) -> None:
        """
        Properly finish the WANDB run and cleanup resources.
        
        Should be called at the end of training to ensure proper cleanup
        and finalization of the WANDB run.
        """
        if self.use_wandb:
            try:
                if wandb.run is not None:
                    wandb.finish()
            except Exception as e:
                logger.warning(f"Error finishing WANDB session: {e}")

    def top_k(self, scores: np.ndarray, k: int) -> np.ndarray:
        """
        Extract top-k indices from beam search scores for each batch element.
        
        Efficiently selects the k highest-scoring actions from the expanded
        search space during beam search decoding. Used to maintain the most
        promising reasoning paths while pruning lower-scoring alternatives.
        
        Args:
            scores: Beam search scores for each batch element and action.
                Shape: [batch_size, k * max_num_actions] where k is current beam size.
            k: Number of top-scoring paths to retain for continued search.
            
        Returns:
            Flattened indices of top-k scoring actions across all batch elements.
            Shape: [batch_size * k] containing action indices that can be
            used to select corresponding actions and beam states.
            
        Note:
            - Reshapes and sorts scores to find global top-k per batch element
            - Returns flattened indices for efficient tensor indexing operations
            - Critical component of beam search maintaining search diversity
        """
        scores = scores.reshape(-1, k * self.max_num_actions)  # [B, (k*max_num_actions)]
        idx = np.argsort(scores, axis=1)
        idx = idx[:, -k:]  # take the last k highest indices # [B , k]
        return idx.reshape((-1))

if __name__ == '__main__':
    # Read command line options and setup logging
    options = read_options()

    # Set logging
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%Y/%m/%d %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    logfile = logging.FileHandler(options['log_file_name'], 'w')
    logfile.setFormatter(fmt)
    logger.addHandler(logfile)

    # Load vocabularies
    logger.info('Reading vocab files (ent & rel to id)...')
    relation_vocab = json.load(open(os.path.join(options['vocab_dir'], 'relation_vocab.json')))
    entity_vocab = json.load(open(os.path.join(options['vocab_dir'], 'entity_vocab.json')))

    logger.info('Total number of entities {}'.format(len(entity_vocab)))
    logger.info('Total number of relations {}'.format(len(relation_vocab)))

    # Configure TensorFlow for deterministic behavior
    save_path = ''
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = False
    config.log_device_placement = False
    config.allow_soft_placement = True

    # Set seed for reproducibility
    set_seeds(options['seed'])

    embedding_server = EmbeddingServer(options['question_tokenizer_name'])

    # Training phase
    if not options['load_model']:
        trainer = TrainerNLQ(
            batch_size=options['batch_size'],
            num_rollouts=options['num_rollouts'],
            positive_reward=options['positive_reward'],
            negative_reward=options['negative_reward'],
            path_length=options['path_length'],
            test_rollouts=options['test_rollouts'],
            data_input_dir=options['data_input_dir'],
            question_tokenizer_name=options['question_tokenizer_name'],
            question_format=options['question_format'],
            cached_QAMetaData_path=options['cached_QAMetaData_path'],
            raw_QAData_path=options['raw_QAData_path'],
            multi_answers=options['multi_answers'],
            max_num_actions=options['max_num_actions'],
            embedding_size=options['embedding_size'],
            hidden_size=options['hidden_size'],
            use_entity_embeddings=options['use_entity_embeddings'],
            train_entity_embeddings=options['train_entity_embeddings'],
            train_relation_embeddings=options['train_relation_embeddings'],
            LSTM_layers=options['LSTM_layers'],
            projection_adapter=options['projection_adapter'],
            projection_layers=options['projection_layers'],
            projection_hidden=options['projection_hidden'],
            learning_rate=options['learning_rate'],
            grad_clip_norm=options['grad_clip_norm'],
            gamma=options['gamma'],
            Lambda=options['Lambda'],
            beta=options['beta'],
            total_iterations=options['total_iterations'],
            eval_every=options['eval_every'],
            output_dir=options['output_dir'],
            model_dir=options['model_dir'],
            path_logger_file=options['path_logger_file'],
            pool=options['pool'],
            use_beam=options['use_beam'],
            seed=options['seed'],
            entity_vocab=entity_vocab, 
            relation_vocab=relation_vocab,
            use_full_graph=options['use_full_graph'],
            use_stop_signal=options['use_stop_signal'],
            use_restart_signal=options['use_restart_signal'],
            embedding_server=embedding_server,
            use_wandb=options.get('track', False)
        )
        with tf.compat.v1.Session(config=config) as sess:
            # Set seeds again after session creation to ensure TF operations are deterministic
            set_seeds(options['seed'])
            sess.run(trainer.initialize())

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

    # Evaluation phase
    trainer = TrainerNLQ(
        batch_size=options['batch_size'],
        num_rollouts=options['num_rollouts'],
        positive_reward=options['positive_reward'],
        negative_reward=options['negative_reward'],
        path_length=options['path_length'],
        test_rollouts=options['test_rollouts'],
        data_input_dir=options['data_input_dir'],
        question_tokenizer_name=options['question_tokenizer_name'],
        question_format=options['question_format'],
        cached_QAMetaData_path=options['cached_QAMetaData_path'],
        raw_QAData_path=options['raw_QAData_path'],
        multi_answers=options['multi_answers'],
        max_num_actions=options['max_num_actions'],
        embedding_size=options['embedding_size'],
        hidden_size=options['hidden_size'],
        use_entity_embeddings=options['use_entity_embeddings'],
        train_entity_embeddings=options['train_entity_embeddings'],
        train_relation_embeddings=options['train_relation_embeddings'],
        LSTM_layers=options['LSTM_layers'],
        projection_adapter=options['projection_adapter'],
        projection_layers=options['projection_layers'],
        projection_hidden=options['projection_hidden'],
        learning_rate=options['learning_rate'],
        grad_clip_norm=options['grad_clip_norm'],
        gamma=options['gamma'],
        Lambda=options['Lambda'],
        beta=options['beta'],
        total_iterations=options['total_iterations'],
        eval_every=options['eval_every'],
        output_dir=options['output_dir'],
        model_dir=options['model_dir'],
        path_logger_file=options['path_logger_file'],
        pool=options['pool'],
        use_beam=options['use_beam'],
        seed=options['seed'],
        entity_vocab=entity_vocab, 
        relation_vocab=relation_vocab,
        use_full_graph=options['use_full_graph'], 
        use_stop_signal=options['use_stop_signal'],
        use_restart_signal=options['use_restart_signal'],
        embedding_server=embedding_server,
        use_wandb=options.get('track', False)  # Enable WANDB for evaluation if tracking is on
    )
    
    with tf.compat.v1.Session(config=config) as sess:
        # Set seeds again after session creation to ensure TF operations are deterministic  
        set_seeds(options['seed'])
        trainer.initialize(restore=save_path, sess=sess) # check if it is fine to initialize an already trained model or if we need to create one before this line

        # trainer.test_rollouts = 100                      # set test rollouts to 100 for evaluation

        # create files to store results
        if options['print_paths'] or options['print_predictions']:
            os.makedirs(os.path.join(path_logger_file, "test_beam"), exist_ok=True)
            trainer.path_logger_file_ = os.path.join(path_logger_file, "test_beam", "paths")
        
        with open(os.path.join(output_dir, 'scores.txt'), 'a') as score_file:
            score_file.write("Test (beam) scores with best model from " + save_path + "\n")

        # Perform Evaluation
        trainer.test(sess, beam=options['use_beam'], print_paths=options['print_paths'], save_model=False, mode='test')
        if options['print_predictions']:
            set_seeds(options['seed']) # Ensure reproducibility for predictions
            trainer.predict(sess, beam=options['use_beam'], mode='test')
    
    logging.info(f"Evaluation completed. Closing Server")
    embedding_server.close()  # Close the embedding server connection
    trainer.finish_wandb()
