import numpy as np
import tensorflow as tf
from typing import Dict, Any, List, Tuple, Union, Optional

tf.compat.v1.disable_eager_execution()

class AgentNLQ(object):
    """
    Reinforcement learning agent for knowledge graph reasoning using MINERVA.
    
    This agent uses an LSTM-based policy network to navigate knowledge graphs
    by selecting relations to follow at each step. The agent learns to find
    paths from start entities to target entities through reinforcement learning.
    
    The architecture combines:
    - Entity and relation embedding lookup tables
    - Multi-layer LSTM for maintaining path history
    - MLP policy network for action selection
    - Attention mechanism between query and candidate actions
    """

    def __init__(self, params: Dict[str, Any], entity_vocab: Dict[str, int], relation_vocab: Dict[str, int]) -> None:
        """
        Initialize the MINERVA agent with embedding tables and policy network.
        
        Args:
            params (Dict[str, Any]): Configuration dictionary containing:
                - embedding_size: Dimension of relation/entity embeddings
                - hidden_size: LSTM hidden state dimension
                - use_entity_embeddings: Whether to use entity embeddings
                - train_entity_embeddings: Whether entity embeddings are trainable
                - train_relation_embeddings: Whether relation embeddings are trainable
                - num_rollouts: Number of parallel rollouts during training
                - test_rollouts: Number of rollouts during testing
                - LSTM_layers: Number of LSTM layers in policy network
                - batch_size: Training batch size
            entity_vocab (Dict[str, int]): Vocabulary mapping for entities
            relation_vocab (Dict[str, int]): Vocabulary mapping for relations
        """

        self.action_vocab_size = len(relation_vocab)
        self.entity_vocab_size = len(entity_vocab)
        self.embedding_size = params['embedding_size']
        self.hidden_size = params['hidden_size']
        self.ePAD = tf.constant(entity_vocab['PAD'], dtype=tf.int32)
        self.rPAD = tf.constant(relation_vocab['PAD'], dtype=tf.int32)
        if params['use_entity_embeddings']:
            self.entity_initializer = tf.keras.initializers.GlorotUniform()
        else:
            self.entity_initializer = tf.zeros_initializer()
        self.train_entities = params['train_entity_embeddings']
        self.train_relations = params['train_relation_embeddings']

        self.num_rollouts = params['num_rollouts']
        self.test_rollouts = params['test_rollouts']
        self.LSTM_Layers = params['LSTM_layers']
        self.batch_size = params['batch_size'] * params['num_rollouts']
        self.dummy_start_label = tf.constant(
            np.ones(self.batch_size, dtype='int64') * relation_vocab['DUMMY_START_RELATION'])

        self.entity_embedding_size = self.embedding_size
        self.use_entity_embeddings = params['use_entity_embeddings']
        self.m = 4 if self.use_entity_embeddings else 2

        with tf.compat.v1.variable_scope("action_lookup_table"):
            self.action_embedding_placeholder = tf.compat.v1.placeholder(tf.float32,
                                                               [self.action_vocab_size, 2 * self.embedding_size])

            self.relation_lookup_table = tf.compat.v1.get_variable("relation_lookup_table",
                                                         shape=[self.action_vocab_size, 2 * self.embedding_size],
                                                         dtype=tf.float32,
                                                         initializer=tf.keras.initializers.GlorotUniform(),
                                                         trainable=self.train_relations)
            self.relation_embedding_init = self.relation_lookup_table.assign(self.action_embedding_placeholder)

        with tf.compat.v1.variable_scope("entity_lookup_table"):
            self.entity_embedding_placeholder = tf.compat.v1.placeholder(tf.float32,
                                                               [self.entity_vocab_size, 2 * self.embedding_size])
            self.entity_lookup_table = tf.compat.v1.get_variable("entity_lookup_table",
                                                       shape=[self.entity_vocab_size, 2 * self.entity_embedding_size],
                                                       dtype=tf.float32,
                                                       initializer=self.entity_initializer,
                                                       trainable=self.train_entities)
            self.entity_embedding_init = self.entity_lookup_table.assign(self.entity_embedding_placeholder)

        # LSTM policy core
        with tf.compat.v1.variable_scope("policy_step"):
            cells = []
            for _ in range(self.LSTM_Layers):
                cells.append(tf.compat.v1.nn.rnn_cell.LSTMCell(self.m * self.hidden_size, use_peepholes=True, state_is_tuple=True))
            self.policy_step = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)

        # Project text query embedding to the policy feature space
        with tf.compat.v1.variable_scope("query_projection"):
            self.query_proj = lambda x: tf.compat.v1.layers.dense(
                x, self.m * self.embedding_size, activation=tf.nn.relu)

    def get_mem_shape(self) -> Tuple[int, int, Optional[int], int]:
        """
        Get the shape of LSTM memory states for the policy network.
        
        Returns:
            Tuple[int, int, Optional[int], int]: Memory shape tuple containing:
                - Number of LSTM layers
                - 2 (for cell state and hidden state)
                - None (variable batch size)
                - Memory dimension (m * hidden_size)
        """
        return (self.LSTM_Layers, 2, None, self.m * self.hidden_size)

    def policy_MLP(self, state: tf.Tensor) -> tf.Tensor:
        """
        Multi-layer perceptron for generating policy representations.
        
        Takes the concatenated state (LSTM output + query + current entity) and
        transforms it into a representation that can be used to score candidate
        actions through dot product attention.
        
        Args:
            state (tf.Tensor): Concatenated state vector containing LSTM output,
                query embedding, and current entity information.
                Shape: [batch_size, state_dim]
                
        Returns:
            tf.Tensor: Policy representation vector used for action scoring.
                Shape: [batch_size, m * embedding_size]
        """
        with tf.compat.v1.variable_scope("MLP_for_policy"):
            hidden = tf.compat.v1.layers.dense(state, 4 * self.hidden_size, activation=tf.nn.relu)
            output = tf.compat.v1.layers.dense(hidden, self.m * self.embedding_size, activation=tf.nn.relu)
        return output

    def action_encoder(self, next_relations: tf.Tensor, next_entities: tf.Tensor) -> tf.Tensor:
        """
        Encode actions (relation-entity pairs) into embedding vectors.
        
        Creates action representations by concatenating relation and entity embeddings
        (if entity embeddings are enabled) or using only relation embeddings.
        These encodings are used for both action selection and as input to the LSTM.
        
        Args:
            next_relations (tf.Tensor): Relation indices for candidate actions.
                Shape: [batch_size, max_actions] or [batch_size]
            next_entities (tf.Tensor): Entity indices for candidate actions.
                Shape: [batch_size, max_actions] or [batch_size]
                
        Returns:
            tf.Tensor: Action embedding vectors. If using entity embeddings,
                concatenates relation and entity embeddings. Otherwise, returns
                only relation embeddings.
                Shape: [batch_size, (max_actions), embedding_dim]
        """
        with tf.compat.v1.variable_scope("lookup_table_edge_encoder"):
            relation_embedding = tf.nn.embedding_lookup(self.relation_lookup_table, next_relations)
            entity_embedding = tf.nn.embedding_lookup(self.entity_lookup_table, next_entities)
            if self.use_entity_embeddings:
                action_embedding = tf.concat([relation_embedding, entity_embedding], axis=-1)
            else:
                action_embedding = relation_embedding
        return action_embedding

    def step(self, next_relations: tf.Tensor, next_entities: tf.Tensor, prev_state: tf.Tensor, 
             prev_relation: tf.Tensor, question_embedding: tf.Tensor, current_entities: tf.Tensor,
             label_action: tf.Tensor, range_arr: tf.Tensor, first_step_of_test: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Execute one step of the policy network for action selection.
        
        This method performs a single step of reasoning:
        1. Encodes the previous action taken
        2. Updates LSTM state with previous action
        3. Generates current state representation 
        4. Scores all candidate actions using attention
        5. Masks invalid actions and samples next action
        6. Computes policy gradient loss
        
        Args:
            next_relations (tf.Tensor): Candidate relation indices for next step.
                Shape: [batch_size, max_actions]
            next_entities (tf.Tensor): Candidate entity indices for next step.
                Shape: [batch_size, max_actions] 
            prev_state (tf.Tensor): Previous LSTM hidden states.
                Shape: [(layers, 2, batch_size, hidden_size)]
            prev_relation (tf.Tensor): Previously selected relation indices.
                Shape: [batch_size]
            question_embedding (tf.Tensor): Question text embedding.
                Shape: [batch_size, embedding_size]
            current_entities (tf.Tensor): Current entity positions.
                Shape: [batch_size]
            label_action (tf.Tensor): Ground truth action indices for training.
                Shape: [batch_size]
            range_arr (tf.Tensor): Range array for indexing operations.
                Shape: [batch_size]
            first_step_of_test (tf.Tensor): Boolean indicating if this is first test step.
                
        Returns:
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]: Contains:
                - loss: Policy gradient loss for this step [batch_size]
                - new_state: Updated LSTM states
                - log_softmax_scores: Log probabilities over actions [batch_size, max_actions]  
                - action_idx: Selected action indices [batch_size]
                - chosen_relation: Selected relation indices [batch_size]
        """

        prev_action_embedding = self.action_encoder(prev_relation, current_entities)
        # 1. one step of rnn
        output, new_state = self.policy_step(prev_action_embedding, prev_state)  # output: [B, 4D]

        # Get state vector
        prev_entity = tf.nn.embedding_lookup(self.entity_lookup_table, current_entities)
        state = tf.concat([output, prev_entity], axis=-1) if self.use_entity_embeddings else output

        # Project query and fuse
        q_proj = self.query_proj(question_embedding)
        state_query_concat = tf.concat([state, q_proj], axis=-1)

        candidate_action_embeddings = self.action_encoder(next_relations, next_entities)

        # MLP for policy
        policy_vec = self.policy_MLP(state_query_concat) # aggressive projection
        policy_vec = tf.expand_dims(policy_vec, axis=1)  # [B, 1, 2D]
        prelim_scores = tf.reduce_sum(tf.multiply(candidate_action_embeddings, policy_vec), axis=2)

        #!! NOTE: Continue from here

        # Masking PAD actions

        comparison_tensor = tf.ones_like(next_relations, dtype=tf.int32) * self.rPAD  # matrix to compare
        mask = tf.equal(next_relations, comparison_tensor)  # The mask
        dummy_scores = tf.ones_like(prelim_scores) * -99999.0  # the base matrix to choose from if dummy relation
        scores = tf.where(mask, dummy_scores, prelim_scores)  # [B, MAX_NUM_ACTIONS]

        # 4 sample action
        action = tf.cast(tf.random.categorical(logits=scores, num_samples=1), tf.int32)  # [B, 1]

        # loss
        # 5a. cross-entropy against sampled action (REINFORCE-style)
        label_action =  tf.squeeze(action, axis=1)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=label_action)  # [B,]

        # 6. Map back to true id
        action_idx = tf.squeeze(action)
        chosen_relation = tf.gather_nd(next_relations, tf.transpose(tf.stack([range_arr, action_idx])))

        return loss, new_state, tf.nn.log_softmax(scores), action_idx, chosen_relation

    def __call__(self, candidate_relation_sequence: List[tf.Tensor], candidate_entity_sequence: List[tf.Tensor], 
                 current_entities: List[tf.Tensor], path_label: List[tf.Tensor], question_embedding: tf.Tensor, 
                 range_arr: tf.Tensor, first_step_of_test: tf.Tensor, T: int = 3, 
                 entity_sequence: int = 0) -> Tuple[List[tf.Tensor], List[tf.Tensor], List[tf.Tensor]]:
        """
        Execute a complete multi-step reasoning episode through the knowledge graph.
        
        Unrolls the policy network for T time steps, where at each step the agent:
        1. Observes candidate actions from current position
        2. Uses LSTM and attention to select best action  
        3. Updates internal state and moves to next position
        4. Computes policy gradient loss for training
        
        This implements the core MINERVA algorithm for multi-hop reasoning.
        
        Args:
            candidate_relation_sequence (List[tf.Tensor]): Sequence of candidate relations
                at each time step. Length T, each element shape: [batch_size, max_actions]
            candidate_entity_sequence (List[tf.Tensor]): Sequence of candidate entities  
                at each time step. Length T, each element shape: [batch_size, max_actions]
            current_entities (List[tf.Tensor]): Current entity positions at each step.
                Length T, each element shape: [batch_size]
            path_label (List[tf.Tensor]): Ground truth action labels for training.
                Length T, each element shape: [batch_size] 
            question_embedding (tf.Tensor): Question text embedding to be answered.
                Shape: [batch_size]
            range_arr (tf.Tensor): Range array for batch indexing operations.
                Shape: [batch_size]
            first_step_of_test (tf.Tensor): Boolean indicating if in test mode.
            T (int, optional): Number of reasoning steps. Defaults to 3.
            entity_sequence (int, optional): Unused parameter. Defaults to 0.
            
        Returns:
            Tuple[List[tf.Tensor], List[tf.Tensor], List[tf.Tensor]]: Contains:
                - all_loss: Policy gradient losses at each step, length T
                - all_logits: Log probabilities over actions at each step, length T  
                - action_idx: Selected action indices at each step, length T
        """

        self.baseline_inputs = []
        # get the query vector
        # query_embedding = tf.nn.embedding_lookup(self.relation_lookup_table, query_relation)  # [B, 2D]
        # query_embedding is now fed from outside (e.g., BERT) with shape [B, Q]
        state = self.policy_step.zero_state(batch_size=self.batch_size, dtype=tf.float32)

        prev_relation = self.dummy_start_label

        all_loss = []  # list of loss tensors each [B,]
        all_logits = []  # list of actions each [B,]
        action_idx = []

        with tf.compat.v1.variable_scope("policy_steps_unroll") as scope:
            for t in range(T):
                if t > 0:
                    scope.reuse_variables()
                next_rel = candidate_relation_sequence[t]  # [B, MAX_NUM_ACTIONS, MAX_EDGE_LENGTH]
                next_ent = candidate_entity_sequence[t]
                cur_ent = current_entities[t]

                label_t = path_label[t]  # [B]

                loss, state, logits, idx, chosen_relation = self.step(
                    next_rel,
                    next_ent,
                    state, 
                    prev_relation, 
                    question_embedding,
                    cur_ent,
                    label_action=label_t,
                    range_arr=range_arr,
                    first_step_of_test=first_step_of_test
                )

                all_loss.append(loss)
                all_logits.append(logits)
                action_idx.append(idx)
                prev_relation = chosen_relation

            # [(B, T), 4D]

        return all_loss, all_logits, action_idx
