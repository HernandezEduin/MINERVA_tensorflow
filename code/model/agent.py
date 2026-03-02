"""
Natural Language Question (NLQ) agent for knowledge graph reasoning using MINERVA.

This module implements the core reinforcement learning agent that learns to navigate
knowledge graphs by following relation-entity paths to answer natural language questions.
The agent uses an LSTM-based policy network with attention mechanisms to select optimal
reasoning paths through multi-hop graph traversal.

The agent architecture consists of:
- Embedding lookup tables for entities and relations
- Multi-layer LSTM for maintaining reasoning state and path history
- MLP policy network with attention for action selection
- Question projection layer to align text embeddings with graph embeddings

Classes:
    AgentNLQ: Main agent class implementing the MINERVA reasoning algorithm
"""

import numpy as np
import tensorflow as tf

from typing import Dict, Any, List, Tuple, Optional

class AgentNLQ(object):
    """
    Reinforcement learning agent for natural language knowledge graph reasoning.
    
    Implements the MINERVA algorithm for multi-hop reasoning over knowledge graphs
    using natural language questions. The agent learns to navigate from query entities
    to answer entities by selecting optimal relation-entity paths through policy
    gradient reinforcement learning.
    
    The agent combines:
    - Entity and relation embedding lookup tables for graph representation
    - Multi-layer LSTM for maintaining reasoning state and path memory
    - MLP policy network with dot-product attention for action selection
    - Question projection layer to align natural language with graph embeddings
    - Action masking to handle invalid transitions and padding
    
    Architecture Details:
    - Action encoding concatenates relation and entity embeddings (if enabled)
    - LSTM state captures multi-hop reasoning history
    - Policy network scores candidate actions via attention mechanism
    - Sampling-based action selection enables exploration during training
    
    Attributes:
        action_vocab_size (int): Number of possible relations/actions
        entity_vocab_size (int): Number of entities in knowledge graph
        embedding_size (int): Dimension of entity/relation embeddings
        hidden_size (int): LSTM hidden state dimension
        m (int): Embedding multiplier factor (4 with entities, 2 without)
        use_entity_embeddings (bool): Whether to use entity embeddings
        policy_step: Multi-layer LSTM cell for reasoning state
        relation_lookup_table: Trainable relation embedding table
        entity_lookup_table: Trainable entity embedding table
        question_proj: Neural network for question embedding projection
        
    Example:
        >>> agent = AgentNLQ(
        ...     embedding_size=100, hidden_size=200, use_entity_embeddings=True,
        ...     train_entity_embeddings=True, train_relation_embeddings=True,
        ...     num_rollouts=20, test_rollouts=100, LSTM_layers=3, batch_size=128,
        ...     entity_vocab=entity_vocab, relation_vocab=relation_vocab
        ... )
        >>> losses, logits, actions = agent(
        ...     candidate_relations, candidate_entities, 
        ...     current_entities, question_emb, range_arr, T=3
        ... )
    """

    def __init__(
        self,
        embedding_size: int,
        hidden_size: int,
        use_entity_embeddings: bool,
        train_entity_embeddings: bool,
        train_relation_embeddings: bool,
        LSTM_layers: int,
        projection_adapter: str,
        projection_layers: int,
        projection_hidden: int,
        entity_vocab: Dict[str, int],
        relation_vocab: Dict[str, int]
    ) -> None:
        """
        Initialize the MINERVA agent with embedding tables and policy network components.
        
        Sets up the complete neural architecture for knowledge graph reasoning including
        embedding lookup tables, LSTM policy network, and question projection layers.
        Configures training parameters and initializes all components for multi-hop
        reasoning episodes.
        
        Args:
            embedding_size: Dimension of entity/relation embeddings
            hidden_size: LSTM hidden state dimension  
            use_entity_embeddings: Whether to include entity embeddings
            train_entity_embeddings: Whether entity embeddings are trainable
            train_relation_embeddings: Whether relation embeddings are trainable
            LSTM_layers: Number of LSTM layers in policy network
            projection_adapter: Type of question projection adapter ('linear', 'mlp', 'residual')
            projection_layers: Number of layers in question projection MLP
            projection_hidden: Hidden dimension size for projection adapter
            entity_vocab: Entity name to integer ID mapping for embedding lookup
            relation_vocab: Relation name to integer ID mapping for embedding lookup
            
        Note:
            - Embedding size is internally doubled for relation embeddings
            - Entity embeddings can be disabled for relation-only reasoning
            - LSTM uses peephole connections for improved memory
            - Question projection aligns text embeddings with graph space
        """

        assert projection_adapter in ['linear', 'mlp', 'residual'], \
            "projection_adapter must be one of 'linear', 'mlp', or 'residual'"

        self.action_vocab_size = len(relation_vocab)                        # number of possible actions
        self.entity_vocab_size = len(entity_vocab)                          # number of possible entities
        self.embedding_size = embedding_size                                # dimension size of entity/relation embeddings (NOTE: this will be doubled) [m*D]
        self.hidden_size = hidden_size                                      # dimension size of LSTM hidden state
        self.ePAD = tf.constant(entity_vocab['PAD'], dtype=tf.int32)        # entity padding token
        self.rPAD = tf.constant(relation_vocab['PAD'], dtype=tf.int32)      # relation padding token
        self.drPad = tf.constant(relation_vocab['DUMMY_START_RELATION'], dtype=tf.int32)    # dummy relation for step 0 NOTE: Might be self loop action 
        if use_entity_embeddings:                                           # whether to use entity embeddings
            self.entity_initializer = tf.keras.initializers.GlorotUniform()
        else:
            self.entity_initializer = tf.zeros_initializer()
        self.train_entities = train_entity_embeddings                      # whether entity embeddings are trainable
        self.train_relations = train_relation_embeddings                   # whether relation embeddings are trainable

        self.LSTM_Layers = LSTM_layers                                      # number of layers in LSTM
        self.projection_adapter = projection_adapter                        # type of question projection adapter ('linear', 'mlp', 'residual')
        self.projection_layers = projection_layers                          # number of layers in question projection MLP
        self.projection_hidden = projection_hidden                          # hidden dimension size for projection adapter

        self.entity_embedding_size = self.embedding_size
        self.use_entity_embeddings = use_entity_embeddings
        self.m = 4 if self.use_entity_embeddings else 2                     # multiplicative factor of the embedding sizes, necessary for later models [B, m*D]

        # NOTE: The lookup tables are very similar to the embeddings of KGE models, but without the pretraining and scoring functions for embedding optimizations
        # Initialize Embedding Lookup for Relations
        with tf.compat.v1.variable_scope("action_lookup_table"):            # Embedding Lookup for Relations
            # Temporary container for loading external weights
            self.action_embedding_placeholder = tf.compat.v1.placeholder(tf.float32,
                                                               [self.action_vocab_size, 2 * self.embedding_size])

            # Actual embedding lookup table
            self.relation_lookup_table = tf.compat.v1.get_variable("relation_lookup_table",
                                                         shape=[self.action_vocab_size, 2 * self.embedding_size],
                                                         dtype=tf.float32,
                                                         initializer=tf.keras.initializers.GlorotUniform(),
                                                         trainable=self.train_relations)
            
            # Transferring of the embeddings
            self.relation_embedding_init = self.relation_lookup_table.assign(self.action_embedding_placeholder)

        # Initialize Embedding Lookup for Entities
        with tf.compat.v1.variable_scope("entity_lookup_table"):
            # Temporary container for loading external weights
            self.entity_embedding_placeholder = tf.compat.v1.placeholder(tf.float32,
                                                               [self.entity_vocab_size, 2 * self.embedding_size])

            # Actual embedding lookup table
            self.entity_lookup_table = tf.compat.v1.get_variable("entity_lookup_table",
                                                       shape=[self.entity_vocab_size, 2 * self.entity_embedding_size],
                                                       dtype=tf.float32,
                                                       initializer=self.entity_initializer, # if it will not be used, will be initialized with zeros
                                                       trainable=self.train_entities)

            # Transferring of the embeddings
            self.entity_embedding_init = self.entity_lookup_table.assign(self.entity_embedding_placeholder)

        # LSTM policy core
        with tf.compat.v1.variable_scope("policy_step"):                                            # Only takes in Rel and Entity Embeddings
            cells = []
            for _ in range(self.LSTM_Layers):                                                       # Create an LSTM for each layer
                cells.append(tf.compat.v1.nn.rnn_cell.LSTMCell(self.m * self.hidden_size, use_peepholes=True, state_is_tuple=True))
            self.policy_step = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)    # Stacked LSTM

        # Project text question embedding to the policy feature space
        with tf.compat.v1.variable_scope("question_projection"):
            self.out_dim = self.m * self.embedding_size

            # Create placeholder for pretrained question projection weights
            if self.projection_adapter == 'linear' or self.projection_adapter == 'mlp':
                self.question_embedding_placeholder = tf.compat.v1.placeholder(tf.float32, [None, self.m * self.embedding_size])
            else:
                self.qproj_kernel_ph = tf.compat.v1.placeholder(tf.float32, shape=[None, self.out_dim], name="qproj_kernel_ph")
                self.qproj_bias_ph   = tf.compat.v1.placeholder(tf.float32, shape=[self.out_dim],      name="qproj_bias_ph")
            
            # Create initialization operation (to be called later if pretrained weights exist)
            self.question_proj_init = None  # Will be set up after first call to question_projr
                    
            if self.projection_adapter == 'linear':
                self.question_proj = self.question_proj_linear
            elif self.projection_adapter == 'mlp':
                self.question_proj = self.question_proj_mlp
            else:  # residual
                self.question_proj = self.question_proj_residual

    def get_mem_shape(self) -> Tuple[int, int, Optional[int], int]:
        """
        Get the memory state shape for the multi-layer LSTM policy network.
        
        Returns the tensor shape specification needed to initialize or manage
        LSTM memory states during reasoning episodes. Each LSTM layer maintains
        both cell state and hidden state tensors.
        
        Returns:
            Tuple containing:
                - num_layers (int): Number of LSTM layers in the policy network
                - state_components (int): 2 for (cell_state, hidden_state) per layer
                - batch_dimension (None): Variable batch size dimension
                - state_size (int): Memory dimension (m * hidden_size)
                
        Note:
            - Memory dimension scales with embedding multiplier 'm'
            - Batch dimension is None to support variable batch sizes
            - Used for state initialization and tensor shape validation
        """
        return (self.LSTM_Layers, 2, None, self.m * self.hidden_size)

    def policy_MLP(self, state: tf.Tensor) -> tf.Tensor:
        """
        Multi-layer perceptron for generating policy query vectors.
        
        Transforms the concatenated state representation (LSTM output + question
        embedding + entity context) into a query vector that can be used to score
        candidate actions through dot-product attention. This is a key component
        of the attention mechanism that enables the agent to focus on relevant
        actions based on current context.
        
        Args:
            state: Concatenated state vector containing:
                - LSTM output from previous reasoning steps
                - Projected question embedding 
                - Current entity embedding (if entity embeddings enabled)
                Shape: [batch_size, state_dimension]
                
        Returns:
            Policy query vector for action scoring via dot-product attention.
            Shape: [batch_size, m * embedding_size] where m is the embedding
            multiplier factor (4 with entities, 2 without)
            
        Note:
            - Uses ReLU activation in both hidden and output layers
            - Hidden layer dimension is 4 * hidden_size for expressiveness
            - Output dimension matches action embedding size for attention
        """
        with tf.compat.v1.variable_scope("MLP_for_policy"):
            hidden = tf.compat.v1.layers.dense(state, 4 * self.hidden_size, activation=tf.nn.relu) # TODO: check if this is the correct second dimension
            output = tf.compat.v1.layers.dense(hidden, self.m * self.embedding_size, activation=tf.nn.relu)
        return output

    def action_encoder(
        self, 
        next_relations: tf.Tensor, 
        next_entities: tf.Tensor
    ) -> tf.Tensor:
        """
        Encode knowledge graph actions into dense embedding representations.
        
        Converts relation-entity pairs (actions) into dense vector representations
        by looking up embeddings and optionally concatenating them. These encodings
        serve dual purposes: as input to the LSTM for state updates and as candidate
        action representations for attention-based scoring.
        
        Args:
            next_relations: Relation indices for candidate or taken actions.
                Shape: [batch_size, max_actions] for candidates or [batch_size] for taken
            next_entities: Entity indices for candidate or taken actions.
                Shape: [batch_size, max_actions] for candidates or [batch_size] for taken
                
        Returns:
            Dense action embedding vectors. Behavior depends on use_entity_embeddings:
            - If True: Concatenates relation and entity embeddings 
              Shape: [batch_size, (max_actions), 4 * embedding_size]
            - If False: Returns only relation embeddings
              Shape: [batch_size, (max_actions), 2 * embedding_size]
              
        Note:
            - Relation embeddings are always included (2 * embedding_size)
            - Entity embeddings are optional (2 * embedding_size when used)
            - Output dimension determines LSTM input size and attention dimension
        """
        with tf.compat.v1.variable_scope("lookup_table_edge_encoder"):
            relation_embedding = tf.nn.embedding_lookup(self.relation_lookup_table, next_relations)
            entity_embedding = tf.nn.embedding_lookup(self.entity_lookup_table, next_entities)
            if self.use_entity_embeddings:
                action_embedding = tf.concat([relation_embedding, entity_embedding], axis=-1)
            else:
                action_embedding = relation_embedding
        return action_embedding

    def step(
        self, 
        next_relations: tf.Tensor, 
        next_entities: tf.Tensor, 
        prev_state: tf.Tensor, 
        prev_relation: tf.Tensor, 
        question_embedding: tf.Tensor, 
        current_entities: tf.Tensor,
        range_arr: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Execute one reasoning step of the MINERVA policy network.
        
        Performs a single step of multi-hop knowledge graph reasoning by:
        1. Encoding the previous action as input to LSTM
        2. Updating LSTM state to capture reasoning history
        3. Constructing current state representation with question context
        4. Scoring candidate actions using dot-product attention
        5. Masking invalid actions and sampling next action stochastically
        6. Computing policy gradient loss for reinforcement learning
        
        This implements the core reasoning step of the MINERVA algorithm.
        
        Args:
            next_relations: Candidate relation indices for next reasoning step.
                Shape: [batch_size, max_actions]
            next_entities: Candidate entity indices for next reasoning step.
                Shape: [batch_size, max_actions] 
            prev_state: Previous LSTM hidden and cell states from all layers.
                Nested tuple structure: [(layers, 2, batch_size, hidden_size)]
            prev_relation: Previously selected relation indices.
                Shape: [batch_size]
            question_embedding: Natural language question embedding vector.
                Shape: [batch_size, embedding_dimension]
            current_entities: Current entity positions in knowledge graph.
                Shape: [batch_size]
            range_arr: Batch indexing array for advanced tensor operations.
                Shape: [batch_size] with values [0, 1, 2, ..., batch_size-1]
                
        Returns:
            Tuple containing:
                - loss: Policy gradient loss (cross-entropy) for this step [batch_size]
                - new_state: Updated LSTM states after processing previous action
                - log_softmax_scores: Log probabilities over candidate actions [batch_size, max_actions]  
                - action_idx: Sampled action indices from categorical distribution [batch_size]
                - chosen_relation: Actual relation IDs selected by the agent [batch_size]
                
        Note:
            - Uses stochastic action sampling for exploration during training
            - Applies action masking to handle invalid transitions (PAD relations)
            - Policy gradient loss computed against sampled actions (REINFORCE)
            - State representation combines LSTM output, entity context, and question
        """

        # Encode previous action (relation and entity indices to embeddings)
        prev_action_embedding = self.action_encoder(prev_relation, current_entities)                # [B, max_actions, m*D]
        
        # One Step of RNN (embeddings + states)
        output, new_state = self.policy_step(prev_action_embedding, prev_state)                     # [B, m*D]

        # State = LSTM output + Previous Entity Embedding [B, m*D + 2D]
        prev_entity = tf.nn.embedding_lookup(self.entity_lookup_table, current_entities)
        state = tf.concat([output, prev_entity], axis=-1) if self.use_entity_embeddings else output
        
        # Project Question Embedding to a lower dimension space
        q_proj = self.question_proj(question_embedding)                                             # [B, m*D]
        state_query_concat = tf.concat([state, q_proj], axis=-1)                                    # [B, 2m*D + 2D]

        # Encode candidate actions (relation and entity indices to embeddings)
        candidate_action_embeddings = self.action_encoder(next_relations, next_entities)            # [B, max_actions, m*D]

        # MLP for policy (attention)
        policy_vec = self.policy_MLP(state_query_concat)                                            # transformation for representation [B, m*D]
        policy_vec = tf.expand_dims(policy_vec, axis=1)                                             # [B, 1, m*D]
        prelim_scores = tf.reduce_sum(tf.multiply(candidate_action_embeddings, policy_vec), axis=2) # dot product attention [B, max_actions]

        # Masking PAD actions & Giving Low Scores (for invalid actions or empty action due to extra pads)
        comparison_tensor = tf.ones_like(next_relations, dtype=tf.int32) * self.rPAD                # matrix to compare
        mask = tf.equal(next_relations, comparison_tensor)                                          # mask for padding, mainly for masking direct action to answer at step 1
        dummy_scores = tf.ones_like(prelim_scores) * -99999.0                                       # the base matrix to choose from if dummy relation
        scores = tf.where(mask, dummy_scores, prelim_scores)                                        # assign the scores where invalid [B, max_actions]

        # Sample the actions based on these scores (not deterministic). Will give the indices/id
        action = tf.cast(tf.random.categorical(logits=scores, num_samples=1), tf.int32)             # [B, 1]

        # Calculate the Loss
        # Cross-entropy against sampled action (REINFORCE-style)
        action_idx =  tf.squeeze(action, axis=1)                                                 # [B,]
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=action_idx)  # [B,]

        # advanced tensor indexing to extract the actual relation IDs that were selected by the agent
        chosen_relation = tf.gather_nd(next_relations, tf.transpose(tf.stack([range_arr, action_idx])))

        return loss, new_state, tf.nn.log_softmax(scores), action_idx, chosen_relation

    # Question Projection Adapters
    def question_proj_linear(self, x):
        """
        Projects question embeddings into the policy/KG feature space using a
        linear transformation. This allows for a simple alignment of the question
        representation with the graph embedding space.
        Linear Projection:
            output = ReLU(W * x + b)
        Args:
            x: Input question embeddings. Shape: [batch_size, input_dimension]
        Returns:
            Projected question embeddings. Shape: [batch_size, out_dim]
        """
        with tf.compat.v1.variable_scope("question_dense", reuse=tf.compat.v1.AUTO_REUSE):
            output = tf.compat.v1.layers.dense(
                x,
                self.out_dim,
                activation=None,
                name="dense"
            )
            
            # Set up initialization operation on first call
            if self.question_proj_init is None:
                # Get the dense layer variables
                proj_vars = tf.compat.v1.get_collection(
                    tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, 
                    scope="question_projection/question_dense/dense"
                )
                if len(proj_vars) >= 2:  # weight and bias
                    weight_var, bias_var = proj_vars[0], proj_vars[1]
                    # Create assignment operations - simplified for compatibility
                    try:
                        weight_assign = weight_var.assign(self.question_embedding_placeholder[:tf.shape(weight_var)[0], :tf.shape(weight_var)[1]])
                        bias_assign = bias_var.assign(self.question_embedding_placeholder[tf.shape(weight_var)[0], :tf.shape(bias_var)[0]])
                        self.question_proj_init = tf.group(weight_assign, bias_assign)
                    except:
                        # If assignment fails, skip pretrained initialization
                        self.question_proj_init = tf.no_op()
            
            return output
    
    # Multi-layer MLP projection
    def question_proj_mlp(self, x):
        """
        Projects question embeddings into the policy/KG feature space using a
        multi-layer perceptron (MLP). This allows for flexible transformation of the
        question representation to better align with the graph embedding space.
        MLP Architecture:
        MLP:
            h_1 = ReLU(W_1 * x + b_1)
            h_2 = ReLU(W_2 * h_1 + b_2)
            ...
            output = W_n * h_{n-1} + b_n
        Args:
            x: Input question embeddings. Shape: [batch_size, input_dimension]
        Returns:
            Projected question embeddings. Shape: [batch_size, out_dim]
        """
        with tf.compat.v1.variable_scope("question_mlp", reuse=tf.compat.v1.AUTO_REUSE):
            h = x
            for layer_idx in range(self.projection_layers - 1):
                h = tf.compat.v1.layers.dense(
                    h,
                    self.projection_hidden,
                    activation=tf.nn.relu,
                    name=f"mlp_fc{layer_idx+1}"
                )
            output = tf.compat.v1.layers.dense(
                h,
                self.out_dim,
                activation=None,
                name=f"mlp_fc{self.projection_layers}"
            )

            if self.question_proj_init is None:
                # Get the dense layer variables
                proj_vars = tf.compat.v1.get_collection(
                    tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, 
                    scope="question_projection/question_mlp"
                )
                if len(proj_vars) >= 2 * self.projection_layers:  # weight and bias for each layer
                    assign_ops = []
                    for layer_idx in range(self.projection_layers):
                        weight_var = proj_vars[2 * layer_idx]
                        bias_var = proj_vars[2 * layer_idx + 1]
                        try:
                            weight_assign = weight_var.assign(self.question_embedding_placeholder[:tf.shape(weight_var)[0], :tf.shape(weight_var)[1]])
                            bias_assign = bias_var.assign(self.question_embedding_placeholder[tf.shape(weight_var)[0], :tf.shape(bias_var)[0]])
                            assign_ops.extend([weight_assign, bias_assign])
                        except:
                            # If assignment fails, skip pretrained initialization
                            self.question_proj_init = tf.no_op()
                            break
                    else:
                        self.question_proj_init = tf.group(*assign_ops)

            return output

    # Residual adapter projection
    def question_proj_residual(self, x):
        """
        Projects question embeddings into the policy/KG feature space using a
        residual MLP architecture. This allows for flexible adaptation of the
        question representation while preserving the original information through
        a skip connection.
        ```
        Residual Adapter:
            z = Linear(x)
            r = MLP(z)
            output = z + r
        ```
        Args:
            x: Input question embeddings. Shape: [batch_size, input_dimension]
        Returns:
            Projected question embeddings. Shape: [batch_size, out_dim]
        """
        with tf.compat.v1.variable_scope("question_residual_adapter", reuse=tf.compat.v1.AUTO_REUSE):
            # 1) Linear down-projection to policy/KG feature space (NO activation)
            z = tf.compat.v1.layers.dense(
                x,
                self.out_dim,
                activation=None,
                use_bias=True,
                name="down"
            )

            # 2) Residual MLP in out_dim space
            h = z
            for layer_idx in range(self.projection_layers - 1):
                h = tf.compat.v1.layers.dense(
                    h,
                    self.projection_hidden,
                    activation=tf.nn.relu,
                    use_bias=True,
                    name=f"res_fc{layer_idx+1}"
                )
            r = tf.compat.v1.layers.dense(
                h,
                self.out_dim,
                activation=None,
                use_bias=True,
                name=f"res_fc{self.projection_layers}"
            )

            output = z + r  # residual add in the same dimension

            # --- Optional: pretrained init for the DOWN projection only ---
            # This initializes only the "down" layer (kernel+bias). Residual MLP stays randomly initialized.
            if self.question_proj_init is None:
                try:
                    down_kernel = tf.compat.v1.get_variable("down/kernel")
                    down_bias   = tf.compat.v1.get_variable("down/bias")

                    # Assert shapes match at runtime where possible
                    self.question_proj_init = tf.group(
                        tf.compat.v1.assign(down_kernel, self.qproj_kernel_ph),
                        tf.compat.v1.assign(down_bias,   self.qproj_bias_ph)
                    )
                except Exception:
                    # If variable names differ or init not needed, safely no-op
                    self.question_proj_init = tf.no_op()

            return output

    def export_embeddings(
        self,
        session: tf.compat.v1.Session
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Export the learned entity and relation embedding lookup tables.

        Args:
            session: Active TensorFlow session.

        Returns:
            Tuple containing:
                - entity_embeddings: [num_entities, 2 * entity_embedding_size]
                - relation_embeddings: [num_relations, 2 * embedding_size]
        """
        with session.graph.as_default():
            entity_embeddings, relation_embeddings = session.run([
                self.entity_lookup_table,
                self.relation_lookup_table
            ])
        return entity_embeddings, relation_embeddings
    
    def export_projection_weights(
        self,
        session: tf.compat.v1.Session
    ) -> Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]]:
        """
        Export the learned question projection weights.

        Returns:
            A dictionary mapping layer names to (kernel, bias) tuples.
            Returns None if no matching projection variables are found.

            Examples:
                linear:
                    {
                        "dense": (kernel, bias)
                    }

                mlp:
                    {
                        "mlp_fc1": (kernel, bias),
                        "mlp_fc2": (kernel, bias),
                        ...
                    }

                residual:
                    {
                        "down": (kernel, bias),
                        "res_fc1": (kernel, bias),
                        ...
                    }
        """
        with session.graph.as_default():
            if self.projection_adapter == "linear":
                scope = "question_projection/question_dense/dense"
                expected_layers = ["dense"]

            elif self.projection_adapter == "mlp":
                scope = "question_projection/question_mlp"
                expected_layers = [f"mlp_fc{i+1}" for i in range(self.projection_layers)]

            elif self.projection_adapter == "residual":
                scope = "question_projection/question_residual_adapter"
                expected_layers = ["down"] + [f"res_fc{i+1}" for i in range(self.projection_layers)]

            else:
                return None

            vars_in_scope = tf.compat.v1.global_variables(scope=scope)
            if not vars_in_scope:
                return None

            var_dict = {v.name: v for v in vars_in_scope}
            exported = {}

            for layer_name in expected_layers:
                kernel_var = None
                bias_var = None

                for var_name, var in var_dict.items():
                    if f"/{layer_name}/kernel:" in var_name:
                        kernel_var = var
                    elif f"/{layer_name}/bias:" in var_name:
                        bias_var = var

                if kernel_var is not None and bias_var is not None:
                    kernel, bias = session.run([kernel_var, bias_var])
                    exported[layer_name] = (kernel, bias)

            return exported if exported else None

    def export_representation_parameters(
        self,
        session: tf.compat.v1.Session,
        output_path: str
    ) -> None:
        """
        Export entity embeddings, relation embeddings, and question projection weights
        into a single .npz file.

        Args:
            session: Active TensorFlow session.
            output_path: Output .npz path.
        """
        entity_embeddings, relation_embeddings = self.export_embeddings(session)
        projection_weights = self.export_projection_weights(session)

        save_dict = {
            "entity_embeddings": entity_embeddings,
            "relation_embeddings": relation_embeddings,
        }

        if projection_weights is not None:
            for layer_name, (kernel, bias) in projection_weights.items():
                save_dict[f"{layer_name}_kernel"] = kernel
                save_dict[f"{layer_name}_bias"] = bias

        np.savez(output_path, **save_dict)

    def __call__(
        self, 
        candidate_relation_sequence: List[tf.Tensor], 
        candidate_entity_sequence: List[tf.Tensor], 
        current_entities: List[tf.Tensor], 
        question_embedding: tf.Tensor, 
        range_arr: tf.Tensor, 
        T: int = 3
    ) -> Tuple[List[tf.Tensor], List[tf.Tensor], List[tf.Tensor]]:
        """
        Execute complete multi-step reasoning episode through knowledge graph.
        
        Implements the full MINERVA reasoning algorithm by unrolling the policy
        network for T time steps. At each step, the agent observes candidate actions,
        uses LSTM memory and attention to select optimal actions, and updates its
        internal state. This produces a reasoning path from start to target entities.
        
        The method performs temporal unrolling of the policy network:
        1. Initialize LSTM state and dummy start relation
        2. For each time step t ∈ [0, T):
           - Observe candidate actions at current position
           - Use step() to select action and update state
           - Collect losses, logits, and action indices
        3. Return all collected information for training
        
        Args:
            candidate_relation_sequence: Time sequence of candidate relation options.
                Length T, each element shape: [batch_size, max_actions]
            candidate_entity_sequence: Time sequence of candidate entity options.
                Length T, each element shape: [batch_size, max_actions]
            current_entities: Time sequence of current entity positions.
                Length T, each element shape: [batch_size]
            question_embedding: Natural language question embeddings to answer.
                Shape: [batch_size, embedding_dimension]
            range_arr: Batch indexing array for advanced tensor operations.
                Shape: [batch_size] containing [0, 1, 2, ..., batch_size-1]
            T: Number of reasoning steps to perform. Defaults to 3 for typical
               multi-hop reasoning tasks.
            
        Returns:
            Tuple containing:
                - all_loss: Policy gradient losses at each time step.
                  List of length T, each element shape: [batch_size]
                - all_logits: Log probability distributions over actions at each step.
                  List of length T, each element shape: [batch_size, max_actions]  
                - action_idx: Selected action indices at each time step.
                  List of length T, each element shape: [batch_size]
                  
        Note:
            - Implements policy gradient reinforcement learning (REINFORCE)
            - Uses variable scope reuse for parameter sharing across time steps
            - LSTM state carries reasoning context across all steps
            - Action selection is stochastic for exploration during training
            - Supports variable path lengths via T parameter
        """

        self.baseline_inputs = []
        batch_size = tf.shape(question_embedding)[0]

        # Initial State for LSTM
        state = self.policy_step.zero_state(batch_size=batch_size, dtype=tf.float32)
        prev_relation = tf.ones(batch_size, dtype=tf.int32) * self.drPad  # dummy relation for step 0 NOTE: Might be self loop action

        all_loss = []       # list of loss tensors each [B,]
        all_logits = []     # list of actions each [B,]
        action_idx = []     # list of actions taken

        with tf.compat.v1.variable_scope("policy_steps_unroll") as scope:
            for t in range(T):
                if t > 0:
                    scope.reuse_variables()
                next_rel = candidate_relation_sequence[t]   # [B, max_actions]
                next_ent = candidate_entity_sequence[t]     # [B, max_actions]
                cur_ent = current_entities[t]               # [B,]

                loss, state, logits, idx, chosen_relation = self.step(
                    next_rel,
                    next_ent,
                    state, 
                    prev_relation, 
                    question_embedding,
                    cur_ent,
                    range_arr=range_arr
                )

                all_loss.append(loss)
                all_logits.append(logits)
                action_idx.append(idx)
                prev_relation = chosen_relation

            # [(B, T), m*D]

        return all_loss, all_logits, action_idx
