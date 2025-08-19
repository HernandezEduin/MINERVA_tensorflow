from __future__ import absolute_import
from __future__ import division
import argparse
import uuid
import os
from pprint import pprint


def read_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_input_dir", default="", type=str)
    parser.add_argument("--input_file", default="train.txt", type=str)
    parser.add_argument("--create_vocab", default=0, type=int)
    parser.add_argument("--vocab_dir", default="", type=str)
    parser.add_argument("--max_num_actions", default=200, type=int)
    parser.add_argument("--path_length", default=3, type=int)
    parser.add_argument("--hidden_size", default=50, type=int)
    parser.add_argument("--embedding_size", default=50, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--grad_clip_norm", default=5, type=int)
    parser.add_argument("--l2_reg_const", default=1e-2, type=float)
    parser.add_argument("--learning_rate", default=1e-3, type=float)
    parser.add_argument("--beta", default=1e-2, type=float)
    parser.add_argument("--positive_reward", default=1.0, type=float)
    parser.add_argument("--negative_reward", default=0, type=float)
    parser.add_argument("--gamma", default=1, type=float)
    parser.add_argument("--log_dir", default="./logs/", type=str)
    parser.add_argument("--log_file_name", default="reward.txt", type=str)
    parser.add_argument("--output_file", default="", type=str)
    parser.add_argument("--num_rollouts", default=20, type=int)
    parser.add_argument("--test_rollouts", default=100, type=int)
    parser.add_argument("--LSTM_layers", default=1, type=int)
    parser.add_argument("--model_dir", default='', type=str)
    parser.add_argument("--base_output_dir", default='', type=str)
    parser.add_argument("--total_iterations", default=2000, type=int)

    parser.add_argument("--Lambda", default=0.0, type=float)
    parser.add_argument("--pool", default="max", type=str)
    parser.add_argument("--eval_every", default=100, type=int)
    parser.add_argument("--use_entity_embeddings", default=0, type=int)
    parser.add_argument("--train_entity_embeddings", default=0, type=int)
    parser.add_argument("--train_relation_embeddings", default=1, type=int)
    parser.add_argument("--model_load_dir", default="", type=str)
    parser.add_argument("--load_model", default=0, type=int)
    # parser.add_argument("--nell_evaluation", default=0, type=int)
    # parser.add_argument("--nell_query", default='all', type=str)

    try:
        parsed = vars(parser.parse_args())
    except IOError as msg:
        parser.error(str(msg))
    parsed['input_files'] = [parsed['data_input_dir'] + '/' + parsed['input_file']]

    parsed['use_entity_embeddings'] = (parsed['use_entity_embeddings'] == 1)
    parsed['train_entity_embeddings'] = (parsed['train_entity_embeddings'] == 1)
    parsed['train_relation_embeddings'] = (parsed['train_relation_embeddings'] == 1)

    parsed['pretrained_embeddings_action'] = ""
    parsed['pretrained_embeddings_entity'] = ""

    parsed['output_dir'] = parsed['base_output_dir'] + '/' + str(uuid.uuid4())[:4]+'_'+str(parsed['path_length'])+'_'+str(parsed['beta'])+'_'+str(parsed['test_rollouts'])+'_'+str(parsed['Lambda'])

    parsed['model_dir'] = parsed['output_dir']+'/'+ 'model/'

    parsed['load_model'] = (parsed['load_model'] == 1)

    ##Logger##
    parsed['path_logger_file'] = parsed['output_dir']
    parsed['log_file_name'] = parsed['output_dir'] +'/log.txt'
    os.makedirs(parsed['output_dir'])
    os.mkdir(parsed['model_dir'])
    with open(parsed['output_dir']+'/config.txt', 'w') as out:
        pprint(parsed, stream=out)

    # print and return
    maxLen = max([len(ii) for ii in parsed.keys()])
    fmtString = '\t%' + str(maxLen) + 's : %s'
    print('Arguments:')
    for keyPair in sorted(parsed.items()): print(fmtString % keyPair)
    return parsed

def read_options_nlq():
    parser = argparse.ArgumentParser(
        description="""
        MINERVA NLQ (Natural Language Question) Trainer Configuration

        This provides the arguments for training and evaluating MINERVA agents for
        knowledge graph reasoning using natural language questions. The agent learns
        to navigate multi-hop paths through knowledge graphs using reinforcement
        learning (REINFORCE) to answer complex questions.

        Key Features:
        - Multi-hop reasoning over knowledge graphs
        - LSTM-based question encoding with transformer embeddings
        - Policy gradient training with baseline variance reduction
        - Beam search evaluation for improved accuracy
        - Comprehensive metrics: Hits@K and Mean Reciprocal Rank (MRR)
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # TODO: Add a config overload with yaml

    # Dataset and Data Processing
    parser.add_argument("--data_input_dir", default="", type=str,
                        help="Directory containing knowledge graph data files")

    # TODO: Add an option to generate the vocab for the user. This argument is currently unused.
    parser.add_argument("--create_vocab", default=0, type=int,
                        help="Whether to create vocabulary files (1) or use existing (0)")
    parser.add_argument("--vocab_dir", default="", type=str,
                        help="Directory containing entity and relation vocabulary files")
    parser.add_argument("--max_num_actions", default=200, type=int,
                        help="Maximum number of relations/actions per entity in the knowledge graph")
    
    # QA Dataset
    parser.add_argument('--raw_QAData_path', type=str, default="",
                         help="Path to the raw QA CSV dataset")
    parser.add_argument('--cached_QAMetaData_path', type=str, default="",
                         help="Path to cached tokenized QA metadata JSON file")
    parser.add_argument('--force_data_prepro', '-f', action="store_true",
                         help="Force re-processing of QA data, even if cache exists")
    
    # Textual Embedding (LLMs)
    parser.add_argument("--question_tokenizer_name", type=str, default="bert-base-uncased",
                         help="Tokenizer name for question embeddings")
    
    # LSTM/Neural Network Architecture
    parser.add_argument("--hidden_size", default=50, type=int,
                        help="Hidden state size for LSTM layers")
    parser.add_argument("--LSTM_layers", default=1, type=int,
                        help="Number of LSTM layers in the agent network")

    # Embedding Configuration
    parser.add_argument("--embedding_size", default=50, type=int,
                        help="Embedding dimension for entities and relations. Keep in mind that each will be doubled.")
    # TODO: Replace with str2bool
    parser.add_argument("--use_entity_embeddings", default=0, type=int,
                        help="Whether to use entity embeddings (1) or not (0)")
    # TODO: Replace with str2bool
    parser.add_argument("--train_entity_embeddings", default=0, type=int,
                        help="Whether to fine-tune entity embeddings during training (1) or keep frozen (0)")
    # TODO: Replace with str2bool
    parser.add_argument("--train_relation_embeddings", default=1, type=int,
                        help="Whether to train relation embeddings (1) or keep frozen (0)")
    
    # Reinforcement Learning
    # TODO: See if we can modify path length to reasoning length
    parser.add_argument("--path_length", default=3, type=int,
                        help="Maximum number of reasoning steps/hops in the knowledge graph")
    parser.add_argument("--num_rollouts", default=20, type=int,
                        help="Number of rollout trajectories per question during training")
    parser.add_argument("--test_rollouts", default=100, type=int,
                        help="Number of rollout trajectories per question during evaluation")
    parser.add_argument("--pool", default="max", type=str, choices=["max", "sum"],
                        help="Pooling method for Evaluation of Rollouts ('max', 'sum')")
    parser.add_argument("--positive_reward", default=1.0, type=float,
                        help="Reward value when agent reaches the correct answer entity")
    parser.add_argument("--negative_reward", default=0, type=float,
                        help="Reward value when agent doesn't reach the correct answer")
    parser.add_argument("--gamma", default=1, type=float,
                        help="Discount factor for future rewards in RL (typically 0.9-1.0)")
    
    # Training Configuration
    parser.add_argument("--batch_size", default=128, type=int,
                        help="Number of questions processed in each training batch")
    parser.add_argument("--learning_rate", default=1e-3, type=float,
                        help="Learning rate for the optimizer (Adam) and baseline regularization.")
    parser.add_argument("--grad_clip_norm", default=5, type=int,
                        help="Maximum gradient norm for gradient clipping (prevents explosion)")
    parser.add_argument("--beta", default=1e-2, type=float,
                        help="Entropy regularization coefficient for exploration")
    parser.add_argument("--Lambda", default=0.0, type=float,
                        help="Baseline regularization parameter")
    parser.add_argument("--total_iterations", default=2000, type=int,
                        help="Total number of training iterations")
    parser.add_argument("--eval_every", default=100, type=int,
                        help="Frequency of evaluation (every N training iterations)")
    
    # Model Loading and Saving
    parser.add_argument("--model_dir", default="", type=str,
                        help="Directory to save trained model checkpoints")
    parser.add_argument("--model_load_dir", default="", type=str,
                        help="Directory to load pre-trained model from")
    # TODO: Replace with str2bool
    parser.add_argument("--load_model", default=0, type=int,
                        help="Whether to load a pre-trained model (1) or train from scratch (0)")
    parser.add_argument("--base_output_dir", default="", type=str,
                        help="Base directory for all output files and logs")
    
    # Logging
    parser.add_argument("--log_file_name", default="reward.txt", type=str,
                        help="Name of the main log file")
    
    # Miscellaneous
    parser.add_argument("--seed", type=int, default=42,
                         help="Random seed for reproducibility")

    try:
        parsed = vars(parser.parse_args())
    except IOError as msg:
        parser.error(str(msg))

    parsed['use_entity_embeddings'] = (parsed['use_entity_embeddings'] == 1)
    parsed['train_entity_embeddings'] = (parsed['train_entity_embeddings'] == 1)
    parsed['train_relation_embeddings'] = (parsed['train_relation_embeddings'] == 1)

    parsed['pretrained_embeddings_action'] = ""
    parsed['pretrained_embeddings_entity'] = ""
    parsed['pretrained_question_projector'] = ""

    parsed['output_dir'] = parsed['base_output_dir'] + '/' + str(uuid.uuid4())[:4]+'_'+str(parsed['path_length'])+'_'+str(parsed['beta'])+'_'+str(parsed['test_rollouts'])+'_'+str(parsed['Lambda'])

    parsed['model_dir'] = parsed['output_dir']+'/'+ 'model/'

    parsed['load_model'] = (parsed['load_model'] == 1)

    ##Logger##
    parsed['path_logger_file'] = parsed['output_dir']
    parsed['log_file_name'] = parsed['output_dir'] +'/log.txt'
    os.makedirs(parsed['output_dir'])
    os.mkdir(parsed['model_dir'])
    with open(parsed['output_dir']+'/config.txt', 'w') as out:
        pprint(parsed, stream=out)

    # print and return
    maxLen = max([len(ii) for ii in parsed.keys()])
    fmtString = '\t%' + str(maxLen) + 's : %s'
    print('Arguments:')
    for keyPair in sorted(parsed.items()): print(fmtString % keyPair)
    return parsed