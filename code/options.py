from __future__ import absolute_import
from __future__ import division

import os
import argparse
import time
import json
from omegaconf import OmegaConf, DictConfig

import wandb

from typing import Dict, Any

def read_options() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="""
        MINERVA: Enhanced Knowledge Graph Reasoning Framework

        This configuration system supports both structured query reasoning and natural
        language question answering over knowledge graphs. MINERVA agents learn to
        navigate multi-hop paths using reinforcement learning to find answers.

        Dual Framework Support:
        - Query-based reasoning: Original MINERVA for structured query answering
        - Natural Language Questions (NLQ): Enhanced framework with transformer integration
        - Reinforcement learning with LSTM-based agents and policy gradients
        - Beam search evaluation and comprehensive metrics (Hits@K, MRR)
        - Modern TensorFlow 2.x compatibility with graph mode preservation

        Key Capabilities:
        - Multi-hop reasoning over large-scale knowledge graphs
        - BERT/transformer integration for question understanding
        - Policy gradient training (REINFORCE) with baseline variance reduction
        - Flexible evaluation with multiple rollouts and beam search
        - Enhanced metrics and path visualization for analysis
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # TODO: Add a config overload with yaml

    # Dataset and Data Processing
    parser.add_argument("--data_input_dir", default="", type=str,
                        help="Directory containing knowledge graph data files")

    # TODO: Add an option to generate the vocab for the user. This argument is currently unused.
    parser.add_argument("--create_vocab", default=0, type=str2bool,
                        help="Whether to create vocabulary files or use existing. Accepts: yes/true/t/y/1 (True) or no/false/f/n/0 (False)")
    parser.add_argument("--vocab_dir", default="", type=str,
                        help="Directory containing entity and relation vocabulary files")
    parser.add_argument("--max_num_actions", default=200, type=int,
                        help="Maximum number of relations/actions per entity in the knowledge graph")
    parser.add_argument("--use_full_graph", type=str2bool, default='True',
                        help="Whether to use the full knowledge graph (train + test + dev) or a subgraph (train).")

    # QA Dataset
    parser.add_argument("--question_format", default="full_text", type=str, choices=["full_text", "relation_only", "graph_only"],
                        help="Format of the question input ('full_text', 'relation_only', 'graph_only')")
    parser.add_argument('--raw_QAData_path', type=str, default="",
                         help="Path to the raw QA CSV dataset. Only required for NLQ Task.")
    parser.add_argument('--cached_QAMetaData_path', type=str, default="",
                         help="Path to cached tokenized QA metadata JSON file. Only required for NLQ Task.")
    parser.add_argument('--force_data_prepro', '-f', action="store_true",
                         help="Force re-processing of QA data, even if cache exists")
    parser.add_argument("--multi_answers", type=str2bool, default='False',
                         help="Whether to handle multiple answers per question. Accepts: yes/true/t/y/1 (True) or no/false/f/n/0 (False)")
    
    # Textual Embedding (LLMs)
    parser.add_argument("--question_tokenizer_name", type=str, default="bert-base-uncased",
                         help="Tokenizer name for question embeddings. Only required for NLQ Task.")

    # LSTM/Neural Network Architecture
    parser.add_argument("--hidden_size", default=50, type=int,
                        help="Hidden state size for LSTM layers")
    parser.add_argument("--LSTM_layers", default=1, type=int,
                        help="Number of LSTM layers in the agent network")

    # Embedding Configuration
    parser.add_argument("--embedding_size", default=50, type=int,
                        help="Embedding dimension for entities and relations. Keep in mind that each will be doubled.")
    parser.add_argument("--use_entity_embeddings", default='False', type=str2bool,
                        help="Whether to use entity embeddings. Accepts: yes/true/t/y/1 (True) or no/false/f/n/0 (False)")
    parser.add_argument("--train_entity_embeddings", default='False', type=str2bool,
                        help="Whether to fine-tune entity embeddings during training. Accepts: yes/true/t/y/1 (True) or no/false/f/n/0 (False)")
    parser.add_argument("--train_relation_embeddings", default='True', type=str2bool,
                        help="Whether to train relation embeddings. Accepts: yes/true/t/y/1 (True) or no/false/f/n/0 (False)")
    parser.add_argument("--projection_adapter", default="linear", type=str, choices=["linear", "mlp", "residual"],
                        help="Type of question projection adapter ('linear', 'mlp', 'residual')")
    parser.add_argument("--projection_layers", default=2, type=int,
                        help="Number of layers in the projection adapter (if applicable)")
    parser.add_argument("--projection_hidden", default=256, type=int,
                        help="Hidden size for each layer in the projection adapter (if applicable)")

    # Reinforcement Learning
    # TODO: See if we can modify path length to reasoning length
    parser.add_argument("--path_length", default=3, type=int,
                        help="Maximum number of reasoning steps/hops in the knowledge graph")
    parser.add_argument("--num_rollouts", default=20, type=int,
                        help="Number of rollout trajectories per question during training")
    parser.add_argument("--test_rollouts", default=100, type=int,
                        help="Number of rollout trajectories per question during evaluation")
    parser.add_argument("--positive_reward", default=1.0, type=float,
                        help="Reward value when agent reaches the correct answer entity")
    parser.add_argument("--negative_reward", default=0, type=float,
                        help="Reward value when agent doesn't reach the correct answer")
    parser.add_argument("--gamma", default=1, type=float,
                        help="Discount factor for future rewards in RL (typically 0.9-1.0)")
    parser.add_argument("--use_stop_signal", default='False', type=str2bool,
                        help="Whether to include a STOP action in the action space. Accepts: yes/true/t/y/1 (True) or no/false/f/n/0 (False)")
    parser.add_argument("--stop_signal_reward", default=0.0, type=float,
                        help="Reward for taking the STOP action (if use_stop_signal is True)")
    parser.add_argument("--stop_signal_penalty", default=0.0, type=float,
                        help="Penalty for taking the STOP action when not at the correct answer (if use_stop_signal is True). Must be a positive value, as it will be subtracted from the reward.")
    parser.add_argument("--length_penalty", default=0.0, type=float,
                        help="Penalty for each step taken to encourage shorter reasoning paths (if use_stop_signal is True). Must be a positive value for the scalar, as it will be subtracted from the reward.")
    parser.add_argument("--use_restart_signal", default='False', type=str2bool,
                        help="Whether to include a RESTART action in the action space. Accepts: yes/true/t/y/1 (True) or no/false/f/n/0 (False)")
    
    # Training Configuration
    parser.add_argument("--batch_size", default=128, type=int,
                        help="Number of questions processed in each training batch")
    parser.add_argument("--test_batch_size", default=128, type=int,
                        help="Number of questions processed in each evaluation batch")
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
    parser.add_argument("--nell_evaluation", default='False', type=str2bool,
                        help="Whether to perform NELL evaluation. Accepts: yes/true/t/y/1 (True) or no/false/f/n/0 (False)")

    # Evaluation Config
    parser.add_argument("--pool", default="max", type=str, choices=["max", "sum"],
                        help="Pooling method for Evaluation of Rollouts ('max', 'sum')")
    parser.add_argument("--use_beam", default='False', type=str2bool,
                        help="Whether to use beam search during decoding. Accepts: yes/true/t/y/1 (True) or no/false/f/n/0 (False)")
    parser.add_argument("--print_paths", default='False', type=str2bool,
                        help="Whether to print the reasoning paths taken by the agent. Accepts: yes/true/t/y/1 (True) or no/false/f/n/0 (False)")
    parser.add_argument("--print_predictions", default='False', type=str2bool,
                        help="Whether to print the final predicted answers by the agent. Accepts: yes/true/t/y/1 (True) or no/false/f/n/0 (False)")

    # Model Loading and Saving
    parser.add_argument("--model_dir", default="", type=str,
                        help="Directory to save trained model checkpoints")
    parser.add_argument("--model_load_dir", default="", type=str,
                        help="Directory to load pre-trained model from")
    parser.add_argument("--load_model", default='False', type=str2bool,
                        help="Whether to load a pre-trained model or train from scratch. Accepts: yes/true/t/y/1 (True) or no/false/f/n/0 (False)")
    parser.add_argument("--base_output_dir", default="", type=str,
                        help="Base directory for all output files and logs")
    
    # Logging and Tracking
    parser.add_argument("--log_file_name", default="reward.txt", type=str,
                        help="Name of the main log file")
    parser.add_argument('--wandb_project', type=str, default='', 
                        help='Weights & Biases project name for experiment tracking')
    parser.add_argument('--wandb_name', type=str, default='',
                        help='Custom name for this specific run (optional)')
    parser.add_argument('--wandb_tags', type=str, nargs='*', default=None,
                        help='Tags for the Weights & Biases run (optional)')
    parser.add_argument('--wandb_notes', type=str, default='',
                        help='Notes for the Weights & Biases run (optional)')
    parser.add_argument('--track', default='False', type=str2bool,
                        help='Enable Weights & Biases tracking. Accepts: yes/true/t/y/1 (True) or no/false/f/n/0 (False)')
    parser.add_argument("--timestamp", type=str, default=None,
                         help="Timestamp for the run. If None, current time is used.")
    
    parser.add_argument('--config_yaml', type=str, default='',
                        help='Path to a YAML configuration file to overload default parameters')

    # Miscellaneous
    parser.add_argument("--seed", type=int, default=42,
                         help="Random seed for reproducibility")

    try:
        parsed = vars(parser.parse_args())
    except IOError as msg:
        parser.error(str(msg))
    
    if parsed['config_yaml'] != '':
        assert os.path.exists(parsed['config_yaml']), f"YAML config file {parsed['config_yaml']} does not exist."
        print(f"Overloading configuration with YAML file: {parsed['config_yaml']}")
        args_namespace = argparse.Namespace(**parsed)
        args_namespace = overload_parse_defaults_with_yaml(parsed['config_yaml'], args_namespace)
        parsed = vars(args_namespace)

    if parsed['timestamp'] is None:
        local_time = time.localtime()
        parsed['timestamp'] = time.strftime("%Y%m%d_%H%M%S", local_time)

    if parsed['track']:
        if parsed['wandb_project'] == '':
            raise ValueError('wandb_project must be specified if tracking is enabled.')
        
        # Extract dataset name for run naming
        dataset_name = parsed['data_input_dir'].split('/')[-1] if parsed['data_input_dir'] else 'unknown'
        run_name = parsed['wandb_name'] if parsed['wandb_name'] else f"minerva-{dataset_name}-{parsed['timestamp']}"
        
        wandb.init(
            project=parsed['wandb_project'],
            name=run_name,
            config=parsed,
            tags=parsed['wandb_tags'] if 'wandb_tags' in parsed else None,
            notes=parsed['wandb_notes'] if 'wandb_notes' in parsed else None,
        )
        
        # Check if we're in a sweep - if so, overwrite parameters with sweep config
        if wandb.run.sweep_id is not None:
            print(f"Running WANDB sweep: {wandb.run.sweep_id}")
            print("Overwriting parameters with sweep configuration...")
            
            # Parameters that should NOT be overwritten by sweeps
            protected_params = {
                'timestamp', 'output_dir', 'model_dir', 'path_logger_file', 
                'log_file_name', 'base_output_dir', 'data_input_dir', 
                'vocab_dir', 'raw_QAData_path', 'cached_QAMetaData_path',
                'wandb_project', 'wandb_name', 'track'
            }
            
            # Update parsed with swept parameters from wandb.config
            for key, value in wandb.config.items():
                if key in parsed and key not in protected_params:
                    old_value = parsed[key]
                    parsed[key] = value
                    print(f"  {key}: {old_value} -> {value}")
                elif key in protected_params:
                    print(f"  {key}: protected from override (keeping: {parsed[key]})")
        else:
            print("Running in tracking mode - keeping original parameters")

    # TODO: Avoid creating a directory if loading a model
    # Preparing Directories for model saving and logging (AFTER parameter override)
    parsed['output_dir'] = os.path.join(parsed['base_output_dir'], parsed['timestamp'])
    parsed['model_dir'] = os.path.join(parsed['output_dir'], 'model/')
    parsed['path_logger_file'] = parsed['output_dir']
    parsed['log_file_name'] = os.path.join(parsed['output_dir'], 'log.txt')
    os.makedirs(parsed['output_dir'])
    os.mkdir(parsed['model_dir'])

    with open(os.path.join(parsed['output_dir'], 'config.json'), 'w') as out:
        # sort keys for consistency
        json.dump(parsed, out, indent=4, sort_keys=True)

    # print and return
    maxLen = max([len(ii) for ii in parsed.keys()])
    fmtString = '\t%' + str(maxLen) + 's : %s (type=%s)'
    print('Arguments:')
    for key in sorted(parsed.keys()):
        print(fmtString % (key, str(parsed[key]), str(type(parsed[key]))))
    return parsed

def str2bool(string: str) -> bool:
    """
    Converts a string input to a boolean value.
    
    Args:
        string (str): The string to convert ('yes', 'true', 'no', 'false', etc.).
    
    Returns:
        bool: The corresponding boolean value.
    
    Raises:
        argparse.ArgumentTypeError: If the input cannot be converted to a boolean.
    """
    if isinstance(string, bool):
       return string
    
    if isinstance(string, int):
        return bool(string)
   
    if string.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif string.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    elif string.lower() in ('none'):
        return None
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def recurse_til_leaf(d: dict, parent_key: str = "") -> dict:
    return_dict = {}
    for k, v in d.items():
        next_key = f"{parent_key}_{k}" if parent_key != "" else k
        if isinstance(v, dict):
            deep_dict = recurse_til_leaf(v, parent_key=next_key)
            return_dict.update(deep_dict)
        else:
            return_dict[next_key] = v
    return return_dict

def overload_parse_defaults_with_yaml(
    yaml_location:str, 
    args: argparse.Namespace,
    resolve: bool = True,
    ) -> argparse.Namespace:
    # check if the yaml file exists
    if not os.path.exists(yaml_location):
        print(f"Yaml file {yaml_location} does not exist, skipping yaml overload")
        return args
    
    print(f"Trying to import the yaml file {yaml_location}")
    ycfg: DictConfig = OmegaConf.load(yaml_location)

    # Optional: flatten nested config like your recurse_til_leaf
    # If your YAML is already flat, you can skip this.
    ydict = OmegaConf.to_container(ycfg, resolve=resolve)
    if not isinstance(ydict, dict):
        raise ValueError(f"YAML root must be a mapping/dict, got {type(ydict)}")

    print(f"Imported yaml with keys {list(ydict.keys())}")

    # Apply overrides onto argparse args
    for k, v in ydict.items():
        if hasattr(args, k):
            setattr(args, k, v)
        else:
            raise ValueError(
                f"Yaml config file {yaml_location} imposes parameter '{k}', "
                f"however this parameter is not found in args"
            )
    return args