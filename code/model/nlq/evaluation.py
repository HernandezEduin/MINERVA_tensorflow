"""
MINERVA evaluation script for natural language question answering over knowledge graphs.

This module implements a standalone evaluation pipeline for pre-trained MINERVA
reinforcement learning agents. It loads trained model checkpoints and assesses
their performance on test datasets without any training or model updates.

Key components:
- Model checkpoint loading and restoration
- Test dataset evaluation with Hits@K and MRR metrics
- Beam search decoding for improved inference performance
- Optional reasoning path visualization and logging
- Deterministic evaluation with reproducible results
- WANDB tracking disabled for evaluation-only runs

Usage:
    This script is designed to evaluate trained models independently from training,
    typically used for final model assessment on held-out test sets.
"""

from __future__ import absolute_import
from __future__ import division

import json
import logging
import os
import sys

import tensorflow as tf
from scipy.special import logsumexp as lse

from code.data.embedding_server import EmbeddingServer
from code.model.nlq.trainer import TrainerNLQ
from code.data.setup import set_seeds
from code.options import read_options

logger = logging.getLogger()
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

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
        cached_QAMetaData_path=options['cached_QAMetaData_path'],
        raw_QAData_path=options['raw_QAData_path'],
        max_num_actions=options['max_num_actions'],
        embedding_size=options['embedding_size'],
        hidden_size=options['hidden_size'],
        use_entity_embeddings=options['use_entity_embeddings'],
        train_entity_embeddings=options['train_entity_embeddings'],
        train_relation_embeddings=options['train_relation_embeddings'],
        LSTM_layers=options['LSTM_layers'],
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
        embedding_server=embedding_server,
        use_wandb=False  # Do not use WANDB for Evaluation
    )
    
    with tf.compat.v1.Session(config=config) as sess:
        # Set seeds again after session creation to ensure TF operations are deterministic  
        set_seeds(options['seed'])
        trainer.initialize(restore=save_path, sess=sess) # check if it is fine to initialize an already trained model or if we need to create one before this line

        # create files to store results
        if options['print_paths']:
            os.makedirs(os.path.join(path_logger_file, "test_beam"), exist_ok=True)
            trainer.path_logger_file_ = os.path.join(path_logger_file, "test_beam", "paths")
        
        with open(os.path.join(output_dir, 'scores.txt'), 'a') as score_file:
            score_file.write("Test (beam) scores with best model from " + save_path + "\n")

        # Perform Evaluation
        trainer.test(sess, beam=options['use_beam'], print_paths=options['print_paths'], save_model=False, mode='test')

    
    logging.info(f"Evaluation completed. Closing Server")
    embedding_server.close()  # Close the embedding server connection
