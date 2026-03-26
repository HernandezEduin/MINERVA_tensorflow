"""
Knowledge Graph Vocabulary Title Mapping Creator for MINERVA

This preprocessing script creates human-readable title mappings for entities and relations
in knowledge graph datasets. It processes node and relation data files to generate
vocabulary files that map knowledge graph identifiers to their human-readable titles.

This code is specifically designed for Freebase or Wikidata-based knowledge graphs that use
machine identifiers (e.g., Q1234, P567) but also provide human-readable titles. It bridges
the gap between machine-readable IDs and human-interpretable descriptions for better
understanding of MINERVA's reasoning processes.

Key Functionality:
- Loads node data and creates entity ID to title mappings from CSV files
- Processes relation data to create relation property to title mappings
- Handles inverse relations by prefixing titles with "(INV)"
- Adds special vocabulary tokens (PAD, UNK, NO_OP, DUMMY_START_RELATION)
- Exports mappings as JSON files for use in MINERVA reasoning pipeline

Input Files:
- node_data.csv: Contains entity information with QID and Title columns
- relation_data.csv: Contains relation information with Property and Title columns

Output Files:
- entity_title.json: Maps entity QIDs to human-readable titles
- relation_title.json: Maps relation properties to human-readable titles (including inverse relations)

Usage:
    python create_vocab_title.py --dataset mquake --root_dir ../../../ --data_dir datasets/data_preprocessed/

This script is essential for interpretability and debugging of MINERVA's reasoning paths,
allowing conversion from knowledge graph identifiers to human-readable descriptions.
"""

import json
import csv
import argparse
import os

import pandas as pd

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Create a graph for the dataset")
    parser.add_argument("--dataset", type=str, default="mquake",
                        help="Name of the dataset to create the graph for")
    parser.add_argument("--root_dir", type=str, default="../../../",
                        help="Root directory for the dataset")
    parser.add_argument("--data_dir", type=str, default="datasets/nlq/",
                        help="Directory where the dataset is located")
    parser.add_argument("--node_data_key", type=str, default="QID",
                        help="Key to use for node data")
    parser.add_argument("--relation_data_key", type=str, default="Property",
                        help="Key to use for relation data")
    return parser.parse_args(args)

if __name__ == '__main__':
    args = parse_args()
    
    dir = os.path.join(args.root_dir, args.data_dir, args.dataset)
    if not os.path.exists(dir):
        raise FileNotFoundError(f"Dataset directory {dir} does not exist.")
    
    vocab_dir = os.path.join(dir, 'vocab/')
    os.makedirs(vocab_dir, exist_ok=True)

    node_df = pd.read_csv(os.path.join(dir, 'node_data.csv')).fillna('')
    entity2title = node_df.set_index(args.node_data_key)['Title'].to_dict()
    entity2title.update({
        'PAD': 'Padding',
        'UNK': 'Unknown'
    })

    rel_df = pd.read_csv(os.path.join(dir, 'relation_data.csv')).fillna('')
    relation2title = rel_df.set_index(args.relation_data_key)['Title'].to_dict()
    relation2title.update({
        '_' + k: "(INV) " + v for k, v in relation2title.items()
    })
    relation2title.update({
        'PAD': 'Padding',
        'UNK': 'Unknown',
        'NO_OP': 'No Operation',
        'STOP': 'Stop',
        'RESTART': 'Restart',
        'DUMMY_START_RELATION': 'Dummy Start Relation'
    })

    with open(os.path.join(vocab_dir, 'entity_title.json'), 'w') as fout:
        json.dump(entity2title, fout, indent=4)
    print(f"Saved entity title mapping for {len(entity2title)} entities to {os.path.join(vocab_dir, 'entity_title.json')}")

    with open(os.path.join(vocab_dir, 'relation_title.json'), 'w') as fout:
        json.dump(relation2title, fout, indent=4)
    print(f"Saved relation title mapping for {len(relation2title)} relations to {os.path.join(vocab_dir, 'relation_title.json')}")