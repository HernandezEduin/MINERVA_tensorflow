import json
import csv
import argparse
import os

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Create a graph for the dataset")
    parser.add_argument("--dataset", type=str, default="kinshiphinton",
                        help="Name of the dataset to create the graph for")
    parser.add_argument("--root_dir", type=str, default="../../../",
                        help="Root directory for the dataset")
    parser.add_argument("--data_dir", type=str, default="datasets/data_preprocessed/",
                        help="Directory where the dataset is located")
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_args()

    dir = os.path.join(args.root_dir, args.data_dir, args.dataset)
    if not os.path.exists(dir):
        raise FileNotFoundError(f"Dataset directory {dir} does not exist.")
    
    vocab_dir = os.path.join(dir, 'vocab/')
    os.makedirs(vocab_dir, exist_ok=True)

    # check if full_graph is present
    files = ['train.txt', 'dev.txt', 'test.txt', 'graph.txt']
    if os.path.exists(os.path.join(dir, 'full_graph.txt')):
        files.append('full_graph.txt')

    entity_vocab = {}
    relation_vocab = {}

    entity_vocab['PAD'] = len(entity_vocab)
    entity_vocab['UNK'] = len(entity_vocab)
    relation_vocab['PAD'] = len(relation_vocab)
    relation_vocab['DUMMY_START_RELATION'] = len(relation_vocab)
    relation_vocab['NO_OP'] = len(relation_vocab)
    relation_vocab['UNK'] = len(relation_vocab)

    entity_counter = len(entity_vocab)
    relation_counter = len(relation_vocab)

    for f in files:
        with open(os.path.join(dir, f)) as raw_file:
            csv_file = csv.reader(raw_file, delimiter='\t')
            for line in csv_file:
                e1,r,e2 = line
                if e1 not in entity_vocab:
                    entity_vocab[e1] = entity_counter
                    entity_counter += 1
                if e2 not in entity_vocab:
                    entity_vocab[e2] = entity_counter
                    entity_counter += 1
                if r not in relation_vocab:
                    relation_vocab[r] = relation_counter
                    relation_counter += 1


    with open(os.path.join(vocab_dir, 'entity_vocab.json'), 'w') as fout:
        json.dump(entity_vocab, fout)

    with open(os.path.join(vocab_dir, 'relation_vocab.json'), 'w') as fout:
        json.dump(relation_vocab, fout)

    print(f"Entity vocab size: {len(entity_vocab)}")
    print(f"Relation vocab size: {len(relation_vocab)}")
    print(f"Vocabularies saved in {vocab_dir}")
    print("Vocabularies created successfully.")

