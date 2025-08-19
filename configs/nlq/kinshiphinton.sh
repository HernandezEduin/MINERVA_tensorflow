#!/usr/bin/env bash

data_input_dir="datasets/data_preprocessed/kinshiphinton/"
vocab_dir="datasets/data_preprocessed/kinshiphinton/vocab"
total_iterations=2000
eval_every=100
path_length=2
hidden_size=50
embedding_size=50
batch_size=16
beta=0.05
Lambda=0.05
use_entity_embeddings=False
train_entity_embeddings=False
train_relation_embeddings=True
base_output_dir="output/kinshiphinton/"
load_model=False
model_load_dir="null"
raw_QAData_path="./datasets/data_preprocessed/kinshiphinton/kinship_hinton_qa_2hop.csv"
cached_QAMetaData_path="./.cache/itl/kinship_hinton_qa_2hop.json"
question_tokenizer_name="bert-base-uncased"
use_beam=False
print_path=False
wandb_project="kinship-hinton-minerva"
wandb_name="test"
track=True