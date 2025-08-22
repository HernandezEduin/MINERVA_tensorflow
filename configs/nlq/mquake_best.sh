#!/usr/bin/env bash

data_input_dir="datasets/data_preprocessed/mquake/"
vocab_dir="datasets/data_preprocessed/mquake/vocab"
total_iterations=40000
eval_every=100
path_length=4
hidden_size=50
embedding_size=50
batch_size=32
beta=0.05
Lambda=0.05
use_entity_embeddings=True
train_entity_embeddings=True
train_relation_embeddings=True
base_output_dir="output/mquake/"
load_model=False
model_load_dir="./saved_models/mquake/qa_nhop_reason_4hop_40k/model/model.ckpt"
raw_QAData_path="./datasets/data_preprocessed/mquake/mquake_qa_nhop.csv"
cached_QAMetaData_path="./.cache/itl/mquake_qa_nhop.json"
question_tokenizer_name="bert-base-uncased"
use_beam=True
print_path=True
wandb_project="mquake-minerva"
wandb_name="QAnhop-Reason4hop"
track=True