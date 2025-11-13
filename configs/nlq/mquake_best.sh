# best configurations for QA hops 2, 3, 4

base_output_dir="output/mquake/"
batch_size=32
beta=0.05
data_input_dir="datasets/data_preprocessed/mquake/"
embedding_size=50
eval_every=100
gamma=1
hidden_size=50
Lambda=0.05
learning_rate=1e-3
load_model=False
max_num_actions=200
model_load_dir="null"
num_rollouts=20
total_iterations=15000
train_entity_embeddings=True
train_relation_embeddings=True
print_path=False
question_tokenizer_name="bert-base-uncased"
use_beam=True
use_entity_embeddings=True
vocab_dir="datasets/data_preprocessed/mquake/vocab"
wandb_project="mquake-minerva"

# Recommended variables to tune
use_full_graph=False
path_length=2
raw_QAData_path="./datasets/data_preprocessed/mquake/mquake_qa_2hop.csv"
cached_QAMetaData_path="./.cache/itl/mquake_qa_2hop.json"
print_predictions=True
wandb_name="empty"
track=False