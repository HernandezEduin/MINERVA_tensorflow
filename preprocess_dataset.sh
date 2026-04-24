export PYTHONPATH="."

dataset_name="$1"
node_data_key="$2"
rel_data_key="$3"

echo "Preprocessing dataset: $dataset_name"

cmd=(python code/data/preprocessing_scripts/create_graph.py --root_dir "./" -f --data_dir "datasets/nlq/" --dataset "$dataset_name")

echo "Executing: ${cmd[*]}"
"${cmd[@]}"

cmd=(python code/data/preprocessing_scripts/create_vocab.py --root_dir "./" --data_dir "datasets/nlq/" --dataset "$dataset_name")
echo "Executing: ${cmd[*]}"
"${cmd[@]}"
# check if node_data_key and rel_data_key are not empty, if not empty, run create_vocab_title.py
if [[ -n "$node_data_key" ]] && [[ -n "$rel_data_key" ]]; then
    cmd=(python code/data/preprocessing_scripts/create_vocab_title.py --root_dir "./" --data_dir "datasets/nlq/" --dataset "$dataset_name" --node_data_key "$node_data_key" --relation_data_key "$rel_data_key")
    echo "Executing: ${cmd[*]}"
    "${cmd[@]}"
fi