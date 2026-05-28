# KGQA Adapted MINERVA
This is the repository for the paper *Theseus in the Graph: Towards Traceable Multi-Hop Graph Navigation* for the adapted MINERVA model on Knowledge Graph Question Answering (KGQA).

This repository is an anonymized train-and-eval release for the double-blind review process. It contains the code and configs for dataset preprocessing, training, and evaluation. The preprocessed datasets and pretrained checkpoints are also available at [Kaggle](https://www.kaggle.com/models/anonymousexpert/minerva-kgqa).

## Review Notice

This repository and the accompanying datasets are provided solely for the double-blind review process.

Please do **not** redistribute, repost, mirror, or publicly release any part of this code or data. A full public release will be provided after the review process concludes.

## Environment Setup

Create and activate a fresh conda environment with **Python 3.9**:

```bash
conda create -n minerva python=3.9 pip -y
conda activate minerva
pip install -r requirements.txt
```

Optionally, you can create the conda environment directly from environment.yml:

```bash
conda env create -f environment.yml
conda activate minerva
```

If you plan to run evaluation on GPU, install the following in your conda environment:

```bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 -y
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

## Kaggle Datasets and Checkpoints
The Kaggle release is organized into cached metadata, preprocessed datasets, and pretrained checkpoints:

```
.cache/
  itl/
    <dataset>_qa_nhop.json                              # cached QA metadata
    <dataset>_qa_nhop_Split-<split>_date-<date>.parquet # cached split files

datasets/
  nlq/
    <dataset>/
      vocab/
        entity_title.json                                # entity ID -> semantic label
        entity_vocab.json                                # entity ID -> Machine ID
        relation_title.json                              # relation ID -> semantic label
        relation_vocab.json                              # relation ID -> Machine ID
      <dataset>_qa_nhop.csv                              # question-answer pairs
      full_graph.txt                                     # evaluation graph
      node_data.csv                                      # node metadata
      relation_data.csv                                  # relation metadata
      triplets.txt                                       # KG triplets

checkpoints/
  <dataset>/
    <run_name>/                                          # e.g. <dataset>_qa_nhop_reason_<path_length>hop_seed<seed>
      model/
        model.ckpt
      test_beam/
        test_paths.txt                                   # predicted label paths for test set
      scores.txt                                         # evaluation scores for test set
```

Here, `<dataset>` is the dataset name, `<split>` is typically `train`, `dev`, or `test`, and `<run_name>` identifies a specific released checkpoint.

## Configs

Released configs follow the pattern `configs/<dataset>/...`, where `<dataset>` is a placeholder for the dataset name.

Use the config files for the dataset you want to preprocess, train, or evaluate.

For the full experimental setup, we trained three models per dataset and report averaged results across seeds. For the double-blind review release, we provide three checkpoints for each dataset, using seeds `0`, `42`, and `100`. By default, the evaluation configs point to the checkpoint with seed `0`. To evaluate the other checkpoints, update the `seed` field in the config to point to the desired checkpoint.

If your local checkpoint path differs from the default one in a config, update:
1. `load_model: True`
2. `model_load_dir: <path-to-checkpoint>`

## Preprocess Data

If you only want to evaluate the provided checkpoints, you can skip data preprocessing and training.

To preprocess a dataset under `datasets/nlq/<dataset>/`, run:

```bash
bash run_data_preprocess.sh <dataset>
```

If the dataset includes `node_data.csv` and `relation_data.csv`, you can also generate title-aware vocab files by passing the corresponding column names:

```bash
bash run_data_preprocess.sh <dataset> <node_data_key> <relation_data_key>
```

This script creates the graph files and vocab files used by training and evaluation.

## Train

Run training on CPU:

```bash
bash run_train.sh configs/<dataset>/<train-config>.yaml
```

Run training on a specific GPU:

```bash
bash run_train.sh configs/<dataset>/<train-config>.yaml 0
```

Training outputs, including model checkpoints, are written according to the selected config.

## Evaluate

Run evaluation on CPU:

```bash
bash run_eval.sh configs/<dataset>/evaluate.yaml
```

Run evaluation on a specific GPU (not ideal for large data like MetaQA):

```bash
bash run_eval.sh configs/<dataset>/evaluate.yaml 0
```

Evaluation outputs are written by the evaluation pipeline according to the selected config.

Train and evaluation configs are mostly the same. The main differences are:
- In training configs, `timestamp` determines the model save folder run_name and `load_model = False`.
- In evaluation configs, `load_model_dir` must point to the full path of the checkpoint model (e.g., `checkpoints/<dataset>/<run_name>/model/model.ckpt`) and `load_model = True`.
