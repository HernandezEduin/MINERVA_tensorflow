# KGQA Adapted MINERVA
This is the repository for the paper *Theseus in the Graph: Towards Traceable Multi-Hop Graph Navigation* for the adapted MINERVA model on Knowledge Graph Question Answering (KGQA).

This repository is an anonymized evaluation-only release for the double-blind review process. It contains the code and configs. The preprocessed datasets and pretrained checkpoints are available at [Kaggle](https://www.kaggle.com/models/anonymousexpert/minerva-kgqa).

## Review Notice

This repository and the accompanying datasets are provided solely for the double-blind review process.

Please do **not** redistribute, repost, mirror, or publicly release any part of this code or data. A full public release will be provided after the review process concludes.

## Environment Setup

Use **Python 3.9** and install the required dependencies:

```bash
pip install -r requirements.txt
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

## Evaluation Configs

Evaluation configs for the released models follow the pattern `configs/<dataset>/evaluation.yaml`, where `<dataset>` is a placeholder for the dataset name.

Use the released evaluation config for the dataset you want to evaluate.

For the full experimental setup, we trained three models per dataset and report averaged results across seeds. For the double-blind review release, we provide three checkpoint for each dataset, using seeds `0`, `42`, and `100`. By default, the evaluation configs point to the checkpoint with seed `0`. To evaluate the other checkpoints, update the `seed` field in the config to point to the desired checkpoint.

If your local checkpoint path differs from the default one in a config, update:
1. `load_model: True`
2. `model_load_dir: <path-to-checkpoint>`

## Evaluate

Run evaluation on CPU:

```bash
bash run_eval.sh configs/<dataset>/evaluation.yaml
```

Run evaluation on a specific GPU (not ideal for large data like MetaQA):

```bash
bash run_eval.sh configs/<dataset>/evaluation.yaml 0
```

Evaluation outputs are written by the evaluation pipeline according to the selected config.
