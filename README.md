# KGQA Adapted MINERVA
Meandering In Networks of Entities to Reach Verisimilar Answers *for Knowledge Graph Question Answering*

This repository is an anonymized evaluation-only release for the double-blind review process. It contains the code, configs, preprocessed data, and pretrained checkpoints needed to evaluate the provided models.

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

## Evaluation Configs

Evaluation configs for the released models follow the pattern `configs/<dataset>/evaluation.yaml`, where `<dataset>` is a placeholder for the dataset name.

Use the released evaluation config for the dataset you want to evaluate.

For the full experimental setup, we trained three models per dataset and report averaged results across seeds. For the double-blind review release, we provide only the `seed0` checkpoint for each dataset.

If your local checkpoint path differs from the default one in a config, update:
1. `load_model: True`
2. `model_load_dir: <path-to-checkpoint>`

## Evaluate

Run evaluation on CPU:

```bash
bash run_eval.sh configs/<dataset>/evaluation.yaml
```

Run evaluation on a specific GPU:

```bash
bash run_eval.sh configs/<dataset>/evaluation.yaml 0
```

Evaluation outputs are written by the evaluation pipeline according to the selected config.
