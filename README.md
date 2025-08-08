# MINERVA
Meandering In Networks of Entities to Reach Verisimilar Answers 

Code and models for the paper [Go for a Walk and Arrive at the Answer - Reasoning over Paths in Knowledge Bases using Reinforcement Learning](https://arxiv.org/abs/1711.05851)

MINERVA is a RL agent which answers queries in a knowledge graph of entities and relations. Starting from an entity node, MINERVA learns to navigate the graph conditioned on the input query till it reaches the answer entity. For example, give the query, (Colin Kaepernick, PLAYERHOMESTADIUM, ?), MINERVA takes the path in the knowledge graph below as highlighted. Note: Only the solid edges are observed in the graph, the dashed edges are unobsrved.
![gif](https://github.com/shehzaadzd/MINERVA/blob/master/images/new.gif)
 _gif courtesy of [Bhuvi Gupta](https://www.linkedin.com/in/bhuvigupta/?originalSubdomain=in)_ 



## Requirements
To install the various Python dependencies (including TensorFlow), make sure you are using **Python 3.9**.
```
pip install -r requirements_cpu_tf2.txt
```

To install the gpu, run the following command in your conda environment

```
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 -y
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```


## Key changes in this fork

This fork ([HernandezEduin/MINERVA_tensorflow](https://github.com/HernandezEduin/MINERVA_tensorflow)) updates the original  
[shehzaadzd/MINERVA](https://github.com/shehzaadzd/MINERVA) codebase for modern environments while preserving original functionality.

- **Environment upgrade:** Python 3.9 + `tensorflow-cpu==2.11.1` (last TF release officially supporting Python 3.9).
- **Graph mode preserved:** Added `tf.compat.v1.disable_eager_execution()` so TF 1.x-style training loops still work.
- **API migrations to `tf.compat.v1`:**
  - `tf.Session` → `tf.compat.v1.Session`
  - `tf.placeholder` → `tf.compat.v1.placeholder`
  - `tf.global_variables_initializer` → `tf.compat.v1.global_variables_initializer`
  - `tf.train.Saver` → `tf.compat.v1.train.Saver`
  - `tf.variable_scope` / `tf.get_variable` → `tf.compat.v1.variable_scope` / `tf.compat.v1.get_variable`
  - Optimizer and gradient functions switched to `tf.compat.v1.train.*` and `tf.compat.v1.gradients`.
- **`tf.contrib` replacements:**
  - `tf.contrib.layers.xavier_initializer()` → `tf.keras.initializers.GlorotUniform()`
  - `tf.contrib.rnn.LSTMCell` / `MultiRNNCell` → `tf.compat.v1.nn.rnn_cell.LSTMCell` / `MultiRNNCell`
- **Removed deprecated functions:**
  - `tf.to_int32` → `tf.cast(..., tf.int32)`
  - `tf.multinomial` → `tf.random.categorical`
  - `tf.layers.dense` → `tf.compat.v1.layers.dense`
  - `tf.div` → `tf.math.divide`
- **Other adjustments:**
  - Updated requirements to modern, Py 3.9-compatible versions.
  - Added `requirements_cpu_tf2.txt` for easy setup.
  - Minor directory creation fixes to avoid errors when rerunning.

The original TF 1.3.0 + Python 3.6 requirement has been removed; all code now runs in a modern environment while retaining original behavior.


## Training
Training MINERVA is easy!. The hyperparam configs for each experiments are in the [configs](https://github.com/HernandezEduin/MINERVA_tensorflow/tree/master/configs) directory. To start a particular experiment, just do
```
bash run.sh configs/${dataset}.sh
```
where the `${dataset}.sh` is the name of the config file. For example, 
```
bash run.sh configs/countries_s3.sh
```

## Testing

We are also releasing pre-trained models so that you can directly use MINERVA for query answering. They are located in the  [saved_models](https://github.com/HernandezEduin/MINERVA_tensorflow/tree/master/saved_models) directory. To load the model, set the ```load_model``` to 1 in the config file (default value 0) and ```model_load_dir``` to point to the saved_model. For example in [configs/countries_s2.sh](https://github.com/HernandezEduin/MINERVA_tensorflow/blob/master/configs/countries_s2.sh), make
```
load_model=1
model_load_dir="saved_models/countries_s2/model.ckpt"
```
## Output
The code outputs the evaluation of MINERVA on the datasets provided. The metrics used for evaluation are Hits@{1,3,5,10,20} and MRR (which in the case of Countries is AUC-PR). Along with this, the code also outputs the answers MINERVA reached in a file.

## Code Structure

The structure of the code is as follows
```
Code
├── Model
│    ├── Trainer
│    ├── Agent
│    ├── Environment
│    └── Baseline
├── Data
│    ├── Grapher
│    ├── Batcher
│    └── Data Preprocessing scripts
│            ├── create_vocab
│            ├── create_graph
│            ├── Trainer
│            └── Baseline

```

## Data Format

To run MINERVA on a custom graph based dataset, you would need the graph and the queries as triples in the form of (e<sub>1</sub>,r, e<sub>2</sub>).
Where e<sub>1</sub>, and e<sub>2</sub> are _nodes_ connected by the _edge_ r.
The graph (train only) and the full graph (train, dev, and test) can be created using the `create_graph.py` in `data/data preprocessing scripts`.
The vocab can be created using `create_vocab.py` in the same folder, stored as JSON: `{'entity/relation': ID}`.
The following shows the directory structure of the Kinship dataset.

```
kinship
    ├── graph.txt
    ├── train.txt
    ├── dev.txt
    ├── test.txt
    └── Vocab
            ├── entity_vocab.json
            └── relation_vocab.json
``` 
## Citation
If you use this code, please cite their paper
```
@inproceedings{minerva,
  title = {Go for a Walk and Arrive at the Answer: Reasoning Over Paths in Knowledge Bases using Reinforcement Learning},
  author = {Das, Rajarshi and Dhuliawala, Shehzaad and Zaheer, Manzil and Vilnis, Luke and Durugkar, Ishan and Krishnamurthy, Akshay and Smola, Alex and McCallum, Andrew},
  booktitle = {ICLR},
  year = 2018
}
```
