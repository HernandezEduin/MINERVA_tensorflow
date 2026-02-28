# MINERVA
Meandering In Networks of Entities to Reach Verisimilar Answers 

Enhanced implementation of the MINERVA reinforcement learning framework for knowledge graph reasoning, based on the paper [Go for a Walk and Arrive at the Answer - Reasoning over Paths in Knowledge Bases using Reinforcement Learning](https://arxiv.org/abs/1711.05851).

This repository extends the original MINERVA with modern TensorFlow compatibility, code improvements, and enhanced natural language question answering capabilities.

## Project Overview

MINERVA is a reinforcement learning agent that answers queries in knowledge graphs by learning to navigate from source entities to answer entities.

**Branch note:** This **main branch** focuses on the **MultiHop KGQA (NLQ)** task (not the original **Knowledge Graph Completion / Query** task). For the **Query task (Knowledge Graph Completion)** version of MINERVA, use: [minerva_tf1](https://github.com/HernandezEduin/MINERVA/tree/minerva_tf1)

- **Query-based reasoning (Knowledge Graph Completion)**: Original MINERVA setup for completing missing links. Given a query in the form **(h, r, ?)**, the agent must navigate the graph to reach the correct tail entity **t**—with the **direct edge (h, r, t)** masked on the first hop so it must find an alternative multi-hop route.
- **Natural Language Question Answering (MultiHop KGQA)**: Multi-hop reasoning from natural language. Given a **question**, a **knowledge graph**, and a **start entity**, the agent navigates the graph to arrive at the correct **answer entity**.

![gif](https://github.com/shehzaadzd/MINERVA/blob/master/images/new.gif)
 _gif courtesy of [Bhuvi Gupta](https://www.linkedin.com/in/bhuvigupta/?originalSubdomain=in)_ 



## Requirements
To install the various Python dependencies (including TensorFlow), make sure you are using **Python 3.9**.
```
pip install -r requirements.txt
```

To install the gpu, run the following command in your conda environment

```
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 -y
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

## Training

Train on natural language questions with multi-hop reasoning:
```bash
bash run_nlq.sh configs/nlq/${dataset}.yaml
```

To train with gpu, run:
```bash
bash run_nlq.sh configs/nlq/${dataset}.yaml ${gpu-id}
```

Example configurations for NLQ are in `configs/nlq/`:
```bash
bash run_nlq.sh configs/nlq/mquake.yaml 0
```

## Testing

Pre-trained models are **not** provided. To evaluate, you must first train a model and then load it for testing.

### Loading Pre-trained Models
To load a previously trained model, update your configuration file:
1. Set `load_model=True` 
2. Set `model_load_dir` to point to the saved model (e.g., `saved_models/{your_dataset}/model/model.ckpt`)

### Evaluating 
For NLQ reasoning, train your own models (or plug in your own datasets) and evaluate with:
```bash
bash run_eval.sh configs/nlq/${your_dataset}.yaml
```

## Natural Language Question Answering

The enhanced `nlq/` framework extends MINERVA with sophisticated natural language understanding for multi-hop KGQA.

### Key Features
- **BERT Integration**: Uses pre-trained language models for question encoding
- **Multi-hop Reasoning**: Handles complex questions requiring multiple reasoning steps  
- **Flexible Embeddings**: Supports various question tokenizers and embedding models
- **Enhanced Evaluation**: Comprehensive metrics including Hits@K, MRR, Edit Distance, and Graph Overlap.
- **Beam Search**: Improved inference with beam search decoding
- **Restart Signal**: Lets the agent jump back to the start entity and retry with remaining hops
- **Stop Signal**: Lets the agent stop early at the current entity (before the hop budget is exhausted)
- **Multi-Answer**: Supports multi-answer training/evaluation (a rollout is correct if it reaches any gold answer)
- **Multi-Graph-Type**: Evaluates generalization across different graph variants (e.g., full graph, incomplete/pruned graph, directed vs. undirected).
- **Multi-Question-Format**: Supports multiple question input formats (e.g., full-text, relation-only, graph-only).
- **Projection Adapter**: Supports multiple question→KG embedding adapters for aligning language embeddings to the KG space (e.g., **Linear**, **MLP**, **Residual**).

### Question Processing Pipeline
1. **Tokenization**: Tokenize questions using a configurable tokenizer
2. **Embedding Generation**: Question embeddings are generated via an external embedding server to avoid TensorFlow 1.x compatibility constraints.
3. **Graph Navigation**: Agent navigates knowledge graph conditioned on question embeddings
4. **Answer Retrieval**: Multi-rollout exploration with beam search for robust answers

### Example Usage
```bash
# Train NLQ model on mquake dataset
bash run_nlq.sh configs/nlq/mquake.yaml

# The framework handles questions like:
# "Who is the father of John's brother?"
# "What team does the player from Boston play for?"
```

## Output

The framework prints and saves a rich set of evaluation artifacts for both **answer accuracy** and **reasoning behavior**.

### 1) Answer Metrics (Ranking Quality)
Standard KGQA ranking metrics computed over the candidate answer set:
- **Hits@{1,3,5,10,20}**: For each question, we generate a fixed number of rollouts (e.g., **100** at test time). Each rollout induces a candidate endpoint answer and is **ranked by the path log-probability** (higher log-prob = more likely under the policy). Hits@K reports the fraction of questions where a gold answer appears among the **top-K rollouts**.
- **MRR (Mean Reciprocal Rank)**: For each question, compute the rank (by path log-probability) of the first rollout that reaches a gold answer, then average `1 / rank` across questions (higher is better).

### 2) Reasoning Diagnostics (Top-Rollout)
Behavioral statistics computed from the **top-scoring rollout/path per question** (useful for debugging navigation policies):
- **Special Step Rate**: Fraction of steps that are *special actions* (e.g., **STOP**, **RESTART**, **NO-OP**), depending on the configured action space.
- **Restart Rate**: How often the agent triggers **RESTART** (jumps back to the start entity and continues with remaining hops).
- **No-Op Rate**: How often the agent takes **NO-OP** (remains at the current entity).
- **Cycle Rate**: Frequency of revisiting nodes/edges (looping behavior).
- **Backtrack Rate**: How often the agent backtracks (more relevant when using an **undirected** graph).
- **Unique Edges**: Average number of distinct edges traversed per episode (higher ⇒ more exploration).
- **Redundancy**: Fraction of traversed edges that are repeats (lower ⇒ less wasted motion).
- **Avg Segment Hops**: Average hops per *effective segment* after truncation and cleanup: the path is truncated between the **last RESTART** and the **first STOP** (if present), and **NO-OP** steps are removed.

### 3) Endpoint Coverage (Multi-Answer Support)
For datasets/questions with **multiple valid answers**, we measure whether the *set of endpoints produced across rollouts* actually covers the *set of gold answers* (as opposed to repeatedly predicting the same few answers):
- **Recall / Precision / F1**: Computed between the set of **unique predicted endpoints** (union over all rollouts for a question) and the set of **gold answer entities**.  
  This reveals whether the policy **covers all valid answers** or **collapses** onto a small subset across its rollouts.

### 4) Faithfulness to Ground-Truth Reasoning (When GT Paths Exist)
If your dataset provides reference reasoning paths, the framework reports faithfulness metrics that compare predicted reasoning against ground-truth (GT) paths.

**Path preprocessing (used for all faithfulness metrics):** We evaluate on a **truncated, cleaned** version of the predicted path: special actions (e.g., **STOP/RESTART/NO-OP**) are removed, and any undirected traversals are converted back into **directed** KG triplets before comparison.

- **GT-Edge Overlap (Recall/Precision/F1)**: Treats each path as a **set of directed triplets** (i.e., a small subgraph) and computes overlap with the GT triplet set. This makes the comparison **permutation-invariant** and supports **subgraph-style** evaluation (edge-set match rather than strict step order).
- **Node-Set Overlap (Recall/Precision/F1)**: Overlap between the **sets of visited entities** in the predicted vs. GT path (order-invariant).
- **Relation-Set Overlap (Recall/Precision/F1)**: Overlap between the **sets of relations** used in the predicted vs. GT path (order-invariant).
- **Path Edit Distance (Normalized)**: For *sequence-level* evaluation, we compute the **normalized path edit distance** between the cleaned predicted path and the GT path (lower is better), capturing ordering and structural deviations that edge-set metrics ignore. 

### 5) Path Logs (Qualitative Debugging)
Optional per-question traces are written for inspection (see `test_path.txt`-style output). Each entry includes:
- **Question / Start Entity / Gold Answer / Predicted Answer**
- **Gold Path vs. Predicted Path** (human-readable), plus the **raw rollout trajectory** with special actions explicitly shown (e.g., **NO-OP**, **RESTART**, **STOP**)
- **Per-example diagnostics** including **Path F1 (↑)**, **Normalized Edit Distance (↓)**, **Negative Log-Probability (↓)**, **Gold vs. Agent hop counts**, and whether the example is solved (**Hit@1**)

## Code Structure

The codebase is organized into two main reasoning frameworks:

```
Code/
├── Model/
│   │── trainer.py               # Enhanced training pipeline with NLQ support
│   │── agent.py                 # Enhanced agent with question understanding
│   │── environment.py           # Environment with question processing
│   └── baseline.py              # Shared baseline estimator
├── Data/
│   ├── grapher.py               # Knowledge graph navigation
│   ├── feed_data.py             # Data batching for NL questions
│   ├── embedding_server.py      # Question embedding generation
│   └── preprocessing_scripts/   # Data preprocessing utilities
└── Configs/
    └── nlq/                     # Configurations for NL questions
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

### Natural Language Question Answering Dataset Format

For the enhanced NLQ framework, you'll need additional data files to support natural language question processing:

#### Required Files for NLQ
1. **Knowledge Graph Files** (same as query-based format):
   - `graph.txt`: Knowledge graph triples in format (entity1, relation, entity2)
   - `vocab/entity_vocab.json`: Entity name to ID mapping
   - `vocab/relation_vocab.json`: Relation name to ID mapping

2. **Question-Answer Dataset** (CSV format):
   A CSV file containing the following columns:
   - `Question`: Natural language question text (e.g., "Who is the father of John's brother?")
   - `Source-Entity`: Starting entity for graph navigation (entity name, not ID)
   - `Answer-Entity`: Target answer entity (entity name, not ID)

#### Example NLQ Dataset Structure
```
dataset_nlq/
├── graph.txt                    # Knowledge graph triples
├── questions.csv                # Natural language questions
└── vocab/
    ├── entity_vocab.json        # Entity vocabulary
    └── relation_vocab.json      # Relation vocabulary
```

#### Sample CSV Content
```csv
Question,Source-Entity,Answer-Entity
"Who is the father of Mary's brother?",Mary,John_Smith
"What team does the Boston player play for?",Boston_Celtics,NBA
"Where was the president of France born?",Emmanuel_Macron,Amiens
```

#### Data Processing Pipeline
1. **Question Tokenization**: Questions are processed using configurable tokenizers (BERT, etc.)
2. **Entity Mapping**: Query and answer entities are mapped to graph node IDs
3. **Embedding Generation**: Question embeddings are generated via the embedding server
4. **Graph Alignment**: Entities are verified to exist in the knowledge graph
5. **Batch Creation**: Questions are grouped into training/evaluation batches

#### Configuration Parameters
Key NLQ-specific parameters in config files:
- `question_tokenizer_name`: Tokenizer model (e.g., "bert-base-uncased")
- `cached_QAMetaData_path`: Path to preprocessed question metadata
- `raw_QAData_path`: Path to raw CSV question data

## Citation

If you use this code, please cite the original MINERVA paper:

```bibtex
@inproceedings{minerva,
  title = {Go for a Walk and Arrive at the Answer: Reasoning Over Paths in Knowledge Bases using Reinforcement Learning},
  author = {Das, Rajarshi and Dhuliawala, Shehzaad and Zaheer, Manzil and Vilnis, Luke and Durugkar, Ishan and Krishnamurthy, Akshay and Smola, Alex and McCallum, Andrew},
  booktitle = {ICLR},
  year = 2018
}
```

If you use the enhanced implementations or natural language question answering capabilities, please also consider citing this repository:

```bibtex
<!-- TODO: Add citation for upcoming paper on enhanced MINERVA implementation
@inproceedings{minerva_enhanced,
  title = {[Paper title to be determined]},
  author = {Hernandez, Eduin and Garcia, Luis, and Askar, Nurassyl, and Rini, Stefano},
  booktitle = {[Conference/Journal to be determined]},
  year = {[Year to be determined]},
  url = {https://github.com/HalcyonSolutions/MINERVA}
}
-->
```

For now, you can reference this repository directly:
- Repository: [HalcyonSolutions/MINERVA](https://github.com/HalcyonSolutions/MINERVA)
- Enhanced TensorFlow implementation with modern compatibility and natural language question answering support
