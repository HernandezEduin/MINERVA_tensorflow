# MINERVA
Meandering In Networks of Entities to Reach Verisimilar Answers 

Enhanced implementation of the MINERVA reinforcement learning framework for knowledge graph reasoning, based on the paper [Go for a Walk and Arrive at the Answer - Reasoning over Paths in Knowledge Bases using Reinforcement Learning](https://arxiv.org/abs/1711.05851).

This repository extends the original MINERVA with modern TensorFlow compatibility, code improvements, and enhanced natural language question answering capabilities.

## Project Overview

MINERVA is a reinforcement learning agent that answers queries in knowledge graphs by learning to navigate from source entities to answer entities. This implementation provides two distinct reasoning frameworks:

- **Query-based reasoning** (`query/`): Original MINERVA implementation for structured query answering
- **Natural Language Question answering** (`nlq/`): Enhanced framework for multi-hop reasoning with natural language questions

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


## Key Improvements in This Implementation

This enhanced implementation ([HalcyonSolutions/MINERVA](https://github.com/HalcyonSolutions/MINERVA)) significantly improves upon the original [shehzaadzd/MINERVA](https://github.com/shehzaadzd/MINERVA) codebase with modern compatibility and code quality enhancements.

### Environment & Compatibility Upgrades
- **Modern Python support:** Upgraded to Python 3.9 + `tensorflow-cpu==2.11.1` 
- **TensorFlow 2.x compatibility:** Preserved graph mode execution with `tf.compat.v1.disable_eager_execution()`
- **API modernization:** Migrated all TensorFlow APIs to `tf.compat.v1` namespace for future compatibility
- **Dependency updates:** Replaced deprecated `tf.contrib` modules with modern equivalents

### Code Quality & Architecture Improvements  
- **Parameter clarity:** Replaced dictionary-based parameter passing with explicit individual variables
- **Enhanced documentation:** Comprehensive docstrings with type hints for better maintainability
- **Improved error handling:** Fixed shape mismatch issues and training/testing mode coordination
- **Modular design:** Clean separation between query and natural language question frameworks

### Framework Extensions
- **Dual reasoning modes:** 
  - `query/`: Original structured query reasoning (preserved from authors)
  - `nlq/`: Enhanced natural language question answering with multi-hop reasoning
- **Flexible execution:** Separate run scripts (`run_query.sh`, `run_nlq.sh`) for different reasoning tasks
- **Enhanced evaluation:** Comprehensive metrics and path visualization for both frameworks


## Training

This implementation supports two distinct reasoning frameworks:

### Query-based Reasoning (Original MINERVA)
Train on structured queries with the original MINERVA framework:
```bash
bash run_query.sh configs/query/${dataset}.sh
```

Example configurations are available in `configs/query/`:
```bash
bash run_query.sh configs/query/countries_s3.sh
bash run_query.sh configs/query/kinship.sh
bash run_query.sh configs/query/fb15k-237.sh
```

### Natural Language Question Answering 
Train on natural language questions with multi-hop reasoning:
```bash
bash run_nlq.sh configs/nlq/${dataset}.sh
```

Example configurations for NLQ are in `configs/nlq/`:
```bash
bash run_nlq.sh configs/nlq/kinshiphinton.sh
```

### Configuration Structure
- `configs/query/`: Original MINERVA configurations for structured query reasoning
- `configs/nlq/`: Enhanced configurations for natural language question answering
- Each config file contains hyperparameters optimized for specific datasets and reasoning tasks

## Testing

Pre-trained models are available in the [saved_models](https://github.com/HalcyonSolutions/MINERVA/tree/master/saved_models) directory for immediate evaluation.

### Loading Pre-trained Models
To use a pre-trained model, modify the configuration file:
1. Set `load_model=1` 
2. Set `model_load_dir` to point to the saved model

#### Query-based Models
For structured query reasoning, use models in `saved_models/` with query configs:
```bash
# Example: configs/query/countries_s2.sh
load_model=1
model_load_dir="saved_models/countries_s2/model.ckpt"
```

#### Natural Language Question Models  
For NLQ reasoning, train your own models or use the framework with your datasets:
```bash
bash run_nlq.sh configs/nlq/your_dataset.sh
```

## Natural Language Question Answering

The enhanced `nlq/` framework extends MINERVA with sophisticated natural language understanding:

### Key Features
- **BERT Integration**: Uses pre-trained language models for question encoding
- **Multi-hop Reasoning**: Handles complex questions requiring multiple reasoning steps  
- **Flexible Embeddings**: Supports various question tokenizers and embedding models
- **Enhanced Evaluation**: Comprehensive metrics including Hits@K and MRR
- **Beam Search**: Improved inference with beam search decoding

### Question Processing Pipeline
1. **Tokenization**: Natural language questions are tokenized using configurable models
2. **Embedding Generation**: Question embeddings created via embedding server
3. **Graph Navigation**: Agent navigates knowledge graph conditioned on question embeddings
4. **Answer Retrieval**: Multi-rollout exploration with beam search for robust answers

### Example Usage
```bash
# Train NLQ model on kinshiphinton dataset
bash run_nlq.sh configs/nlq/kinshiphinton.sh

# The framework handles questions like:
# "Who is the father of John's brother?"
# "What team does the player from Boston play for?"
```

## Output
The framework outputs comprehensive evaluation metrics:
- **Hits@{1,3,5,10,20}**: Answer ranking accuracy at different cutoffs
- **MRR (Mean Reciprocal Rank)**: Average reciprocal rank of correct answers
- **Path Visualization**: Detailed reasoning trajectories for analysis
- **Answer Files**: Complete answer sets with confidence scores

For Countries dataset, MRR corresponds to AUC-PR (Area Under Precision-Recall curve).

## Performance (ICASSP)

### Kinship
#### Summary Table (Hits@1)

| Model / QA Hop Size          | Graph-Type | 1-Hop  | 2-Hop  | 3-Hop  | n-Hop  |
| ---------------------------- | ---------- | ------ | ------ | ------ | ------ |
| RW-End                       | Full       | 1.72e-1 | 8.19e-2 | 7.41e-2 | 8.13e-2 |
| RW-End                       | Train      | 8.11e-2 | 7.46e-2 | 7.09e-2 | 7.43e-2 |
| RW-Gold                      | Full       | 8.59e-2 | 7.11e-3 | 6.04e-4 | 3.55e-3 |
| MINERVA ($d_{KG}$=12)        | Full       | 9.38e-1 | 9.84e-1 | 7.55e-1 | 5.20e-1 |
| MINERVA ($d_{KG}$=12)        | Train      | 9.38e-2 | 7.58e-1 | 6.98e-1 | 6.30e-1 |

We report **Hits@1** under two graph regimes:

* **Complete KG** (train+valid+test triples as edges)
* **Incomplete KG** (train-only edges)

All hyperparameters are shared across hop buckets; only the hop budget **H** varies. For mixed-hop buckets (e.g., 2–4), H is fixed to the bucket maximum.

---

### Test Results for MINERVA ($d_{KG}$=12) on Full Graph

| QA & Reasoning | MRR     | Hits@1  | Hits@3  | Hits@5  | Hits@10 |
| -------------- | ------- | ------- | ------- | ------- | ------- |
| 1-Hop          | 9.69e-1 | 9.38e-1 | 1.00    | 1.00    | 1.00    |
| 2-Hop          | 9.92e-1 | 9.84e-1 | 1.00    | 1.00    | 1.00    |
| 3-Hop          | 8.77e-1 | 7.55e-1 | 1.00    | 1.00    | 1.00    |
| n-Hop          | 5.79e-1 | 5.20e-1 | 5.90e-1 | 6.70e-1 | 7.60e-1 |

#### Example Predicted Paths
```
1 Hop Question:
who is the husband of margaret?
KG Start : Margaret
KG GT Ans: Arthur
Agent Ans: Arthur
Predicted Path: Margaret --[_wife]--> Arthur
==============
2 Hop Question:
who is the husband of the mother of arthur?
KG Start : Arthur
KG GT Ans: Christopher
Agent Ans: Christopher
Predicted Path: Arthur --[father]--> Christopher --[NO_OP]--> Christopher
==============
3 Hop Question:
who is the mother of the brother of the wife of charles?
KG Start : Charles
KG GT Ans: Christine
Agent Ans: Christine
Predicted Path: Charles --[NO_OP]--> Charles --[wife]--> Jennifer --[mother]--> Christine
```
---

### Test Results for MINERVA ($d_{KG}$=12) on Train Graph

| QA & Reasoning | MRR     | Hits@1  | Hits@3  | Hits@5  | Hits@10 |
| -------------- | ------- | ------- | ------- | ------- | ------- |
| 1-Hop          | 2.86e-1 | 9.38e-2 | 3.44e-1 | 5.63e-1 | 6.88e-1 |
| 2-Hop          | 8.11e-1 | 7.58e-1 | 8.39e-1 | 8.87e-1 | 9.52e-1 |
| 3-Hop          | 8.23e-1 | 6.98e-1 | 9.62e-1 | 1.00    | 1.00    |
| n-Hop          | 7.34e-1 | 6.30e-1 | 7.70e-1 | 8.80e-1 | 9.80e-1 |

#### Example Predicted Paths
```
1 Hop Question:
who is the husband of margaret?
KG Start : Margaret
KG GT Ans: Arthur
Agent Ans: Colin
Predicted Path: Margaret --[nephew]--> Colin
==============
2 Hop Question:
who is the husband of the mother of arthur?
KG Start : Arthur
KG GT Ans: Christopher
Agent Ans: Christopher
Predicted Path: Arthur --[mother]--> Penelope --[husband]--> Christopher
==============
3 Hop Question:
who is the mother of the brother of the wife of charles?
KG Start : Charles
KG GT Ans: Christine
Agent Ans: Christine
Predicted Path: Charles --[NO_OP]--> Charles --[wife]--> Jennifer --[mother]--> Christine
```

### MQuAKE-ST
#### Summary Table (Hits@1)

| Model / QA Hop Size        | Graph-Type | 2-Hop   | 3-Hop   | 4-Hop   | n-Hop   |
| -------------------------- | ---------- | ------- | ------- | ------- | ------- |
| RW-End                     | Full       | 8.53e-3 | 3.61e-3 | 1.55e-3 | 3.39e-3 |
| RW-End                     | Train      | 8.53e-3 | 3.61e-3 | 1.56e-3 | 3.40e-3 |
| RW-Gold                    | Full       | 1.55e-3 | 6.69e-6 | 5.00e-8 | 3.26e-5 |
| **MINERVA ($d_{KG}$=100)** | Full       | 7.16e-1 | 3.62e-1 | 9.06e-1 | 8.16e-1 |
| **MINERVA ($d_{KG}$=100)** | Train      | 8.63e-1 | 8.54e-1 | 9.61e-1 | 8.65e-1 |
| MultiHopKG ($d_{KG}$=100)  | Full       | 4.33e-1 | 4.82e-1 | 3.55e-1 | 2.96e-1 |
| MultiHopKG ($d_{KG}$=100)  | Train      | 3.72e-1 | 3.51e-1 | 3.26e-1 | 3.47e-1 |
| **SQUIRE ($d_{KG}$=100)**  | Full       | 8.38e-1 | 8.68e-1 | 6.64e-1 | 8.35e-1 |
| **SQUIRE ($d_{KG}$=100)**  | Train      | 4.99e-1 | 8.15e-1 | 6.74e-1 | 6.75e-1 |

---

### Test Results for MINERVA ($d_{KG}$=100) on Full Graph

| QA & Reasoning | Hits@1  | Hits@3  | Hits@5  | Hits@10     | Hits@20 | MRR         |
| -------------- | ------- | ------- | ------- | ----------- | ------- | ----------- |
| **2-Hop**      | 7.16e-1 | 8.12e-1 | 8.34e-1 | 8.47e-1     | 8.57e-1 | 7.68e-1     |
| **3-Hop**      | 3.62e-1 | 3.78e-1 | 3.90e-1 | 4.06e-1     | 4.09e-1 | 3.75e-1     |
| **4-Hop**      | 9.06e-1 | 9.39e-1 | 9.50e-1 | 9.67e-1     | 9.67e-1 | 9.26e-1     |
| **n-Hop**      | 8.16e-1 | 8.85e-1 | 9.03e-1 | 9.12e-1     | 9.16e-1 | 8.53e-1     |

#### Example Predicted Paths
```
2-Hop:
Question: what was the genre of the book authored by x1 known as "the early asimov"?
KG Start : The Early Asimov
KG GT Ans: science fiction
Agent Ans: science fiction
Path: The Early Asimov --[author]--> Isaac Asimov --[genre]--> science fiction
==============

3-Hop:
Question: what is the birthplace of the founder of the religion associated with maria christina of austria?
KG Start : Maria Christina of Austria
KG GT Ans: Bethlehem
Agent Ans: Catherine Henriette de Bourbon
Path: Maria Christina of Austria --[given name]--> Henriette --[No Operation]--> Henriette --[(INV) given name]--> Catherine Henriette de Bourbon
==============

4-Hop / n-Hop example:
Question: on which continent was the founder of the company that developed windows live messenger born?
KG Start : Windows Live Messenger
KG GT Ans: Asia
Agent Ans: Asia
Path: Windows Live Messenger --[developer]--> Microsoft --[board member]--> Satya Nadella --[country of citizenship]--> India --[continent]--> Asia
```

### Test Results for MINERVA ($d_{KG}$=100) on Train Graph

| QA & Reasoning | Hits@1  | Hits@3  | Hits@5  | Hits@10     | Hits@20     | MRR     |
| -------------- | ------- | ------- | ------- | ----------- | ----------- | ------- |
| **2-Hop**      | 8.63e-1 | 9.49e-1 | 9.63e-1 | 9.73e-1     | 9.82e-1     | 9.08e-1 |
| **3-Hop**      | 8.54e-1 | 8.98e-1 | 9.17e-1 | 9.53e-1     | 9.61e-1     | 8.85e-1 |
| **4-Hop**      | 9.61e-1 | 9.72e-1 | 9.78e-1 | 9.83e-1     | 9.83e-1     | 9.68e-1 |
| **n-Hop**      | 8.65e-1 | 9.28e-1 | 9.41e-1 | 9.59e-1     | 9.67e-1     | 9.00e-1 |

#### Example Predicted Paths
```
2-Hop:
Question: what was the genre of the book authored by x1 known as "the early asimov"?
KG Start : The Early Asimov
KG GT Ans: science fiction
Agent Ans: science fiction
Path: The Early Asimov --[author]--> Isaac Asimov --[genre]--> science fiction
==============

3-Hop:
Question: what is the birthplace of the founder of the religion associated with maria christina of austria?
KG Start : Maria Christina of Austria
KG GT Ans: Bethlehem
Agent Ans: Bethlehem
Path: Maria Christina of Austria --[religion or worldview]--> Catholic Church --[founded by]--> Jesus Christ --[place of birth]--> Bethlehem
==============

4-Hop / n-Hop example:
Question: on which continent was the founder of the company that developed windows live messenger born?
KG Start : Windows Live Messenger
KG GT Ans: Asia
Agent Ans: Asia
Path: Windows Live Messenger --[developer]--> Microsoft --[board member]--> Satya Nadella --[country of citizenship]--> India --[continent]--> Asia
```

## Code Structure

The codebase is organized into two main reasoning frameworks:

```
Code/
├── Model/
│   ├── query/                    # Original MINERVA (from authors)
│   │   ├── trainer.py           # Training pipeline for structured queries
│   │   ├── agent.py             # RL agent for query reasoning
│   │   └── environment.py       # Knowledge graph environment
│   ├── nlq/                     # Natural Language Question answering
│   │   ├── trainer.py           # Enhanced training pipeline with NLQ support
│   │   ├── agent.py             # Enhanced agent with question understanding
│   │   └── environment.py       # Environment with question processing
│   └── baseline.py              # Shared baseline estimator
├── Data/
│   ├── grapher.py               # Knowledge graph navigation
│   ├── feed_data.py             # Data batching for queries
│   ├── feed_nlq_data.py         # Data batching for NL questions
│   ├── embedding_server.py      # Question embedding generation
│   └── preprocessing_scripts/   # Data preprocessing utilities
└── Configs/
    ├── query/                   # Configurations for structured queries
    └── nlq/                     # Configurations for NL questions
```

### Key Components
- **query/**: Preserved original MINERVA implementation for structured query reasoning
- **nlq/**: Enhanced framework supporting natural language questions with BERT embeddings
- **Shared components**: Baseline estimator, knowledge graph navigator, and utilities
- **Dual configuration**: Separate configs for each reasoning mode

## Technical Improvements

### Code Quality Enhancements
- **Type Safety**: Comprehensive type hints throughout the codebase
- **Parameter Clarity**: Replaced dictionary-based parameter passing with explicit variables
- **Documentation**: Detailed docstrings for all classes and methods
- **Error Handling**: Improved error messages and validation

### Performance & Stability
- **Shape Consistency**: Fixed TensorFlow shape mismatch issues between training and testing
- **Memory Management**: Optimized episode processing and garbage collection
- **Training Stability**: Enhanced baseline variance reduction and gradient clipping
- **Mode Coordination**: Proper environment mode switching between train/dev/test

### Modern TensorFlow Integration
- **TF 2.x Compatibility**: Full migration to `tf.compat.v1` namespace
- **Deprecated API Replacement**: Updated all deprecated TensorFlow functions
- **Graph Mode Preservation**: Maintained original training dynamics while enabling modern deployment

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
