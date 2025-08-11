import os
import argparse

from tqdm import tqdm

from transformers import AutoTokenizer, TFAutoModel
import code.data.utils as data_utils

def get_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    'Datasets & File Paths'
    # KG Dataset
    ap.add_argument('--data_dir', type=str, default="./datasets/data_preprocessed/kinshiphinton", help='Root directory for KG triples and metadata (default: ./data/FB15k)')

    # QA Dataset
    ap.add_argument('--raw_QAData_path', type=str, default="./datasets/data_preprocessed/kinshiphinton/kinship_hinton_qa_1hop.csv", help="Path to the raw QA CSV dataset (default: FreebaseQA)")
    ap.add_argument('--cached_QAMetaData_path', type=str, default="./.cache/itl/kinship_hinton_qa_1hop.json", help="Path to cached tokenized QA metadata JSON file")
    ap.add_argument('--force_data_prepro', '-f', action="store_true", help="Force re-processing of QA data, even if cache exists")

    'Textual Embedding (LLMs)'
    ap.add_argument("--question_tokenizer_name", type=str, default="bert-base-uncased", help="Tokenizer name for question embeddings")
    ap.add_argument("--answer_tokenizer_name", type=str, default="facebook/bart-base", help="Tokenizer name for answer embeddings")

    return ap.parse_args()

if __name__ == "__main__":

    args = get_args()

    # Load the KGE Dictionaries
    ent2id, rel2id, id2ent, id2rel = data_utils.load_dictionary(args.data_dir)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # question_tokenizer = AutoTokenizer.from_pretrained(args.question_tokenizer_name)
    question_encoder = TFAutoModel.from_pretrained(args.question_tokenizer_name, from_pt=False)

    # TODO: Check if it is necessary to convert the entity/relation to ids in the MINERVA algorithm
    train_df, dev_df, test_df, train_metadata = data_utils.load_qa_data(
        cached_metadata_path=args.cached_QAMetaData_path,
        raw_QAData_path=args.raw_QAData_path,
        question_tokenizer_name=args.question_tokenizer_name,
        answer_tokenizer_name=args.answer_tokenizer_name, 
        entity2id=ent2id,
        relation2id=rel2id,
        logger=None,
        force_recompute=args.force_data_prepro,
    )

    batch_size = 32

    print(f"Training on {len(train_df)} samples, with batch size {batch_size}")

    for sample_offset_idx in tqdm(range(0, len(train_df), batch_size), desc="Training Batches", leave=False):
        mini_batch = train_df[sample_offset_idx : sample_offset_idx + batch_size]

        # Deconstruct the batch
        questions = mini_batch["Question"].tolist()
        query_ent = mini_batch["Query-Entity"].tolist()
        query_rel = mini_batch["Query-Relation"].tolist()
        answer_id = mini_batch["Answer-Entity"].tolist()
        hops = mini_batch["Hops"].tolist() if "Hops" in mini_batch.columns else None
        paths = mini_batch["Paths"].tolist() if "Paths" in mini_batch.columns else None

        question_embeddings = data_utils.ids_to_embeddings_tf(
            token_id_batches=questions,
            model=question_encoder,
            pad_id=0,                  # adjust if your PAD differs
            add_special_tokens=True,   # set False if your IDs already include [CLS]/[SEP]
            cls_id=101,
            sep_id=102,
            max_length=128,
        )
        # question_embeddings.shape -> [batch, hidden] (e.g., [32, 768])

        # # For debugging purposes
        # for element_id in range(len(mini_batch)):
        #     questions_txt = question_tokenizer.decode(questions[element_id])
        #     paths_txt = [[id2ent[head], id2rel[rel], id2ent[tail]] for head, rel, tail in paths[element_id]]
        #     print(f"Batch {sample_offset_idx // batch_size + 1} - Questions: {questions_txt}, Query Entity: {id2ent[query_ent[element_id]]}, Query Relation: {id2rel[query_rel[element_id]]}, Answer ID: {id2ent[answer_id[element_id]]}, Hops: {hops[element_id]}, Paths: {paths_txt}")
