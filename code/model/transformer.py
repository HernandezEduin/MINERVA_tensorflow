import argparse

import tensorflow as tf

from code.data.feed_nlq_data import QuestionBatcher

def get_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    'Datasets & File Paths'
    ap.add_argument('--batch_size', type=int, default=32, help='Batch size for training and evaluation (default: 32)')
    ap.add_argument('--mode', type=str, default='train', choices=['train', 'dev', 'test'], help='Mode for the batcher (default: train)')

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

    batcher = QuestionBatcher(
        input_dir=args.data_dir,
        batch_size=args.batch_size,
        question_tokenizer_name = args.question_tokenizer_name,
        cached_QAMetaData_path = args.cached_QAMetaData_path,
        raw_QAData_path = args.raw_QAData_path,
        force_data_prepro = args.force_data_prepro,
        mode = args.mode
    )

    # Testing Disabling Eager Execution for MINERVA's compatibility
    tf.compat.v1.disable_eager_execution()

    next_batch_func = batcher.yield_next_batch_train if args.mode == 'train' else batcher.yield_next_batch_test
    max_questions = batcher.get_question_num()
    counter = 0
    for data in next_batch_func():
        questions, q_embeddings, source_ent, ans_ent = data
        question_text = batcher.translate_questions(questions)
        ent_names = batcher.translate_entities(source_ent)
        ans_ent_name = batcher.translate_entities(ans_ent)

        for i0 in range(source_ent.shape[0]):
            print(f"Batch Questions: {question_text[i0]}, Source Entity: {ent_names[i0]}, Answer Entity: {ans_ent_name[i0]}")
        
        counter += len(questions)

        if counter >= max_questions:
            break

    batcher.embedding_server.close()