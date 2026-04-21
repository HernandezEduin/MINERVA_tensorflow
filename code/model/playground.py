import argparse

import tensorflow as tf

from code.data.feed_data import QuestionBatcher

import sys

def get_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    'Datasets & File Paths'
    ap.add_argument('--batch_size', type=int, default=32, help='Batch size for training and evaluation (default: 32)')
    ap.add_argument('--mode', type=str, default='train', choices=['train', 'dev', 'test'], help='Mode for the batcher (default: train)')

    # KG Dataset
    ap.add_argument('--data_dir', type=str, default="./datasets/nlq/kinshiphinton_v2", help='Root directory for KG triples and metadata (default: ./data/FB15k)')

    # QA Dataset
    ap.add_argument('--question_format', type=str, default="full_text", choices=['full_text', 'paraphrased', 'relation_only', 'graph_only'], help="Format of the question input")
    ap.add_argument("--evaluate_paraphrases", action='store_true',
                        help="Whether to evaluate on paraphrased questions instead of original text.")
    # ap.add_argument('--raw_QAData_path', type=str, default="./datasets/nlq/mquake_st/mquake_sa_qa_nhop.csv", help="Path to the raw QA CSV dataset (default: FreebaseQA)")
    # ap.add_argument('--cached_QAMetaData_path', type=str, default="./.cache/itl/mquake_sa_qa_nhop.json", help="Path to cached tokenized QA metadata JSON file")
    ap.add_argument('--raw_QAData_path', type=str, default="./datasets/nlq/kinshiphinton_v2/kinship_qa_nhop.csv", help="Path to the raw QA CSV dataset (default: FreebaseQA)")
    ap.add_argument('--cached_QAMetaData_path', type=str, default="./.cache/itl/kinship_qa_nhop.json", help="Path to cached tokenized QA metadata JSON file")
    ap.add_argument('--force_data_prepro', '-f', action="store_true", help="Force re-processing of QA data, even if cache exists")
    ap.add_argument('--use_weighted_hop_sampling', action='store_true', help="Whether to use weighted hop-based sampling for training batches")

    'Textual Embedding (LLMs)'
    ap.add_argument("--question_tokenizer_name", type=str, default="bert-base-uncased", help="Tokenizer name for question embeddings")
    ap.add_argument("--answer_tokenizer_name", type=str, default="facebook/bart-base", help="Tokenizer name for answer embeddings")

    ap.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')

    return ap.parse_args()

if __name__ == "__main__":

    args = get_args()

    batcher = QuestionBatcher(
        input_dir=args.data_dir,
        batch_size=args.batch_size,
        test_batch_size=args.batch_size,
        question_tokenizer_name = args.question_tokenizer_name,
        question_format=args.question_format,
        use_weighted_hop_sampling = args.use_weighted_hop_sampling,
        evaluate_paraphrases=args.evaluate_paraphrases,
        cached_QAMetaData_path = args.cached_QAMetaData_path,
        raw_QAData_path = args.raw_QAData_path,
        force_data_prepro = args.force_data_prepro,
        mode = args.mode,
    )

    # Testing Disabling Eager Execution for MINERVA's compatibility
    tf.compat.v1.disable_eager_execution()

    next_batch_func = batcher.yield_next_batch_train if args.mode == 'train' else batcher.yield_next_batch_test
    max_questions = batcher.get_question_num()
    is_multi_answer = batcher.has_multi_answers()
    has_paths = batcher.has_paths()
    has_path_keys = batcher.has_path_keys()

    counter = 0
    for data in next_batch_func():
        questions, q_embeddings, source_ent, ans_ent, paths, path_keys, hops, ques_ids = data

        question_text = batcher.translate_questions(questions)
        ent_names = batcher.translate_entities(source_ent)
        ans_ent_name = batcher.translate_entities(ans_ent, dynamic_list=is_multi_answer)

        for i0 in range(source_ent.shape[0]):
            hop_info = f" ({hops[i0]}-hop)"

            print(f"Batch Questions (ID {ques_ids[i0]}){hop_info}: {question_text[i0]}, Source Entity: {ent_names[i0]}, Answer Entity: {ans_ent_name[i0]}")

        break

        counter += len(questions)

        if counter >= max_questions:
            break

    batcher.embedding_server.close()