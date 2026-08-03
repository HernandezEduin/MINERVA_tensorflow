import argparse
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm

from baseline_path_fidelity import (
    aggregate_per_question,
    answer_set,
    build_grapher,
    compute_path_fidelity,
    default_cache_path,
    effective_reference_length,
    exact_reference_match,
    format_metric,
    load_eval_frames,
    make_episode,
    metric_availability_notes,
    row_id,
    shortest_path_to_answer,
    write_results,
)


METRIC_NAMES = ["PED", "RED", "F1_SG", "F1_Rel"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Answer-oracle shortest-path baseline for MINERVA/THESEUS evaluation.")
    parser.add_argument("--triplet-path", "--data-input-dir", dest="data_input_dir", default="./datasets/nlq/kinshiphinton_v2/",
                        help="Directory containing graph files and vocab/.")
    parser.add_argument("--question-path", default="./datasets/nlq/kinshiphinton_v2/kinship_qa_nhop.csv",
                        help="Raw QA CSV file.")
    parser.add_argument("--cached-qa-metadata-path", default=None,
                        help="Cached QA metadata JSON. Defaults to ./.cache/itl/<question basename>.json.")
    parser.add_argument("--question-tokenizer-name", default="bert-base-uncased")
    parser.add_argument("--force-data-prepro", action="store_true")
    parser.add_argument("--num-rollout-steps", type=int, default=4,
                        help="Preserved for CLI compatibility; shortest paths are evaluated at their actual length.")
    parser.add_argument("--max-hops", type=int, default=None,
                        help="Episode hop budget for evaluator compatibility. Does not cap BFS unless --cap-search-at-hop-budget is set.")
    parser.add_argument("--cap-search-at-hop-budget", action="store_true",
                        help="If set, only search paths up to the configured hop budget.")
    parser.add_argument("--max-num-actions", type=int, default=200,
                        help="Maximum action slots per entity, matching MINERVA grapher truncation.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Evaluate only the first N examples from each selected split.")
    parser.add_argument("--test-only", action="store_true",
                        help="If set, only evaluate test questions.")
    parser.add_argument("--use-self-loops", action="store_true",
                        help="Accepted for compatibility. NO_OP self-loops are ignored by shortest-path search.")
    parser.add_argument("--use-full-graph", action="store_true",
                        help="If set, use full_graph.txt instead of graph.txt.")
    parser.add_argument("--include-inverse-relations", action="store_true",
                        help="If set, allow inverse relation actions. By default the directed evaluator action space is used.")
    parser.add_argument("--use-stop-signal", action="store_true")
    parser.add_argument("--use-restart-signal", action="store_true")
    parser.add_argument("--path-segment-policy", default="final_segment_truncate",
                        choices=["raw", "truncate_at_stop", "final_segment", "final_segment_truncate"])
    parser.add_argument("--output", default="shortcut_oracle_stats.json")
    parser.add_argument("--output-format", choices=["json", "csv"], default="json")
    return parser.parse_args()


def evaluate_split(name: str, df: Any, metadata: Dict[str, Any], grapher, args: argparse.Namespace):
    hop_budget = args.max_hops if args.max_hops is not None else args.num_rollout_steps
    episode = make_episode(
        grapher=grapher,
        df=df,
        metadata=metadata,
        path_length=hop_budget,
        use_stop_signal=args.use_stop_signal,
        use_restart_signal=args.use_restart_signal,
    )
    multi_answers = bool(metadata.get("is_multi_answer", False))
    rows: List[Dict[str, Any]] = []

    for local_idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc=f"Oracle {name}", leave=False)):
        start = int(row["Source-Entity"])
        answers = answer_set(row["Answer-Entity"], multi_answers)
        path = shortest_path_to_answer(
            grapher=grapher,
            start=start,
            answers=answers,
            include_self_loops=False,
        )
        if args.cap_search_at_hop_budget and path is not None and len(path) > hop_budget:
            path = None

        reached = path is not None and ((path[-1][2] if path else start) in answers)
        metrics = compute_path_fidelity(episode, path or [], local_idx, args.path_segment_policy) if path is not None else {
            "PED": None,
            "RED": None,
            "F1_SG": None,
            "F1_Rel": None,
        }
        exact_match = exact_reference_match(episode, path or [], local_idx) if path is not None else None
        reference_len = effective_reference_length(episode, local_idx)
        shortest_len = len(path) if path is not None else None

        per_question: Dict[str, Any] = {
            "split": name,
            "question_id": row_id(row, local_idx),
            "start_entity": start,
            "valid_answer_entities": sorted(answers),
            "reference_path_length": reference_len,
            "predicted_path_length": shortest_len,
            "answer_reached": bool(reached),
            "exact_reference_match": exact_match,
            "reaches_answer_different_path": bool(reached and exact_match is False),
        }
        if reference_len not in (None, 0) and shortest_len is not None:
            per_question["shortest_to_reference_length_ratio"] = float(shortest_len) / float(reference_len)
        else:
            per_question["shortest_to_reference_length_ratio"] = None
        per_question.update(metrics)
        rows.append(per_question)

    return rows


def print_summary(summary: Dict[str, Any], output_path: str) -> None:
    print("\n" + "=" * 88)
    print("Answer-oracle shortest-path path-fidelity baseline")
    print("=" * 88)
    print(f"Evaluated questions:            {summary['num_evaluated_questions']}")
    print(f"Missing-path questions:         {summary['missing_path_count']} ({format_metric(summary['missing_path_rate'])})")
    print(f"Answer success rate:            {format_metric(summary['answer_success_rate'])}")
    print(f"Average shortest-path length:   {format_metric(summary['average_shortest_path_length'])}")
    print(f"Average annotated-path length:  {format_metric(summary['average_annotated_path_length'])}")
    print(f"Average shortest/ref ratio:     {format_metric(summary['average_shortest_to_reference_length_ratio'])}")
    print(f"Exact annotated-path matches:   {summary['exact_match_count']} ({format_metric(summary['exact_match_rate'])})")
    print(f"Correct answer, different path: {summary['different_path_success_count']} ({format_metric(summary['different_path_success_rate'])})")
    print("-" * 88)
    print(f"Average PED:    {format_metric(summary.get('average_PED'))}")
    print(f"Average RED:    {format_metric(summary.get('average_RED'))}")
    print(f"Average F1_SG:  {format_metric(summary.get('average_F1_SG'))}")
    print(f"Average F1_Rel: {format_metric(summary.get('average_F1_Rel'))}")
    if summary.get("metric_availability_notes"):
        print("Metric availability notes:")
        for metric, note in summary["metric_availability_notes"].items():
            print(f"  {metric}: {note}")
    print("-" * 88)
    if output_path:
        print(f"Machine-readable output: {output_path}")


def main() -> None:
    args = parse_args()
    hop_budget = args.max_hops if args.max_hops is not None else args.num_rollout_steps
    frames, metadata, ent2id, rel2id, _, _, _, _ = load_eval_frames(
        data_input_dir=args.data_input_dir,
        question_path=args.question_path,
        cached_qa_metadata_path=args.cached_qa_metadata_path,
        question_tokenizer_name=args.question_tokenizer_name,
        seed=args.seed,
        force_data_prepro=args.force_data_prepro,
        test_only=args.test_only,
    )
    grapher = build_grapher(
        data_input_dir=args.data_input_dir,
        ent2id=ent2id,
        rel2id=rel2id,
        max_num_actions=args.max_num_actions,
        use_full_graph=args.use_full_graph,
        use_directed_graph=not args.include_inverse_relations,
        use_stop_signal=args.use_stop_signal,
        use_restart_signal=args.use_restart_signal,
    )

    per_question: List[Dict[str, Any]] = []
    for split_name, df in frames:
        if args.max_samples is not None:
            df = df.head(args.max_samples)
        per_question.extend(evaluate_split(split_name, df, metadata, grapher, args))

    summary = aggregate_per_question(per_question, METRIC_NAMES)
    n = len(per_question)
    missing = sum(1 for row in per_question if row["predicted_path_length"] is None)
    successes = sum(1 for row in per_question if row["answer_reached"])
    exact_matches = sum(1 for row in per_question if row["exact_reference_match"] is True)
    different_successes = sum(1 for row in per_question if row["reaches_answer_different_path"])

    shortest_lengths = [row["predicted_path_length"] for row in per_question if row["predicted_path_length"] is not None]
    reference_lengths = [row["reference_path_length"] for row in per_question if row["reference_path_length"] is not None]
    ratios = [row["shortest_to_reference_length_ratio"] for row in per_question if row["shortest_to_reference_length_ratio"] is not None]
    summary.update({
        "baseline": "answer_oracle_shortest_path",
        "seed": args.seed,
        "max_samples": args.max_samples,
        "hop_budget": hop_budget,
        "graph": "full_graph.txt" if args.use_full_graph else "graph.txt",
        "use_directed_graph": not args.include_inverse_relations,
        "self_loops_ignored_by_search": True,
        "tie_breaking": "BFS with outgoing actions sorted by (relation_id, target_entity_id)",
        "metric_availability_notes": metric_availability_notes(metadata),
        "answer_success_rate": float(successes / n) if n else None,
        "missing_path_count": missing,
        "missing_path_rate": float(missing / n) if n else None,
        "average_shortest_path_length": float(np.mean(shortest_lengths)) if shortest_lengths else None,
        "average_annotated_path_length": float(np.mean(reference_lengths)) if reference_lengths else None,
        "average_shortest_to_reference_length_ratio": float(np.mean(ratios)) if ratios else None,
        "exact_match_count": exact_matches,
        "exact_match_rate": float(exact_matches / n) if n else None,
        "different_path_success_count": different_successes,
        "different_path_success_rate": float(different_successes / n) if n else None,
    })

    payload = {"summary": summary, "per_question": per_question}
    output_path = write_results(args.output, args.output_format, payload)
    print_summary(summary, output_path)


if __name__ == "__main__":
    main()
