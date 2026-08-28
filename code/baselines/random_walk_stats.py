import argparse
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm

from code.baselines.baseline_path_fidelity import (
    aggregate_per_question,
    answer_set,
    build_grapher,
    cleaned_path_and_relations,
    compute_path_fidelity,
    default_cache_path,
    effective_reference_length,
    format_metric,
    load_eval_frames,
    make_episode,
    metric_availability_notes,
    row_id,
    valid_actions,
    write_results,
)


METRIC_NAMES = ["PED", "RED", "F1_SG", "F1_Rel"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Random-walk path-fidelity baseline for MINERVA/THESEUS evaluation.")
    parser.add_argument("--triplet-path", "--data-input-dir", dest="data_input_dir", default="./datasets/nlq/kinshiphinton_v2/",
                        help="Directory containing graph files and vocab/.")
    parser.add_argument("--question-path", default="./datasets/nlq/kinshiphinton_v2/kinship_qa_nhop.csv",
                        help="Raw QA CSV file.")
    parser.add_argument("--cached-qa-metadata-path", default=None,
                        help="Cached QA metadata JSON. Defaults to ./.cache/itl/<question basename>.json.")
    parser.add_argument("--question-tokenizer-name", default="bert-base-uncased")
    parser.add_argument("--force-data-prepro", action="store_true")
    parser.add_argument("--num-rollout-steps", type=int, default=4,
                        help="Fixed hop budget. Preserved legacy name.")
    parser.add_argument("--max-hops", type=int, default=None,
                        help="Alias for the fixed hop budget. Overrides --num-rollout-steps when set.")
    parser.add_argument("--max-num-actions", type=int, default=200,
                        help="Maximum action slots per entity, matching MINERVA grapher truncation.")
    parser.add_argument("--num-walks", type=int, default=20,
                        help="Monte Carlo random walks per question.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Evaluate only the first N examples from each selected split.")
    parser.add_argument("--test-only", action="store_true",
                        help="If set, only evaluate test questions.")
    parser.add_argument("--use-self-loops", action="store_true",
                        help="If set, allow NO_OP self-loop actions during random-walk generation.")
    parser.add_argument("--use-full-graph", action="store_true",
                        help="If set, use full_graph.txt instead of graph.txt.")
    parser.add_argument("--include-inverse-relations", action="store_true",
                        help="If set, allow inverse relation actions. By default the directed evaluator action space is used.")
    parser.add_argument("--use-stop-signal", action="store_true")
    parser.add_argument("--use-restart-signal", action="store_true")
    parser.add_argument("--path-segment-policy", default="final_segment_truncate",
                        choices=["raw", "truncate_at_stop", "final_segment", "final_segment_truncate"])
    parser.add_argument("--use-ideal-paths", action="store_true",
                        help="Preserve legacy RW-Path reporting: sampled exact annotated-path hit rate when entity paths are available.")
    parser.add_argument("--output", default="random_walk_stats.json")
    parser.add_argument("--output-format", choices=["json", "csv"], default="json")
    return parser.parse_args()


def sample_walk(grapher, start_entity: int, hop_budget: int, rng: np.random.Generator, include_self_loops: bool) -> List[tuple]:
    current = int(start_entity)
    path: List[tuple] = []
    for _ in range(hop_budget):
        actions = valid_actions(grapher, current, include_self_loops=include_self_loops, include_stop_restart=False)
        if not actions:
            if include_self_loops:
                path.append((current, int(grapher.rNO_OP), current))
            break
        relation, target = actions[int(rng.integers(0, len(actions)))]
        path.append((current, int(relation), int(target)))
        current = int(target)
    return path


def metric_means(metrics: List[Dict[str, Optional[float]]]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for name in METRIC_NAMES:
        vals = [m[name] for m in metrics if m.get(name) is not None]
        out[name] = float(np.mean(vals)) if vals else None
    return out


def evaluate_split(name: str, df: Any, metadata: Dict[str, Any], grapher, args: argparse.Namespace, rng: np.random.Generator):
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

    for local_idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc=f"Random walk {name}", leave=False)):
        start = int(row["Source-Entity"])
        answers = answer_set(row["Answer-Entity"], multi_answers)
        walk_metrics: List[Dict[str, Optional[float]]] = []
        answer_hits: List[float] = []
        exact_path_hits: List[float] = []
        predicted_lengths: List[int] = []

        for _ in range(args.num_walks):
            raw_path = sample_walk(grapher, start, hop_budget, rng, args.use_self_loops)
            cleaned, _ = cleaned_path_and_relations(episode, raw_path, args.path_segment_policy)
            predicted_lengths.append(len(cleaned))
            final_entity = raw_path[-1][2] if raw_path else start
            answer_hits.append(1.0 if int(final_entity) in answers else 0.0)
            walk_metrics.append(compute_path_fidelity(episode, raw_path, local_idx, args.path_segment_policy))

            if args.use_ideal_paths and episode.paths_exists:
                exact_path_hits.append(1.0 if [tuple(edge) for edge in cleaned] == [tuple(edge) for edge in episode.paths[local_idx]] else 0.0)

        means = metric_means(walk_metrics)
        per_question: Dict[str, Any] = {
            "split": name,
            "question_id": row_id(row, local_idx),
            "start_entity": start,
            "valid_answer_entities": sorted(answers),
            "reference_path_length": effective_reference_length(episode, local_idx),
            "predicted_path_length": float(np.mean(predicted_lengths)) if predicted_lengths else 0.0,
            "answer_reached": float(np.mean(answer_hits)) if answer_hits else 0.0,
            "num_walks": args.num_walks,
            "seed": args.seed,
        }
        per_question.update(means)
        if args.use_ideal_paths and episode.paths_exists:
            per_question["RW_Path"] = float(np.mean(exact_path_hits)) if exact_path_hits else None
        rows.append(per_question)

    return rows


def print_summary(summary: Dict[str, Any], output_path: Optional[str]) -> None:
    print("\n" + "=" * 88)
    print("Unbiased random-walk path-fidelity baseline")
    print("=" * 88)
    print(f"Evaluated questions: {summary['num_evaluated_questions']}")
    print(f"Walks/question:      {summary['num_walks']}")
    print(f"Seed:                {summary['seed']}")
    print(f"Hop budget:          {summary['hop_budget']}")
    print(f"Graph:               {summary['graph']}")
    print(f"Self-loops sampled:  {summary['use_self_loops']}")
    print("-" * 88)
    print(f"Average PED:    {format_metric(summary.get('average_PED'))}")
    print(f"Average RED:    {format_metric(summary.get('average_RED'))}")
    print(f"Average F1_SG:  {format_metric(summary.get('average_F1_SG'))}")
    print(f"Average F1_Rel: {format_metric(summary.get('average_F1_Rel'))}")
    print(f"RW-Ans:         {format_metric(summary.get('RW_Ans'))}")
    if summary.get("metric_availability_notes"):
        print("Metric availability notes:")
        for metric, note in summary["metric_availability_notes"].items():
            print(f"  {metric}: {note}")
    if "RW_Path" in summary:
        print(f"RW-Path:        {format_metric(summary.get('RW_Path'))}")
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

    rng = np.random.default_rng(args.seed)
    per_question: List[Dict[str, Any]] = []
    for split_name, df in frames:
        if args.max_samples is not None:
            df = df.head(args.max_samples)
        per_question.extend(evaluate_split(split_name, df, metadata, grapher, args, rng))

    summary = aggregate_per_question(per_question, METRIC_NAMES)
    summary.update({
        "baseline": "unbiased_random_walk",
        "num_walks": args.num_walks,
        "seed": args.seed,
        "max_samples": args.max_samples,
        "hop_budget": hop_budget,
        "graph": "full_graph.txt" if args.use_full_graph else "graph.txt",
        "use_self_loops": bool(args.use_self_loops),
        "use_directed_graph": not args.include_inverse_relations,
        "path_segment_policy": args.path_segment_policy,
        "metric_availability_notes": metric_availability_notes(metadata),
        "RW_Ans": float(np.mean([row["answer_reached"] for row in per_question])) if per_question else None,
    })
    if args.use_ideal_paths and per_question and "RW_Path" in per_question[0]:
        summary["RW_Path"] = float(np.mean([row["RW_Path"] for row in per_question if row.get("RW_Path") is not None]))

    payload = {"summary": summary, "per_question": per_question}
    output_path = write_results(args.output, args.output_format, payload)
    print_summary(summary, output_path)


if __name__ == "__main__":
    main()
