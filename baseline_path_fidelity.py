import csv
import json
import math
import os
from collections import deque
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from code.data.data_utils import load_dictionary, load_qa_data
from code.data.grapher import RelationEntityGrapher
from code.model.environment import EpisodeNLQ


MetricDict = Dict[str, Optional[float]]
Edge = Tuple[int, int, int]


def default_cache_path(question_path: str) -> str:
    base = os.path.splitext(os.path.basename(question_path))[0]
    return os.path.join(".", ".cache", "itl", f"{base}.json")


def load_eval_frames(
    data_input_dir: str,
    question_path: str,
    cached_qa_metadata_path: Optional[str],
    question_tokenizer_name: str,
    seed: int,
    force_data_prepro: bool,
    test_only: bool,
):
    ent2id, rel2id, id2ent, id2rel, ent2name, rel2name = load_dictionary(data_input_dir)
    cached_qa_metadata_path = cached_qa_metadata_path or default_cache_path(question_path)
    train_df, dev_df, test_df, metadata = load_qa_data(
        cached_metadata_path=cached_qa_metadata_path,
        raw_QAData_path=question_path,
        question_tokenizer_name=question_tokenizer_name,
        entity2id=ent2id,
        relation2id=rel2id,
        seed=seed,
        logger=None,
        force_recompute=force_data_prepro,
    )

    frames = [("test", test_df)] if test_only else [
        ("train", train_df),
        ("dev", dev_df),
        ("test", test_df),
    ]
    return frames, metadata, ent2id, rel2id, id2ent, id2rel, ent2name, rel2name


def build_grapher(
    data_input_dir: str,
    ent2id: Dict[str, int],
    rel2id: Dict[str, int],
    max_num_actions: int,
    use_full_graph: bool,
    use_directed_graph: bool,
    use_stop_signal: bool = False,
    use_restart_signal: bool = False,
) -> RelationEntityGrapher:
    graph_file = "full_graph.txt" if use_full_graph else "graph.txt"
    return RelationEntityGrapher(
        triple_store=os.path.join(data_input_dir, graph_file),
        relation_vocab=rel2id,
        entity_vocab=ent2id,
        max_num_actions=max_num_actions,
        use_stop_signal=use_stop_signal,
        use_restart_signal=use_restart_signal,
        use_directed_graph=use_directed_graph,
    )


def answer_set(answer_value: Any, multi_answers: bool) -> Set[int]:
    if multi_answers:
        return {int(x) for x in answer_value}
    return {int(answer_value)}


def row_id(row: Any, fallback: int) -> Any:
    if "Question-Number" in row.index:
        value = row["Question-Number"]
        try:
            return int(value)
        except Exception:
            return value
    return int(fallback)


def valid_actions(
    grapher: RelationEntityGrapher,
    entity: int,
    include_self_loops: bool,
    include_stop_restart: bool = False,
) -> List[Tuple[int, int]]:
    raw = grapher.return_next_raw_actions(np.asarray([int(entity)], dtype=np.int32))[0]
    invalid_entities = {grapher.ePAD, grapher.eUNKNOWN}
    invalid_relations = {grapher.rPAD, grapher.rUNKNOWN, grapher.rDUMMY}
    if not include_self_loops:
        invalid_relations.add(grapher.rNO_OP)
    if not include_stop_restart:
        invalid_relations.update({grapher.rSTOP, grapher.rRESTART})

    out: List[Tuple[int, int]] = []
    seen = set()
    for target, relation in raw:
        target = int(target)
        relation = int(relation)
        if target in invalid_entities or relation in invalid_relations:
            continue
        key = (relation, target)
        if key in seen:
            continue
        seen.add(key)
        out.append((relation, target))
    return out


def make_episode(
    grapher: RelationEntityGrapher,
    df: Any,
    metadata: Dict[str, Any],
    path_length: int,
    use_stop_signal: bool = False,
    use_restart_signal: bool = False,
) -> EpisodeNLQ:
    starts = df["Source-Entity"].to_numpy(dtype=np.int32)
    multi_answers = bool(metadata.get("is_multi_answer", False))
    answers = df["Answer-Entity"].tolist() if multi_answers else df["Answer-Entity"].to_numpy(dtype=np.int32)
    paths = df["Paths"].tolist() if metadata.get("paths_column") is not None and "Paths" in df.columns else None
    path_keys = df["Path-Key"].tolist() if metadata.get("path_keys_column") is not None and "Path-Key" in df.columns else None
    hops = df["Hops"].tolist() if "Hops" in df.columns else None
    dummy_questions = [[] for _ in range(len(df))]
    dummy_embeddings = np.zeros((len(df), 1), dtype=np.float32)
    return EpisodeNLQ(
        graph=grapher,
        question_tokens=dummy_questions,
        question_embeddings=dummy_embeddings,
        start_entities=starts,
        end_entities=answers,
        batch_size=len(df),
        path_len=path_length,
        num_rollouts=1,
        positive_reward=1.0,
        negative_reward=0.0,
        mode="test",
        multi_answers=multi_answers,
        paths=paths,
        path_keys=path_keys,
        path_hops=hops,
    )


def effective_reference_length(episode: EpisodeNLQ, idx: int) -> Optional[int]:
    if episode.paths_exists:
        return len(episode.paths[idx])
    if episode.path_key_exists:
        return len(episode.path_keys[idx])
    return None


def cleaned_path_and_relations(
    episode: EpisodeNLQ,
    raw_path: Sequence[Edge],
    path_segment_policy: str,
) -> Tuple[List[Edge], List[int]]:
    cleaned = episode.clean_pred_path_for_eval(raw_path, policy=path_segment_policy)
    return cleaned, [int(step[1]) for step in cleaned]


def compute_path_fidelity(
    episode: EpisodeNLQ,
    pred_path: Sequence[Edge],
    idx: int,
    path_segment_policy: str = "final_segment_truncate",
) -> MetricDict:
    cleaned, relations = cleaned_path_and_relations(episode, pred_path, path_segment_policy)
    metrics: MetricDict = {"PED": None, "RED": None, "F1_SG": None, "F1_Rel": None}

    if episode.paths_exists:
        metrics["PED"] = float(episode.get_path_edit_distance(cleaned, idx))
        metrics["F1_SG"] = float(episode.get_subgraph_overlap(cleaned, idx)[2])

    if episode.paths_exists or episode.path_key_exists:
        metrics["RED"] = float(episode.get_relation_edit_distance(relations, idx))
        metrics["F1_Rel"] = float(episode.get_relation_coverage(relations, idx)[2])

    return metrics


def mean_optional(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not vals:
        return None
    return float(np.mean(vals))


def std_optional(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not vals:
        return None
    return float(np.std(vals))


def aggregate_per_question(rows: List[Dict[str, Any]], metric_names: Sequence[str]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"num_evaluated_questions": len(rows)}
    for name in metric_names:
        summary[f"average_{name}"] = mean_optional(row.get(name) for row in rows)
    return summary


def shortest_path_to_answer(
    grapher: RelationEntityGrapher,
    start: int,
    answers: Set[int],
    include_self_loops: bool = False,
) -> Optional[List[Edge]]:
    start = int(start)
    if start in answers:
        return []

    queue = deque([(start, [])])
    visited = {start}

    while queue:
        entity, path = queue.popleft()
        actions = valid_actions(
            grapher,
            entity,
            include_self_loops=include_self_loops,
            include_stop_restart=False,
        )
        actions = [(relation, target) for relation, target in actions if target != entity]
        for relation, target in sorted(actions, key=lambda x: (x[0], x[1])):
            edge = (int(entity), int(relation), int(target))
            new_path = path + [edge]
            if target in answers:
                return new_path
            if target not in visited:
                visited.add(target)
                queue.append((target, new_path))
    return None


def canonical_semantic_path(episode: EpisodeNLQ, pred_path: Sequence[Edge]) -> List[Edge]:
    return [
        episode.canon_edge(h, r, t)
        for h, r, t in pred_path
        if r not in episode.special_tokens and r != episode.grapher.rNO_OP
    ]


def exact_reference_match(episode: EpisodeNLQ, pred_path: Sequence[Edge], idx: int) -> Optional[bool]:
    if episode.paths_exists:
        return canonical_semantic_path(episode, pred_path) == [tuple(edge) for edge in episode.paths[idx]]
    if episode.path_key_exists:
        pred_rels = [
            episode.canon_rel(r)
            for _, r, _ in pred_path
            if r not in episode.special_tokens and r != episode.grapher.rNO_OP
        ]
        return pred_rels == list(episode.path_keys[idx])
    return None


def safe_float(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {k: safe_float(v) for k, v in value.items()}
    if isinstance(value, list):
        return [safe_float(v) for v in value]
    return value


def write_results(path: Optional[str], output_format: str, payload: Dict[str, Any]) -> Optional[str]:
    if not path:
        return None
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = safe_float(payload)

    if output_format == "json":
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        return path

    if output_format == "csv":
        rows = payload.get("per_question", [])
        with open(path, "w", newline="") as f:
            fieldnames = sorted({key for row in rows for key in row.keys()})
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        summary_path = path + ".summary.json"
        with open(summary_path, "w") as f:
            json.dump(payload.get("summary", {}), f, indent=2, sort_keys=True)
        return f"{path} (+ {summary_path})"

    raise ValueError(f"Unsupported output format: {output_format}")



def metric_availability_notes(metadata: Dict[str, Any]) -> Dict[str, str]:
    notes: Dict[str, str] = {}
    if metadata.get("paths_column") is None:
        notes["PED"] = "unavailable: no entity-level reference Paths column; evaluator only has relation-chain Path-Key data"
        notes["F1_SG"] = "unavailable: no entity-level reference Paths column; evaluator only has relation-chain Path-Key data"
    if metadata.get("paths_column") is None and metadata.get("path_keys_column") is None:
        notes["RED"] = "unavailable: no Paths or Path-Key reference data"
        notes["F1_Rel"] = "unavailable: no Paths or Path-Key reference data"
    return notes


def format_metric(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.6f}"
