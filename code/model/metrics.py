import numpy as np

from collections import namedtuple

from typing import Any, List, Set, Tuple, Dict, Optional

EvaluationMetrics = namedtuple('EvaluationMetrics', [
    'hits_at_1', 'hits_at_3', 'hits_at_5', 'hits_at_10', 'hits_at_20', 
    'answer_recall', 'answer_precision', 'answer_f1',
    'path_recall', 'path_precision', 'path_f1',
    'node_recall', 'node_precision', 'node_f1',
    'rel_recall', 'rel_precision', 'rel_f1',
    'edit_distance',
    'special_step_rate', 'restart_rate', 'no_op_rate', 'cycle_rate', 
    'backtrack_rate', 'unique_edges', 'redundancy',
    'termination_steps', 'termination_rollout', 'segment_hops',
    'stop_rate', 'correct_stop_rate', 'incorrect_stop_rate', 'hit_wo_stop_rate',
    'stop_rate_rollout', 'correct_stop_rate_rollout', 'incorrect_stop_rate_rollout', 'hit_wo_stop_rate_rollout',
    'restart_any_rate', 'post_restart_success_rate', 'restart_and_hit_rate',
    'restart_any_rate_rollout', 'post_restart_success_rate_rollout', 'restart_and_hit_rate_rollout',
    'mrr', 'max_hits_at_1', 'max_mrr',
    'valid_action_count', 'question_entropy', 'path_entropy',
    'hop_accuracy', 'hop_mrr',
])

def _as_float(x):
    return None if x is None else float(x)

def entropy_from_log_probs(
    log_probs: np.ndarray,
    axis: int = -1,
    base: float = np.e
) -> np.ndarray:
    """
    Compute entropy from natural-log probabilities, expressed
    in the requested logarithmic base.

    Args:
        log_probs: array of shape [..., num_actions]
        axis: axis corresponding to the action dimension
        base: output log base (np.e for nats, 2 for bits, 10 for base-10)

    Returns:
        Entropy with shape log_probs.shape with `axis` removed.
    """
    probs = np.exp(log_probs)
    entropy_nats = -np.sum(probs * log_probs, axis=axis)

    if base == np.e:
        return entropy_nats
    return entropy_nats / np.log(base)

def edit_distance(seq1: List[int], seq2: List[int]) -> int:
    """
    Compute the edit distance between two sequences of integers.

    Edit distance is defined as the minimum number of insertions, deletions, or substitutions
    required to transform seq1 into seq2.

    Args:
        seq1: First sequence of integers.
        seq2: Second sequence of integers.

    Returns:
        edit_distance (int): The computed edit distance between seq1 and seq2.
    """
    m = len(seq1)
    n = len(seq2)

    if m == 0 and n == 0:
        return 0.0, m, n
    if m == 0 or n == 0:
        return max(m, n), m, n

    dp = np.zeros((m + 1, n + 1), dtype=int)

    for i0 in range(m + 1):
        dp[i0][0] = i0  # Deletion cost
    for j0 in range(n + 1):
        dp[0][j0] = j0  # Insertion cost

    for i0 in range(1, m + 1):
        for j0 in range(1, n + 1):
            if seq1[i0 - 1] == seq2[j0 - 1]:
                dp[i0][j0] = dp[i0 - 1][j0 - 1]  # No cost if elements match
            else:
                dp[i0][j0] = min(
                    dp[i0 - 1][j0] + 1,    # Deletion
                    dp[i0][j0 - 1] + 1,    # Insertion
                    dp[i0 - 1][j0 - 1] + 1 # Substitution
                )
    return dp[m][n], m, n

def compute_precision_recall_f1(pred: Set[Any], gt: Set[Any]) -> Tuple[float, float, float]:
    """
    Compute precision, recall, and F1 score between two sets of items.
    Precision is the fraction of predicted items that are correct (in the ground truth).
    Recall is the fraction of ground truth items that are correctly predicted.
    F1 score is the harmonic mean of precision and recall.

    Args:
        pred: Set of predicted items.
        gt: Set of ground truth items.
    Returns:
        precision: Precision score.
        recall: Recall score.
        f1_score: F1 score.
    """
    tp = len(pred & gt)
    fp = len(pred - gt)
    fn = len(gt - pred)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1_score = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1_score
