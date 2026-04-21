import numpy as np

from collections import namedtuple

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