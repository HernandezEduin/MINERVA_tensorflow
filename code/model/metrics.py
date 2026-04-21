import numpy as np

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