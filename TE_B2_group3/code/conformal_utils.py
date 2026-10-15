import numpy as np


def finite_sample_quantile(scores, coverage):
    """Return the split-conformal order statistic or infinity if it is unavailable."""
    values = np.sort(np.asarray(scores, dtype=float))
    if len(values) == 0:
        raise ValueError("cannot calibrate an interval with no scores")
    rank = int(np.ceil((len(values) + 1) * coverage))
    if rank > len(values):
        return np.inf
    return values[rank - 1]
