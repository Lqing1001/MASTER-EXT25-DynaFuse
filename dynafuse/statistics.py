"""Daily paired Newey-West inference and moving-block bootstrap."""
import math
import numpy as np
def hac_mean_test(values: np.ndarray, lag: int=10) -> dict[str, float]:
    values = values[np.isfinite(values)]
    n = len(values)
    centered = values - values.mean()
    long_run = float(centered @ centered / n)
    for k in range(1, min(lag, n - 1) + 1):
        gamma = float(centered[k:] @ centered[:-k] / n)
        long_run += 2.0 * (1.0 - k / (lag + 1.0)) * gamma
    se = math.sqrt(max(long_run, 0.0) / n)
    t = float(values.mean() / se) if se > 0 else float('nan')
    p_two_sided = math.erfc(abs(t) / math.sqrt(2.0)) if np.isfinite(t) else float('nan')
    return {'n': n, 'mean_difference': float(values.mean()), 'hac_se': se, 'hac_t': t, 'p_two_sided': p_two_sided}

def moving_block_ci(values: np.ndarray, rng: np.random.RandomState, block: int=20, draws: int=5000):
    values = values[np.isfinite(values)]
    n = len(values)
    starts = np.arange(max(1, n - block + 1))
    means = np.empty(draws, dtype=np.float64)
    blocks_needed = int(math.ceil(n / block))
    for i in range(draws):
        chosen = rng.choice(starts, size=blocks_needed, replace=True)
        sample = np.concatenate([values[s:s + block] for s in chosen])[:n]
        means[i] = sample.mean()
    return {'block_length': block, 'draws': draws, 'ci95_low': float(np.quantile(means, 0.025)), 'ci95_high': float(np.quantile(means, 0.975)), 'probability_positive': float(np.mean(means > 0))}
