import numpy as np



def zscore_norm(d, indices):
    mean = np.mean(d[indices], axis=0)
    std = np.std(d[indices], axis=0)
    return (d - mean) / (std + 1e-9), {
        "mean": mean,
        "std": std,
    }

def robust_zscore_norm(d, indices):
    median = np.median(d[indices], axis=0)
    mad = np.median(np.abs(d[indices] - median), axis=0)
    return (d - median) / (mad * 1.4826 + 1e-9)