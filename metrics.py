import numpy as np

def compute_ece(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (y_prob >= lo) & (y_prob < hi)
        if not np.any(mask):
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (np.sum(mask) / N) * abs(acc - conf)
    return float(ece)

def compute_pace(y_true, p_hat, L: int = 2, M: int = 2000, rng: np.random.Generator | None = None) -> float:
    y_true = np.asarray(y_true).astype(int)
    p_hat = np.asarray(p_hat).astype(float)
    N = len(y_true)
    if rng is None:
        rng = np.random.default_rng()
    errors = []
    for _ in range(M):
        idx = rng.choice(N, size=L, replace=False)
        p_parlay = float(np.prod(p_hat[idx]))
        y_parlay = 1.0 if np.all(y_true[idx] == 1) else 0.0
        errors.append(abs(p_parlay - y_parlay))
    return float(np.mean(errors))