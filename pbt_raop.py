"""Part B (revised): phase I / phase II monitoring on the Random Acts of Pizza log.
Reference = intervals 0..ONSET-1, monitoring = ONSET..T-1, one-sided (deterioration) CUSUM.
In-control SD of each monitored series per interval is estimated from a null in which group
labels are permuted within intervals (keeps the platform's real common drift)."""
import json, os, sys
import numpy as np
from pbt_applied import load_raop, boot, inject, benign, series, T, ONSET, OUT
DETECTORS = ["pbt", "pbt_res", "raw", "gap"]

def lower_cusum(q, mu, sd, k=0.5):
    z = (q - mu) / sd
    s = 0.0; m = 0.0
    for t in range(ONSET, T):
        s = max(0.0, s - z[t] - k); m = max(m, s)
    return m

def permuted(y, X, sub, tt, rng):
    sub = sub.copy()
    for t in range(T):
        m = np.flatnonzero(tt == t); sub[m] = rng.permutation(sub[m])
    return sub

def main(path, n=400):
    y0, X0, sub0, tt0, act0 = load_raop(path)
    rng = np.random.default_rng(11)
    # null distribution of each series (for per-interval SD)
    null = {d: [] for d in DETECTORS}
    for s in range(500):
        y, X, sub, tt, act = boot(y0, X0, sub0, tt0, act0, rng)
        q = series(y, X, permuted(y, X, sub, tt, rng), tt, T)
        for d in DETECTORS: null[d].append(q[d])
    sd = {d: np.array(null[d]).std(0) + 1e-6 for d in DETECTORS}
    def stat(y, X, sub, tt):
        q = series(y, X, sub, tt, T)
        return {d: lower_cusum(q[d], q[d][:ONSET].mean(), sd[d]) for d in DETECTORS}
    # threshold: 5% alarm rate on bootstrap of the unmodified real log (no injected change)
    base = [stat(*boot(y0, X0, sub0, tt0, act0, rng)[:4]) for s in range(n)]
    h = {d: float(np.quantile([b[d] for b in base], 0.95)) for d in DETECTORS}
    res = {"h": h, "n_boot": n, "alarm_rate": {}}
    for label, st in [("inject_0.1", .1), ("inject_0.2", .2), ("inject_0.3", .3), ("inject_0.5", .5), ("benign_shift", None)]:
        al = {d: 0 for d in DETECTORS}
        for s in range(n):
            y, X, sub, tt, act = boot(y0, X0, sub0, tt0, act0, rng)
            if st is None: y, X, sub, tt = benign(y, X, sub, tt, act)
            else: y = inject(y, sub, tt, st, rng)
            r = stat(y, X, sub, tt)
            for d in DETECTORS: al[d] += r[d] > h[d]
        res["alarm_rate"][label] = {d: al[d] / n for d in DETECTORS}
        print(label, res["alarm_rate"][label], flush=True)
    json.dump(res, open(os.path.join(OUT, "raop_pbt_results.json"), "w"), indent=1)

if __name__ == "__main__":
    main(sys.argv[1])
