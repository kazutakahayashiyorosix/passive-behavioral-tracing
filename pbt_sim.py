"""Synthetic validation for Passive Behavioral Tracing (PBT).

Everything here is simulated from the stylized model below; no platform data is used.
Run:  python pbt_sim.py            -> results/*.json, results/*.csv
"""
import json, os, sys
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

T = 12                     # observation intervals
N_MAIN, N_SUB = 80, 20     # participants
N_CTX = 4                  # naturally occurring contexts
SIGMA = 0.5                # response noise
FLOOR = 0.05               # responses are positive (an allocation, a response time inverse, ...)
REF = 4                    # Drift Detection reference window: first REF intervals
EPS = 1e-9


def simulate(cond, rng, n_sub=N_SUB, onset=T // 2, strength=1.0):
    """Return dict of arrays, shape (participants, T)."""
    n = N_MAIN + n_sub
    sub = np.r_[np.zeros(N_MAIN, bool), np.ones(n_sub, bool)]
    ctx = rng.integers(0, N_CTX, size=(n, T))
    ctx_eff = np.array([0.0, 0.3, 0.6, 0.9])[ctx]
    c = rng.gamma(2.0, 2.0, size=(n, T))                 # contribution
    fair = 1.0 + 1.0 * c + ctx_eff
    unfair = 0.6 * (1.0 + 0.35 * c + ctx_eff)           # weakly coupled to contribution, suppressed level
    unfair = (1 - strength) * fair + strength * unfair     # graded unfairness (strength 1 = the main experiment)
    y = fair.copy()
    if cond == "sustained":
        y[sub] = unfair[sub]
    elif cond == "onset":
        y[sub, onset:] = unfair[sub, onset:]
    y = np.maximum(FLOOR, y + rng.normal(0, SIGMA, size=(n, T)))
    return dict(sub=sub, ctx=ctx, c=c, y=y)


def tc(y_u, ctx_u):
    """Trace Consistency for one participant: pairs that share a context."""
    terms = []
    for k in range(N_CTX):
        idx = np.flatnonzero(ctx_u == k)
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                a, b = y_u[idx[i]], y_u[idx[j]]
                terms.append(abs(a - b) / (max(a, b) + EPS))
    return 1 - np.mean(terms) if terms else np.nan


def br(c_u, y_u):
    return np.corrcoef(c_u, y_u)[0, 1]


def nif(S, min_support=20, n_bins=5):
    """1 - max over contexts of total-variation distance between the groups' response distributions."""
    y, ctx, sub = S["y"], S["ctx"], S["sub"]
    edges = np.quantile(y, np.linspace(0, 1, n_bins + 1)[1:-1])
    d = []
    for k in range(N_CTX):
        a = y[~sub][ctx[~sub] == k]; b = y[sub][ctx[sub] == k]
        if len(a) < min_support or len(b) < min_support:
            continue
        pa = np.bincount(np.searchsorted(edges, a), minlength=n_bins) / len(a)
        pb = np.bincount(np.searchsorted(edges, b), minlength=n_bins) / len(b)
        d.append(0.5 * np.abs(pa - pb).sum())
    return 1 - max(d)


def drift_series(S):
    """Q_t: subgroup's mean response relative to what the main group's own response rule predicts
    for the subgroup's contributions and contexts in that interval (1 = treated like everyone else)."""
    y, c, ctx, sub = S["y"], S["c"], S["ctx"], S["sub"]
    q = np.empty(T)
    for t in range(T):
        X = np.c_[np.ones((~sub).sum()), c[~sub, t], ctx[~sub, t]]
        beta, *_ = np.linalg.lstsq(X, y[~sub, t], rcond=None)
        pred = np.c_[np.ones(sub.sum()), c[sub, t], ctx[sub, t]] @ beta
        q[t] = y[sub, t].mean() / max(pred.mean(), EPS)
    return q


def cusum_flags(q, h, k=0.5):
    """Two-sided tabular CUSUM (Page 1954) in units of the reference-window SD.
    The reference (in-control) level is estimated from the series' own first REF intervals,
    so the procedure detects CHANGE relative to how the system behaved at the start of the window."""
    mu, sd = q[:REF].mean(), q[:REF].std(ddof=1) + 1e-3
    z = (q - mu) / sd
    sp = sn = 0.0
    flags = np.zeros(T, bool)
    for t in range(REF, T):
        sp = max(0.0, sp + z[t] - k); sn = max(0.0, sn - z[t] - k)
        flags[t] = sp > h or sn > h
    return flags


def run_one(cond, seed, h, n_sub=N_SUB, onset=T // 2, strength=1.0):
    rng = np.random.default_rng(seed)
    S = simulate(cond, rng, n_sub=n_sub, onset=onset, strength=strength)
    sub = S["sub"]
    tcs = np.array([tc(S["y"][u], S["ctx"][u]) for u in np.flatnonzero(sub)])
    brs = np.array([br(S["c"][u], S["y"][u]) for u in np.flatnonzero(sub)])
    tc_all = np.array([tc(S["y"][u], S["ctx"][u]) for u in range(len(sub))])
    br_all = np.array([br(S["c"][u], S["y"][u]) for u in range(len(sub))])
    q = drift_series(S)
    f = cusum_flags(q, h)
    first = int(np.argmax(f)) if f.any() else -1
    n = nif(S)
    delta = f.mean()
    return dict(cond=cond, seed=seed, n_sub=n_sub, onset=onset,
                br_sub=float(np.nanmean(brs)), tc_sub=float(np.nanmean(tcs)), nif=float(n),
                delta_drift=float(delta), detected=bool(f.any()), first_flag=first,
                pbt_ethics=float(n * (1 - delta)), pbt_coop=float(np.nanmean(tc_all * br_all)))


def calibrate_h(n_runs=2000, target_fpr=0.05):
    """CUSUM threshold so that a fair series raises any alarm in at most target_fpr of runs."""
    peaks = []
    for s in range(n_runs):
        q = drift_series(simulate("fair", np.random.default_rng(10_000_000 + s)))
        mu, sd = q[:REF].mean(), q[:REF].std(ddof=1) + 1e-3
        z = (q - mu) / sd; sp = sn = 0.0; m = 0.0
        for t in range(REF, T):
            sp = max(0.0, sp + z[t] - 0.5); sn = max(0.0, sn - z[t] - 0.5); m = max(m, sp, sn)
        peaks.append(m)
    return float(np.quantile(peaks, 1 - target_fpr))


def summarize(rows, keys):
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows], float)
        out[k] = [float(np.mean(v)), float(np.std(v))]
    return out


def main(n=1000):
    h = calibrate_h()
    res = {"h": h, "n_runs": n, "conditions": {}}
    rows_all = []
    for ci, cond in enumerate(["fair", "sustained", "onset"]):
        rows = [run_one(cond, 1_000 * ci + s, h) for s in range(n)]
        rows_all += rows
        sm = summarize(rows, ["br_sub", "nif", "tc_sub", "delta_drift", "pbt_ethics", "pbt_coop"])
        sm["detection_rate"] = float(np.mean([r["detected"] for r in rows]))
        if cond == "onset":
            ff = np.array([r["first_flag"] for r in rows if r["detected"]])
            sm["detection_delay_median"] = float(np.median(ff - T // 2))
            sm["flag_before_onset_rate"] = float(np.mean(ff < T // 2))
        res["conditions"][cond] = sm
    # detection of sustained unfairness by a BR threshold calibrated on half the fair runs
    fair_br = np.array([r["br_sub"] for r in rows_all if r["cond"] == "fair"])
    sus_br = np.array([r["br_sub"] for r in rows_all if r["cond"] == "sustained"])
    on_br = np.array([r["br_sub"] for r in rows_all if r["cond"] == "onset"])
    cal, test = fair_br[: n // 2], fair_br[n // 2:]
    thr = float(np.quantile(cal, 0.05))
    res["br_threshold"] = {"threshold": thr, "flag_sustained": float(np.mean(sus_br < thr)),
                           "flag_onset": float(np.mean(on_br < thr)), "flag_heldout_fair": float(np.mean(test < thr))}
    # power: graded unfairness strength x subgroup size; thresholds calibrated on fair runs of the same size
    power = []
    for n_sub in (10, 20, 40):
        fair = [run_one("fair", 8_000_000 + 1000 * n_sub + s, h, n_sub=n_sub) for s in range(300)]
        br_thr = float(np.quantile([r["br_sub"] for r in fair[:150]], 0.05))
        nif_thr = float(np.quantile([r["nif"] for r in fair[:150]], 0.05))
        fa = dict(drift=float(np.mean([r["detected"] for r in fair])),
                  br=float(np.mean([r["br_sub"] < br_thr for r in fair[150:]])),
                  nif=float(np.mean([r["nif"] < nif_thr for r in fair[150:]])))
        for st in (0.1, 0.2, 0.3, 0.5, 1.0):
            sus = [run_one("sustained", 9_000_000 + 1000 * n_sub + int(st * 100) * 7 + s, h, n_sub=n_sub, strength=st) for s in range(150)]
            on = [run_one("onset", 9_500_000 + 1000 * n_sub + int(st * 100) * 7 + s, h, n_sub=n_sub, strength=st) for s in range(150)]
            power.append(dict(n_sub=n_sub, strength=st,
                              drift_onset=float(np.mean([r["detected"] for r in on])),
                              drift_sustained=float(np.mean([r["detected"] for r in sus])),
                              br_sustained=float(np.mean([r["br_sub"] < br_thr for r in sus])),
                              nif_sustained=float(np.mean([r["nif"] < nif_thr for r in sus])),
                              false_alarm=fa))
    res["power"] = power
    json.dump(res, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    import csv
    with open(os.path.join(OUT, "runs.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_all[0])); w.writeheader(); w.writerows(rows_all)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1000)
