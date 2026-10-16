"""
pbt_applied.py -- baseline comparisons and a semi-synthetic test on a public log
for Passive Behavioral Tracing (PBT).

Part A (synthetic): PBT's model-relative drift statistic vs two common monitoring
baselines (CUSUM on the subgroup's raw mean response; CUSUM on the raw group gap),
and PBT's reciprocity threshold vs a demographic-parity ratio threshold, under five
conditions including two benign shifts (subgroup contribution falls; everyone's
treatment changes).

Part B (semi-synthetic, public data): the Random Acts of Pizza log (Althoff et al.,
ICWSM 2014). Response = request succeeded; contribution = requester's prior activity
and request length; naturally occurring group = accounts younger than 30 days;
12 equal-count time intervals. Onset unfairness is injected by withholding a share
of the subgroup's successes from interval 6; a benign shift is created by removing
the subgroup's most active requesters from interval 6.

Usage: python3 pbt_applied.py /path/to/pizza_request_dataset.json
"""
import json, os, sys
import numpy as np
from pbt_sim import simulate as base_simulate, T, N_MAIN, N_SUB, REF, EPS

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
ONSET = T // 2
DETECTORS = ["pbt", "raw", "gap"]


# ---------------------------------------------------------------- statistics
def cusum_peak(q):
    mu, sd = q[:REF].mean(), q[:REF].std(ddof=1) + 1e-3
    z = (q - mu) / sd
    sp = sn = 0.0; m = 0.0; first = -1
    peaks = np.zeros(len(q))
    for t in range(REF, len(q)):
        sp = max(0.0, sp + z[t] - 0.5); sn = max(0.0, sn - z[t] - 0.5)
        peaks[t] = max(sp, sn)
    return peaks


def series(y, X, sub, tt, nT):
    """Return the three monitored series over intervals."""
    q = {"pbt": np.empty(nT), "pbt_res": np.empty(nT), "raw": np.empty(nT), "gap": np.empty(nT)}
    # pooled main-group model with interval intercepts (for the residual form)
    D = np.c_[X[:, 1:], (tt[:, None] == np.arange(nT)[None, :]).astype(float)]
    bp, *_ = np.linalg.lstsq(D[~sub], y[~sub], rcond=None)
    res_sub = y[sub] - D[sub] @ bp
    for t in range(nT):
        q["pbt_res"][t] = res_sub[tt[sub] == t].mean()
        m = tt == t
        mm, ms = m & ~sub, m & sub
        beta, *_ = np.linalg.lstsq(X[mm], y[mm], rcond=None)
        pred = X[ms] @ beta
        q["pbt"][t] = y[ms].mean() / max(pred.mean(), EPS)
        q["raw"][t] = y[ms].mean()
        q["gap"][t] = y[ms].mean() - y[mm].mean()
    return q


# ---------------------------------------------------------------- Part A
def sim(cond, rng):
    if cond in ("fair", "sustained", "onset"):
        S = base_simulate(cond, rng)
    else:
        S = base_simulate("fair", rng)
        n = N_MAIN + N_SUB
        sub = S["sub"]
        ctx_eff = np.array([0.0, 0.3, 0.6, 0.9])[S["ctx"]]
        c = S["c"].copy()
        shift = np.zeros_like(c)
        if cond == "covshift":       # subgroup contributes half as much from ONSET; treated fairly
            c[np.ix_(sub, np.arange(ONSET, T))] *= 0.5
        if cond == "commonshift":    # everyone's treatment rises by 1.0 from ONSET; fair
            shift[:, ONSET:] = 1.0
        y = 1.0 + c + ctx_eff + shift
        y = np.maximum(0.05, y + rng.normal(0, 0.5, size=y.shape))
        S = dict(sub=sub, ctx=S["ctx"], c=c, y=y)
    return S


def flatten(S):
    n, nT = S["y"].shape
    tt = np.tile(np.arange(nT), (n, 1)).ravel()
    sub = np.repeat(S["sub"], nT)
    X = np.c_[np.ones(n * nT), S["c"].ravel(), S["ctx"].ravel()]
    return S["y"].ravel(), X, sub, tt


def br_sub(S):
    return float(np.nanmean([np.corrcoef(S["c"][u], S["y"][u])[0, 1] for u in np.flatnonzero(S["sub"])]))


def dp_ratio(S):
    return float(S["y"][S["sub"]].mean() / S["y"][~S["sub"]].mean())


def part_a(n=500):
    conds = ["fair", "sustained", "onset", "covshift", "commonshift"]
    # calibrate each detector's CUSUM threshold and the level thresholds on separate fair runs
    cal = {d: [] for d in DETECTORS}; cal_br, cal_dp = [], []
    for s in range(1000):
        S = sim("fair", np.random.default_rng(50_000_000 + s))
        q = series(*flatten(S), T)
        for d in DETECTORS: cal[d].append(cusum_peak(q[d]).max())
        cal_br.append(br_sub(S)); cal_dp.append(dp_ratio(S))
    h = {d: float(np.quantile(cal[d], 0.95)) for d in DETECTORS}
    thr_br, thr_dp = float(np.quantile(cal_br, 0.05)), float(np.quantile(cal_dp, 0.05))
    out = {"h": h, "thr_br": thr_br, "thr_dp": thr_dp, "n_runs": n, "alarm_rate": {}}
    for ci, cond in enumerate(conds):
        al = {d: 0 for d in DETECTORS}; lv = {"br": 0, "dp": 0}
        for s in range(n):
            S = sim(cond, np.random.default_rng(60_000_000 + 100_000 * ci + s))
            q = series(*flatten(S), T)
            for d in DETECTORS: al[d] += cusum_peak(q[d]).max() > h[d]
            lv["br"] += br_sub(S) < thr_br; lv["dp"] += dp_ratio(S) < thr_dp
        out["alarm_rate"][cond] = {**{f"drift_{d}": al[d] / n for d in DETECTORS},
                                   "level_br": lv["br"] / n, "level_dp": lv["dp"] / n}
        print("A", cond, out["alarm_rate"][cond], flush=True)
    return out


# ---------------------------------------------------------------- Part B
def load_raop(path):
    d = json.load(open(path))
    d.sort(key=lambda r: r["unix_timestamp_of_request_utc"])
    y = np.array([float(r["requester_received_pizza"]) for r in d])
    act = np.log1p([r["requester_number_of_comments_at_request"] + r["requester_number_of_posts_at_request"] for r in d])
    ln = np.log1p([len(r["request_text_edit_aware"]) for r in d])
    sub = np.array([r["requester_account_age_in_days_at_request"] < 30 for r in d])
    tt = np.minimum((np.arange(len(d)) * T) // len(d), T - 1)
    X = np.c_[np.ones(len(d)), act, ln]
    return y, X, sub, tt, act


def boot(y, X, sub, tt, act, rng):
    idx = np.concatenate([rng.choice(np.flatnonzero(tt == t), (tt == t).sum()) for t in range(T)])
    return y[idx].copy(), X[idx], sub[idx], tt[idx], act[idx]


def inject(y, sub, tt, strength, rng):
    y = y.copy()
    hit = sub & (tt >= ONSET) & (y == 1) & (rng.random(len(y)) < strength)
    y[hit] = 0.0
    return y


def benign(y, X, sub, tt, act):
    med = np.median(act[sub])
    keep = ~(sub & (tt >= ONSET) & (act > med))
    return y[keep], X[keep], sub[keep], tt[keep]


def part_b(path, n=400):
    y0, X0, sub0, tt0, act0 = load_raop(path)
    info = {"n": int(len(y0)), "sub_share": float(sub0.mean()),
            "rate_sub": float(y0[sub0].mean()), "rate_main": float(y0[~sub0].mean()),
            "rate_by_interval": [float(y0[tt0 == t].mean()) for t in range(T)]}
    # null: permute group labels within interval (no group-specific change, real common drift kept)
    rng = np.random.default_rng(7)
    cal = {d: [] for d in DETECTORS}
    for s in range(1000):
        y, X, sub, tt, act = boot(y0, X0, sub0, tt0, act0, rng)
        sub = sub.copy()
        for t in range(T):
            m = np.flatnonzero(tt == t); sub[m] = rng.permutation(sub[m])
        q = series(y, X, sub, tt, T)
        for d in DETECTORS: cal[d].append(cusum_peak(q[d]).max())
    h = {d: float(np.quantile(cal[d], 0.95)) for d in DETECTORS}
    res = {"info": info, "h": h, "n_boot": n}
    q = series(y0, X0, sub0, tt0, T)
    res["unmodified_alarm"] = {d: bool(cusum_peak(q[d]).max() > h[d]) for d in DETECTORS}
    res["unmodified_series"] = {d: [float(v) for v in q[d]] for d in DETECTORS}
    rows = {}
    for label, strength in [("unmodified", 0.0), ("inject_0.1", 0.1), ("inject_0.2", 0.2),
                            ("inject_0.3", 0.3), ("inject_0.5", 0.5), ("benign_shift", None)]:
        al = {d: 0 for d in DETECTORS}
        for s in range(n):
            y, X, sub, tt, act = boot(y0, X0, sub0, tt0, act0, rng)
            if strength is None:
                y, X, sub, tt = benign(y, X, sub, tt, act)
            elif strength > 0:
                y = inject(y, sub, tt, strength, rng)
            q = series(y, X, sub, tt, T)
            for d in DETECTORS: al[d] += cusum_peak(q[d]).max() > h[d]
        rows[label] = {d: al[d] / n for d in DETECTORS}
        print("B", label, rows[label], flush=True)
    res["alarm_rate"] = rows
    return res


if __name__ == "__main__":
    A = part_a()
    B = part_b(sys.argv[1])
    json.dump({"part_a": A, "part_b": B}, open(os.path.join(OUT, "applied_results.json"), "w"), indent=1)
