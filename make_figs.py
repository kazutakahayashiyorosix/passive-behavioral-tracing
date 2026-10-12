import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
r = json.load(open("results/summary.json"))
os.makedirs("figs", exist_ok=True)
C = r["conditions"]; conds = ["fair", "sustained", "onset"]; lab = ["Fair", "Sustained-unfair", "Onset-unfair"]
keys = [("br_sub", "BR (subgroup)"), ("nif", "NIF"), ("tc_sub", "TC (subgroup)"), ("delta_drift", r"$\delta_{drift}$")]
cols = ["#4a6fa5", "#9fb8d9", "#c9822b", "#e8b77a"]
fig, ax = plt.subplots(figsize=(6.6, 2.9))
w = 0.2; x = np.arange(3)
for i, (k, name) in enumerate(keys):
    m = [C[c][k][0] for c in conds]; s = [C[c][k][1] for c in conds]
    b = ax.bar(x + (i - 1.5) * w, m, w, yerr=s, capsize=2, color=cols[i], label=name, edgecolor="black", linewidth=0.4)
    for xi, v, sd in zip(x + (i - 1.5) * w, m, s):
        ax.text(xi, v + sd + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5)
ax.set_xticks(x); ax.set_xticklabels(lab); ax.set_ylim(0, 1.12); ax.set_ylabel("Value (mean ± SD, 1,000 runs)")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(ncol=4, fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.13))
fig.tight_layout(); fig.savefig("figs/fig1_constructs.png", dpi=300); fig.savefig("figs/fig1_constructs.pdf")

P = r["power"]; sizes = [10, 20, 40]
fig, axs = plt.subplots(1, 3, figsize=(6.8, 2.75), sharey=True)
for a, n in zip(axs, sizes):
    rows = [p for p in P if p["n_sub"] == n]; st = [p["strength"] for p in rows]
    a.plot(st, [p["drift_onset"] for p in rows], "o-", color="#c9822b", label="Drift, onset-unfair")
    a.plot(st, [p["br_sustained"] for p in rows], "s-", color="#4a6fa5", label="BR threshold, sustained")
    a.plot(st, [p["nif_sustained"] for p in rows], "^-", color="#9fb8d9", label="NIF threshold, sustained")
    a.plot(st, [p["drift_sustained"] for p in rows], "x--", color="#888888", label="Drift, sustained")
    a.axhline(0.05, color="#bbbbbb", lw=0.8, ls=":")
    a.set_title(f"Subgroup of {n}", fontsize=9); a.set_xlabel("Unfairness strength"); a.set_xticks([0.1, 0.3, 0.5, 1.0])
    a.spines[["top", "right"]].set_visible(False)
axs[0].set_ylabel("Detection rate")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, ncol=4, fontsize=7, frameon=False, loc="lower center")
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig("figs/fig2_power.png", dpi=300); fig.savefig("figs/fig2_power.pdf")
print("ok")
