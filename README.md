# Passive Behavioral Tracing

Code and results for the paper **"Passive Behavioral Tracing: A Non-Interventional Fairness Protocol for Cooperative AI"** (Kazutaka Hayashi, YOROSIX, Osaka, Japan).
The preprint is in `paper.pdf`.

PBT audits fairness in AI-mediated cooperative systems using only logs the system already keeps: Trace Consistency, Non-Interventional Fairness, Behavioral Reciprocity and Drift Detection (two-sided CUSUM), combined into PBT-Ethics and PBT-Coop.

## Contents
| File | What it does |
|---|---|
| `pbt_sim.py` | Stylized generative model, the four constructs, the Monte Carlo study and the power analysis |
| `make_figs.py` | Figures 1 and 2 |
| `results/summary.json`, `results/runs.csv` | Outputs used in the paper |
| `figs/` | Figures |

## Reproduce
```bash
pip install -r requirements.txt
python pbt_sim.py 1000   # about 10 minutes; fixed seeds
python make_figs.py
```

All data are simulated from the model in Section 4.7 of the paper; no platform data is used.

## License
Code: MIT. See `LICENSE`.
