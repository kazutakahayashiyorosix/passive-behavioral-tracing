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

## Comparative and public-data evaluation (v2)

```
python3 pbt_applied.py /path/to/pizza_request_dataset.json   # Part A baselines + benign shifts -> results/applied_results.json
python3 pbt_raop.py    /path/to/pizza_request_dataset.json   # Part B Random Acts of Pizza, phase I/II -> results/raop_pbt_results.json
```

The Random Acts of Pizza dataset (Althoff, Danescu-Niculescu-Mizil & Jurafsky, ICWSM 2014) is not redistributed here; obtain it from its authors.
Note: `results/applied_results.json` part B is a superseded first design (two-sided CUSUM, four-interval reference); the reported public-log results are in `results/raop_pbt_results.json`.
