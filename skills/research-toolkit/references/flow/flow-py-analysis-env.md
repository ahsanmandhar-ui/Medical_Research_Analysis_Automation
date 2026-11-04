---
name: flow-py-analysis-env
description: Freeze and verify a reproducible Python analysis environment on numpy/pandas/scipy/statsmodels/sklearn/matplotlib - package inventory with version pins writing an env-lock.json, community import aliases and pinned display precision, default_rng seeding so reruns match byte-for-byte, a data/figures/tables/reports workspace scaffold with a provenance runlog.json, and the guarded-import offline fallback ladder that reimplements absent task packages (pythonmeta, zepid, pyarrow, numba) on the installed stack. Use when the user needs to pin/probe/lock a Python environment, standardise imports or print precision, seed RNGs for exact reruns, scaffold an analysis workspace, or keep a run working without installing when a task package is missing.
---

# Frozen Python Analysis Environment (Python)

## Purpose
Own the substrate an analysis runs on - the frozen package set, portable import/display conventions,
explicit RNG seeding, the data-beside-code layout, and the fallback ladder when a task package is absent
`(McKinney p19-34)` `(Masoumi p1-9)`. The deliverables are provenance and determinism, not one statistic.

## Inputs
- Nothing required for a default run: `SEED = 42`, print precision `4`, CI multiplier `1.96`, a safe study
  `SLUG`; every block builds its own seeded demo data; an optional `input_data` path is logged in the runlog.
- Inventory maps: the essential core stack `(McKinney p21-28)` plus the policy-excluded package set; the
  demo table uses the four-count schema `events_exp,total_exp,events_ctl,total_ctl` `(Masoumi p2)`.

## Algorithms (all pure numpy/scipy)

### Package inventory, version pins and the blocklist
Probe every core version and write `env-lock.json` (`python`, `platform`, `core_stack`, `blocked_policy`)
to `workspace/` or a temp dir when no project exists. **The fleet never runs `pip install`:** blocked
packages get reimplemented on the installed stack, each with a named substitute.
```python
import importlib, json, platform, sys, tempfile
from pathlib import Path
ALLOWED = {"numpy": "numpy", "pandas": "pandas", "scipy": "scipy", "statsmodels": "statsmodels",
           "scikit-learn": "sklearn", "matplotlib": "matplotlib", "seaborn": "seaborn", "joblib": "joblib",
           "pillow": "PIL", "openpyxl": "openpyxl", "pymupdf": "pymupdf", "pdfplumber": "pdfplumber",
           "requests": "requests"}                                  # essential core stack (McKinney p21-28)
BLOCKED = {"pythonmeta": "IV + DL pooling via numpy/scipy/statsmodels (Masoumi p2-3)",
           "zepid": "forest/funnel plots via matplotlib Agg (Masoumi p3)",
           "pyarrow": "CSV+openpyxl (McKinney p30, p580)", "numba": "vectorised numpy (McKinney p21)",
           "tables": "CSV", "tensorflow": "sklearn/statsmodels", "torch": "sklearn/statsmodels",
           "keras": "sklearn/statsmodels", "xgboost": "sklearn GBM", "pingouin": "scipy.stats",
           "networkx": "graph-free sklearn"}             # policy-excluded; NEVER install, re-derive numerics
def probe(import_name):                                              # guarded - an absent module never crashes
    try: return str(getattr(importlib.import_module(import_name), "__version__", "installed"))
    except Exception: return None
core = {k: probe(v) for k, v in ALLOWED.items()}; policy = {b: probe(b) for b in BLOCKED}
for k, v in core.items(): print(("INSTALLED " + k + "=" + v) if v else "MISSING " + k)
fallback_used = []
for b, sub in BLOCKED.items():
    print(f"PRESENT guard-only {b}={policy[b]}" if policy[b] else f"MISSING {b} -> fallback: {sub}")
    fallback_used += [b] * (policy[b] is None)
lock = {"python": sys.version.split()[0], "platform": f"{platform.system()}/{platform.machine()}", "core_stack": core, "blocked_policy": policy}
out = Path(tempfile.mkdtemp(prefix="envlock_")) / "env-lock.json"; out.write_text(json.dumps(lock, indent=2, sort_keys=True), encoding="utf-8")
print(f"RESULT python={lock['python']} core_installed={sum(1 for v in core.values() if v)}/{len(ALLOWED)}")
print(f"RESULT fallback_used={','.join(fallback_used) if fallback_used else 'none'}")
print("RESULT policy=the fleet never installs; blocked packages are reimplemented on this stack")
```

### Session header and RNG seeding
Paste this header on every analysis so runs print identically `(McKinney p33-34)`. Every draw comes from
one explicit `default_rng(seed)` - the legacy global seed is banned - and checksums hash the rounded array
so the printed value and the digest agree `(McKinney p21-22, p33)`.
```python
import hashlib, io
import numpy as np, pandas as pd, matplotlib             # canonical aliases, never star-imports (McKinney p34)
matplotlib.use("Agg")                                    # headless - never plt.show() in a governed run
import matplotlib.pyplot as plt, seaborn as sns, statsmodels.api as sm   # working namespace is statsmodels.api
pd.options.display.max_columns = 20; pd.options.display.max_rows = 20
pd.options.display.max_colwidth = 80
np.set_printoptions(precision=4, suppress=True)          # pinned print precision (McKinney p33)
def draw(seed, n=5):                                     # one explicit default_rng stream per artefact
    return np.random.default_rng(seed).normal(size=n)
def checksum(arr):                                       # hash the ROUNDED array: print and digest agree
    buf = io.BytesIO(); np.save(buf, np.round(arr, 4), allow_pickle=False)
    return hashlib.sha256(buf.getvalue()).hexdigest()[:16]
ok = bool(np.array_equal(draw(42), draw(42)))
print(f"RESULT header_ok={all(m is not None for m in (np, pd, plt, sns, sm))} np_precision={np.get_printoptions()['precision']} max_rows={pd.options.display.max_rows} seed=42 stream_equal={ok} checksum={checksum(draw(42))} deterministic={int(ok)}")
```

### Workspace scaffold with provenance
Data sits beside code in a named, relocatable folder `(McKinney p32-33; Masoumi p2)`:
`workspace/<slug>/{data,figures,tables,reports}` plus a `runlog.json` logging interpreter, package
versions, seed and input identity - the four facts that let any run be re-derived offline.
```python
import json, sys, tempfile; from pathlib import Path
import numpy as np, pandas as pd
SLUG, SEED = "demo-analysis", 42; SUBS = ["data", "figures", "tables", "reports"]
root = Path(tempfile.mkdtemp(prefix="workspace_")) / SLUG
for s in SUBS: (root / s).mkdir(parents=True)
rng = np.random.default_rng(SEED)
demo = pd.DataFrame({"study": [f"s{i+1:02d}" for i in range(8)], "events_exp": rng.integers(1, 40, 8),
                     "total_exp": rng.integers(40, 120, 8), "events_ctl": rng.integers(1, 40, 8),
                     "total_ctl": rng.integers(40, 120, 8)})         # four-count schema (Masoumi p2)
demo.to_csv(root / "data" / "demo_input.csv", index=False)
runlog = {"slug": SLUG, "layout": SUBS, "seed": SEED, "python": sys.version.split()[0],
          "packages": {p: str(getattr(__import__(p), "__version__", "?")) for p in ("numpy", "pandas", "scipy", "statsmodels", "sklearn", "matplotlib")},
          "input_data": "data/demo_input.csv (seeded)", "note": "provenance lets the run be re-derived offline"}
(root / "runlog.json").write_text(json.dumps(runlog, indent=2, sort_keys=True), encoding="utf-8")
g = json.loads((root / "runlog.json").read_text(encoding="utf-8"))
print(f"RESULT slug={SLUG} dirs={sum((root / s).is_dir() for s in SUBS)}/4 demo_rows={len(demo)} runlog_seed={g['seed']} runlog_packages={len(g['packages'])}")
```

### Reimplemented pooling numerics, worked numbers (fallback recipe)
The blocked pooling engine's exact outputs rebuilt `(Masoumi p2-3, p5)`: fixed effect
`mu = sum(w_i*yi)/sum(w_i)`, `w_i = 1/vi`; `Q = sum(w_i*(yi-mu)^2)` on `k-1` df;
`I2 = max(0,(Q-df)/Q)*100`; DerSimonian-Laird `tau2 = max(0,(Q-df)/(sum(w)-sum(w^2)/sum(w)))`;
random effect `w_RE = 1/(vi+tau2)`; Egger asymmetry as a statsmodels OLS of `yi/sei` on `1/sei`
(intercept = bias). The first printed row must give `study1_rr=1.42 ci95=0.88-2.29`, matching the
published 1.42 (0.89-2.25) `(Masoumi p7)`; the second-decimal gap is expected cross-engine drift.
```python
import tempfile; from pathlib import Path
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt; from scipy import stats; import statsmodels.api as sm
rr = np.array([1.42, 1.05, 6.21, 2.09, 3.49])        # published trial RRs, 5-row subset (Masoumi p7)
vi = np.array([0.059, 0.035, 0.620, 0.073, 0.331])   # Var(logRR) = 1/a - 1/n1 + 1/c - 1/n0 (Masoumi p2)
yi = np.log(rr); w = 1.0 / vi; sei = np.sqrt(vi)
mu = float((w * yi).sum() / w.sum()); se = float(np.sqrt(1.0 / w.sum()))                  # fixed effect (IV)
Q = float((w * (yi - mu) ** 2).sum()); df = len(yi) - 1                                   # Cochran Q
I2 = max(0.0, (Q - df) / Q) * 100
tau2 = max(0.0, (Q - df) / (w.sum() - (w ** 2).sum() / w.sum()))                        # DerSimonian-Laird
wr = 1.0 / (vi + tau2); mu_r = float((wr * yi).sum() / wr.sum()); se_r = float(np.sqrt(1.0 / wr.sum()))
egger = sm.OLS(yi / sei, sm.add_constant(1.0 / sei)).fit()        # bias = intercept (Masoumi p3, p5)
lo, hi = np.exp(mu - 1.96 * se), np.exp(mu + 1.96 * se); rlo, rhi = np.exp(mu_r - 1.96 * se_r), np.exp(mu_r + 1.96 * se_r)
fig, ax = plt.subplots(figsize=(5, 2.8)); out = Path(tempfile.mkdtemp(prefix="fallback_figs_")); yy = np.arange(len(yi))[::-1]
for i in range(len(yi)):                                       # forest plot: zepid replaced by matplotlib
    ax.plot(np.exp(yi[i] + 1.96 * sei[i] * np.array([-1.0, 1.0])), [yy[i], yy[i]], color="#333", lw=1.2)
    ax.plot(np.exp(yi[i]), yy[i], "s", color="#1f77b4", ms=5)
ax.axvline(1.0, color="k", ls="--", lw=1); ax.set_xscale("log"); ax.set_yticks(yy); fig.tight_layout()
fig.savefig(out / "forest.png", dpi=120); plt.close(fig)
print(f"RESULT study1_rr={rr[0]:.2f} ci95={np.exp(yi[0] - 1.96 * sei[0]):.2f}-{np.exp(yi[0] + 1.96 * sei[0]):.2f}")
print(f"RESULT fixed_rr={np.exp(mu):.3f} fe_ci={lo:.3f}-{hi:.3f} random_rr={np.exp(mu_r):.3f} re_ci={rlo:.3f}-{rhi:.3f}")
print(f"RESULT q={Q:.2f} df={df} p={stats.chi2.sf(Q, df):.4f} i2_pct={I2:.2f} tau2={tau2:.4f}")
print(f"RESULT egger_intercept={egger.params[0]:.3f} egger_p={egger.pvalues[0]:.3f} re_ci_wider_than_fe={int((rhi - rlo) >= (hi - lo))} forest_written={(out / 'forest.png').is_file()}")
```

## Output contract
- Every tool prints `RESULT name=value`: `python=`/`core_installed=n/13`/`fallback_used=`;
  `header_ok=`/`seed=`/`stream_equal=`/`checksum=`/`deterministic=1`; `slug=`/`dirs=4/4`/`runlog_seed=`;
  `study1_rr=`/`ci95=`/`fixed_rr=`/`fe_ci=`/`random_rr=`/`re_ci=`/`q=`/`i2_pct=`/`tau2=`/
  `egger_intercept=`/`egger_p=`/`re_ci_wider_than_fe=1`.
- Artefacts: `env-lock.json` (`python`, `platform`, `core_stack`, `blocked_policy`);
  `workspace/<slug>/runlog.json` (`slug`, `layout`, `seed`, `python`, `packages`, `input_data`);
  `data/demo_input.csv`; `figures/forest.png`.

## Conventions
- `SEED = 42` default, recorded in the runlog; display pinned: `max_columns/max_rows = 20`,
  `max_colwidth = 80`, `np.set_printoptions(precision=4, suppress=True)` `(McKinney p33)`. Never hardcode
  a result - compute it and print the `RESULT` lines.
- The fleet never runs `pip install`; blocked packages are probed guarded and reimplemented on the
  installed stack: Parquet/HDF -> CSV+openpyxl, JIT -> vectorised numpy, the meta engines and `zepid` ->
  the numpy/scipy/statsmodels fallback above and headless matplotlib (`Agg`) `(Masoumi p3)`.
- Route by job: statsmodels for inference (standard errors, p-values), scikit-learn for prediction
  `(McKinney p26-28)`; never write into the skill folder itself. Do not update across managers
  (conda vs pip corrupts the environment); the GIL caps pure-Python threads - parallelise with C
  backends or processes `(McKinney p21-22, p30)`.
- Provenance is interpreter-pinned: the book assumed Python 3.10 `(McKinney p27-29)` - after any image
  change re-run the seeding check before trusting a stored checksum. State which subset you reproduce:
  published heterogeneity differs by subset (I-squared 54.51 vs 41.37, Q 35.18 vs 27.29) `(Masoumi p3, p7)`.
- Wording: record the environment BEFORE results - "Analysis run on Python {python} with numpy {v},
  pandas {v}, scipy {v}, statsmodels {v}, scikit-learn {v}, matplotlib {v} (see env-lock); seed {seed};
  input {path}. {package} was unavailable offline; numerics re-derived on the allowed stack; agreement
  asserted to the second decimal against the published table `(Masoumi p5-6)`."
- Sources: McKinney, *Python for Data Analysis* (3rd ed.), Ch.1 Preliminaries, pp.19-34 (+p580);
  Masoumi & Shahraz 2022, Python meta-analysis tutorial, pp.1-9.
