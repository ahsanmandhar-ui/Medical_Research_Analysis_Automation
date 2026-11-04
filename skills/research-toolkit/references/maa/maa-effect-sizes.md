---
name: maa-effect-sizes
description: Computes study-level effect sizes for a meta-analytic input table in pure numpy/scipy - risk ratio (RR), odds ratio (OR, on the log-OR scale) and risk difference (RD) from 2x2 counts, each with its large-sample variance, standard error and 95% CI, plus the continuous-outcome family (mean difference, Cohen d, Hedges g with its bias correction J and variance). Covers the zero-cell continuity correction and the dropout/missing-data labels (ACS, ICA-0, ICA-1, ICA-best, ICA-worst) that decide what enters the 2x2 before anything is computed. pythonmeta and zEpid are NOT used - every formula is reimplemented from first principles because neither is installed. Reproduces the per-study log-RR standard errors and IV weight shares printed in Masoumi & Shahraz 2022 Table 2 (p7) to within 0.2 weight points. Use when asked for the OR/RR/RD or SMD of individual studies, the variance/s.e./CI behind one effect size, how to handle a zero cell or dropouts, or to build the es/se table a meta-analysis runs on.
---

# Effect Sizes for Meta-Analytic Input Tables (Python)

## Purpose
Turn per-study data into one row per study of `effect size + variance + s.e. + 95% CI` - the only object a
meta-analysis can pool. The worked example is the 17-trial haloperidol-versus-placebo Cochrane set, whose
selected effect size is the risk ratio, RR > 1 favouring haloperidol (Masoumi & Shahraz 2022 p2).

## Inputs
- Binary (`CATE`) study: counts `a,b,c,d` = responders/non-responders, treatment vs control, i.e. the
  article's `resp.h, fail.h, resp.p, fail.p`; PythonMeta takes `resp.h, resp.p, T.h, T.p` instead, so
  `T.h = a+b`, `T.p = c+d` (Masoumi & Shahraz 2022 p2).
- Continuous (`CONT`) study: `m1, sd1, n1, m2, sd2, n2` -> MD/SMD family (Masoumi & Shahraz 2022 p2).
- Dropouts `drop.h, drop.p` plus a declared assignment rule. `ALPHA = 0.05`, `Z = 1.959964`.

## Algorithms (all pure numpy/scipy)
### RR, OR (log-OR) and RD from one 2x2, with variance and CI
```python
import numpy as np
from scipy import stats
Z = stats.norm.ppf(1 - 0.05 / 2)
a, b, c, d = 38.0, 62.0, 25.0, 75.0      # illustrative counts on the p2 schema, not the paper's data
p1, p2 = a / (a + b), c / (c + d)
rows = [("RD", p1 - p2, p1 * (1 - p1) / (a + b) + p2 * (1 - p2) / (c + d), None),
        ("RR", np.log(p1 / p2), (1 - p1) / ((a + b) * p1) + (1 - p2) / ((c + d) * p2), np.exp),
        ("OR", np.log((a * d) / (b * c)), 1 / a + 1 / b + 1 / c + 1 / d, np.exp)]
assert np.isclose((a * d) / (b * c), (p1 / (1 - p1)) / (p2 / (1 - p2)))   # cross-product = odds ratio
for name, es, var, back in rows:
    se = float(np.sqrt(var)); lo, hi = es - Z * se, es + Z * se
    nat = es if back is None else float(back(es))
    if back is not None:
        assert np.isclose(nat, np.sqrt(np.exp(lo) * np.exp(hi)))           # CI log-symmetric
        lo, hi = back(lo), back(hi)
    print(f"RESULT {name}={nat:.4f} scale_es={es:+.4f} var={var:.6f} se={se:.4f} ci95=[{lo:.4f},{hi:.4f}]")
```
RD sits on the natural scale; the RR and OR variances are the delta-method (Woolf) ones on the log scale, so
those CIs are asymmetric naturally but geometrically symmetric. Keep `scale_es` - downstream weights use `1/var`
on the log scale, never on the printed ratio, and the RR/OR "no effect" value is 1 while RD's is 0.

### Zero cells and dropouts: what enters the 2x2
```python
import numpy as np
from scipy import stats
Z = stats.norm.ppf(1 - 0.05 / 2)
def logrr(a, b, c, d):
    p1, p2 = np.float64(a) / (a + b), np.float64(c) / (c + d)   # np floats: a zero cell gives inf, not a raise
    return np.log(p1 / p2), (1 - p1) / ((a + b) * p1) + (1 - p2) / ((c + d) * p2)
h, hf, hd, q, qf, qd = 22.0, 28.0, 6.0, 14.0, 36.0, 10.0   # resp.h fail.h drop.h resp.p fail.p drop.p
lab = {"ACS": (h, hf, q, qf), "ICA-0": (h, hf + hd, q, qf + qd), "ICA-1": (h + hd, hf, q + qd, qf),
       "ICA-b": (h + hd, hf, q, qf + qd), "ICA-w": (h, hf + hd, q + qd, qf)}
for k, cells in lab.items():
    es, var = logrr(*cells); se = np.sqrt(var)
    print(f"RESULT {k} RR={np.exp(es):.3f} se={se:.4f} ci95=[{np.exp(es - Z * se):.3f},{np.exp(es + Z * se):.3f}]")
with np.errstate(divide="ignore"):
    raw_es, raw_var = logrr(0.0, 30.0, 12.0, 28.0)          # a zero cell: log RR -inf, var inf
assert np.isneginf(raw_es) and np.isinf(raw_var)
es, var = logrr(0.5, 30.5, 12.5, 28.5); se = np.sqrt(var)    # 0.5 added to all four cells
print(f"RESULT zero_cell_raw={raw_es} cc_logRR={es:.4f} cc_RR={np.exp(es):.4f} cc_se={se:.4f} cc_ci95=[{np.exp(es - Z * se):.4f},{np.exp(es + Z * se):.4f}]")
```
`ACS` keeps observed outcomes only; `ICA-0`/`ICA-1` assume none/all of the dropouts had the event; `ICA-b`/`ICA-w` are
the best/worst case (all experimental dropouts respond and none of the control's, then the reverse) (Masoumi & Shahraz
2022 p3). Each label is a different 2x2, so a different RR - say which one you report and never drop a zero-cell study.

### Reproducing the article's printed per-study log-RR s.e. and IV weights (Table 2, p7)
```python
import numpy as np
from scipy import stats
TAB = ("Arvanitis1997 1.42 0.89 2.25 18.86;Beasley1996 1.05 0.73 1.50 31.22;Bechelli1983 6.21 1.52 25.35 2.05;"
       "Borison1992 7.00 0.40 122.45 0.49;Chouinard1993 3.49 1.11 10.95 3.10;Durost1964 8.68 1.26 59.95 1.09;"
       "Garry1962 1.75 0.58 5.24 3.37;Howard1974 2.04 0.67 6.21 3.27;Marder1994 1.36 0.75 2.47 11.37;"
       "Nishikawa1982 3.00 0.14 65.91 0.42;Nishikawa1984 9.20 0.58 145.77 0.53;Reschke1974 3.79 1.06 13.60 2.48;"
       "Selman1976 1.48 0.94 2.35 19.11;Serafetinides1972 8.40 0.50 142.28 0.51;Simpson1967 2.35 0.13 43.53 0.48;"
       "Spencer1992 11.00 1.67 72.40 1.14;Vichaiya1971 19.00 1.16 311.97 0.52")   # study RR lo hi IV weight %
R = np.array([[float(x) for x in t.split()[1:]] for t in TAB.split(";")])
nm = [t.split()[0] for t in TAB.split(";")]
rr, lo, hi, w_pr = R[:, 0], R[:, 1], R[:, 2], R[:, 3]
Z = stats.norm.ppf(1 - 0.05 / 2)
se = (np.log(hi) - np.log(lo)) / (2 * Z)                    # s.e. read back out of the printed CI width
rl, rh = np.exp(np.log(rr) - Z * se), np.exp(np.log(rr) + Z * se)
assert np.max(np.abs(rl - lo) / lo) < 0.02 and np.max(np.abs(rh - hi) / hi) < 0.02   # bounds reproduce
w = 1 / se ** 2
dev = 100 * w / w.sum() - w_pr                               # IV weight share vs the printed column
assert np.abs(dev).max() < 0.20                              # tolerance: 0.20 weight points
k = int(np.argmax(np.abs(dev)))
print(f"RESULT studies={len(se)} se_logRR=[{se.min():.4f},{se.max():.4f}] max_abs_weight_dev_pp={abs(dev[k]):.3f} worst={nm[k]}")
print(f"RESULT check {nm[1]} RR={rr[1]:.2f} logRR={np.log(rr[1]):+.4f} se={se[1]:.4f} ci95=[{rl[1]:.2f},{rh[1]:.2f}] weight={w_pr[1] + dev[1]:.2f} printed={w_pr[1]}")
```
The article prints RR (95% CI) and the fixed-effect IV weight shares but never the s.e. (Masoumi & Shahraz 2022
p7). Inverting the interval, `se(log RR) = (ln hi - ln lo)/(2 x 1.96)`, rebuilds every printed bound to within 2%,
and `100 x (1/se^2)/sum(1/se^2)` reproduces the weight column to 0.149 points (worst row Selman1976: 19.11 printed
vs 19.26 recomputed; Beasley1996: 31.22 vs 31.18) - residual is 2-dp rounding of the bounds, not a formula gap. The
17 raw 2x2 tables sit in a supplementary file, so the printed RRs are used as given and never re-fitted into counts.

### Continuous arms: mean difference, Cohen d, Hedges g with variance and CI
```python
import numpy as np
from scipy import stats
rng = np.random.default_rng(42)
n1, n2 = 45, 38
x1 = rng.normal(1.20, 1.0, n1); x2 = rng.normal(0.55, 1.0, n2)
m1, m2, s1, s2 = x1.mean(), x2.mean(), x1.std(ddof=1), x2.std(ddof=1)
df = n1 + n2 - 2
sp = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / df)      # pooled sd
md, d = m1 - m2, (m1 - m2) / sp
J = 1 - 3 / (4 * df - 1)                                          # small-sample bias correction
v_d = (n1 + n2) / (n1 * n2) + d ** 2 / (2 * (n1 + n2))
g, v_g, se_g = d * J, J ** 2 * v_d, J * np.sqrt(v_d)
Zt = stats.t.ppf(1 - 0.05 / 2, df)
assert g < d and np.isclose(v_g, se_g ** 2)
print(f"RESULT sp={sp:.4f} MD={md:.4f} cohend={d:.4f} J={J:.5f} hedges_g={g:.4f}")
print(f"RESULT var_d={v_d:.6f} var_g={v_g:.6f} se_g={se_g:.4f} g_ci95_t=[{g - Zt * se_g:.4f},{g + Zt * se_g:.4f}] g_ci95_z=[{g - 1.959964 * se_g:.4f},{g + 1.959964 * se_g:.4f}]")
```
PythonMeta offers MD and SMD for `CONT` data (Masoumi & Shahraz 2022 p2); carry g forward, since d is
biased upward at small `df`. Here "no effect" is 0, and both arms must score in the same direction.

## Output contract
One row per study, never pooled: `study`, `measure` (`RR|OR|RD|MD|SMD_g|SMD_d`), `scale` (`log|natural`), `es`,
`var`, `se`, `ci_lo`, `ci_hi`, `weight_iv = 1/var`, `correction` (`none|cc0.5`), `imputation` (`ACS|ICA-0|ICA-1|ICA-b|ICA-w`).
Print each quantity as `RESULT name=value`; write any tabulated output only inside a `tempfile` directory.

## Validity conditions
- Integer counts per arm, event vs non-event, responder label fixed first: RR > 1 means haloperidol is
  better only if `resp.h`/`resp.p` count clinical improvement (Masoumi & Shahraz 2022 p2).
- Every cell > 0 for the log-scale measures, else a flagged 0.5 correction; RD stays finite at a zero cell.
- Declare the dropout rule before computing - the article re-runs the whole model under each assumption
  (Masoumi & Shahraz 2022 p3); reassigned dropouts change the effect size, they do not clean the data.
- These variances are large-sample approximations; sparse cells give wildly asymmetric intervals, as the
  printed 0.14-65.91 around a RR of 3.00 shows (Masoumi & Shahraz 2022 p7).
- SMD needs a per-arm SD and `df >= 2`; MD needs both arms on one measurement scale.

## Reporting conventions
Quote each study as `RR (lo-hi)` at 2 dp with the measure and scale named, the way Table 2 does (Masoumi &
Shahraz 2022 p7), and put the imputation label plus any correction in the same line. Ratios are reported on
the natural scale, never in log units; a CI covering 1 (RR/OR) or 0 (RD) is "no effect", not "no data".

## Out of scope
Pooling: the MH/IV/DL fixed and random estimates and their summary rows (text p3 gives FE 2.09, CI
1.69-2.59 and RE 2.28, CI 1.54-3.37, Table 2 p7 gives fixed 1.57 (1.28-1.92) and random 2.09 (1.49-2.92) -
different methods and data subsets, so the text's pooled CI does not follow from the table and is reported
as a source inconsistency). Also heterogeneity (Q, I2, tau2), subgroup and meta-regression analysis,
Egger's test and funnel plots (Table 1 p6), forest plots, imputation beyond building the 2x2 - sibling
skills own each of those.

## Conventions
- `pythonmeta` and `zepid`, the article's two packages, are not installed: every effect size, variance,
  s.e. and CI above is reimplemented from first formulas in `numpy`/`scipy.stats`.
- `ALPHA = 0.05`, `Z = stats.norm.ppf(0.975) = 1.959964`; the article prints 95% CIs throughout
  (Masoumi & Shahraz 2022 p7). Ratios are computed on the log scale, back-transformed only to print.
- Demo 2x2 counts are illustrative and labelled as such; the only source numbers quoted are Table 2's
  printed RR/CI/weights. Continuous demos use `numpy.random.default_rng(42)`; nothing is hardcoded.
- The article names the measures (OR/RR/RD, MD/SMD) but prints no variance formulas (p2); the ones used
  here are the standard Woolf/delta-method forms, checked against its own printed weight column.
- Source: Masoumi & Shahraz, *Meta-analysis using Python: a hands-on tutorial*, BMC Med Res Methodol
  2022;22:193, pp.1-8.
