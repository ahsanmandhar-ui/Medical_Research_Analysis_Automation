---
name: maa-publication-bias
description: Detect publication bias and small-study effects in a meta-analysis in pure Python - contour-enhanced funnel plot (matplotlib zones from scipy.stats.norm), Egger's regression as a statsmodels WLS of effect on standard error with weights 1/se^2, Begg rank correlation (scipy.stats.kendalltau) and a Duval & Tweedie trim-and-fill iterative loop, each asserted on seeded synthetic data with injected asymmetry, plus a reproduction of the haloperidol Egger Table 1 of Masoumi & Shahraz (BMC Med Res Methodol 2022;22:193); pythonmeta and zepid are unavailable and are re-implemented. Use when a funnel plot looks asymmetric, small studies seem to exaggerate the pooled effect, Egger or Begg P-values, a contour-enhanced funnel, or a trim-and-fill adjusted pooled estimate are requested.
---

# Publication Bias and Small-Study Effects in Meta-Analysis (Python)

## Purpose
Diagnose whether small studies systematically differ from large ones - the small-study-effect /
publication-bias step of the source workflow (Masoumi & Shahraz 2022 p3-5).

## Inputs
- `y`: per-study effects on a log scale (log RR here; RR above 1 favours treatment) (Masoumi & Shahraz 2022 p2).
- `se`: per-study standard errors, same length. From published CIs: `se = (log hi - log lo) / (2*1.96)`.
- `ALPHA = 0.05` (two-sided), `SEED = 42` via `numpy.random.default_rng`, contour cuts at p = 0.001 / 0.01 / 0.05.
- Worked example: 17 haloperidol-vs-placebo trials (p2); Table 2's per-study RRs with 95% CIs (p6-7) rebuild `(y, se)`.

## Algorithms (all pure numpy/scipy)

### Contour-enhanced funnel plot (matplotlib, zones via scipy.stats.norm)
```python
import numpy as np, os, tempfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
TRUE = 0.45
rng = np.random.default_rng(42)
se = rng.uniform(0.08, 0.45, 40); y = TRUE + rng.normal(0.0, 1.0, 40) * se
p_all = 2.0 * stats.norm.sf(np.abs(y) / se)
supp = int(np.sum((se > 0.25) & (p_all >= 0.05)))                        # planted: unpublished small studies
keep = ~((se > 0.25) & (p_all >= 0.05)); se_o, y_o = se[keep], y[keep]
w = 1.0 / se_o**2; est = float((y_o * w).sum() / w.sum())                # IV pooled of the surviving set
gap = int(np.sum((se_o > 0.25) & (2.0 * stats.norm.sf(np.abs(y_o) / se_o) >= 0.05)))
S, Y = np.meshgrid(np.linspace(0.50, 0.05, 240), np.linspace(-0.6, 2.2, 240))
C = 2.0 * stats.norm.sf(np.abs(Y) / S)                                   # two-sided p vs no-effect per pixel
fig, ax = plt.subplots(figsize=(6.2, 5.0), dpi=110)
ax.contourf(S, Y, C, levels=[0.0, 0.001, 0.01, 0.05, 1.0], colors=["#023858", "#3c8dc4", "#a9d3ec", "#f2f2f2"], alpha=0.9)
ax.scatter(y_o, se_o, c="k", s=22, zorder=3)
ax.axvline(0.0, c="gray"); ax.axvline(est, ls="--", c="crimson", zorder=2)
ax.set_ylim(ax.get_ylim()[::-1]); ax.set_xlabel("log effect size"); ax.set_ylabel("standard error")
assert supp >= 4 and gap == 0 and est > TRUE + 0.03                       # planted hole + inflated pooled line
with tempfile.TemporaryDirectory() as td:
    png = os.path.join(td, "funnel.png"); fig.savefig(png, bbox_inches="tight"); plt.close(fig)
    assert os.path.isfile(png) and os.path.getsize(png) > 1000
    print(f"RESULT png_bytes={os.path.getsize(png)} n_observed={len(y_o)}_of_40 pooled_iv={est:.4f} true={TRUE}")
    print(f"RESULT planted_suppressed={supp} contour_gap_small_nonsig={gap} asymmetry_detection=PASS")
```
Funnel: `y` vs `se`, inverted SE axis, dashed vertical line at the fixed-effect pooled log RR as in Fig. 5
(p6). Contours demark zones of statistical significance, helping discern asymmetry due to publication bias
(p5): a hole in the non-significant small-study zone (planted here) suggests suppression (p4-5).

### Egger's regression test (statsmodels WLS of `y` on `se`, weights `1/se**2`)
```python
import numpy as np, statsmodels.api as sm
rng = np.random.default_rng(42)
se = rng.uniform(0.08, 0.45, 25)
y = 0.45 + 1.50 * (se - se.min()) + rng.normal(0.0, 0.6, 25) * se         # injected small-study inflation
res = sm.WLS(y, sm.add_constant(se), weights=1.0 / se**2).fit()
b0, b1 = res.params
o = sm.OLS(y / se, sm.add_constant(1.0 / se)).fit()                        # standardised effect on precision
L = np.linalg.lstsq(np.column_stack([np.ones(25), 1.0 / se]), y / se, rcond=None)[0]
assert b1 > 1.0 and res.pvalues[1] < 0.05 and abs(o.params[0] - b1) < 1e-8 and np.allclose(L, o.params, atol=1e-9)
print(f"RESULT egger_bias={b1:.4f} se={res.bse[1]:.4f} t={res.tvalues[1]:.2f} P={res.pvalues[1]:.3g} injected=1.50 detection=PASS")
print(f"RESULT egger_intercept={b0:.4f} se={res.bse[0]:.4f} P={res.pvalues[0]:.3g}")
```
Egger's test asks whether the association between estimated effects and study size is greater than chance,
applied through a statsmodels linear regression (Masoumi & Shahraz 2022 p3). WLS of `y` on `se` with weights `1/se**2` is algebraically identical to regressing standardised effect `y/se` on precision `1/se` - asserted.

### Haloperidol worked example: Egger's Table 1 reproduced
```python
import numpy as np, statsmodels.api as sm
rr = [1.42,1.05,6.21,7.00,3.49,8.68,1.75,2.04,1.36,3.00,9.20,3.79,1.48,8.40,2.35,11.00,19.00]
lo = [0.89,0.73,1.52,0.40,1.11,1.26,0.58,0.67,0.75,0.14,0.58,1.06,0.94,0.50,0.13,1.67,1.16]
hi = [2.25,1.50,25.35,122.45,10.95,59.95,5.24,6.21,2.47,65.91,145.77,13.60,2.35,142.28,43.53,72.40,311.97]
y = np.log(rr); se = (np.log(hi) - np.log(lo)) / (2 * 1.959963985)         # Table 2 rows, Python column (p6-7)
res = sm.WLS(y, sm.add_constant(se), weights=1.0 / se**2).fit()
assert abs(res.params[0] + 0.1463) < 0.01 and abs(res.params[1] - 1.7894) < 0.01
assert abs(res.bse[0] - 0.109) < 0.01 and abs(res.bse[1] - 0.257) < 0.01 and res.pvalues[1] < 0.001
print(f"RESULT intercept={res.params[0]:.4f} t={res.tvalues[0]:.3f} P={res.pvalues[0]:.3f} ci_low={res.conf_int()[0][0]:.3f} ci_high={res.conf_int()[0][1]:.3f} (paper -0.1463, -1.342, 0.200, -0.379 to 0.086)")
print(f"RESULT bias={res.params[1]:.4f} t={res.tvalues[1]:.3f} P={res.pvalues[1]:.2g} (paper 1.7894, 6.950, 0.000)")
print(f"RESULT iv_fixed_pooled_logRR={(y/se**2).sum()/(1/se**2).sum():.4f} RR={np.exp((y/se**2).sum()/(1/se**2).sum()):.2f} (Table 2 Fixed row 1.57, p7)")
```
Table 1 prints intercept -0.1463 (95% CI -0.379 to 0.086, P = 0.200) and Bias 1.7894 (P < 0.001)
(Masoumi & Shahraz 2022 p6); the block above reproduces both to within 0.01 despite the 2-3-digit CIs.
Note two source defects kept visible, not fudged: the p5 text says the intercept CI "does not include zero"
although Table 1's interval does (P = 0.200), and the narrative pooled RR of 2.09 (p3) is the Mantel-Haenszel
fixed-effect estimate while IV pooling of the Table 2 rows gives 1.57, matching the Table 2 "Fixed" row (p7).

### Begg's rank-correlation test (scipy.stats.kendalltau)
```python
import numpy as np
from scipy import stats
rng = np.random.default_rng(42)
se = rng.uniform(0.08, 0.45, 25)
y = 0.45 + 1.50 * (se - se.min()) + rng.normal(0.0, 0.6, 25) * se          # same injected asymmetry
w = 1.0 / se**2; fe = float((y * w).sum() / w.sum()); v = 1.0 / w.sum()
adj = (y - fe) / np.sqrt(np.maximum(se**2 - v, 1e-12))                     # Begg-Mazumdar adjusted effects
tau, p = stats.kendalltau(adj, se**2)
assert tau > 0.3 and p < 0.05
print(f"RESULT begg_tau={tau:.4f} P={p:.3g} n=25 fe_pooled={fe:.4f} detection=PASS injected_bias=1.50")
```
Complement to Egger: Kendall's tau between adjusted effects and variances; a positive significant tau means
small studies carry larger effects, the signal the paper reads off the asymmetric funnel (p3-4).

### Trim-and-fill sensitivity analysis (Duval & Tweedie L0, iterative)
```python
import numpy as np
from scipy import stats
rng = np.random.default_rng(42)
se = rng.uniform(0.08, 0.50, 30); y = 0.55 + rng.normal(0.0, 1.0, 30) * se
keep = ~((se > 0.30) & (y < 0.45)); se, y = se[keep], y[keep]              # planted suppression, 7 dropped
n_drop = 30 - len(y); w = 1.0 / se**2; obs = float((y * w).sum() / w.sum())
def trim_fill(y, se, iters=50):
    w = 1.0 / se**2; est = float((y * w).sum() / w.sum()); n = len(y); k = 0
    for _ in range(iters):                                                  # iterate until k0 stabilises
        d = y - est; Tn = float(stats.rankdata(np.abs(d))[d > 0].sum())    # rank sum of positive side
        k_new = max(0, int(np.ceil((4.0 * Tn - n * (n + 1)) / (2.0 * n - 1))))
        if k_new == k: break
        k = k_new; idx = np.argsort(d)[: n - k]                             # trim k most extreme positive
        est = float((y[idx] * w[idx]).sum() / w[idx].sum())
    d = y - est; t_idx = np.argsort(d)[-k:]                                 # fill: mirror them about est
    ay, aSe = np.concatenate([y, 2.0 * est - y[t_idx]]), np.concatenate([se, se[t_idx]]); aw = 1.0 / aSe**2
    return k, est, float((ay * aw).sum() / aw.sum())
k, est_trim, adj = trim_fill(y, se)
assert k >= 1 and abs(k - n_drop) <= 3 and adj < obs - 0.01
print(f"RESULT planted_suppressed={n_drop} k0_imputed={k} n_observed={len(y)} trimmed_center={est_trim:.4f}")
print(f"RESULT observed_pooled={obs:.4f} adjusted_pooled={adj:.4f} reduction={obs - adj:.4f} detection=PASS")
```
Trim-fill is "yet to be developed in Python" (Masoumi & Shahraz 2022 p5), so this is an explicit Duval &
Tweedie loop: estimate the pooled centre, count the rank-based excess (L0) on the over-represented side,
trim, re-estimate until `k0` stops changing, mirror the trimmed studies and re-pool. Detection means `k0`
recovers the planted drop count and the adjusted pooled moves back toward the truth (0.55).

## Output contract
Each block prints `RESULT name=value` lines and writes at most one headless PNG into a private temp dir.
Fields: `png_bytes`, `n_observed`, `pooled_iv`, `planted_suppressed`, `contour_gap_small_nonsig`, `egger_bias`,
`egger_intercept`, `t`, `P`, `begg_tau`, `k0_imputed`, `trimmed_center`, `observed_pooled`, `adjusted_pooled`;
report each test P against ALPHA and, for trim-and-fill, naive vs adjusted pooled side by side.

## Validity conditions
- Studies plotted as effect vs SE on one common scale; log RR here, RR > 1 favouring treatment (Masoumi & Shahraz 2022 p2).
- Symmetric scatter about the pooled line = no small-study effect; asymmetry is "greater than expected by chance" only when the test P < ALPHA (p3).
- Contour zones classify studies by their own significance; small studies sitting in significant zones point to mechanisms other than publication bias (p5).
- Funnel asymmetry is not proof: clinical heterogeneity remains an unchecked source of variability (p6).

## Out of scope
Effect-size construction from counts/means (sibling `maa-effect-sizes`), fixed/random pooling and
Mantel-Haenszel (sibling `maa-pooling-models`), Q / I-squared / tau-squared (sibling `maa-heterogeneity`)
and forest plots (sibling `maa-forest-plots`). The inverse-variance pooled value used here is only the funnel's
reference line. Missing-data imputations and subgroup analyses of the source paper are also not covered (p3).

## Conventions
- ALPHA = 0.05 two-sided; all synthetic data from `numpy.random.default_rng(42)`; never invent a value - compute it and cite the page.
- `pythonmeta` and `zepid` are not installed and are never imported: Egger is `statsmodels.WLS` (the paper's own statsmodels route, p3), Begg is `scipy.stats.kendalltau`, trim-and-fill and contours are inline reimplementations; funnel drawn with matplotlib because PythonMeta's plot is unavailable (p1-2).
- Figures are headless: `matplotlib.use("Agg")` before pyplot, savefig only inside `tempfile.TemporaryDirectory()`, PNG existence and size asserted.
- Egger convention matches Table 1: `sm.WLS(y, add_constant(se), weights=1/se**2)`, so the "Bias" row is the `se` coefficient and the "Intercept" row is the constant; CI-derived SEs use z = 1.959963985 on log scale.
- Cites use the article's PDF pagination (pages 1-9); printed values quoted to the paper's digits, reproduction tolerance 0.01.
