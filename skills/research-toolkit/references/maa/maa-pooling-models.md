---
name: maa-pooling-models
description: Fixed- and random-effects pooling of meta-analytic evidence in pure numpy/scipy - inverse-variance fixed-effect weights, Mantel-Haenszel and Peto pooling for binary counts, DerSimonian-Laird / Paule-Mandel / REML between-study tau-squared, Cochran's Q, the pooled estimate and its 95% CI, plus a rule for choosing fixed vs random with justification; reproduces the haloperidol Cochrane example. pythonmeta is documented as the drop-in. Use when asked to combine several study effect sizes into one pooled result, whether to use a fixed or random effects model, what each study's weight is, how tau-squared is estimated, or how to get a Mantel-Haenszel / Peto pooled risk without R or STATA.
---

# Meta-Analysis Pooling Models (Python)

## Purpose
Pooling models turn k per-study effect measures `yi` with variances `vi` into one summary estimate,
a fixed-effect (one common effect) or random-effects (effects vary by study) weighted mean, its 95% CI,
the study weights and the between-study variance tau-squared (Masoumi & Shahraz 2022 p2). Effect-size
computation, heterogeneity statistics beyond Q, publication bias and forest plots are sibling skills.

## Inputs
- `yi`: per-study effect on the pooling scale (log RR / log OR / SMD), length k; `vi`: its variance.
- Binary counts (MH / Peto): `a`, `n1` (treatment events / total) and `c`, `n0` (control events / total).
- Only published RR with 95% CI is available on the face of most tables, so `vi` is rebuilt from the CI
  half-width: `vi = ((ln(hi)-ln(lo))/(2*zc))**2` (Masoumi & Shahraz 2022 p7, Table 2 columns).
- `ALPHA = 0.05`; `zc = scipy.stats.norm.ppf(0.975)`.

## Algorithms (all pure numpy/scipy)
### Reconstruct vi; inverse-variance fixed effect and weights
Fixed-effect weights are `wi = 1/vi`; the pooled estimate is the weighted mean and its SE is
`sqrt(1/sum(wi))`. Haloperidol benchmark: 17 Cochrane trials; fixed-effect RR is reported as 1.57
(95% CI 1.28-1.92) and study 1/2 weights as 18.86% / 31.22% (Masoumi & Shahraz 2022 p7).
```python
import numpy as np
from scipy import stats
zc = stats.norm.ppf(0.975)
# 17 haloperidol trials, per-study RR (95% CI) as tabulated (Masoumi & Shahraz 2022 p7)
rr = np.array([1.42,1.05,6.21,7.00,3.49,8.68,1.75,2.04,1.36,3.00,9.20,3.79,1.48,8.40,2.35,11.00,19.00])
lo = np.array([0.89,0.73,1.52,0.40,1.11,1.26,0.58,0.67,0.75,0.14,0.58,1.06,0.94,0.50,0.13,1.67,1.16])
hi = np.array([2.25,1.50,25.35,122.45,10.95,59.95,5.24,6.21,2.47,65.91,145.77,13.60,2.35,142.28,43.53,72.40,311.97])
yi = np.log(rr)                                     # pool on the log scale
vi = ((np.log(hi) - np.log(lo)) / (2 * zc)) ** 2    # variance rebuilt from the CI half-width
wi = 1.0 / vi                                       # inverse-variance fixed-effect weights
fe = (wi * yi).sum() / wi.sum()                     # fixed-effect pooled log RR
se = np.sqrt(1.0 / wi.sum())
ci = np.exp([fe - zc * se, fe + zc * se])
wt = wi / wi.sum() * 100
assert abs(np.exp(fe) - 1.57) < 0.03 and abs(ci[0] - 1.28) < 0.03 and abs(ci[1] - 1.92) < 0.03
assert abs(wt[0] - 18.86) < 0.5 and abs(wt[1] - 31.22) < 0.5   # FE % weights (Table 2 p7)
print(f"RESULT k={len(rr)} fixed_effect_RR={np.exp(fe):.3f} ci95=[{ci[0]:.3f},{ci[1]:.3f}] weight1={wt[0]:.2f} weight2={wt[1]:.2f}")
```
### tau-squared (DL / PM / REML), Q, random-effects pool, and model choice
Random effects add tau-squared to every variance: `w*_i = 1/(vi + tau2)`. tau2 is estimated by
DerSimonian-Laird (closed form), Paule-Mandel (root-find) or REML (bounded optimisation) - all
deterministic. Cross-checks: Q 27.29 (df 16), tau2_DL 0.146, random-effects RR 2.09 (95% CI 1.49-2.92)
(Masoumi & Shahraz 2022 p7). The default RE method in pythonmeta is DL (Masoumi & Shahraz 2022 p2).
```python
import numpy as np
from scipy import stats, optimize
zc = stats.norm.ppf(0.975)
rr = np.array([1.42,1.05,6.21,7.00,3.49,8.68,1.75,2.04,1.36,3.00,9.20,3.79,1.48,8.40,2.35,11.00,19.00])
lo = np.array([0.89,0.73,1.52,0.40,1.11,1.26,0.58,0.67,0.75,0.14,0.58,1.06,0.94,0.50,0.13,1.67,1.16])
hi = np.array([2.25,1.50,25.35,122.45,10.95,59.95,5.24,6.21,2.47,65.91,145.77,13.60,2.35,142.28,43.53,72.40,311.97])
yi = np.log(rr); vi = ((np.log(hi)-np.log(lo))/(2*zc))**2; k = len(rr); wi = 1.0/vi
fe = (wi*yi).sum()/wi.sum(); Q = (wi*(yi-fe)**2).sum()             # Cochran's Q, df = k-1
C = wi.sum() - (wi**2).sum()/wi.sum(); tau_dl = max(0.0, (Q-(k-1))/C)
def pm(f):                                                        # Paule-Mandel estimating function
    w = 1.0/(vi+f); mu = (w*yi).sum()/w.sum(); return (w*(yi-mu)**2).sum() - (k-1)
tau_pm = optimize.brentq(pm, 0.0, 10.0) if pm(0.0) > 0 else 0.0
def reml_neg(f):                                                  # restricted -log-likelihood
    f = max(f, 1e-12); w = 1.0/(vi+f); mu = (w*yi).sum()/w.sum()
    return 0.5*(np.log(vi+f).sum() + np.log(w.sum()) + (w*(yi-mu)**2).sum())
tau_reml = optimize.minimize_scalar(reml_neg, bounds=(0.0, 2.0), method="bounded").x
w = 1.0/(vi+tau_dl); mu_dl = (w*yi).sum()/w.sum(); se = np.sqrt(1.0/w.sum())
ci = np.exp([mu_dl - zc*se, mu_dl + zc*se])
assert abs(Q - 27.29) < 0.5 and abs(tau_dl - 0.146) < 0.01 and abs(np.exp(mu_dl) - 2.09) < 0.03
pick = "random" if stats.chi2.sf(Q, k-1) < 0.10 else "fixed"      # heterogeneity at p<0.10 (Masoumi & Shahraz 2022 p3)
print(f"RESULT Q={Q:.2f} df={k-1} p={stats.chi2.sf(Q,k-1):.4f} I2={max(0,(Q-(k-1))/Q)*100:.2f}")
print(f"RESULT tau2_DL={tau_dl:.3f} tau2_PM={tau_pm:.3f} tau2_REML={tau_reml:.3f}")
print(f"RESULT random_DL_RR={np.exp(mu_dl):.3f} ci95=[{ci[0]:.3f},{ci[1]:.3f}] fixed_RR={np.exp(fe):.3f} model_choice={pick}")
```
Fixed RR 1.57 vs random 2.09 with a wider CI (1.49-2.92 vs 1.28-1.92): significant Q plus the authors'
a priori assumption of varying effects justify the random-effects model, which down-weights the huge
studies and lets the small high-RR trials pull the mean up (Masoumi & Shahraz 2022 p2-3).
### Binary pooling: Mantel-Haenszel and Peto variants
pythonmeta's fixed-effect default is Mantel-Haenszel (MH), switchable to Peto or inverse-variance
(Masoumi & Shahraz 2022 p2). MH RR = `sum(a*n0/N)/sum(c*n1/N)`; the log-RR variance keeps only the two
arm variances (arms are independent, so numerator/denominator are uncorrelated). Peto uses the
observed-minus-expected score: `ln OR = sum(O-E)/sum(V)`, valid for rare events and OR near 1. The raw
haloperidol 2x2 counts live in the article's supplementary files, so these are illustrative counts.
```python
import numpy as np
from scipy import stats
from statsmodels.stats.contingency_tables import StratifiedTable
zc = stats.norm.ppf(0.975)
a = np.array([300,250,200]); n1 = np.array([1000,1000,1000])   # treatment events / total
c = np.array([200,170,140]); n0 = np.array([1000,1000,1000])   # control events / total
b = n1 - a; d = n0 - c; N = n1 + n0
U = (a*n0/N).sum(); V = (c*n1/N).sum(); rr_mh = U/V
var_rr = (((n0/N)**2*(a*b/n1)).sum())/U**2 + (((n1/N)**2*(c*d/n0)).sum())/V**2
se = np.sqrt(var_rr); ci = np.exp([np.log(rr_mh)-zc*se, np.log(rr_mh)+zc*se])
a0,c0,n10,n00 = a[0],c[0],n1[0],n0[0]; u0=a0*n00/(n10+n00); v0=c0*n10/(n10+n00)
one = ((n00/(n10+n00))**2*a0*(n10-a0)/n10)/u0**2 + ((n10/(n10+n00))**2*c0*(n00-c0)/n00)/v0**2
assert abs(one - (1/a0-1/n10+1/c0-1/n00)) < 1e-12               # single-stratum = crude lnRR variance
Ro=(a*d/N).sum(); So=(b*c/N).sum(); or_mh=Ro/So
st = StratifiedTable(np.array([[[ai,bi],[ci,di]] for ai,bi,ci,di in zip(a,b,c,d)]).transpose(1,2,0))
assert abs(or_mh - st.oddsratio_pooled) < 1e-9
E = n1*(a+c)/N; Vp = n1*n0*(a+c)*(b+d)/(N*N*(N-1))
ln_or = (a-E).sum()/Vp.sum(); se_p = 1/np.sqrt(Vp.sum()); pct = np.exp([ln_or-zc*se_p, ln_or+zc*se_p])
print(f"RESULT MH_RR={rr_mh:.3f} ci95=[{ci[0]:.3f},{ci[1]:.3f}] MH_OR={or_mh:.3f} statsmodels_OR={st.oddsratio_pooled:.3f}")
print(f"RESULT Peto_OR={np.exp(ln_or):.3f} ci95=[{pct[0]:.3f},{pct[1]:.3f}]")
```

## Output contract
One `RESULT name=value` line per block: `fixed_effect_RR`/`ci95`/`weight*`, `Q`/`df`/`p`/`I2`,
`tau2_DL`/`tau2_PM`/`tau2_REML`, `random_DL_RR`/`ci95`/`model_choice`, and `MH_RR`/`MH_OR`/`Peto_OR`
with intervals. All on the ratio scale (exponentiated); SEs, CIs and weights are the reported quantities.

## Validity conditions
- Fixed effect is only meaningful if the studies truly share one effect; otherwise it understates the CI
  width, which the random model widens by adding tau2 (Masoumi & Shahraz 2022 p2).
- Pool RRs/ORs on the log scale and exponentiate - never average the raw ratios.
- Peto is biased toward the null when the OR is far from 1 or events are common; prefer MH or IV there.
- A 2x2 MH/Peto stratum needs all four arm totals positive; zero-event studies must be handled by the
  continuity-constant policy, not silently dropped.

## Conventions
- ALPHA=0.05; `zc = norm.ppf(0.975)`. Fixed-effect weight `1/vi`; random weight `1/(vi+tau2)`.
- `pythonmeta` (PythonMeta) and `zepid` are NOT installed and must never be pip-installed; every estimator
  is reimplemented in numpy/scipy/statsmodels so the numbers reproduce. When present, pythonmeta is the
  drop-in: fixed default Mantel-Haenszel (options `IV`, `Peto`), random default DerSimonian-Laird
  (Masoumi & Shahraz 2022 p2). statsmodels `StratifiedTable` is the MH-OR cross-check oracle (tol 1e-9).
- Paule-Mandel solved by `optimize.brentq`; REML by `optimize.minimize_scalar(method="bounded")`; both
  deterministic and converge in milliseconds on k=17. Never invent a value - compute it and cite the page.

## Out of scope
Computing per-study effect sizes from raw data (sibling effect-sizes), I2/tau inference and prediction
intervals (heterogeneity), Egger's test / funnel plots / trim-fill (publication bias), forest plots, and
subgroup or meta-regression models - this skill only pools a supplied set of effect estimates.
