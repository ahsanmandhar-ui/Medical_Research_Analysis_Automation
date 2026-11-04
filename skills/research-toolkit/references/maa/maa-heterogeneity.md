---
name: maa-heterogeneity
description: Heterogeneity statistics for a meta-analysis computed from first formulas in pure Python (numpy/scipy) - Cochran's Q with chi-square P, I-squared in every variant ((Q-df)/Q, tau2/(tau2+typical-variance), 1-1/H2), DerSimonian-Laird tau-squared, H-squared, and the Higgins-Thompson-Spiegelhalter prediction interval, plus the fixed/random inverse-variance pooled log-effects and their Z tests. Works on per-study effect sizes and standard errors (log RR/OR/HR or SMD). Use when asked to test or quantify between-study heterogeneity, report Q and its P value, compute I-squared, tau-squared, H-squared, or build a prediction interval, and decide whether a random-effects model is warranted. NOT for computing effect sizes/weights, choosing the pooling estimator, publication bias (Egger/funnel) or drawing forest plots.
---

# Meta-Analysis Heterogeneity Statistics (Python)

## Purpose
Given per-study effect sizes and their standard errors, decide whether the studies share one true effect:
run Cochran's Q, express inconsistency as I-squared (all algebraic forms), estimate the between-study
variance tau-squared by DerSimonian-Laird, report H-squared, and give the prediction interval that an
individual future study is expected to fall inside (Masoumi & Shahraz 2022 p3).

## Inputs
- `yi`: 1-D array of per-study effect sizes on the LOG/additive scale (log RR, log OR, log HR, or SMD);
  never the raw ratio.
- `sei`: 1-D array of each study's standard error (> 0). `v_i = sei**2`.
- `CONF = 0.95`, `ALPHA = 0.05`; `df = k - 1` with `k = len(yi)`.
- Worked source data: the haloperidol-vs-placebo log-RR table (17 trials) reconstructed from the printed
  RR and 95% CI, `sei = (log(hi)-log(lo)) / (2*1.96)`, as reported for the Python column (Masoumi & Shahraz 2022 p7, Table 2).

## Algorithms (all pure numpy/scipy)

### Effect sizes from printed RRs, then Q, P, tau2, I2 (all variants), H2 (Masoumi & Shahraz 2022 p3, p7)
```python
import numpy as np
from scipy import stats
CONF = 0.95
# (label, RR, CI_lo, CI_hi) for the 17 haloperidol trials, Python column, Table 2 (p7)
tab = [("Arvanitis1997",1.42,0.89,2.25),("Beasley1996",1.05,0.73,1.50),
("Bechelli1983",6.21,1.52,25.35),("Borison1992",7.00,0.40,122.45),
("Chouinard1993",3.49,1.11,10.95),("Durost1964",8.68,1.26,59.95),
("Garry1962",1.75,0.58,5.24),("Howard1974",2.04,0.67,6.21),
("Marder1994",1.36,0.75,2.47),("Nishikawa1982",3.00,0.14,65.91),
("Nishikawa1984",9.20,0.58,145.77),("Reschke1974",3.79,1.06,13.60),
("Selman1976",1.48,0.94,2.35),("Serafetinides1972",8.40,0.50,142.28),
("Simpson1967",2.35,0.13,43.53),("Spencer1992",11.00,1.67,72.40),
("Vichaiya1971",19.00,1.16,311.97)]
yi = np.array([np.log(r[1]) for r in tab])                       # log effect size
zc = stats.norm.ppf(1 - (1 - CONF) / 2)
sei = np.array([(np.log(r[3]) - np.log(r[2])) / (2 * zc) for r in tab])  # SE back from the CI
vi = sei ** 2; w = 1 / vi; k = len(yi); df = k - 1               # inverse-variance weights
y_fe = (w * yi).sum() / w.sum()                                  # fixed-effect (IV) pooled log effect
Q = float((w * (yi - y_fe) ** 2).sum())                          # Cochran's Q (Higgins)
p_Q = float(stats.chi2.sf(Q, df))                                # Q ~ chi-square with k-1 df
C = w.sum() - (w ** 2).sum() / w.sum()                           # DL denominator
tau2 = max(0.0, (Q - df) / C)                                    # DerSimonian-Laird between-study var
s2 = df / C                                                      # Higgins "typical" within-study variance
I2_q = max(0.0, (Q - df) / Q) * 100                             # variant 1: (Q-df)/Q
I2_tau = tau2 / (tau2 + s2) * 100                               # variant 2: tau2/(tau2+typical)
H2 = Q / df; I2_h = max(0.0, 1 - 1 / H2) * 100                 # variant 3 via H2 = Q/(k-1)
print(f"RESULT k={k} df={df} y_fe={y_fe:.4f} RR_FE={np.exp(y_fe):.3f}")
print(f"RESULT Q={Q:.2f} p_Q={p_Q:.4f} tau2_DL={tau2:.4f} H2={H2:.4f} sqrt_H2={np.sqrt(H2):.4f}")
print(f"RESULT I2_from_Q={I2_q:.2f} I2_from_tau2={I2_tau:.2f} I2_from_H2={I2_h:.2f}")
# Assert against the article's own reported Python statistics (Table 2, p7): Q=27.29 I2=41.37 tau2=0.146.
assert abs(Q - 27.29) < 0.05, Q                                  # 27.27 reproduced (CI rounding)
assert abs(I2_q - 41.37) < 0.10, I2_q                           # 41.32 reproduced
assert abs(tau2 - 0.146) < 0.001, tau2                          # 0.1458 reproduced
assert abs(I2_q - I2_tau) < 0.05 and abs(I2_q - I2_h) < 0.05    # all three I2 forms agree
print("RESULT assert_Table2 OK")
```
Prints `Q=27.27 p_Q=0.0386 tau2_DL=0.1458 H2=1.7041`, `I2_from_Q=41.32 I2_from_tau2=41.31 I2_from_H2=41.32`
against the article's reported `Q = 27.29, P = 0.03, I2 = 41.37%, tau2 = 0.146` (Masoumi & Shahraz 2022 p7); the tiny
gap is CI rounding, tolerance stated in `## Conventions`.

### Fixed/random pooled log effects, Z tests, and the prediction interval (Masoumi & Shahraz 2022 p3)
```python
import numpy as np
from scipy import stats
CONF = 0.95
tab = [("Arvanitis1997",1.42,0.89,2.25),("Beasley1996",1.05,0.73,1.50),
("Bechelli1983",6.21,1.52,25.35),("Borison1992",7.00,0.40,122.45),
("Chouinard1993",3.49,1.11,10.95),("Durost1964",8.68,1.26,59.95),
("Garry1962",1.75,0.58,5.24),("Howard1974",2.04,0.67,6.21),
("Marder1994",1.36,0.75,2.47),("Nishikawa1982",3.00,0.14,65.91),
("Nishikawa1984",9.20,0.58,145.77),("Reschke1974",3.79,1.06,13.60),
("Selman1976",1.48,0.94,2.35),("Serafetinides1972",8.40,0.50,142.28),
("Simpson1967",2.35,0.13,43.53),("Spencer1992",11.00,1.67,72.40),
("Vichaiya1971",19.00,1.16,311.97)]
yi = np.array([np.log(r[1]) for r in tab])
zc = stats.norm.ppf(0.975)
sei = np.array([(np.log(r[3]) - np.log(r[2])) / (2 * zc) for r in tab])
vi = sei ** 2; w = 1 / vi; k = len(yi); df = k - 1
y_fe = (w * yi).sum() / w.sum(); se_fe = np.sqrt(1 / w.sum())
Q = float((w * (yi - y_fe) ** 2).sum()); C = w.sum() - (w ** 2).sum() / w.sum()
tau2 = max(0.0, (Q - df) / C)                                    # DL tau2
wr = 1 / (vi + tau2)                                             # random-effects (DL) weights
mu_re = (wr * yi).sum() / wr.sum(); se_re = np.sqrt(1 / wr.sum())
tcrit = stats.t.ppf(0.975, k - 2)                                # t with k-2 df for the PI
sd_pi = np.sqrt(tau2 + se_re ** 2)                               # PI spread (Higgins et al.)
pi_lo, pi_hi = mu_re - tcrit * sd_pi, mu_re + tcrit * sd_pi
print(f"RESULT RR_FE={np.exp(y_fe):.3f} CI=[{np.exp(y_fe-zc*se_fe):.3f},{np.exp(y_fe+zc*se_fe):.3f}] Z_FE={y_fe/se_fe:.2f}")
print(f"RESULT RR_RE={np.exp(mu_re):.3f} CI=[{np.exp(mu_re-zc*se_re):.3f},{np.exp(mu_re+zc*se_re):.3f}] Z_RE={mu_re/se_re:.2f}")
print(f"RESULT tau2={tau2:.4f} PI_RR=[{np.exp(pi_lo):.3f},{np.exp(pi_hi):.3f}] t_crit_df={k-2}")
# Assert the Python Table 2 pooled estimates (p7): Random 2.09(1.49-2.92), Fixed 1.57(1.28-1.92), Z 4.37-4.27.
assert abs(np.exp(mu_re) - 2.09) < 0.01 and abs(np.exp(y_fe) - 1.57) < 0.01
assert abs(mu_re / se_re - 4.27) < 0.02 and abs(y_fe / se_fe - 4.37) < 0.02
print("RESULT assert_pooled OK")
```
Prints `RR_RE=2.085 CI=[1.488,2.920] Z_RE=4.27`, `RR_FE=1.567 CI=[1.283,1.918] Z_FE=4.38`, matching Table 2 (p7).
The prediction interval here is `[0.854, 5.089]`; the article's Fig-1 printout quotes a wider `0.73-7.17` from a
different DL printout (RE 2.28, tau2 approx 0.247) - see `## Validity conditions` (Masoumi & Shahraz 2022 p3).

### Generic seeded demo: heterogeneity does not depend on source data
```python
import numpy as np
from scipy import stats
rng = np.random.default_rng(42)
k = 12
yi = rng.normal(0.40, 0.30, k)                                    # log-ORs scattered about 0.4
sei = rng.uniform(0.12, 0.35, k)
vi = sei ** 2; w = 1 / vi; df = k - 1
y_fe = (w * yi).sum() / w.sum()
Q = float((w * (yi - y_fe) ** 2).sum()); p_Q = float(stats.chi2.sf(Q, df))
C = w.sum() - (w ** 2).sum() / w.sum()
tau2 = max(0.0, (Q - df) / C); s2 = df / C
I2 = max(0.0, (Q - df) / Q) * 100; H2 = Q / df
wr = 1 / (vi + tau2); mu = (wr * yi).sum() / wr.sum(); se = np.sqrt(1 / wr.sum())
t = stats.t.ppf(0.975, k - 2); sd = np.sqrt(tau2 + se ** 2)
print(f"RESULT k={k} Q={Q:.2f} p_Q={p_Q:.4f} I2={I2:.2f} H2={H2:.3f} tau2={tau2:.4f}")
print(f"RESULT PI_OR=[{np.exp(mu-t*sd):.3f},{np.exp(mu+t*sd):.3f}]")
```
Seeded synthetic data (not a source example); shows the same five quantities run on any `yi/sei` pair and that a
prediction interval on the additive scale is exponentiated back to a ratio scale.

## Output contract
One line per quantity as `RESULT name=value`: `k`, `df`, `y_fe`/`RR_FE`, `Q`, `p_Q`, `tau2_DL`, `H2`, `sqrt_H2`,
the three I-squared forms `I2_from_Q`, `I2_from_tau2`, `I2_from_H2`, the pooled `RR_RE` with `CI=[lo,hi]`,
`Z_FE`, `Z_RE`, and the prediction interval `PI_RR=[lo,hi]`. Ratios are exponentiated log effects; I-squared is a
percentage; tau2/H2 live on the same (log) scale as `yi`.

## Validity conditions
- Effect sizes and SEs must be on the log/additive scale; never feed raw ratios (Masoumi & Shahraz 2022 p2).
- Q needs `k >= 2`; it has low power when few studies, so a non-significant Q is weak evidence of homogeneity, not proof of it (p3).
- I-squared bands: about 25% low, 50% moderate, 75% high; the haloperidol 41% is "moderate heterogeneity" (p3).
- The prediction interval is meaningful only under a random-effects model and uses `t` on `k-2` df, not `z`.
- The Fig-1 narrative set (`Q=35.18, I2=54.51%, PI 0.73-7.17`, RE 2.28, tau2 approx 0.247) is self-consistent but is
  a DIFFERENT printout from Table 2; it cannot be rebuilt from Table 2's per-study RRs, so the PI asserted here is `[0.854,5.089]`.

## Out of scope
Computing per-study effect sizes/weights, choosing the pooling estimator (MH/Peto/IV, DL/REML/PM), subgroup and
meta-regression tests, publication-bias assessment (Egger, funnel/trim-fill) and drawing forest plots - each is a
sibling skill; only the heterogeneity statistics live here (Masoumi & Shahraz 2022 p2-3).

## Conventions
- `CONF = 0.95`, `ALPHA = 0.05`, `df = k - 1`, PI `t = stats.t.ppf(0.975, k-2)`, `zc = stats.norm.ppf(0.975)`.
- `pythonmeta` and `zepid` are NOT installed and are never installed; their Q/I2/tau2/prediction-interval output is
  reimplemented from the formulas above with `numpy` and `scipy.stats` (`chi2`, `norm`, `t`).
- Tolerances reflect SEs back-derived from 2-dp printed CIs: Q within 0.05, I2 within 0.10, tau2 within 0.001, pooled RR/Z within 0.01/0.02.
- DL tau2 and every I2 form are floored at 0 when `Q < df`; never report a negative heterogeneity estimate.
- Seeded demo via `numpy.random.default_rng(42)`; no figures drawn. Source: Masoumi & Shahraz, *Meta-analysis using
  Python: a hands-on tutorial*, BMC Med Res Methodol 2022;22:193 (PDF pages 1-9).
