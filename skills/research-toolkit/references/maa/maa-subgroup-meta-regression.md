---
name: maa-subgroup-meta-regression
description: Subgroup contrasts, meta-regression and sensitivity/robustness analysis for a meta-analysis in pure Python (numpy/scipy + statsmodels WLS) - per-group DerSimonian-Laird random-effects pooled estimates, the between-group (subgroup) chi-square test via a mixed-effects dummy model, meta-regression on continuous and categorical moderators (weighted least squares with tau2 added to the weights), heterogeneity-explained R2, and sensitivity analysis - an explicit leave-one-out loop, one-study-removed pooled estimates with significance flips, and effect-size-family swaps. Takes per-study effect sizes (log RR/OR/HR) and standard errors. Use when asked to run a subgroup/meta-regression analysis, test whether study characteristics explain a pooled effect, check if a conclusion is robust by dropping studies, or swap the effect-size family. NOT for deriving Q/I2/tau2 as the deliverable, choosing the pooling estimator, effect-size computation, or publication-bias (Egger/funnel) diagnostics - those are siblings.
---

# Meta-Analysis: Subgroups, Meta-Regression & Sensitivity (Python)

## Purpose
Take per-study effect sizes that already carry a heterogeneity signal (sibling `maa-heterogeneity` owns the Q/I2/tau2
deliverable) and explain/robustness-test it: pool studies inside each subgroup, run the between-group chi-square test,
regress the effect on moderators with a mixed-effects (tau2-inflated) model, then stress-test the headline conclusion
with leave-one-out, one-study-removed pooling and an effect-size-family swap (Masoumi & Shahraz 2022 p3).

## Inputs
- `yi`: per-study log/additive effects (log RR/OR/HR, SMD); `sei`: standard errors, `v_i = sei**2`.
- `X`: moderator design matrix - a centred continuous column, or a 0/1 dummy for a two-level categorical moderator.
- Anchor data: haloperidol-vs-placebo log-RR table (17 trials) rebuilt from the printed RR/95% CI, `sei = (log(hi)-log(lo)) / (2*1.96)`, the Python column (Masoumi & Shahraz 2022 p7, Table 2).
- `ALPHA = 0.05`, `CONF = 0.95`, `zc = stats.norm.ppf(0.975)`; `df_between = G - 1`; `df_resid = k - p`.

## Algorithms (all pure numpy/scipy + statsmodels WLS)
Every block re-pools inline: IV weights `w=1/v`, DL `tau2=max(0,(Q-df)/C)`, then mixed/random weights `1/(v_i + tau2)` (tau2 ADDED to the within-study variance) for `mu = sum(wr*yi)/sum(wr)`.

### Subgroup pooled estimates and the between-group chi-square test (Masoumi & Shahraz 2022 p3)
Pool each subgroup by DL RE then contrast them; the article prints `chi2 = 5.60, DF = 1, p = 0.02` for its missing/non-missing split (p3). Here the categorical moderator is publication year (before/after 1990), rebuildable from Table 2: `Q_between = Q_total - sum(Q_within)`, chi-square on `G-1` df.
```python
import numpy as np
from scipy import stats
def re_pool(yi, sei):
    v = np.asarray(sei)**2; w = 1.0/v; y_fe = (w*yi).sum()/w.sum()
    Q = float((w*(yi-y_fe)**2).sum()); df = len(yi)-1
    C = w.sum()-(w**2).sum()/w.sum(); tau2 = max(0.0,(Q-df)/C)
    wr = 1.0/(v+tau2); mu = (wr*yi).sum()/wr.sum(); se = np.sqrt(1.0/wr.sum())
    return mu, se, tau2, Q
zc = stats.norm.ppf(0.975)
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
sei = np.array([(np.log(r[3]) - np.log(r[2]))/(2*zc) for r in tab])
year = np.array([int(r[0][-4:]) for r in tab]); grp = year >= 1990
print(f"RESULT full_RR={np.exp(re_pool(yi,sei)[0]):.3f}")          # source anchor ~2.09 (Table 2, p7)
for g, lab in [(False,"pre-1990"),(True,"1990-onward")]:
    m = grp == g; mu, se, t2, _ = re_pool(yi[m], sei[m])
    print(f"RESULT group[{lab}] k={int(m.sum())} RR={np.exp(mu):.3f} CI=[{np.exp(mu-zc*se):.3f},{np.exp(mu+zc*se):.3f}] tau2={t2:.4f}")
Q_total = re_pool(yi, sei)[3]
Q_between = sum(re_pool(yi[grp==g], sei[grp==g])[3] for g in [False, True])
df_bg = 1; p_bg = float(stats.chi2.sf(Q_total - Q_between, df_bg))
print(f"RESULT Q_between={Q_total-Q_between:.3f} df={df_bg} p={p_bg:.4f}")
assert df_bg == 1 and (Q_total - Q_between) >= 0.0
```

### Meta-regression with mixed-effects (tau2 added to the weights) (Masoumi & Shahraz 2022 p6)
Fit `yi = X b + e`, `e ~ N(0, v_i + tau2)` by iterative WLS: regress on weights `1/(v_i+tau2)`, re-estimate `tau2` from the weighted residual SS, repeat to a fixed point - the same engine as the Egger bias regression in Table 1 (p6), minus its bias reading.
```python
import numpy as np
import statsmodels.api as sm
from scipy import stats
zc = stats.norm.ppf(0.975)
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
sei = np.array([(np.log(r[3]) - np.log(r[2]))/(2*zc) for r in tab])
v = sei**2; k = len(yi)
year = np.array([int(r[0][-4:]) for r in tab], float)
X = sm.add_constant((year - year.mean())/10.0)      # continuous moderator, per 10 years
def metareg_mixed(y, v, X):
    tau2 = 0.0
    for _ in range(200):
        w = 1.0/(v + tau2); fit = sm.WLS(y, X, weights=w).fit()
        r = y - X @ fit.params; Qr = float((w*r**2).sum()); p = X.shape[1]
        C = w.sum() - (w**2).sum()/w.sum()
        tau2_new = max(0.0, (Qr - (k - p))/C)        # DL-style moment update on residuals
        if abs(tau2_new - tau2) < 1e-12: tau2 = tau2_new; break
        tau2 = tau2_new
    return fit, tau2
fit, tau2_resid = metareg_mixed(yi, v, X)
w0 = 1.0/v; y_fe = (w0*yi).sum()/w0.sum(); Q0 = float((w0*(yi-y_fe)**2).sum())
tau2_null = max(0.0, (Q0-(k-1))/(w0.sum()-(w0**2).sum()/w0.sum()))
R2 = max(0.0, (tau2_null - tau2_resid)/tau2_null)*100 if tau2_null > 0 else 0.0
b, seb = fit.params[1], fit.bse[1]; z = b/seb; pval = float(2*stats.norm.sf(abs(z)))
print(f"RESULT tau2_null={tau2_null:.4f} tau2_resid={tau2_resid:.4f} R2_explained={R2:.2f}")
print(f"RESULT beta_year_per10yr={b:.4f} se={seb:.4f} z={z:.3f} p={pval:.4f}")
assert tau2_resid >= 0.0 and np.isfinite(pval)
```
`R2_explained = (tau2_null - tau2_resid)/tau2_null` is the between-study variance the moderator removes; the slope is a Wald z.

### Sensitivity: leave-one-out, one-study-removed pooling, effect-size-family swap (Masoumi & Shahraz 2022 p3)
Explicit deterministic loop removes each study, re-pools and flags significance flips; a synthetic 2x2 shows the conclusion
survives swapping log RR for log OR (the article's own motif: RR moves only 1.97-2.71 across imputation scenarios, p3).
```python
import numpy as np
from scipy import stats
def re_pool(yi, sei):
    v = np.asarray(sei)**2; w = 1.0/v; y_fe = (w*yi).sum()/w.sum()
    Q = float((w*(yi-y_fe)**2).sum()); df = len(yi)-1
    C = w.sum()-(w**2).sum()/w.sum(); tau2 = max(0.0,(Q-df)/C)
    wr = 1.0/(v+tau2); mu = (wr*yi).sum()/wr.sum(); se = np.sqrt(1.0/wr.sum())
    return mu, se
zc = stats.norm.ppf(0.975)
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
sei = np.array([(np.log(r[3]) - np.log(r[2]))/(2*zc) for r in tab])
mu_full, se_full = re_pool(yi, sei); sig_full = abs(mu_full) > zc*se_full
rrs, flips = [], 0
for drop in range(len(yi)):
    keep = np.ones(len(yi), bool); keep[drop] = False
    mu, se = re_pool(yi[keep], sei[keep]); rrs.append(np.exp(mu))
    if (abs(mu) > zc*se) != sig_full:
        flips += 1; print(f"FLIP drop={tab[drop][0]} RR={np.exp(mu):.3f}")
print(f"RESULT LOO_min_RR={min(rrs):.3f} LOO_max_RR={max(rrs):.3f} LOO_flip_count={flips}")
assert min(rrs) <= max(rrs) and 0 <= flips <= len(yi)
a = np.array([28,40,12,55,30,18]); n1 = np.array([45,90,20,80,60,25])
c = np.array([10,22,4,30,15,6]);  n0 = np.array([50,88,18,75,55,22])
logRR = np.log((a/n1)/(c/n0)); vRR = 1/a - 1/n1 + 1/c - 1/n0
logOR = np.log((a/(n1-a))/(c/(n0-c))); vOR = 1/a + 1/(n1-a) + 1/c + 1/(n0-c)
muR, seR = re_pool(logRR, np.sqrt(vRR)); muO, seO = re_pool(logOR, np.sqrt(vOR))
print(f"RESULT family_RR={np.exp(muR):.3f} sigRR={abs(muR)>zc*seR}")
print(f"RESULT family_OR={np.exp(muO):.3f} sigOR={abs(muO)>zc*seO}")
assert (abs(muR)>zc*seR) == (abs(muO)>zc*seO)       # robust: same call across families
```

## Output contract
`RESULT name=value` lines: `full_RR`, per-group `group[...] RR/CI/tau2`, `Q_between/df/p` (subgroup test); `tau2_null`, `tau2_resid`, `R2_explained`, `beta_year_per10yr/se/z/p` (meta-regression); `LOO_min_RR`, `LOO_max_RR`, `LOO_flip_count`, and `family_RR`/`family_OR` with `sigRR`/`sigOR`. All ratios are exponentiated log effects.

## Validity conditions
- Subgroups/moderators are pre-specified, not data-mined; a p near 0.05 on an ad-hoc split is not evidence (Masoumi & Shahraz 2022 p3).
- Between-group chi-square needs `k_g >= 2` per subgroup and is under-powered with few studies per group (p3).
- Report residual tau2 after a "significant" moderator - a good study-level fit still leaves unexplained heterogeneity.
- Leave-one-out flips matter only near the significance boundary; a robust conclusion shows zero flips.

## Conventions
- `pythonmeta` and `zepid` are NOT installed and never installed: subgroup pooling, the between-group chi-square and the mixed-effects meta-regression are reimplemented with `numpy`, `scipy.stats` and `statsmodels.WLS` (weights `1/(v_i+tau2)`).
- `ALPHA=0.05`, `CONF=0.95`, `zc=stats.norm.ppf(0.975)`; tau2 floored at 0; year moderator centred and scaled per 10 years.
- The article's Fig-1 narrative prediction interval (0.73-7.17) does NOT reproduce from Table 2 (a sibling found [0.854, 5.090]), so every number here is asserted against reproducible Table-2 quantities only (Masoumi & Shahraz 2022 p3, p7).
- Family-swap uses a synthetic 2x2 (deterministic, no rng needed); no figures drawn; nothing written to disk.

## Out of scope
Computing per-study effect sizes and weights, choosing the FE/RE pooling estimator, deriving Q/I2/tau2/H2/prediction intervals as the deliverable (sibling `maa-heterogeneity`), publication-bias diagnostics (Egger/funnel/trim-and-fill) and missing-data imputation schemes - none live here; only subgroup contrast, meta-regression and sensitivity/robustness.
