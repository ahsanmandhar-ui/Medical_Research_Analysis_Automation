---
name: srm-hypothesis-tests-categorical
description: Chi-square (X2) tests that compare proportions/odds from a two-way count table, pure Python (numpy/scipy/matplotlib) - expected counts under H0, the all-expected-counts >= 5 validity rule, Pearson X2 with df and always-two-tailed P, the z-equivalent sqrt(X2/df), odds ratio with Woolf 95% CI, phi and Cramer's V effect sizes, Fisher's exact fallback for small expected counts, and hand-built reference-level odds ratios for 2xN tables. Use when asked to compare percentages/proportions or odds across groups, test whether two categorical variables are associated or independent, run a chi-square test of independence/homogeneity, check the expected-count condition, get the OR and its confidence interval from a 2x2 table, or fall back to Fisher's exact test on small counts. NOT for comparing means (t-tests) or the general P-value framework.
---

# Chi-Square Tests for Categorical Counts (Python)

## Purpose
Decide whether the odds/proportions of an outcome are the same across groups, or whether two qualitative
variables are associated, from a two-way count table: H0 OR = 1, H1 OR not 1, alternative always two-tailed (Dunn p451-453).

## Inputs
- `a`: 2-D array of whole, non-negative counts (>= 2x2); rows = outcome levels, columns = groups. Source
  tables: eating `52,105;2,24` (Dunn p452), pet birds `98,101;141,328` (Dunn p459), B12/diet `8,26;8,82`
  (Dunn p463), EV purchase `24,8;51,29` (Dunn p468), kerbside `22,18,15;6,36,1` (Dunn p465).
- `ALPHA = 0.05`; `MIN_EXPECTED = 5` (validity threshold, Dunn p458); `CORRECTION = False` (uncorrected
  Pearson, the book's number); reference column for 2xN odds ratios.

## Algorithms (all pure numpy/scipy)

### X2, expected counts, validity, effect size, OR + Woolf CI, Fisher fallback (Dunn p454-458)
```python
import math
import numpy as np
from scipy import stats
ALPHA, MIN_EXPECTED = 0.05, 5.0
a = np.array([[52., 105.], [2., 24.]])                   # Dunn Table 31.1 (p452)
n = a.sum(); df = (a.shape[0] - 1) * (a.shape[1] - 1)    # 2x2 -> df 1 (Dunn p456)
exp_hand = np.outer(a.sum(axis=1), a.sum(axis=0)) / n    # E = rowtotal*coldtotal/n (Dunn p454-455)
chi2, p, _, exp = stats.chi2_contingency(a, correction=False)   # uncorrected Pearson
z_eq = math.sqrt(chi2 / df)                              # z-equivalent sanity check (Dunn p456)
phi = math.sqrt(chi2 / n); v = math.sqrt(chi2 / (n * (min(a.shape) - 1)))
below = int((exp < MIN_EXPECTED).sum())
odds = a[0] / a[1]                                       # group odds = successes/failures (Dunn p452)
orv = float(odds[0] / odds[1]); se = math.sqrt((1 / a).sum()); zc = stats.norm.ppf(1 - ALPHA / 2)
lo, hi = math.exp(math.log(orv) - zc * se), math.exp(math.log(orv) + zc * se)   # Woolf log-OR CI
f_or, f_p = stats.fisher_exact(a)                        # exact remedy when expected counts are small
print("RESULT expected=", np.round(exp.ravel(), 2), "hand_maxdiff=", round(float(np.abs(exp_hand - exp).max()), 9))
print(f"RESULT chi2={chi2:.3f} df={df} p={p:.4f} z_eq={z_eq:.3f} min_expected={exp.min():.2f} cells_below5={below} valid={below == 0}")
print(f"RESULT phi={phi:.3f} cramers_v={v:.3f} odds={np.round(odds, 3)} OR={orv:.3f} ci95=[{lo:.2f},{hi:.2f}] fisher_p={f_p:.4f} n={n:.0f}")
for corr in (False, True):                               # scipy defaults to Yates on 2x2 tables
    c, q = stats.chi2_contingency(a, correction=corr)[:2]
    print(f"RESULT correction={corr} chi2={c:.3f} p={q:.4f}")
```
Prints `expected=[46.33 110.67 7.67 18.33] hand_maxdiff= 0.0`, `chi2=6.934 df=1 p=0.0085 z_eq=2.633
min_expected=7.67 cells_below5=0 valid=True`, `phi=cramers_v=0.195 odds=[26. 4.375] OR=5.943 ci95=[1.35,26.11]
fisher_p=0.0093 n=183`, Pearson `6.934/0.0085` vs Yates `5.765/0.0163` (Dunn p455-456, p458). Smallest *observed* count 2, smallest *expected* 7.67 > 5 -> valid.

### Very-strong and no-evidence 2x2 tables (Dunn p459-461, p468)
```python
import math
import numpy as np
from scipy import stats
for label, a in (("petbirds", np.array([[98., 101.], [141., 328.]])),
                 ("evpurchase", np.array([[24., 8.], [51., 29.]]))):
    chi2, p, _, exp = stats.chi2_contingency(a, correction=False)
    odds = a[0] / a[1]; orv = float(odds[0] / odds[1])
    se = math.sqrt((1 / a).sum()); zc = stats.norm.ppf(0.975)
    print(f"RESULT {label}: chi2={chi2:.3f} p={p:.6f} OR={orv:.3f} odds={np.round(odds, 3)} "
          f"ci95=[{math.exp(math.log(orv) - zc * se):.2f},{math.exp(math.log(orv) + zc * se):.2f}] "
          f"min_expected={exp.min():.2f} V={math.sqrt(chi2 / a.sum()):.3f}")
```
Pet birds `chi2=22.374 p=0.000002 OR=2.257 odds=[0.695 0.308] ci95=[1.61,3.17] V=0.183` - very strong evidence
(book: OR 2.26, CI 1.6-3.2, Dunn p461). EV purchase `chi2=1.308 p=0.252816 OR=1.706 ci95=[0.68,4.28]` - no
evidence of a difference in the population odds, never "proven equal" (Dunn p468).

### Expected counts below 5 -> quote Fisher's exact as well (Dunn p463-464)
```python
import math
import numpy as np
from scipy import stats
a = np.array([[8., 26.], [8., 82.]])                    # B12 vs diet, Dunn Table 31.7 (p463)
chi2, p, df, exp = stats.chi2_contingency(a, correction=False)
below = int((exp < 5).sum()); odds = a[0] / a[1]; orv = float(odds[0] / odds[1])
se = math.sqrt((1 / a).sum()); zc = stats.norm.ppf(0.975); f_or, f_p = stats.fisher_exact(a)
print(f"RESULT chi2={chi2:.3f} df={df} p={p:.4f} min_expected={exp.min():.3f} cells_below5={below} "
      f"pct_below5={100 * below / exp.size:.1f} valid={below == 0}")
print(f"RESULT odds={np.round(odds, 4)} OR={orv:.3f} "
      f"ci95=[{math.exp(math.log(orv) - zc * se):.3f},{math.exp(math.log(orv) + zc * se):.3f}] fisher_p={f_p:.4f}")
```
`chi2=4.707 p=0.0300 min_expected=4.387 cells_below5=1 pct_below5=25.0 valid=False`: the chi-square "may be statistically
invalid", so pair it with Fisher's exact `fisher_p=0.0388`; `OR=3.154 ci95=[1.077,9.238]` matches the book; one cell *just* under 5 is noted, not fatal (Dunn p463-464).

### 2xN tables: X2 with df, then reference-level odds ratios by hand (Dunn p465-467)
```python
import math
import numpy as np
from scipy import stats
k = np.array([[22., 18., 15.], [6., 36., 1.]])          # kerbside dumping, Dunn Table 31.9 (p465)
chi2, p, _, exp = stats.chi2_contingency(k, correction=False)
df = (k.shape[0] - 1) * (k.shape[1] - 1)                # 2x3 -> df 2 (Dunn p456)
odds = k[0] / k[1]; ors = odds / odds[0]               # reference level = column 0
pct = 100 * k[0] / k.sum(axis=0)
print(f"RESULT chi2={chi2:.3f} df={df} p={p:.6f} z_eq={math.sqrt(chi2 / df):.3f} "
      f"V={math.sqrt(chi2 / (k.sum() * (min(k.shape) - 1))):.3f} min_expected={exp.min():.2f} cells_below5={int((exp < 5).sum())}")
print(f"RESULT odds={np.round(odds, 3)} OR_vs_ref={np.round(ors, 3)} pct={np.round(pct, 1)} coln={k.sum(axis=0).astype(int).tolist()}")
```
`chi2=26.318 df=2 p=0.000002 z_eq=3.628 V=0.518 min_expected=7.02 cells_below5=0`, `odds=[3.667 0.5 15.] OR_vs_ref=[1. 0.136 4.091]
pct=[78.6 33.3 93.8] coln=[28, 54, 16]`: an observed count of 1 is fine when every expected count exceeds 5, and ORs = groups - 1 by hand because no CI exists beyond 2x2 (Dunn p467-468).

### Seeded sample -> table -> test -> headless plot
```python
import os, tempfile
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
rng = np.random.default_rng(42)
grp = rng.choice(np.array(["A", "B"]), size=300)
ev = rng.random(300) < np.where(grp == "A", 0.70, 0.50)
a = np.array([[float(((grp == g) & ev).sum()) for g in ("A", "B")],
              [float(((grp == g) & ~ev).sum()) for g in ("A", "B")]])
chi2, p, df, exp = stats.chi2_contingency(a, correction=False)
png = os.path.join(tempfile.mkdtemp(prefix="chi2_"), "chi2_counts.png")
fig, ax = plt.subplots(figsize=(5, 3)); x = np.arange(2); w = 0.36
ax.bar(x - w / 2, a[0], width=w, label="observed"); ax.bar(x + w / 2, exp[0], width=w, label="expected under H0")
ax.set_xticks(x); ax.set_xticklabels(["A", "B"]); ax.legend(fontsize=8)
fig.suptitle(f"X2={chi2:.3f}, df={df}, two-tailed P={p:.4f}"); fig.tight_layout()
fig.savefig(png, dpi=100); plt.close(fig)
print(f"RESULT table={a.astype(int).tolist()} chi2={chi2:.3f} p={p:.4f} "
      f"OR={float((a[0][0] / a[1][0]) / (a[0][1] / a[1][1])):.3f} min_expected={exp.min():.2f} "
      f"cells_below5={int((exp < 5).sum())} png_exists={os.path.isfile(png)}")
```
Prints `table=[[107, 70], [48, 75]] chi2=13.343 p=0.0003 OR=2.388 min_expected=59.45 cells_below5=0 png_exists=True`; seeded demo data, not a source example, and the PNG is written only inside `tempfile`.

## Output contract
One line per quantity as `RESULT name=value`: `chi2`, `df`, `p` (always two-tailed), `z_eq`, `expected`, `min_expected`,
`cells_below5`, `valid`, `phi`, `cramers_v`, `odds`, `OR`, `ci95=[lo,hi]`, `fisher_p`, `n`; for 2xN also `OR_vs_ref` per column and the `reference` column. Plot: `chi2_counts.png` inside a `tempfile` dir.

## Validity conditions
- Both variables are qualitative counts recorded on the same units; means/SDs do not belong in the table (Dunn p451-452).
- Validity needs **every expected count >= 5** - the rule is on expected, never observed, counts (Dunn p458); below 5,
  report Fisher's exact test, which carries no expected-count assumption (Dunn p464).
- Independent observations; matched/paired binary data need McNemar, not this independence test.
- The alternative is always two-tailed - software never gives a one-sided P here (Dunn p453).
- OR and its CI exist only for 2x2; wider tables report X2 with df plus a manual reference-level summary (Dunn p459, p468).
- Effect size reading: V near 0.1 small, 0.3 moderate, 0.5 large (eating 0.195 small, kerbside 0.518 moderate-strong).

## Reporting conventions
Three parts: strength of evidence, the evidence (`X2 = value; two-tailed P = value`, or `df = value` for bigger
tables), and the sample summary with each group's odds, the OR and its 95% CI (Dunn p458). Paraphrased template:
"strong evidence (X2 = 6.934; two-tailed P = 0.008) that the population odds of eating off-campus differ between
students living with parents (odds: 26) and those not (odds: 4.375; OR: 5.94; 95% CI 1.35 to 26.1)". Spell out what
the odds describe and which groups the OR compares (odds depend on which row is first); an OR of 5.9 is not 5.9x
the probability; keep RQ, hypotheses and conclusion in one language; grade the P (very strong/moderate/little/no
evidence); association is not causation (Dunn p452-454, p461).

## Out of scope
Tests comparing means (t-tests), CIs for differences/ORs, single-proportion tests, the general H0/P-value and
test-selection framework, McNemar for paired binary data.

## Conventions
- `ALPHA = 0.05`, `CONF = 0.95`, `MIN_EXPECTED = 5`, `df = (rows-1)*(cols-1)`, Woolf z-critical
  `stats.norm.ppf(1 - ALPHA/2)`, `correction=False` unless Yates is explicitly asked for.
- Seeded demo data via `numpy.random.default_rng(42)`; matplotlib `Agg`; write only inside `tempfile`; counts stay
  whole, percentages describe, odds (successes/failures) build the OR.
- The book used jamovi/SPSS; `scipy.stats.chi2_contingency` plus `fisher_exact` reproduce its numbers, so no extra
  package is needed and nothing is installed at run time.
- Source: Dunn, *Scientific Research and Methodology* (Springer, 2024) Ch.31, pp.451-476.
