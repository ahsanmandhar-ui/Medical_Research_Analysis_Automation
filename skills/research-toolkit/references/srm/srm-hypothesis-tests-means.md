---
name: srm-hypothesis-tests-means
description: Student t-tests for quantitative means in pure Python (numpy/scipy.stats) - one sample mean against a hypothesised mu, paired before/after mean differences mu_d, and two independent means with Welch (Satterthwaite df) as the default and pooled variance as the contrast. Runs from raw columns or from summary stats only, computing the standard error, the t-score (statistic minus assumed parameter over s.e.), one- and two-tailed P-values, the n>=25 statistical-validity flag and Dunn's three-part conclusion sentence. Use when asked for a one-sample t-test, a paired/dependent t-test, a two-sample or Welch t-test, whether to assume equal variances, how to turn a t into a P-value, or how to word "there is strong evidence (t = ...; P = ...) that ...". NOT for proportions/chi-square, confidence intervals or the general P-value framework.
---

# Hypothesis Tests for Means (Python)

## Purpose
Test a claim about one population mean `mu`, a mean difference `mu_d` (paired data), or the difference between
two independent means: assume H0, locate the statistic on its sampling distribution with a t-score, weigh it with a P-value (Dunn p394-396, p441).

## Inputs
- Raw data: one numeric column (one mean); two aligned columns or a column plus a pair id (paired); one value
  column plus a two-level group column (independent). State the difference order (Before minus After) and keep it (Dunn p422, p439).
- Summary stats are enough: `xbar, s, n`; `dbar, sd, n` of the differences; `xA, sA, nA, xB, sB, nB`. Where the
  source prints the s.e. of a difference that cannot be rebuilt from rounded sds, pass `se_given` (Dunn p444).
- `MU0` (0 for differences), `TAIL` = `two` | `less`/`greater` fixed by the RQ before seeing data, `ALPHA = 0.05`.

## Algorithms (all pure numpy/scipy)
### One mean from a column: ttest_1samp (Dunn p402-403, p406)
```python
import numpy as np
from scipy import stats
rec = np.array([14, 18, 12, 10, 8, 28, 24, 3, 9, 9, 26, 0, 4, 21, 24, 2, 14], float)      # Table 27.2 (p402)
ene = np.array([5260, 5640, 6390, 6515, 7515, 8770, 5470, 6180, 7515, 6805, 8230], float)  # Table 27.3 (p406)
for lbl, x, mu0, tail in (("recovery_days", rec, 15.0, "less"), ("energy_kJ", ene, 7725.0, "two-sided")):
    r1 = stats.ttest_1samp(x, mu0, alternative=tail); r2 = stats.ttest_1samp(x, mu0)
    print(f"RESULT {lbl}: n={x.size} xbar={x.mean():.4f} s={x.std(ddof=1):.4f} se={x.std(ddof=1)/np.sqrt(x.size):.4f} "
          f"t={r1.statistic:.4f} p_tail={r1.pvalue:.4f} p_two={r2.pvalue:.4f} valid={x.size >= 25}")
```
`t=-0.7915 p_tail=0.2201 p_two=0.4402` reproduces the book's t = -0.79, one-tailed P = 0.220 (Dunn p403);
energy gives `t=-2.8208 p_two=0.0181`. Both have `n < 25`, so both need a near-normal population.

### One mean from summaries only: Eq 27.1 (Dunn p393-396, p400-401, p405)
```python
import math
from scipy import stats
MIN_N = 25
def one_mean(xbar, s, n, mu0, tail="two"):
    se = s / math.sqrt(n)                              # s.e.(x_bar) = s/sqrt(n) (Dunn p394)
    t = (xbar - mu0) / se                              # t = (statistic - assumed parameter)/s.e. (p396)
    p2 = 2 * stats.t.sf(abs(t), n - 1)                 # two-tailed, df = n-1
    return se, t, p2, (p2 if tail == "two" else p2 / 2), n >= MIN_N  # one tail = half the two tail (p399)
for lbl, xb, sd, n, mu0, tail in (("body_temp", 36.8051, 0.40732, 130, 37.0, "two"), ("speed_limit", 96.56, 13.874, 400, 90.0, "greater"),
                                  ("protein_intake", 14.0, 3.0, 50, 13.0, "two")):
    se, t, p2, p, valid = one_mean(xb, sd, n, mu0, tail)
    print(f"RESULT {lbl}: se={se:.6f} t={t:.4f} p_two={p2:.6f} " + ("" if tail == "two" else f"p_{tail}={p:.6f} ") + f"p_lt_0.001={p < 0.001} valid={valid}")
```
Body temp `se=0.035724 t=-5.4557 P<0.001`: the book divides by the rounded s.e. and prints -5.453, reporting
`t = -5.45` either way (Dunn p395, p399); speeds `t=9.4565` (book 9.46, p401); protein `se=0.424264 t=2.3570` (p405).

### Paired data: one-sample t on the differences (Dunn p422-427, p429-430, p432-433)
```python
import math
import numpy as np
from scipy import stats
MIN_N = 25
before = np.array([12.1, 11.0, 14.1, 13.8, 15.5, 12.2, 12.8, 9.9, 10.8, 12.7])
after = np.array([12.0, 10.6, 13.4, 11.2, 15.3, 13.6, 12.6, 8.8, 9.6, 12.4])   # Table 29.1 (p422)
d = before - after                                   # saving = Before minus After; the order must be stated
r = stats.ttest_rel(before, after); se = d.std(ddof=1) / math.sqrt(d.size)
print(f"RESULT insulation_raw: n={d.size} dbar={d.mean():.4f} sd={d.std(ddof=1):.4f} se={se:.4f} "
      f"t={r.statistic:.4f} p_two={r.pvalue:.4f} p_greater={r.pvalue / 2:.4f} valid={d.size >= MIN_N}")
def paired(dbar, n, sd=None, se_given=None, tail="two"):
    se = se_given if se_given is not None else sd / math.sqrt(n)   # s.e.(d_bar) = sd/sqrt(n) (Dunn p422)
    t = (dbar - 0.0) / se                                          # H0: mu_d = 0 (Dunn p422)
    p2 = 2 * stats.t.sf(abs(t), n - 1)
    return se, t, p2, (p2 if tail == "two" else p2 / 2), n >= MIN_N
for lbl, dbar, sd, se_g, n, tail in (("covid_pa", -2.68, 51.30, None, 213, "two"), ("species_butterfly_minus_beetle", 0.77, 1.07, None, 210, "greater"),
        ("dbp_first_minus_second", 1.95, 8.026, None, 141, "two"), ("stroke_joint_pain", 1.27, None, 0.57, 34, "two"),
        ("broccoli_dip", 5.2, None, 3.06, 101, "greater")):
    se, t, p2, p, valid = paired(dbar, n, sd=sd, se_given=se_g, tail=tail)
    print(f"RESULT {lbl}: se={se:.6f} t={t:.4f} p_two={p2:.6f} " + ("" if tail == "two" else f"p_{tail}={p:.6f} ") + f"valid={valid}")
```
Reproduces `t=1.6813 p_two=0.1270 p_greater=0.0635` (Dunn p424-425), `t=-0.7624 p_two=0.4466` i.e. the book's
`P > 0.10` (p426-427), `t=10.4284` (p429) and `t=2.8850`, `P=0.005` to three places (p430).

### Two independent means: Welch default, pooled as contrast (Dunn p439-445, p447-448)
```python
import math
from scipy import stats
MIN_N = 25
def welch(xA, sA, nA, xB, sB, nB, tail="two", se_given=None):
    a, b = sA ** 2 / nA, sB ** 2 / nB
    se = se_given if se_given is not None else math.sqrt(a + b)   # Welch s.e. of xA - xB (Dunn p441)
    df = (a + b) ** 2 / (a ** 2 / (nA - 1) + b ** 2 / (nB - 1))   # Satterthwaite df
    t = (xA - xB) / se                                            # H0: muA - muB = 0 (Dunn p439)
    p2 = 2 * stats.t.sf(abs(t), df)
    return se, df, t, p2, (p2 if tail == "two" else p2 / 2), min(nA, nB) >= MIN_N
def pooled(xA, sA, nA, xB, sB, nB):
    sp2 = ((nA - 1) * sA ** 2 + (nB - 1) * sB ** 2) / (nA + nB - 2)
    se = math.sqrt(sp2 * (1 / nA + 1 / nB)); t = (xA - xB) / se
    return se, nA + nB - 2, t, 2 * stats.t.sf(abs(t), nA + nB - 2)
CASES = (("phone_reaction_ms", (585.1875, 89.6460558116231, 32, 533.59375, 65.3599756065616, 32), "two", None),   # p440
         ("gray_whale_length_m", (12.70, 0.611, 260, 12.07, 0.705, 139), "two", None), ("bhadp_disability", (31.83, 7.73, 132, 25.07, 4.80, 137), "two", 0.80285),
         ("lean_forward_deg", (30.7, 2.58, 10, 16.20, 4.44, 5), "greater", 2.167), ("paramedic_grip_kg", (51.0, 8.0, 18, 56.0, 9.0, 11), "two", 3.3044),
         ("paramedic_pushups", (36.0, 10.0, 18, 47.0, 11.0, 11), "two", 4.0689))                                   # p443-445, p447-448
for lbl, args, tail, se_g in CASES:
    se, df, t, p2, p, valid = welch(*args, tail=tail, se_given=se_g)
    print(f"RESULT {lbl}: se={se:.5f} df={df:.2f} t={t:.4f} p_two={p2:.6f} " + ("" if tail == "two" else f"p_{tail}={p:.6f} ") + f"valid={valid}")
for lbl, args in (("pooled_equal_n", CASES[0][1]), ("pooled_unequal_n", CASES[3][1])):
    se, df, t, p2 = pooled(*args)
    print(f"RESULT {lbl}: se={se:.5f} df={df} t={t:.4f} p_two={p2:.6f}  # the book quotes Welch only (p439)")
```
Welch gives `t=2.6307 p_two=0.010953` (book t = 2.631, two-tailed P = 0.011, Dunn p441), whales `t=8.8993` (p443), BHADP
`t=8.4200` from the given s.e. (p444), face-plant `t=6.6913 p_two=0.000834` i.e. book P = 0.001 with `p_greater=0.000417`
(the book halves its own rounded 0.001 to 0.0005, p445). Equal n makes pooled and Welch identical; unequal n does not.

### Seeded raw data, then the sampling-distribution picture (Dunn p394-395, p440)
```python
import os, tempfile, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
rng = np.random.default_rng(42); xA = rng.normal(585.0, 90.0, 32); xB = rng.normal(534.0, 65.0, 32)  # seeded stand-in for Table 30.1 (p438)
w = stats.ttest_ind(xA, xB, equal_var=False); pl = stats.ttest_ind(xA, xB, equal_var=True)
tt = np.linspace(-7, 7, 800); png = os.path.join(tempfile.mkdtemp(prefix="ttest_"), "sampling_dist.png")
fig, ax = plt.subplots(figsize=(5, 3)); ax.plot(tt, stats.t.pdf(tt, w.df), lw=1.2)
ax.fill_between(tt, 0, stats.t.pdf(tt, w.df), where=np.abs(tt) >= abs(w.statistic), color="grey", alpha=.6); ax.axvline(w.statistic, color="crimson")
fig.suptitle(f"Welch t = {w.statistic:.3f}, df = {w.df:.1f}, two-tailed area shaded"); fig.tight_layout()
fig.savefig(png, dpi=100); plt.close(fig)
print(f"RESULT seeded: welch_t={w.statistic:.4f} df={w.df:.2f} p_two={w.pvalue:.6f} pooled_t={pl.statistic:.4f} "
      f"decision_at_0.05={'evidence against H0' if w.pvalue < 0.05 else 'no evidence against H0'} "
      f"decision_matches_book_p441={(w.pvalue < 0.05) == (0.011 < 0.05)} png_exists={os.path.isfile(png)}")  # book P = 0.011 (p441)
```

## Output contract
`RESULT name=value` per test with `n` or `nA`/`nB`, the statistic (`xbar`/`dbar`/difference of means), `se`, `t`, `df`
(Welch: Satterthwaite; else n-1), `p_two`, `p_<tail>`, `p_lt_0.001`, `valid`, `decision_at_0.05`; `sampling_dist.png`
in a `tempfile` dir is the only artefact.

## Validity conditions
- Sample size of at least 25 - `n` for one mean, the number of *differences* for paired data, **both** `nA` and
  `nB` for two means; below 25 the population(s) must be near-normal. 25 is rough, some books say 30 (Dunn p401, p427, p441).
- A large n means the individual values need no normal distribution - only the sampling distribution of the mean
  does, which is what justifies the t-score and the 68-95-99.7 P approximation (Dunn p395, p401).
- Welch (`equal_var=False`, "Equal variance not assumed") is the default: more general, fewer assumptions, so
  pooled is a contrast only; internal validity needs a well-designed study, external validity a random sample (Dunn p401, p439, p442).
- Paired units are analysed as the differences on the same unit, never as two independent means (Dunn p422, p437-438).

## Reporting conventions
- Three parts: the RQ answer, the evidence (`t = value; one-/two-tailed P = value`), and the sample summary
  (means, sds, ns) plus the CI from the comparison-CI skill; never "accept H0" (Dunn p399-400).
- A P-value is never 0: software `0.000` reads `P < 0.001`; one-tailed P = half the two-tailed P (Dunn p397, p399).
  Name which group or visit is higher and how differences were computed; a null result reads "no evidence
  exists in the sample ...", not "no effect" (Dunn p404, p431).

## Out of scope
Chi-square/odds tests for categorical data (srm-hypothesis-tests-categorical); forming H0/H1, what a P-value
means, evidence bands, test selection, significance vs practical importance (srm-hypothesis-testing-framework);
every confidence interval (srm-confidence-intervals-single/-comparisons); proportions; rank tests; 3+ groups.

## Conventions
- `ALPHA = 0.05`, `MIN_N = 25`, `MU0 = 0` for differences, two-tailed unless the RQ names a direction (Dunn p399, p422).
- Raw data: `scipy.stats.ttest_1samp` / `ttest_rel` / `ttest_ind(equal_var=False)`; summary-only cases use `scipy.stats.t.sf`;
  `numpy.random.default_rng(42)` for demo data and `matplotlib.use("Agg")` for figures; writes only inside `tempfile`.
  The book's jamovi/SPSS output is reproduced with scipy alone, so nothing is installed at run time.
- Never invent a value - compute it and cite the page. Source: Dunn, *Scientific Research and Methodology*
  (Springer, 2024) Ch.27 p391-408, Ch.29 p421-436, Ch.30 p437-450.
