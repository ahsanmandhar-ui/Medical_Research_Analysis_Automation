---
name: srm-statistical-reference
description: Lookup/reference layer for Dunn, Scientific Research and Methodology - notation dictionary and parameter/statistic/s.e. formula index (Appendix C), normal-distribution table lookups re-derived with scipy.stats norm.cdf/ppf and asserted to the printed 4-dp grid (Tables B.2-B.3), the 68-95-99.7 rule exact vs printed, random-digit-table selection with a seeded numpy replacement (Table B.1), and worked-answer self-check patterns reproducing printed Appendix D values. Use when you need a z-area, a backwards percentile lookup, a symbol or s.e. formula, or want to verify your code against the book's printed answers.
---

# Statistical Reference - Tables, Notation and Formula Index (Python)

## Purpose
Serve as the book's lookup card: notation/formula index, distribution-table access via `scipy.stats`
instead of printed pages, random-number selection, and self-check patterns that assert code reproduces
printed values (Dunn p557-562, p595-602). Sibling skills own the actual inference.

## Inputs
- `z`: a 2-dp z-score exactly as printed in the table grid; `q`: a left-tail probability for backwards lookups.
- `mu`, `sd`: mean and standard deviation for unstandardizing `x = mu + z*sd` (Dunn p601-602).
- For self-checks, the book's own inputs: counts `x`,`n`, sample `s`, hypothesised `mu0`, target margin `MOE`.
- `SEED = 42` via `numpy.random.default_rng(42)`; the 68-95-99.7 rule reads 68/95/99.7% within 1/2/3 sd (Dunn p595).

## Notation dictionary (Appendix C, Dunn p561-562)
| Parameter | Statistic | s.e. formula | Definition ref |
|---|---|---|---|
| proportion `p` | `p_hat` | sqrt(p_hat*(1-p_hat)/n) | Def. 20.2 |
| mean `mu` | `x_bar` | s/sqrt(n) | Def. 22.1 |
| mean difference (paired) `mu_d` | `d_bar` | s_d/sqrt(n) | Def. 23.2 |
| diff. between two means `mu1-mu2` | `x1_bar - x2_bar` | computed in software | - |
| odds ratio, correlation `rho`, slope `beta1`, intercept `beta0`, `R2` | sample OR, `r`, `b1`, `b0`, `R2` | no s.e. for ORs | App. C |
Symbols: `H0`/`H1` null/alternative (Sect. 28.2), `df` and chi-square statistic (Sect. 31.4), CI (Chap. 21),
s.e. = standard error (Def. 18.3), `n` = sample size (Dunn p561).

## Formula index (Dunn p561-562)
- CI: `statistic +/- (multiplier x s.e.(statistic))`; multiplier approx 2 for an approx 95% CI; the
  second term is the margin of error; odds-ratio CIs are the exception (Dunn p561).
- Test statistic: `t = (statistic - parameter)/s.e.(statistic)`; a chi-square statistic is approx like a
  z-score of value `sqrt(chi2/df)` (Dunn p561-562).
- z-score `z = (x - mu)/sd` and unstandardizing `x = mu + z*sd` (Dunn p601-602); sample size
  `n = 1/MOE^2` (proportion) and `n = (2*s/MOE)^2` (mean), always rounded up (Dunn p576-578).

## Algorithms (all pure numpy/scipy)

### z-table lookup, replaces printed Tables B.2-B.3
```python
from scipy.stats import norm
grid = [(-1.87, 0.0307), (1.87, 0.9693), (-2.00, 0.0228), (-0.61, 0.2709), (-0.22, 0.4129)]
bad = [z for z, book in grid if round(float(norm.cdf(z)), 4) != book]
print(f"RESULT lookups={len(grid)} mismatches={len(bad)} area(-1.87)={float(norm.cdf(-1.87)):.6f}")
assert bad == []
```
The printed grid gives the area *less than* the z looked up; the block asserts scipy agrees to 4 dp
with the book's worked examples z=-1.87 (Dunn p559) and z=+1.87 (Dunn p560).

### Backwards lookup (ppf) and unstandardizing - tree diameters
```python
from scipy.stats import norm
mu, sd, q = 8.8, 2.7, 0.85          # 15% largest diameters, Dunn p575 (Exercise 17.1)
z = float(norm.ppf(q)); x = float(norm.ppf(q, mu, sd))
print(f"RESULT z_85th={z:.4f} book_z=1.04 x_unstd={x:.3f} book_x=11.6")
assert round(z, 2) == 1.04 and round(x, 1) == 11.6
```

### Mensa cut-off (IQ ~ N(100, 15), top 2%)
```python
from scipy.stats import norm
z = float(norm.ppf(0.98)); x = 100 + z * 15
print(f"RESULT z_mensa={z:.4f} book_z=2.05 x_exact={x:.2f} x_book_rounded_z=130.75")
assert round(z, 2) == 2.05
```
The book unstandardizes with the rounded z, 100 + 2.05*15 = 130.75, "an IQ greater than about 130" (Dunn p575).

### 68-95-99.7 rule: exact normal areas vs the printed approximation
```python
from scipy.stats import norm
for k, book in ((1, 0.68), (2, 0.95), (3, 0.997)):
    exact = float(norm.cdf(k) - norm.cdf(-k))
    print(f"RESULT within_{k}sd={exact:.6f} rule={book} err={exact - book:+.6f}")
assert abs(float(norm.cdf(2) - norm.cdf(-2)) - 0.95) < 0.005
```
The rule (called the empirical rule in the glossary, Dunn p595) undercuts the exact 2-sd area by about
0.0045 - the origin of the "multiplier of 2" in every CI.

### Random digits (Table B.1) and the seeded generator replacement
```python
import numpy as np
row_a = [1,1,8,4,2, 4,7,2,9,6, 0,7,1,3,1, 5,4,7,4,1, 5,1,4,7,8]   # B.1 row A, cols I-V (Dunn p557)
digits = np.random.default_rng(42).integers(0, 10, size=25)
print(f"RESULT rowA_mean={sum(row_a)/len(row_a)} rowA_sevens={row_a.count(7)} rng_first5={list(digits[:5])}")
assert sum(row_a) == 102 and row_a.count(7) == 4
```
Printed tables pick random starting rows/columns by chance; `default_rng(42)` gives the same job
reproducibly, so never report the printed digits as if they were your sample.

### Worked-answer self-checks (formula index vs printed answers, Dunn p563-590)
```python
import math
checks = {
  "se_p":    math.sqrt((2182/6882)*(1-2182/6882)/6882),        # Ex 20.1, p_hat=0.317059, Dunn p576
  "se_xbar": 0.43/math.sqrt(45),                               # Ex 22.1, Dunn p577
  "t":       (6753.64-7725)/(1142.123/math.sqrt(11)),          # Ex 27.1, Dunn p581
  "z_chi":   math.sqrt(4.593/1),                               # Ex 31.1, Dunn p585
  "r_neg":   -math.sqrt(0.9929),                               # Ex 34.4, Dunn p586
  "n_prop":  math.ceil(1/0.05**2),                             # Ex 20.4, Dunn p576
  "n_mean":  math.ceil((2*651.1/50)**2),                       # Ex 22.4, Dunn p578
}
print("RESULT " + " ".join(f"{k}={v:.9g}" for k, v in checks.items()))
assert (round(checks["se_p"], 9), round(checks["se_xbar"], 8), round(checks["t"], 3),
        round(checks["z_chi"], 2), round(checks["r_neg"], 4), checks["n_prop"], checks["n_mean"]) \
       == (0.005609244, 0.06410062, -2.821, 2.14, -0.9964, 400, 679)
```
Each tuple element is the book's printed answer; reproducing them at the book's own rounding depth is
the smoke test before trusting any sibling inference skill.

## Output contract
Every block prints `RESULT name=value` lines: `area(z)` table lookups, `z_*`/`x_*` percentile lookups,
`within_Ksd` exact areas, `rowA_*`/`rng_*` random-digit values, and `se_p`, `se_xbar`, `t`, `z_chi`,
`r_neg`, `n_prop`, `n_mean` self-checks. Return these to the caller verbatim; a lookup that fails its
`assert` is a transcription error, not a math error - re-read the grid, never adjust the assert.

## Validity conditions
- Read grids in 2 dp: table rows are the z integer+tenth, columns the hundredth (Dunn p559-560).
- A percentage is a proportion multiplied by 100 - work in proportions (Dunn p599); `r` takes the sign
  of the relationship, so `r = -sqrt(R2)` when the slope is negative (Dunn p586).
- Random numbers only give a simple random sample if every digit string maps to a listed frame member (Dunn p557).

## Out of scope
Forming CIs, running hypothesis tests, correlation/regression modelling and graph choice belong to the
sibling inference and summary skills; this file only looks up, translates notation and verifies numbers.

## Conventions
- `scipy.stats.norm.cdf`/`.ppf` replace printed Tables B.2/B.3; asserts at the book's printed depth.
- Seeds fixed with `numpy.random.default_rng(42)`; figures (none here) would use `matplotlib.use("Agg")`.
- The book's printed tables are the authority; if a value cannot be reproduced honestly, ship the
  ones that reproduce and report the discrepancy instead of fudging. No external package beyond
  numpy/scipy is needed for these lookups.
