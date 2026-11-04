---
name: srm-numerical-summaries-qualitative
description: Numerically summarise qualitative (categorical) data in pure Python (numpy/pandas) - overall, row and column proportions/percentages from a two-way count table, odds (event vs not-event), odds ratios (division and cross-product form) with reading rules, an end-to-end pandas crosstab pipeline, and a Simpson's-paradox stratify-vs-pool check. Use when asked for the percentage/proportion in each category, "success rate by group", row/column percentages, odds or an odds ratio from a contingency table, or whether combining groups reverses a comparison.
---

# Numerical Summaries: Qualitative Data (Python)

## Purpose
Describe one or two categorical variables from counts: proportions/percentages (overall, by row, by
column), odds, and the odds ratio that collapses a 2x2 table into one number, plus rules for reading
them (Dunn p237-248). Every quantity is a *statistic* (p-hat, sample odds) estimating a *parameter*
(p, population odds) (Dunn p238, p241, p244).

## Inputs
- Two-way counts `n`: rows = groups (e.g. treatment method), columns = outcomes (e.g. success/failure);
  kidney-stone example rows A/B x cols Success/Failure = [[81,6],[234,36]] (Dunn p238).
- Or long-form records (group label, outcome label) -> `pandas.crosstab`.
- A single category count `x` and total `N` for a one-variable proportion.

## Algorithms (all pure numpy/scipy)

### Overall, row and column proportions/percentages (Dunn p238-240)
A proportion is a fraction of a total (0 to 1); a percentage is that times 100 (Def 14.1-14.2). Row
percentages compare the outcome *within each row-group*; column percentages say where each outcome
came from; both are computable, pick the comparison you need.
```python
import numpy as np
n = np.array([[81, 6], [234, 36]])                 # small kidney stones: rows A/B, cols Success/Failure
p_hat = n[:, 0].sum() / n.sum()                    # overall success proportion, p-hat
row_pct = 100 * n[:, 0] / n.sum(axis=1)            # % success within each method (Dunn p239)
col_pct = 100 * n[0] / n.sum(axis=0)               # % of each outcome coming from Method A (Dunn p240)
print(f"RESULT overall_p={p_hat:.3f} overall_pct={100*p_hat:.1f} row_pct=[{row_pct[0]:.1f},{row_pct[1]:.1f}] col_pct_succ_A={col_pct[0]:.1f} col_pct_fail_A={col_pct[1]:.1f}")
```
Prints `overall_p=0.882 overall_pct=88.2 row_pct=[93.1,86.7] col_pct_succ_A=25.7 col_pct_fail_A=14.3`
(315/357; 81/87; 81/315, Dunn p238-240).

### Odds (Dunn p241-243)
Odds = (times it happens) / (times it doesn't) - divide by the *remaining* number, not the total;
from percentages the same value results (86.7/13.3 = 6.52 for Method B, Dunn p243).
```python
import numpy as np
n = np.array([[81, 6], [234, 36]])                 # small kidney stones (Dunn p238)
odds = n[:, 0] / n[:, 1]                           # success odds per method, row 1 vs row 2 style
odds_B_from_pct = 86.7 / 13.3                      # same odds from row percentages (Dunn p243)
pct_fem = 67                                       # about 67% of students female (Dunn p242)
odds_from_pct = pct_fem / (100 - pct_fem)
print(f"RESULT odds_A={odds[0]:.1f} odds_B={odds[1]:.2f} odds_B_from_pct={odds_B_from_pct:.2f} odds_67pct={odds_from_pct:.2f}")
```
Prints `odds_A=13.5 odds_B=6.50 odds_B_from_pct=6.52 odds_67pct=2.03`. Reading: odds > 1 event more
likely than not; = 1 equally; < 1 less likely (Dunn p243).

### Odds ratio from the 2x2 table (Dunn p243-244)
OR = how many times greater the group-1 odds are than the group-2 odds; equivalently the
cross-product (a*d)/(b*c). Reversing the group order gives the reciprocal, so always state the order.
```python
import numpy as np
n = np.array([[81, 6], [234, 36]])                 # rows = Method A (row 1) then B (row 2)
or_div = (n[0, 0] / n[0, 1]) / (n[1, 0] / n[1, 1]) # odds of row 1 / odds of row 2
or_cross = (n[0, 0] * n[1, 1]) / (n[0, 1] * n[1, 0])
or_inv = 1 / or_div                                # Method B vs Method A
print(f"RESULT OR_A_vs_B={or_div:.2f} OR_cross={or_cross:.2f} OR_B_vs_A={or_inv:.3f}")
```
Prints `OR_A_vs_B=2.08 OR_cross=2.08 OR_B_vs_A=0.481` - A's success odds are 2.08 times B's (Dunn p243-244).

### End-to-end with pandas.crosstab (Dunn p245-246)
```python
import numpy as np, pandas as pd
fem = np.array(["skips"] * 2383 + ["eats"] * (6640 - 2383))   # Iranian children, Table 14.6 (Dunn p246)
male = np.array(["skips"] * 1944 + ["eats"] * (6846 - 1944))
sex = np.array(["F"] * fem.size + ["M"] * male.size)
tab = pd.crosstab(sex, np.concatenate([fem, male]), margins=False)
pF, pM = tab.loc["F", "skips"] / tab.loc["F"].sum(), tab.loc["M", "skips"] / tab.loc["M"].sum()
oddsF, oddsM = tab.loc["F", "skips"] / tab.loc["F", "eats"], tab.loc["M", "skips"] / tab.loc["M", "eats"]
print(f"RESULT p_F={pF:.3f} p_M={pM:.3f} odds_F={oddsF:.4f} odds_M={oddsM:.4f} OR={oddsF/oddsM:.2f}")
```
Prints `p_F=0.359 p_M=0.284 odds_F=0.5598 odds_M=0.3966 OR=1.41` (Table 14.7 gives OR 1.412, Dunn p245-246).

### Stratify vs pool: Simpson's paradox check (Dunn p240-241)
Recompute the row proportions in each stratum AND in the pooled table; if the sign of the
group difference flips, a recorded confounder (stone size) is being ignored and the pooled
conclusion can be wrong.
```python
import numpy as np
small = np.array([[81, 6], [234, 36]])             # small stones (Dunn p238)
large = np.array([[192, 71], [55, 25]])            # large stones (Dunn p240)
pool = small + large                               # Table 14.5, ignoring size
succ = lambda t: t[:, 0] / t.sum(axis=1)           # success proportion per method (row)
a_s, b_s = succ(small); a_l, b_l = succ(large); a_p, b_p = succ(pool)
print(f"RESULT A_small={a_s:.3f} B_small={b_s:.3f} A_large={a_l:.3f} B_large={b_l:.3f} A_pool={a_p:.2f} B_pool={b_p:.2f}")
```
Prints `A_small=0.931 B_small=0.867 A_large=0.730 B_large=0.688 A_pool=0.78 B_pool=0.83` - A wins in
both strata but pooled favours B: size is the confounder/lurking variable (Dunn p240-241).

## Output contract
Report: `counts` (two-way table with margins), `overall_p`/`overall_pct`, `row_p`/`row_pct` per group,
`col_p`/`col_pct` per column, `odds` per group, `odds_ratio` with the explicit comparison order
(first row/column vs second), and the `>`/`=`/`<` 1 reading. Print one `RESULT name=value` line per
quantity, as in the blocks above.

## Validity conditions
- Choose the denominator by context: percentages divide by the total relevant number; odds divide by
  the remaining number - mixing them changes the answer (Dunn p239, p242).
- A difference in sample proportions/odds has two possible explanations - sampling variation or a real
  population difference; describing it is not deciding between them (Dunn p244-245, p247).
- If a confounding variable was recorded (stone size), analyse within strata before pooling; the
  pooled OR can point to the wrong group (Dunn p240-241).
- Odd/OR from a sample estimates the population odds/OR; report sample vs population distinctly (Dunn p241, p244).

## Reporting conventions
- Odds: "13.5 times as many successes as failures", equivalently "1350 successes for every 100
  failures" (Example 14.1, Dunn p242).
- OR: "the odds of females skipping breakfast are 1.41 times the odds of males" - name both groups
  and the order (Dunn p245).
- Use proportions (0.359) or percentages (35.9%) consistently to 1-3 decimals; report per-group
  sample sizes alongside (Table 14.7 style, Dunn p246).

## Out of scope
Means/SDs/medians/IQR for quantitative variables (Table 14.8) belong to the quantitative-summaries
skill; bar/pie/stacked charts to the graphical-summaries skill; chi-square tests of the relationship
and CIs for the OR belong to the categorical-tests and comparison-CI skills.

## Conventions
- Proportion 0-1; percentage = proportion x 100 (Def 14.1-14.2, Dunn p238).
- jamovi/SPSS convention: odds compare row 1 vs row 2 (or col 1 vs col 2), OR relative to the first
  row/column - keep it and say the order (Dunn p242, p244).
- numpy/pandas only (`crosstab`, array slicing); scipy/statsmodels not needed; if a chart is asked for,
  headless `matplotlib.use("Agg")`.
- No randomness used; if simulating category labels, seed with `numpy.random.default_rng(42)`.
- Never invent a value - always compute it and cite the source page for the method.
- Source: Dunn, *Scientific Research and Methodology* (Springer, 2024) Ch.14, pp.237-250.
