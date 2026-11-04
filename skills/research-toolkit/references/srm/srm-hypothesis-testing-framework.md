---
name: srm-hypothesis-testing-framework
description: The general skeleton every hypothesis test shares, in pure Python (numpy/scipy). Forms H0/H1 from the research question (H0 always contains an 'equals' meaning no difference/change/relationship; one- or two-tailed from the RQ, never from the data), standardises the observation into a t/z test statistic (statistic minus assumed parameter over standard error), converts it to a one-/two-tailed P-value (one-tail = half the two-tail; 68-95-99.7 approximation), maps P to Dunn's evidence-strength wording (Table 28.1), separates statistical significance from practical importance, and routes variable types + design to the correct test. Use when asked what a P-value means or does not mean, whether P=0.018 or P=0.0501 supports H1, how to word a hypothesis-test conclusion, which hypotheses are valid, or which test to select.
---

# Hypothesis-Testing Framework (Python)

## Purpose
Run the common machinery of ALL hypothesis tests - the decision-type twin of confidence intervals:
form H0/H1, compute a test statistic, read the P-value, word the decision, and select the right test.
The general approach is identical for any test (Dunn p409); this skill computes none of the specific tests.

## Inputs
- RQ stating the POPULATION parameter (mu, p, difference, OR), the assumed value, and the direction
  (two-tailed unless the RQ names one side; the RQ, not the data, decides) (Dunn p411).
- Observed statistic + standard error (or a ready t/z-score) and its P-value with a one-/two-tailed flag.
- A context-specific practical-importance cut (e.g. a weight-loss threshold).
- Wording bands from Table 28.1; the 0.05 'significance' cut is traditional but arbitrary (Dunn p414).

## Workflow
Six steps: ask the RQ, design, collect, describe, analyse, report (Dunn p389). Hypothesis testing
formalises the decision process assumption -> expectation -> observation (Dunn p389-390, p417).
With raw data, produce graphical and numerical summaries first and always check the statistical
validity conditions before trusting the P-value (Dunn p410).

## Algorithms (all pure numpy/scipy)

### Forming H0/H1 (Dunn p410-411)
```python
MU0, XBAR = 37.0, 36.8051          # body-temperature example (Dunn p410-411, p418)
def h0_h1(tail):                   # hypotheses always concern the population parameter mu, never xbar
    h0 = f"mu = {MU0}"             # H0 always contains an 'equals': no difference/change/relationship
    h1 = f"mu {'!=' if tail == 'two' else tail} {MU0}"
    return h0, h1
for t in ("two", ">", "<"):        # one-tailed only if the RQ says so - decided before seeing data
    h0, h1 = h0_h1(t)
    print(f"RESULT h0='{h0}' h1='{h1}' tail={t}")
print(f"RESULT xbar={XBAR} onus='data must refute H0; testing starts by assuming H0 true'")
```
Hypothesising about the sample mean (e.g. `xbar = 36.8051`) is pointless - we can just look at it;
P-values exist to learn about the unknown population (Dunn p410, p414).

### Test statistic and P-value from it (Dunn p412-413)
```python
from scipy import stats
def band2(t):                      # two-tailed tail area via the 68-95-99.7 rule (Dunn p413)
    a = abs(t)
    return ">0.32" if a <= 1 else "0.05-0.32" if a <= 2 else "0.003-0.05" if a <= 3 else "<0.003"
for t in (3.4, -2.9, 1.2, -0.95, -0.2, 6.7):   # Exercise 28.1 t-scores (Dunn p418)
    p2 = 2 * stats.norm.sf(abs(t))
    print(f"RESULT t={t} two_sided_band={band2(t)} p_two_exact={p2:.4f} p_one_exact={p2/2:.4f}")
print("RESULT form='t=(statistic - assumed parameter)/s.e.(statistic)' one_tailed='halve the two-tailed P'")
```
The t-score is a z-score whose denominator is a standard error (variation of the statistic); use z if
the population sd is known, t if it is estimated from the sample (Dunn p412-413).

### P-value -> evidence wording (Table 28.1, Dunn p413-414)
```python
def wording(p):                    # interpret in context; thresholds are guidelines only
    if p > 0.10:  return "Insufficient evidence to support H1"
    if p > 0.05:  return "Slight evidence to support H1"
    if p > 0.01:  return "Moderate evidence to support H1"
    if p > 0.001: return "Strong evidence to support H1"
    return "Very strong evidence to support H1"
for p, ctx in ((0.018, "energy intake vs 7725 kJ, n=11 women (Dunn p419)"),
               (0.0501, "boundary case (Dunn p418)"), (0.0499, "boundary case (Dunn p418)"),
               (0.70, "ALDI vs Energizer battery life (Dunn p419)")):
    print(f"RESULT p={p} wording='{wording(p)}' sig_at_0.05={p < 0.05} context='{ctx}'")
print(f"RESULT p={0.0005} wording='{wording(0.0005)}' never_zero='report a 0.000 output as P<0.001'")
```

### Selecting the test - decision tree (Table 32.1, Dunn p477; Table 26.1, Dunn p390)
```python
def select_test(response, explanatory, paired=False):
    if response == "quantitative" and explanatory == "none":
        return ("t-test for mean differences", "CI for mean difference", "Ch.23/29") if paired else ("one-sample t", "CI for one mean", "Ch.22/27")
    if response == "quantitative" and explanatory == "quantitative":
        return ("correlation/regression slope test", "CI for r or slope", "Sect.34.4/35.7")
    if response == "quantitative":
        return ("t-test comparing two means", "CI for difference of means", "Ch.24/30")
    if explanatory == "none":
        return ("one-sample proportion test/CI", "CI for one proportion", "Ch.20")
    if explanatory == "qualitative":
        return ("chi-square test of percentages/odds", "CI for OR", "Ch.25/31")
    return ("not among the tests studied; hundreds of tests exist (Kanji 2006)", "-", "-")
for r, e, p in [("quantitative", "none", False), ("quantitative", "none", True),
                ("quantitative", "qualitative", False), ("quantitative", "quantitative", False),
                ("qualitative", "none", False), ("qualitative", "qualitative", False)]:
    test, ci, ch = select_test(r, e, p)
    print(f"RESULT response={r} explanatory={e} paired={p} test='{test}' ci='{ci}' ref='{ch}'")
print("RESULT table32_scenarios=4 validity_to_check='near-normal sampling distribution; simple random sample; n<10% of population'")
```
Table 32.1 studied four scenarios; deciding requires the response/explanatory variable types and the
design (paired vs independent), not the data (Dunn p477).

### Statistical significance vs practical importance (Dunn p415-416)
```python
import numpy as np
from scipy import stats
rng = np.random.default_rng(42)
wt, wt_cut, p_wt = 1.61, 2.5, 0.0005     # herbal weight loss P<0.001 but 1.61 kg < 2.5 kg practical cut (Dunn p416)
ba, bb, p_bat = 5.73, 5.44, 0.70         # ALDI 5.73 h vs Energizer 5.44 h, P=0.70 (Dunn p419)
print(f"RESULT weight_diff={wt}kg practical_cut={wt_cut}kg practically_important={wt >= wt_cut} p={p_wt} statistically_significant={p_wt < 0.05}")
print(f"RESULT battery_diff={ba - bb:.2f}h p={p_bat} statistically_significant={p_bat < 0.05}")
x = rng.normal(37.0 + 0.01, 0.35, 10000) # tiny 0.01 difference + large n -> small P (Dunn p415)
t = (x.mean() - 37.0) / (x.std(ddof=1) / np.sqrt(x.size))
p_two = 2 * stats.t.sf(abs(t), x.size - 1)
print(f"RESULT sim_delta=0.01 n={x.size} t={t:.2f} p_two={p_two:.4f} statistically_significant={p_two < 0.05}")
```
Statistical significance asks 'could the difference be chance?'; practical importance asks 'does the
difference matter, given context?' - separate questions, both important (Dunn p415-416).

## Output contract
Report per test: `h0`, `h1`, `tail` (two|>|<), `test_statistic` (t or z), `p_value` with
`p_direction`, `evidence_wording` (Table 28.1 band), `statistically_significant` (bool at the
arbitrary 0.05), `practically_important` (bool vs the stated cut), and from the tree: `selected_test`,
`selected_ci`, `ref`. Each block above prints its `RESULT name=value` lines; no files are written.

## Validity conditions
- Statistical validity conditions vary per test; they exist so the sampling distribution is near-normal
  and the 68-95-99.7 P-value holds - unmet, the P-value and conclusion may be inappropriate (Dunn p416).
- All computations assume a simple random sample; also need n < 10% of the population size (Dunn p417).
- External validity requires internal validity (a well-designed study); internal validity can only be
  discussed with design details, and non-random samples limit the population the test speaks about (Dunn p416-417).
- Form RQ and hypotheses before looking at the data (Dunn p411).

## Reporting conventions
- A P-value is the likelihood of the observed statistic (or more extreme) under H0 - NOT the probability
  H0 is true; it proves nothing; a big P does not make H0 true; a small P does not mean a big difference,
  only that chance cannot reasonably explain it (Dunn p414).
- Conclusions are worded as the level of evidence supporting H1 and always about the population; a test
  never proves anything from one sample. A conclusion carries: RQ answer + evidence (t, P, one/two-tailed)
  + sample summary with a CI (Dunn p415).
- Check whether software reports one- or two-tailed P-values (some always two-tailed); halve for one-tailed (Dunn p413).
- Avoid the bare word 'significant' unless statistical significance is meant; use 'substantial' for size (Dunn p414).

## Out of scope
Computing the specific tests belongs to siblings: chi-square/odds comparisons
(srm-hypothesis-tests-categorical), one-sample/paired/two-sample t-tests from raw data
(srm-hypothesis-tests-means), CI construction (srm-confidence-intervals-single/-comparisons),
correlation/regression inference (srm-correlation, srm-linear-regression).

## Conventions
- Default to two-tailed tests; one-tailed only when the RQ says so; P reported to 4 dp, never as exactly 0.
- `scipy.stats.norm` for the 68-95-99.7 framework, `scipy.stats.t` for exact tails; statsmodels for the
  internals the sibling skills own.
- Seeded demo data only via `numpy.random.default_rng(42)`; headless `matplotlib` (`Agg`) if a normal-curve
  P-value diagram is requested (Dunn p413). Never invent a value - compute it and cite the source page.
- Source: Dunn, *Scientific Research and Methodology* (Springer, 2024) Ch.26 p389-390, Ch.28 p409-420, Ch.32 p477-478.
