---
name: srm-probability-and-decision-logic
description: Probability from three approaches plus the logic of statistical decision-making, pure Python (numpy/itertools/math). Applies the classical (equally-likely count), relative-frequency (past proportion) and subjective approaches; converts probability to odds; computes joint, marginal and conditional probabilities; tests independence via the multiplication rule; turns a probability into an expected count/value; and runs the assumption -> expectation -> observation simulation that decides whether a sample statistic contradicts an assumed population parameter. Use when asked for a probability or odds of an event, "are these two events independent", conditional probability given another event, expected number of outcomes, or how a sample drives a decision about a population.
---

# Probability and Decision Logic (Python)

## Purpose
Quantify the chance an event occurs (classical, relative-frequency or subjective), relate probability
to odds, joint/conditional events and independence, and reproduce the assumption -> expectation ->
observation loop that turns a sample statistic into a decision about a population parameter.

## Inputs
- Discrete experiment: a finite sample space of equally-likely outcomes (die faces, cards, coin flips).
- Count data: `x` occurrences out of `n` past "attempts" (relative frequency), or a small contingency table.
- A testable assumption about a population parameter (e.g. fair pack, proportion `p = 0.5`) and the
  statistic to observe in a sample of size `n`.

## Algorithms (all pure numpy/scipy)

### Classical approach and odds (Dunn p265-266)
Probability = (equally-likely outcomes in the event) / (total equally-likely outcomes);
odds = (outcomes in event) / (outcomes not in event). Enumerate the sample space with `itertools`.
```python
import itertools
ss = list(itertools.product(range(1, 7)))          # one fair six-sided die, Dunn p264-266
n = len(ss)
p = lambda ev: sum(1 for o in ss if o[0] in ev) / n
odds = lambda ev: p(ev) / (1 - p(ev))
print(f"RESULT P(roll_4)={p({4}):.4f} P(even)={p({2,4,6}):.4f} odds(even)={odds({2,4,6}):.1f} "
      f"P(roll_2)={p({2}):.4f} odds(roll_2)={odds({2}):.1f} P(1_or_2)={p({1,2}):.4f}")
```
Prints `P(roll_4)=0.1667 P(even)=0.5000 odds(even)=1.0 P(roll_2)=0.1667 odds(roll_2)=0.2 P(1_or_2)=0.3333`
(the 1/6, 0.5, odds-1, 1/5=0.2 and 2/6 examples, Dunn p265-266).

### Relative frequency, joint and conditional probability (Dunn p267-268)
Relative frequency uses a past proportion; a conditional probability divides a joint count by the
count of the given condition.
```python
male, total = 157088, 305377                       # Australia live births 2015, Dunn p267
pb = round(male / total, 3)                        # 0.514 -> estimated P(boy)
rows = {"SHSS": {"u30": 56, "o30": 40}, "SSE": {"u30": 68, "o30": 91}}  # library table, Dunn p268
grand = sum(sum(r.values()) for r in rows.values())
sse = sum(rows["SSE"].values())
shss_o30 = rows["SHSS"]["o30"]; shss = sum(rows["SHSS"].values())
print(f"RESULT P_boy={pb:.4f} odds_boy={pb/(1-pb):.4f} "
      f"P_SSE={sse/grand:.4f} P_SHSS_and_over30={shss_o30/grand:.4f} "
      f"P_over30_given_SHSS={shss_o30/shss:.4f}")
```
Prints `P_boy=0.5140 odds_boy=1.0583 P_SSE=0.6235 P_SHSS_and_over30=0.1569 P_over30_given_SHSS=0.4167`
(0.514, odds 1.058, 159/255, 40/255, 40/96 = 41.7%, Dunn p267-268).

### Independence and the multiplication rule (Dunn p266, p269)
Two events are independent when P(second | first) equals P(second). Drawing without replacement
breaks that; the coin fallacy (two tails = 1/3) breaks equipotence.
```python
import numpy as np
rng = np.random.default_rng(42)
p_ace1 = 4 / 52
p_ace2_given_ace1 = 3 / 51                         # an Ace is gone -> probability changes
p_ace2_given_no_ace1 = 4 / 51
print(f"RESULT P(Ace_1)={p_ace1:.4f} P(Ace_2|Ace_1)={p_ace2_given_ace1:.4f} "
      f"P(Ace_2|not_Ace_1)={p_ace2_given_no_ace1:.4f} dependent={p_ace2_given_ace1!=p_ace2_given_no_ace1}")
n_trial = 200_000
key = rng.random((n_trial, 52)); pick = np.argsort(key, axis=1)[:, :2]
is_ace = pick < 4                                   # cards 0..3 are the four Aces
p_aa_sim = is_ace.all(axis=1).mean()
p_aa_rule = p_ace1 * p_ace2_given_ace1              # multiplication rule (dependent)
coins = rng.integers(0, 2, size=(n_trial, 2)); p_tt = (coins == 0).all(axis=1).mean()
print(f"RESULT P(A,A)_rule={p_aa_rule:.5f} P(A,A)_sim={p_aa_sim:.5f} P(TT)={p_tt:.4f} not_1_3={abs(p_tt-1/3)>0.02}")
```
Prints `P(Ace_1)=0.0769 P(Ace_2|Ace_1)=0.0588 P(Ace_2|not_Ace_1)=0.0784 dependent=True` and `P(TT)=0.25`
(3/51 vs 4/51, two Tails = 1/4 not 1/3, Dunn p266, p269, p272).

### Probability as expected value / expected count (Dunn p265, p271)
A probability is the expected proportion; over `n` trials the expected count is `n x p` (a binomial mean).
```python
p_four, rolls = 1 / 6, 600
exp_fours = rolls * p_four                        # binomial mean = n x p (expected count)
sens, spec, m = 0.99, 0.98, 100                   # diagnostic test, Dunn p271
exp_pos_diseased = m * sens                       # expected true positives
exp_pos_healthy  = m * (1 - spec)                 # expected false positives
print(f"RESULT expected_fours={exp_fours:.1f} of {rolls} exp_pos_diseased={exp_pos_diseased:.0f} "
      f"exp_pos_healthy={exp_pos_healthy:.0f}")
```
Prints `expected_fours=100.0 of 600 exp_pos_diseased=99 exp_pos_healthy=2` (0.167 expected proportion,
0.99 and 0.98 giving 99 and 2 positives of 100, Dunn p265, p271).

### Decision logic: assumption -> expectation -> observation (Dunn p255-256, p259-260)
Assume a fair pack (`p = 0.5` red), expect about half red in a hand of 15, then simulate the statistic
under that assumption and check how rare the observed value is.
```python
import numpy as np, matplotlib, tempfile, os
from math import comb
matplotlib.use("Agg")
import matplotlib.pyplot as plt
rng = np.random.default_rng(42)
HAND, NREP = 15, 1000                             # 15-card hands, 1000 repetitions, Dunn p259-260
col = np.array([1]*26 + [0]*26)                   # 1 = red, 26 red in the pack
phat = col[np.argsort(rng.random((NREP, 52)), axis=1)[:, :HAND]].sum(axis=1) / HAND
obs_15red = int((phat == 1.0).sum())
exact_15red = comb(26, HAND) / comb(52, HAND)     # true P(15 red | fair pack), no replacement
fig, ax = plt.subplots(figsize=(4, 3))
ax.hist(phat, bins=np.arange(0, 1.01, 0.1), color="steelblue")
ax.set_xlabel("sample proportion red"); ax.set_ylabel("frequency")
out = os.path.join(tempfile.mkdtemp(), "hands.png"); fig.savefig(out, dpi=80); plt.close(fig)
print(f"RESULT phat_mean={phat.mean():.3f} phat_sd={phat.std(ddof=1):.4f} "
      f"hands_all_red_in_1000={obs_15red} exact_P_15red={exact_15red:.3e} contradicts_assumption={obs_15red==0}")
```
Prints `phat_mean=0.496 phat_sd=0.107 hands_all_red_in_1000=0 exact_P_15red=1.724e-06
contradicts_assumption=True` - all 15 red never occurred in 1000 hands, so the fair-pack assumption is
probably wrong, exactly as Dunn reasons (Dunn p255-256, p259-260).

## Output contract
Print `RESULT name=value` per quantity. Return/report: `probability`, `odds`, `joint_prob`,
`conditional_prob`, `is_independent` (bool), `expected_count`, and for decisions `assumption`,
`expected_statistic` (mean/sd), `observed_statistic`, `rare_under_assumption` (bool). Odds, proportions
and probabilities are distinct quantities - label which one is being reported (Dunn p266).

## Validity conditions
- The classical approach is only valid when all outcomes are equally likely; a loaded die invalidates it
  (Dunn p265).
- Relative frequency estimates need a large, representative history; a past proportion is an estimate of
  a future probability, not a guarantee (Dunn p267).
- Probability is stated before the outcome is known; once it has occurred, report a proportion, not a
  "probability it just happened" (Dunn p266).
- A sample statistic that is merely uncommon under the assumption does not contradict it; only values
  that are rare-by-chance do (e.g. 4 red of 15 = p-hat 0.267 is uncommon but happens, Dunn p260).

## Conventions
- Seeded simulation only via `numpy.random.default_rng(42)`; never hardcode a result - compute and print
  the `RESULT` lines above.
- Exact enumeration with `itertools` for small sample spaces; hypergeometric `math.comb` for draws
  without replacement; binomial mean `n x p` for expected counts.
- Headless `matplotlib` (`Agg`); write any figure only inside a `tempfile.mkdtemp()` directory.
- Report probabilities to 4 dp and odds to 1-3 dp; keep full precision in the working.
- Subjective probabilities (weather, interest rates) come from expert models and cannot be computed here;
  state them as judgement, not code (Dunn p267-268).
- Source: Dunn, *Scientific Research and Methodology* (Springer, 2024) Ch.15-16, pp.253-272.

## Out of scope
Formal hypothesis tests (p-values, significance level, Type I/II errors) and confidence intervals - the
machinery that turns "rare under the assumption" into a decision threshold - belong to the hypothesis-test
and confidence-interval skills; the normal model for sampling distributions is the distributions skill.
