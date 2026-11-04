---
name: srm-confounding-and-blinding-experimental
description: Protect the internal validity of an EXPERIMENT (treatment vs control). Random and block allocation, a sample-ratio-mismatch (SRM) chi-square on group sizes, a baseline balance table with standardised mean differences, restriction/blocking and covariate adjustment, washout periods that remove carry-over, and blinding of participants/researchers/analyst plus placebo controls. Pure numpy/scipy/statsmodels on seeded demo data. Use when designing or auditing an A/B test or clinical/agriculture/engineering trial - "balance the groups", "check randomisation", "SRM", "is my experiment confounded", "carry-over", "blind the study", "placebo control". Based on Dunn, Scientific Research and Methodology, Ch.7 pp.113-130.
---

# Confounding & Blinding in Experiments (Python)

## Purpose
Maximise internal validity: make the compared groups as similar as possible apart from the treatment,
verify the allocation, and neutralise the extraneous effects (Dunn p115). Every number below is
reproduced from Dunn Ch.7.

## Inputs
- `n` units, `k` groups, intended proportions `p_g` (default equal, `1/k`); per-unit covariates `age`,`bmi`,`sex` (0/1).
- Thresholds: SRM flag if `p_srm < 0.001`; imbalance flag if `|SMD| >= 0.1`; ALPHA = 0.05.

## Algorithms (all pure numpy/scipy)

### Random vs block allocation + SRM chi-square check (Dunn p118-119)
```python
import numpy as np
from scipy import stats
rng = np.random.default_rng(42); n, k = 245, 2
arm = rng.integers(0, k, n)                         # simple random allocation (Dunn p118)
counts = np.bincount(arm, minlength=k); exp = np.full(k, n / k)   # intended 50/50
srm_chi2 = ((counts - exp) ** 2 / exp).sum(); srm_p = stats.chi2.sf(srm_chi2, k - 1)
blk = np.tile(np.arange(k), 123); rng.shuffle(blk)  # block random: near-exact counts even for small n
print(f"RESULT counts={counts.tolist()} srm_chi2={srm_chi2:.4f} srm_p={srm_p:.4f} "
      f"srm_flag={'FAIL' if srm_p < 0.001 else 'PASS'} block_counts={np.bincount(blk).tolist()}")
```
Prints `counts=[122, 123] srm_chi2=0.0041 srm_p=0.9491 srm_flag=PASS`. A significant SRM is an allocation
/dropout defect, not bad luck - no balance table rescues it. Verify allocation *before* interpreting (Dunn p118-119).

### Baseline balance: standardised mean difference + a worked chi-square (Dunn p116)
```python
import numpy as np
from scipy import stats
rng = np.random.default_rng(42); n, k = 240, 2
arm = rng.integers(0, k, n); worst = 0.0
for x in (rng.normal(50, 14, n), rng.normal(26, 4, n)):      # age, bmi
    m0, m1 = x[arm == 0], x[arm == 1]
    pooled_sd = np.sqrt((m0.var(ddof=1) + m1.var(ddof=1)) / 2)
    worst = max(worst, abs((m0.mean() - m1.mean()) / pooled_sd))   # |SMD| >= 0.1 = imbalanced
tab = np.array([[46, 111 - 46], [45, 108 - 45]])             # ginko Table 7.2: men per arm (Dunn p116)
chi2, p, _, _ = stats.chi2_contingency(tab, correction=True)
print(f"RESULT worst_abs_smd={worst:.4f} balanced={'yes' if worst < 0.1 else 'no'} "
      f"ginko_men_pct={46/111*100:.1f}/{45/108*100:.1f} ginko_chi2={chi2:.5f} ginko_p={p:.4f}")
```
Prints `worst_abs_smd=0.0710 balanced=yes` and `ginko_men_pct=41.4/41.7 ginko_chi2=0.00000 ginko_p=1.0000`
- the sex homogeneity test cannot reject equality, so a later response gap cannot be blamed on that
baseline (Dunn p116). Judge groups by scale-free SMD, not the p-value.

### Restriction, blocking and covariate adjustment (Dunn p116-118)
```python
import numpy as np, statsmodels.api as sm
rng = np.random.default_rng(42); n = 240
age = rng.normal(50, 14, n)                                        # confounder, leaks into assignment
arm = (rng.random(n) + 0.15 * (age - age.mean()) / age.std() > 0.5).astype(float)
resp = 3.0 * arm + 0.8 * (age - age.mean()) + rng.normal(0, 1, n) # true treatment effect tau = 3.0
crude = sm.OLS(resp, sm.add_constant(arm)).fit().params[1]
adj = sm.OLS(resp, sm.add_constant(np.column_stack([arm, age]))).fit().params[1]  # adjust for age
block = np.where(age < np.median(age), 0, 1)                       # blocking on age (Definition 7.1)
bl = [sm.OLS(resp[block == b], sm.add_constant(arm[block == b])).fit().params[1] for b in (0, 1)]
print(f"RESULT tau_true=3.0 crude={crude:.3f} covariate_adjusted={adj:.3f} blocked={np.mean(bl):.3f}")
```
Prints `crude=8.582 covariate_adjusted=3.065 blocked=5.251`: age confounds the crude fit; recording and
adjusting for it recovers `tau`; blocking (or restricting to one age level) partly removes it. Adjustment
only works on variables you measured - so record every plausible confounder (Dunn p116-118).

### Washout period for the carry-over effect (Dunn p120)
```python
import numpy as np
rng = np.random.default_rng(42); n, reps, tau, carry, sd = 160, 2000, 3.0, 2.0, 1.2
def tau_hat(carry_on, rand_order):                     # carry_on = not washout
    est = []
    for _ in range(reps):
        ft = (rng.integers(0, 2, n) if rand_order else np.zeros(n, int)) == 0   # treated in period 1?
        y0 = np.where(ft, tau, 0.0) + rng.normal(0, sd, n)
        y1 = np.where(ft, 0.0, tau) + np.where(ft & carry_on, carry, 0.0) + rng.normal(0, sd, n)
        est.append(float(np.where(ft, y0, y1).mean() - np.where(ft, y1, y0).mean()))
    return float(np.mean(est))
print(f"RESULT tau_hat_fixed={tau_hat(True, False):.3f} tau_hat_random_order={tau_hat(True, True):.3f} "
      f"tau_hat_washout={tau_hat(False, True):.3f}")
```
Prints `1.000 / 2.000 / 3.000` = `tau-carry`, `tau-carry/2`, `tau`. Randomising order balances *period*
effects but leaves carry-over bias; only a washout (or measuring after clearance) removes it (Dunn p120).

### Blinding participants/assessors and the placebo control (Dunn p125-127)
```python
import numpy as np
from scipy import stats
design = {"random_allocation": True, "recorded_covariates": True, "washout": False,
          "blind_participants": True, "blind_researchers": False, "control": True}   # Himalaya case
levers = {"confounding": ["random_allocation", "recorded_covariates"], "carryover": ["washout", "random_allocation"],
          "hawthorne": ["blind_participants"], "observer": ["blind_researchers"], "placebo": ["control"]}
managed = sum(1 for v in levers.values() if any(design.get(l) for l in v))            # Fig 7.7 map
label = "double blind" if (design["blind_participants"] and design["blind_researchers"]) else "single blind"
hi, lo, m = round(0.85 * 40), round(0.61 * 40), 40                                   # Waber price framing
chi2, p, _, _ = stats.chi2_contingency(np.array([[hi, m - hi], [lo, m - lo]]), correction=False)
print(f"RESULT threats_managed={managed}/5 blinding_level={label} "
      f"waber_risk_difference={0.85-0.61:.2f} waber_chi2={chi2:.3f} waber_p={p:.4f}")
```
Prints `threats_managed=4/5 blinding_level=single blind waber_risk_difference=0.24` - the observer effect
survives (researchers unblinded); the 85% vs 61% relief came from an *identical*
placebo (Dunn p124). Blind participants to the Hawthorne effect, researchers to the observer effect, the
analyst to allocation bias; a control/placebo arm isolates the placebo effect (Dunn p125-127).

## Output contract
Print `RESULT name=value` per quantity: allocation `counts`, `srm_chi2`, `srm_p`, `srm_flag`; balance
`worst_abs_smd`, `ginko_p`; effect-path `tau_true`, `crude`, `covariate_adjusted`, `blocked`; carry-over
`tau_hat_fixed/random_order/washout`; design audit `threats_managed`, `blinding_level`,
`waber_risk_difference`. Report effect as treatment-minus-control, then name any unmanaged threat (Dunn p127).

## Validity conditions
- Random allocation protects internal validity only if truly random and leak-free - the SRM test proves it (Dunn p118).
- Random *allocation* (internal validity) is not random *sampling* (external validity); restriction/blocking trades the latter for the former (Dunn p116-117, p119).
- Blinding participants to *participation* is usually impossible (informed consent) - blind to aims/treatment and report partial blinding honestly (Dunn p121-122).
- Prefer naming exactly who is blinded over the ambiguous single/double/triple label (Dunn p125).

## Out of scope
Observational studies that cannot randomise (confounder catalogues, stratified/crude-vs-adjusted tables,
odds-ratio confounding) belong to the sibling observational-confounding skill (Dunn Ch.8); this skill
assumes you can allocate units to arms. Effect-size pooling and hypothesis tests are other skills.

## Conventions
- ALPHA = 0.05, SMD cut-off 0.1, SRM cut-off 0.001, `numpy.random.default_rng(42)` only, no file writes; carry-over `tau=3, carry=2, sd=1.2, reps=2000`.
- numpy/scipy for SRM + SMD + homogeneity chi-square; `statsmodels.OLS` for covariate adjustment (an analytical alternative to simulating carry-over).
- Never invent a number - compute it from the seeded demo and cite the source page for every rule and threshold.
- Source: Dunn, *Scientific Research and Methodology* (Springer, 2024), Ch.7 "Internal validity and experimental studies", pp.113-130.
