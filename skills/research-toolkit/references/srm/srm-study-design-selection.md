---
name: srm-study-design-selection
description: Pick and encode the right quantitative design from a research question's POCI components and the researcher's allocation choices - descriptive vs observational (retrospective/prospective/cross-sectional) vs experimental (true vs quasi), rank the internal-validity trade-off, assign recorded variables as extraneous/confounding/lurking, and emit a pre-analysis protocol dict/JSON, pure Python (pandas/itertools/numpy). Use when asked what type of study a scenario is, which design answers an RQ, whether cause-and-effect may be claimed, or to draft a study-design protocol.
---

# Study Design Selection and Pre-Analysis Protocol (Python)

## Purpose
Turn an RQ into the right quantitative design: classify descriptive vs observational vs experimental
from the POCI components, pick the observational/experimental subtype, rank the internal-validity
trade-off, assign variable roles, and encode the choice in a pre-analysis protocol
(Dunn ch.3 p65-76, ch.6 p103-112).

## Inputs
- `C_present`, `I_imposed`: RQ flags; P and O are always present (Dunn Table 3.3 p71).
- `outcome_when`, `comparison_when`: one of `"now" | "earlier" | "later"` (Dunn Table 3.1 p68).
- `allocates_who`, `allocates_treatment`: the two allocation questions (Dunn Table 3.2 p70).
- 0/1 columns for response, explanatory and candidate covariates plus a `measured` flag
  (Dunn Table 6.1 p108); `SEED = 42`, `ASSOC_TOL = 0.05` for the demo thresholds.

## Algorithms (all pure numpy/scipy)

### RQ type -> study design
```python
import pandas as pd
TABLE = pd.DataFrame([("Descriptive",0,0,"Descriptive"),("Relational",1,0,"Observational"),
                      ("Interventional",1,1,"Experimental")], columns=["rq","C","I","study"])
def choose_design(C_present, I_imposed):
    return "experimental" if I_imposed else ("observational" if C_present else "descriptive")
for rq, C, I in [("heart_rate_after_walk",0,0), ("echinacea_self_chosen",1,0), ("echinacea_assigned",1,1)]:
    print(f"RESULT {rq}={choose_design(C,I)}")
```
Prints `descriptive / observational / experimental` for the three worked RQs (Dunn p66-70).

### Observational subtype (timing) and case-control matching
```python
def obs_subtype(outcome_when, comparison_when):
    return {("now","earlier"):"retrospective", ("later","now"):"prospective",
            ("now","now"):"cross-sectional"}.get((outcome_when, comparison_when), "not_observational")
for k,(o,c) in [("smnd_metals",("now","earlier")), ("softdrink_gout",("later","now")),
                ("food_security",("now","now")), ("legionella_case_control",("now","earlier"))]:
    print(f"RESULT {k}={obs_subtype(o,c)}")
n_cases, n_controls = 25, 75                 # legionella case-control study (Dunn p69)
print(f"RESULT matching_controls_per_case={n_controls // n_cases}")
```
All four examples classify as in the source; the legionella study paired 75 controls with 25 cases,
i.e. 3 controls per case, matched on age (within 5 years), sex and postcode (Dunn p69).

### Experiment taxonomy, factorial grid, validity ladder
```python
import itertools, math
GRID = list(itertools.product(["low-fat","high-fat"], ["average-protein","high-protein"]))
N_DIET = 811                                 # randomised diet trial (Dunn p75)
IV_RANK = {"true_experiment":3, "quasi_experiment":2, "observational":1}
def exp_type(allocates_who, allocates_treatment):
    if allocates_who and allocates_treatment: return "true_experiment"
    if allocates_treatment: return "quasi_experiment"
    return "observational" if not allocates_who else "not_applicable"
for k,(w,t) in [("concrete_beams",(1,1)), ("overlays_vs_mattresses",(0,1)), ("sacks_diets",(1,1))]:
    et = exp_type(bool(w), bool(t))
    print(f"RESULT {k}={et} iv_rank={IV_RANK[et]} cause_and_effect={et == 'true_experiment'}")
print(f"RESULT factorial_cells={len(GRID)} target_per_cell={math.ceil(N_DIET / len(GRID))}")
```
Prints `factorial_cells=4 target_per_cell=203` (811 adults over a 2x2 fat x protein grid, rounded up).
Internal validity ranks true > quasi > observational; only a well-designed true experiment licenses a
cause-and-effect claim (Dunn p71, p73).

### Variable roles: extraneous vs confounding vs lurking
```python
import numpy as np
rng = np.random.default_rng(42)
n = 10000
smoker  = (rng.random(n) < 0.30).astype(float)
passive = (rng.random(n) < 0.25).astype(float)
lighter = (rng.random(n) < 0.10 + 0.60 * smoker).astype(float)             # explanatory
cancer  = (rng.random(n) < 0.02 + 0.20 * smoker + 0.05 * passive).astype(float)  # response
def rr(a, b): return np.corrcoef(a, b)[0, 1]
def role(r_y, r_x, measured, tol=0.05):
    if abs(r_y) > tol and abs(r_x) > tol: return "confounding" if measured else "lurking"
    return "extraneous" if abs(r_y) > tol else "chance_variation"
for k, m in [("smoker_recorded",1), ("smoker_not_recorded",0), ("passive_smoking_recorded",1)]:
    s = smoker if k.startswith("smoker") else passive
    print(f"RESULT {k}={role(rr(s,cancer), rr(s,lighter), bool(m))}")
```
Reproduces the cigarette-lighter example: smoking is a confounder when recorded and a lurking variable
when not; passive smoking moves only the response, so it stays extraneous (Dunn p106-108).

### Chance: many individuals beats many repeats
```python
import numpy as np
BETWEEN_SD, WITHIN_SD = 12.0, 4.0            # seeded demo magnitudes; source gives no typing SDs
def se_grand_mean(n_people, n_repeats):
    return np.sqrt(BETWEEN_SD**2 / n_people + WITHIN_SD**2 / (n_people * n_repeats))
opts = {"one_person_30x": (1,30), "30_people_1x": (30,1), "10_people_3x": (10,3)}
se = {k: se_grand_mean(*v) for k, v in opts.items()}
for k, v in se.items(): print(f"RESULT {k}_se={v:.3f}")
print(f"RESULT best_design={min(se, key=se.get)}")
```
`30_people_1x` prints the smallest SE: with one person repeated 30 times the between-individual
variation is invisible; between-individual variation usually dominates, so prefer more individuals
over more repeats (Dunn p109).

### Pre-analysis protocol
```python
import json, math, pathlib, tempfile
protocol = {"P": "811 overweight adults", "O": "change in body weight after 2 years",
            "C": "four diets", "I": "researchers randomly assigned the diets",
            "design": "true_experiment", "allocates_who": 1, "allocates_treatment": 1,
            "n": 811, "factor_levels": {"fat": 2, "protein": 2},
            "record_baselines": ["age", "sex", "other candidate extraneous variables"],
            "cause_and_effect_claim": True, "external_validity": "random sample of the intended population"}
protocol["cells"] = protocol["factor_levels"]["fat"] * protocol["factor_levels"]["protein"]
protocol["target_per_cell"] = math.ceil(protocol["n"] / protocol["cells"])
path = pathlib.Path(tempfile.mkdtemp()) / "protocol.json"
path.write_text(json.dumps(protocol), encoding="utf-8")
print(f"RESULT protocol_cells={protocol['cells']} target_per_cell={protocol['target_per_cell']} bytes={path.stat().st_size}")
```
The protocol is a plain dict + JSON written into a `tempfile` dir; the cause-and-effect flag is only
settable for a true experiment (Dunn p71-72).

## Output contract
Report fields: `study` (Descriptive | Observational | Experimental), `subtype`
(`retrospective | prospective | cross-sectional | true_experiment | quasi_experiment`), `iv_rank`
(3|2|1), `cause_and_effect` (bool), `variable_roles` (extraneous/confounding/lurking/chance per
covariate), `matching_controls_per_case` where relevant, and the protocol dict above written to
`protocol.json` in a temp dir. Every block prints `RESULT name=value`.

## Validity conditions
- Internal validity means the outcome-comparison association is attributable to the comparison,
  with other explanations minimised (Dunn p72, p104).
- External validity needs internal validity first and ideally a random sample; it concerns the
  intended population, not everyone wider (Dunn p72).
- The groups being compared must look similar at baseline - compare baseline characteristics before
  trusting any contrast (Dunn p73).
- Plan for all four impacts on the response: explanatory variable, extraneous variables, design
  issues, chance; unrecorded extraneous effects simply become chance variation (Dunn p104, p107).
- Experiments can be ruled out by ethical, financial, practical or logistical constraints; then the
  causal claim is off the table (Dunn p72).

## Out of scope
Implementing randomisation, allocation checks, blinding, washouts or confounding adjustment inside a
chosen design (sibling confounding/blinding skills own that); sampling method and sample-size planning
(Dunn ch.5, own skill); CI and test arithmetic. This skill stops at "design chosen + protocol encoded".

## Conventions
- Seed every simulation with `numpy.random.default_rng(42)`; write only inside a `tempfile` dir;
  no figures (if one is added, `matplotlib.use("Agg")`).
- Timing labels are exactly `"now" | "earlier" | "later"`; answer the two allocation questions strictly
  per the experiment comparison table (Dunn p70).
- `ASSOC_TOL = 0.05` is a demo correlation threshold for variable roles; replace with domain judgement.
- pandas + `itertools.product` for the design grid; stdlib `json` for the protocol. Never invent a
  value - always compute it and cite the source page for the method.
- Source: Dunn, *Scientific Research and Methodology* (Springer, 2024) Ch.3 pp.65-76, Ch.6 pp.103-112.
