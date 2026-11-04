---
name: srm-research-question-design
description: Turn a topic into an answerable quantitative research question in pure Python (numpy/pandas/re) - extract the POCI elements (population, outcome, comparison/connection, intervention), classify it descriptive/relational/interventional, frame it as estimation vs hypothesis-testing, test whether two operational definitions of one concept disagree, count units of observation vs units of analysis (sample size), and build the data-entry frame (one row per unit of analysis, one column per variable). Use when asked to write, refine or critique an RQ, identify P/O/C/I, name the response and explanatory variables, spot pseudo-replication, or set up jamovi/SPSS-style data entry.
---

# Research Question Design (Python)

## Purpose
Turn a vague topic into an answerable quantitative RQ: pull out POCI, classify the RQ type, choose the
estimation or decision framing, pin the operational definitions, and fix the units so the data-entry frame
matches the population; the RQ drives every other aspect of the study (Dunn p33).

## Inputs
- `rq`: one English RQ string; `concept`: a term needing a conceptual plus an operational definition (Dunn p34-35).
- `P, O, C, I`: 0/1 flags - population, outcome, a comparison between distinct subsets *or* a connection with
  another quantity that varies, and an intervention imposed by the researchers (Dunn p37, p40-41).
- Design: `units_of_analysis`, `observations_per_unit` and the operational-definition thresholds; no
  significance level applies, because nothing here is estimated or tested.

## Algorithms (all pure numpy/scipy)
### POCI flags -> RQ type (Table 2.2, Dunn p47)
```python
import numpy as np
# the coeliac topic ladder, labelled by Dunn on the same page: D -> R -> I (Dunn p47)
ladder = [
    ("proportion of Australian adults who are coeliacs", (1, 1, 0, 0)),
    ("coeliac percentage the same for females and males", (1, 1, 1, 0)),
    ("coeliacs given a diet with oats vs a diet without oats", (1, 1, 1, 1)),
]
flags = np.array([f for _, f in ladder])
for topic, poci in ladder:
    P, O, C, I = poci
    assert P and O, "an RQ without a Population or an Outcome cannot be answered"
    poc = "".join(ch for ch, on in zip("POCI", poci) if on)
    kind = "interventional" if I else ("relational" if C else "descriptive")
    print(f"RESULT poc={poc} topic={topic[:24]} rq_type={kind}")
print(f"RESULT n_ladder_rqs={len(ladder)} elements_present={flags.sum(axis=0).tolist()}")
```
Prints `rq_type=descriptive`, `relational`, `interventional` and `elements_present=[3, 3, 2, 1]`: all three
have P and O, C appears in the last two, I only in the last one, and research grows up this ladder as the topic
matures (Table 2.2, Dunn p46-47).
### Estimation vs decision framing (Dunn p47-49)
```python
import re
EST = r"how much|how many|how large|how strong|what is the difference"
YESNO = r"^\s*(?:is|are|was|were|do|does|did)\b|\bthe same\b|\bwhether\b|\brelated to\b|\b(?:shorter|longer|higher|lower|better)\b"
# one echinacea study asked both ways (Example 2.18, Example 2.19, Dunn p48-49)
rq_est = "Among Australian teens with a cold, how much shorter are cold symptoms on average for teens given echinacea?"
rq_dec = "Among Australian teens with a cold, is the average duration of cold symptoms shorter for teens given echinacea?"
for tag, rq in (("Ex2.18", rq_est), ("Ex2.19", rq_dec)):
    es, ds = len(re.findall(EST, rq, re.I)), len(re.findall(YESNO, rq, re.I))
    framing = "estimation/CI" if es else ("decision/hypothesis-testing" if ds else "unclear-rewrite")
    print(f"RESULT example={tag} est_score={es} decision_score={ds} framing={framing}")
```
Returns `estimation/CI` then `decision/hypothesis-testing`, as the source labels them (Dunn p48-49). A "how
much / what is the difference" RQ asks how precisely a population value (a difference, or the strength of a
relationship) is estimated; a yes/no RQ asks whether it exists at all, and a size phrase wins because a size
cannot be reduced to two answers (Dunn p47-49).

### Units of observation vs units of analysis (Dunn p55-57)
```python
import numpy as np
# study: (units of analysis, units of observation nested inside each one)
designs = {
    "hair_thickness_2_men": (2, 100),      # 100 strands per man, Dunn p55-56
    "tyre_wear_10_cars": (10, 4),          # 4 tyres fitted to each car, Dunn p57
    "fragileX_16_mice": (16, 10),          # 10 neurons per mouse, Dunn p57
    "school_PA_44_kids": (44, 1),          # program allocated to each child, Dunn p57
}
arr = np.array(list(designs.values()))
obs = arr[:, 0] * arr[:, 1]
for name, (u_a, per) in designs.items():
    print(f"RESULT design={name} observations={u_a * per} sample_size={u_a} nesting={per}")
print(f"RESULT observations_total={int(obs.sum())} sample_size_total={int(arr[:, 0].sum())}")
```
The hair study gives 200 observations but `sample_size=2`, since one man's strands share shampoo, sun and genes,
and the 160 neurons of 16 mice are 16 units of analysis: sample size counts units of analysis, and studies
compare those, never units of observation (Dunn p55-57).

### Two operational definitions of one concept can disagree (Dunn p34-36)
```python
import numpy as np
rng = np.random.default_rng(42)
n, THR2, THR4 = 60, 5, 3                    # Meline (2006) studies 2 and 4 thresholds, Dunn p36
per150 = rng.poisson(4.6, n)                # within-word disfluences per 150 spoken words
permin = rng.poisson(2.7, n)                # stuttered words per minute
s2, s4 = per150 > THR2, permin > THR4
print(f"RESULT concept=stuttering threshold_study2={THR2}_per_150_words threshold_study4={THR4}_per_minute n={n}")
print(f"RESULT classified_study2={int(s2.sum())} classified_study4={int(s4.sum())} disagree={int((s2 != s4).sum())}")
```
Same people, two defensible operational definitions, different counts and a non-zero `disagree`: always say
which one you used (Dunn p36). A conceptual definition states *what* is measured, the operational one *how* the
value is captured, and commonly accepted definitions should be used unless there is a good reason (Dunn p34-35).

### Data-entry frame: one row per unit of analysis (Dunn p58-59)
```python
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
rng = np.random.default_rng(42)
active, sedentary = 28, 28                  # the Woolf et al. sample, Dunn p38
frame = pd.DataFrame({
    "ID": np.arange(1, active + sedentary + 1),                    # identifier, not a variable
    "Activity": ["active"] * active + ["sedentary"] * sedentary,   # explanatory variable = the C
    "IronStatus": np.round(rng.normal(0.0, 1.0, active + sedentary), 2),  # response variable -> the O
})
for group, mean in frame.groupby("Activity", observed=True)["IronStatus"].mean().items():
    print(f"RESULT outcome_average_{group}={mean:.3f}")
print(f"RESULT rows={frame.shape[0]} variable_columns={frame.shape[1] - 1} id_column=ID")
with tempfile.TemporaryDirectory() as td:
    entry = Path(td) / "entry.csv"
    frame.to_csv(entry, index=False)
    print(f"RESULT csv_roundtrip_shape={pd.read_csv(entry).shape}")
```
56 rows = the 28 + 28 women = 56 units of analysis, 2 variable columns plus one ID column, with variable names
as column headers rather than a data row (Dunn p58-59). Outcome -> response variable and comparison/connection
-> explanatory variable is the mapping that sets those columns (Dunn p53-54).

## Output contract
Print/report `poc` (flag string), `rq_type` (`descriptive` | `relational` | `interventional`), `framing`
(`estimation/CI` | `decision/hypothesis-testing` | `unclear-rewrite`), `conceptual_definition` plus
`operational_definitions` per term, `response_variable`, `explanatory_variable`, `observations`,
`units_of_analysis` (= `sample_size`), `nesting`, and `data_entry_frame` as `rows` x `variable_columns` (+
optional `ID`), optionally round-tripped to a CSV in a temp dir. One `RESULT name=value` line per quantity.

## Validity conditions
- An RQ is about the population the results generalise to, never the sample studied; opening with "utilising a
  convenience sample at ..." makes it a poor RQ (Dunn p50).
- The outcome summarises the population or its subsets numerically, not one individual's value (Dunn p40, p52).
- A comparison needs two or more distinct subsets treated differently; measuring the right and the left arm on
  everyone is not a comparison - the outcome is then the mean difference and the RQ stays descriptive (Dunn p41, p45).
- An intervention exists only if the researchers allocate the C; using a measuring device is not one (Dunn p41-42).
- Units of analysis must not share a common underlying source, and two individuals cannot show a relationship
  (a 6-day vs a 5-day cold) (Dunn p52, p56).
- FINER: feasible, interesting, novel, ethical, relevant; a project RQ must at least be feasible and ethical,
  and ethical is not negotiable (Dunn p49-50).

## Reporting conventions
- Templates: descriptive "Among {P}, what is {O}?"; relational "Among {P}, is {O} the same for {the groups}?"
  or "is {O} related to {something else}?"; interventional says subjects were given or allocated the treatment,
  and it is worth making that explicit (Dunn p43-46).
- Refine a vague topic ("is it better to take echinacea?") by choosing P, then O, then C, then whether to impose
  I; "better" stays unanswerable until the comparison is named (Dunn p51).
- Name a variable as the aspect that varies across individuals, never as one of its values: "how the phone is
  being used", not "texting" (Dunn p60). POCI need not be exhaustive in the RQ: dose belongs in the operational
  definitions stated elsewhere (Dunn p52).

## Out of scope
- Study-design selection, sampling method and measurement-scale typing of variables belong to sibling skills;
  only the POCI decomposition and the response/explanatory role mapping are done here (Dunn p53-54).
- Computing the interval or running the test for a framed RQ belongs to the confidence-interval and
  hypothesis-test skills; the whole six-step end-to-end pipeline is a separate pipeline skill (Dunn p33).

## Conventions
- Seed every demo sample with `numpy.random.default_rng(42)`; never hardcode a printed number - compute it.
- POCI flag order is P, O, C, I, and the RQ type is read off those flags (Table 2.2, Dunn p47).
- `numpy` for the counting, `pandas` for the entry frame, `re` for the framing classifier (a cue check, no NLP
  or design package); writes go only into a `tempfile` dir and any figure uses headless `matplotlib` (`Agg`).
- Source: Dunn, *Scientific Research and Methodology* (Springer, 2024) Ch.2, PDF pages 33-62.
