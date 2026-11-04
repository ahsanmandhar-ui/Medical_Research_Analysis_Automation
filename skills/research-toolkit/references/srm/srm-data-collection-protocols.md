---
name: srm-data-collection-protocols
description: Write and audit a data-collection protocol plus a survey instrument, then compile the instrument into an executable data-entry schema and validate pilot responses against it, pure Python (numpy, pandas, re; a minimal jsonschema subset reimplemented inline). Runs the STAR reporting gates, the pilot-study gate, the survey framing rules (leading, ambiguous, uninformed, double-barrelled, confidentiality, unclear wording, plus the structural options rule) and numeric answer-option audits that flag a value fitting two bands or none. Use when asked to draft or review a data collection protocol, questionnaire or survey questions, multiple-choice answer options, a pilot study plan, or a validated data-entry schema.
---

# Data Collection Protocols and Survey Instruments (Python)

## Purpose
Two linked jobs: document exactly how data will be obtained (protocol, pilot, STAR reporting), then compile the questionnaire
into a machine-checkable data-entry schema, so weak wording and defective answer options are caught before real collection rather than after it (Dunn p151-156).

## Inputs
- `STAR` / `PILOT`: short strings - the reporting gates a protocol must satisfy and the purposes a pilot study serves.
- `QNS`: `[(q_id, question_text)]` - the instrument, one row per question.
- `SPANS`/`SETS`: `{question: {option_label: (lo, hi, lower_closed, upper_closed)}}` - numeric multiple-choice options.
- `WORDS`: the two wording orders to randomise over; pilot `RESPONSES` are seeded draws, one record per respondent.

## Algorithms (all pure numpy/scipy)

### Protocol register, STAR gates and pilot gate (Dunn p152-153)
```python
import json, tempfile
from pathlib import Path
import pandas as pd
STAR = ["Structured: organised logically, detailed", "Transparent: all needed info, accurate",
        "Accessible: easy to reach, follow, comprehend", "Reporting: whole and in detail, replicable"]
PILOT = ["protocol feasibility", "unforeseen challenges", "data for sample-size calculations", "time and money saved"]
DTYPES = {"item": "string", "kind": "category", "settled": "bool"}
reg = pd.DataFrame([{"item": s, "kind": "STAR", "settled": True} for s in STAR]
                   + [{"item": p, "kind": "pilot", "settled": False} for p in PILOT]).astype(DTYPES)
with tempfile.TemporaryDirectory() as td:
    fp = Path(td) / "protocol.json"
    fp.write_text(json.dumps({"star_all_met": bool(reg.loc[reg["kind"] == "STAR", "settled"].all()),
                              "pilot_steps_open": int((~reg["settled"]).sum())}), encoding="utf-8")
    doc = json.loads(fp.read_text(encoding="utf-8"))
print(f"RESULT dtypes_ok={all(str(reg[c].dtype) == d for c, d in DTYPES.items())} star_components={len(STAR)} "
      f"pilot_purposes={len(PILOT)} doc={doc}")
```
Prints `star_components=4 pilot_purposes=4 doc={'star_all_met': True, 'pilot_steps_open': 4}`. A draft protocol becomes final only after a
small test run has checked it, and the record must let someone else approximately repeat the study, including how blinding and random allocation happen (Dunn p152).

### Framing-rule engine over the instrument (Dunn p153-155)
```python
import re
QNS = [(1, "Because bottled water creates enormous landfill, do you support a ban?"),
       (2, "Do you like this new orthotic?"), (3, "Do children run faster now?"),
       (4, "Do you jog and swim for exercise?"), (5, "Is fibre-composite use waterside likely to make it swell?"),
       (6, "Do you have a water tank installed without council permission?"),
       (7, "Do you agree: I avoid low-fat food unless it is also low-calorie, salt irrelevant?"),
       (8, "Do you like or dislike this new orthotic?"), (9, "Should dogs be required to be vaccinated or not?")]
TEXT_RULES = {"not_leading": r"\bbecause\b|enormous|incredible|unwavering|concerned|support a ban",
              "not_ambiguous": r"\bnow\b|too much|faster", "informed_only": r"fibre-composite|bromodosis",
              "single_barrel": r"\w+ and \w+", "confidential": r"without council permission|illegally",
              "wording_clear": r"unless|irrelevant|not necessarily"}
n_poles = lambda t: len([p for p in ("like", "dislike") if re.search(rf"\b{p}\b", t, re.I)])
flagged, split = {}, {}
for qid, text in QNS:
    hits = [k for k, pat in TEXT_RULES.items() if re.search(pat, text, re.I)]
    if n_poles(text) == 1: hits.append("one_pole_only")
    if hits: flagged[qid] = hits
    if "single_barrel" in hits: split[qid] = len(re.findall(r"\b\w+ and \w+\b", text)) + 1
print(f"RESULT n_rules={len(TEXT_RULES) + 1} n_questions={len(QNS)} n_flagged={len(flagged)} n_clean={len(QNS) - len(flagged)}")
print(f"RESULT split_questions_needed={split} poles_before={n_poles(QNS[1][1])} poles_after={n_poles(QNS[7][1])}")
```
Prints `n_rules=7 n_questions=9 n_flagged=7 n_clean=2` and `split_questions_needed={4: 2} poles_before=1 poles_after=2`: six wording detectors
plus the structural options rule; a double-barrelled item becomes two questions, and a one-pole item gains its opposite pole (Dunn p154-155).

### Option gate and the data-entry schema compiled from it (Dunn p154-157)
```python
import numpy as np, pandas as pd
SETS = {"care_time": {"0-5 min": (0,5,True,True), "5-10 min": (5,10,True,True), ">10 min": (10,np.inf,False,False)},
        "age": {"under 18": (-np.inf,18,False,False), "over 18": (18,np.inf,False,False)},
        "care_time_fixed": {"<5 min": (0,5,True,False), "5-<10 min": (5,10,True,False), ">=10 min": (10,np.inf,True,False)},
        "age_fixed": {"<18": (-np.inf,18,True,False), ">=18": (18,np.inf,True,False)}}
PROBES = {"care_time": range(0,11), "age": range(10,31), "care_time_fixed": range(0,11), "age_fixed": range(10,31)}
def hit(x, spans): return [k for k,(lo,hi,clo,chi) in spans.items() if (x>=lo if clo else x>lo) and (x<=hi if chi else x<hi)]
def assign(x, spans):
    h = hit(x, spans); return h[0] if len(h) == 1 else None
WORDS = ["like or dislike", "dislike or like"]
def schema(min_spans, age_spans):
    return {"type": "object", "required": ["pid","age_band","care_minutes","wording","consent"],
            "properties": {"pid": {"type":"integer","minimum":1}, "age_band": {"enum": list(age_spans)},
                           "care_minutes": {"enum": list(min_spans)}, "wording": {"enum": WORDS}, "consent": {"type":"boolean"}}}
def check(v, r, p="$"):
    t, e = r.get("type"), []
    ty = {"integer": lambda x: isinstance(x,int) and not isinstance(x,bool), "boolean": lambda x: isinstance(x,bool),
          "string": lambda x: isinstance(x,str), "object": lambda x: isinstance(x,dict)}
    if t and not ty[t](v): return [f"{p}: not {t}"]
    if "minimum" in r and v < r["minimum"]: e.append(f"{p}: {v} below min {r['minimum']}")
    if "enum" in r and v not in r["enum"]: e.append(f"{p}: {v!r} not in enum")
    if isinstance(v, dict):
        e += [f"$.{k}: required, absent" for k in r.get("required", []) if k not in v]
        for k, sub in r.get("properties", {}).items():
            if k in v: e += check(v[k], sub, f"$.{k}")
    return e
for name, spans in SETS.items():
    c = {x: len(hit(x, spans)) for x in PROBES[name]}
    print(f"RESULT {name} exclusive={not any(v>1 for v in c.values())} exhaustive={not any(v==0 for v in c.values())} "
          f"shared={sorted(v for v,n in c.items() if n>1)} unassigned={sorted(v for v,n in c.items() if n==0)}")
rng = np.random.default_rng(42); recs = []
for pid in range(1, 11):
    m, a = int(rng.choice([0,5,9,10])), int(rng.choice([17,18,19]))
    base = {"pid": pid, "wording": str(rng.choice(WORDS)), "consent": True}
    recs.append({**base, "age_band": assign(a, SETS["age"]), "care_minutes": assign(m, SETS["care_time"]),
                 "age_fix": assign(a, SETS["age_fixed"]), "min_fix": assign(m, SETS["care_time_fixed"])})
df = pd.DataFrame(recs)
df["err_as_asked"] = [len(check({k: r[k] for k in ("pid","age_band","care_minutes","wording","consent")},
                                 schema(SETS["care_time"], SETS["age"]))) for _, r in df.iterrows()]
df["err_fixed"] = [len(check({**{k: r[k] for k in ("pid","wording","consent")}, "age_band": r["age_fix"],
                                "care_minutes": r["min_fix"]}, schema(SETS["care_time_fixed"], SETS["age_fixed"]))) for _, r in df.iterrows()]
print(f"RESULT n_records={len(df)} n_rejected_as_asked={int((df.err_as_asked>0).sum())} n_rejected_after_option_fix={int((df.err_fixed>0).sum())} n_options={len(SETS['care_time'])}")
```
The gate prints `shared=[5]` (an answer of 5 minutes fits two of the 3 options as asked, Dunn p155-156) and `unassigned=[18]` (exactly 18 fits neither age
option, Dunn p157); closing each lower bound and opening each upper bound prints `exclusive=True exhaustive=True`. Validation then prints
`n_records=10 n_rejected_as_asked=7 n_rejected_after_option_fix=0`: seven of ten seeded pilot records fail the schema compiled from the as-asked bands, and all
pass once the bands are repaired. The `wording` field also randomises which pole is offered first (Dunn p154), and direct entry is the point: paper forms have to be
re-keyed by hand, which is slow, costly and error-prone, while online entry cuts transcription error but risks a lower response rate (Dunn p156).

## Output contract
- Protocol register: DataFrame `item | kind (STAR|pilot) | settled (bool)`, plus `protocol.json` written only inside a `tempfile` dir with `star_all_met` (bool) and `pilot_steps_open` (int).
- Wording audit: `{q_id: [rule_name, ...]}`, `n_clean`, and `split_questions_needed` (how many separate questions a double-barrelled item becomes).
- Option gate: per question `exclusive`, `exhaustive`, `shared`, `unassigned` probe values; repair by moving each bound to half-open.
- Data-entry schema: JSON-Schema-shaped dict (`type`, `required`, `properties`, `enum`, `minimum`) plus per-record `err_as_asked` / `err_fixed` error lists. Every quantity is reported as a `RESULT name=value` line.

## Validity conditions
- A protocol records design, implementation and collection; it is meaningful only if another researcher could approximately repeat the study from it, so pilot before finalising (Dunn p152).
- Options must be mutually exclusive and exhaustive; otherwise a response cannot be assigned to one category and the reported percentage is meaningless (Dunn p154, p156-157).
- Uninformed respondents still answer, and such data is worthless response bias; technical questions require an informed sample or a screening filter (Dunn p154-155).
- Questions that breach confidentiality draw denials and close ethics-committee scrutiny; keeping one needs justification (Dunn p154).
- Open-ended answers feed qualitative analysis and closed ones feed counts; pick the type before building the schema (Dunn p153).

## Reporting conventions
- Report in order: draft protocol, pilot findings and refinements, final protocol, then the instrument with its flag table and repaired option bands (Dunn p151-152).
- Name the offending value, not just the verdict ("5 minutes matches 2 of 3 options"), give the repair, and re-run the gate until the record set validates (Dunn p155-157).

## Out of scope
Sampling frames and study design, ethics submission procedure, non-response weighting, and all downstream cleaning and inference (intervals, tests, correlation) belong to sibling skills.

## Conventions
- `numpy.random.default_rng(42)` for the pilot draws and wording order; never type a result as a literal - compute and print it.
- Probes cover the whole option domain (minutes 0-10, ages 10-30); default to half-open bands (lower closed, upper open).
- `pandas` dtypes pinned to `string` / `category` / `bool`; writes go only into a `tempfile` dir; use the `Agg` backend for any figure.
- The validator is a hand-rolled schema subset (`type`, `enum`, `minimum`, `required`, `properties`) because `jsonschema` is not assumed installed.
- Paraphrase, never copy respondent-facing wording; source: Dunn, *Scientific Research and Methodology* (Springer, 2024) Ch.10, p151-158.
