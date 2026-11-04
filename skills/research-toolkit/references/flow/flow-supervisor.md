---
name: flow-supervisor
description: Fleet supervisor for the Python analysis-skill fleet - plan and dispatch worker waves (hard cap 10 concurrent agents, one skill per agent, run_in_background), gate builds by re-execution instead of worker claims, run the 2-attempt-then-blocker retry ladder, keep a single scribe for the progress ledger, print per-wave counters, and route any user task by scoring the frontmatter description of every SKILL.md installed under the skills root (pure lexical overlap, stdlib only). Use when asked to orchestrate, dispatch a wave, gate or validate the fleet, retry a failed worker, produce a fleet status roll-up, or answer which skill should handle this.
---

# Fleet Supervisor (Python)

## Purpose
Single orchestrator for a 69-skill fleet in 7 waves: plan the next batch, dispatch background worker
agents, gate every claim by re-running validation, re-queue failures under a bounded retry ladder,
report per-wave status, route unfamiliar tasks to the owning skill. Orchestration and shared plumbing
only - never domain maths `(Dunn p25)`.

## Inputs
- `SKILLS_ROOT = ~/.qoder/skills` - each skill at `<skill>\SKILL.md` with a `description:` frontmatter
  key; routing reads only these files.
- Embedded wave table in the plan block (69 skills; wave sizes 10,10,10,10,10,10,9; categories
  flow=6 srm=22 pda=13 mlk=21 maa=7) - the only run-time routing data.
- Constants `BATCH_CAP = 10`, `MAX_ATTEMPTS = 2`, `CONFIDENT_GAP = 3`; ledger dicts `attempts` and `results`.

## Doctrine (Dunn p25 - using software in research)
Dunn p25's spreadsheet warning - silent transformation, hidden formula errors, no analysis record, bugs
in basics, invisible human error, wrong-tool use - maps to six rules:
- **S1 verify, never trust**: a reported PASS is not evidence until the gate re-runs it; print counters
  a reader can re-count. **S2 mechanical validation loop**: catch drift by execution, not review.
- **S3 single-scribe ledger**: only the supervisor edits the progress document, workers never do; the
  fixed worker prompt keeps the record reproducible from the ledger alone.
- **S4 bounded retries**: 2 attempts then blocker - a third blind "fix" can be worse.
- **S5 small legible batches**: cap 10 concurrent, one skill per agent, blocker list beside every plan.
- **S6 route before you run**: the supervisor answers no domain question itself; zero score escalates.

## Dispatch protocol
- One `Agent` call per skill: `subagent_type: general-purpose`, `run_in_background: true`; never more
  than 10 concurrent - no 11th starts while any wave worker runs; one skill per agent, because bundling
  breaks per-skill retry/blocking (S5).
- Prompt template, fixed verbatim text for every worker: `read <tasks-dir>/<skill>.md and execute it`.
- Await background completion notifications, no sleep-polling; gate-not-claim: record `built` only when
  the validator that executes the skill's code blocks passes - "worker reported built" and
  "gate-confirmed built" stay distinct until re-run (S1/S2).
- Retry ladder: 1st failure re-queues (eligible in the next plan), 2nd moves to the blocker list
  needing a human decision; explicit unblock is the only override; a pass resets attempts (S4).
- Waves ascend 1..7; start wave k+1 only once earlier blockers are resolved or consciously waived.

## Algorithms (all pure numpy/scipy)
### Plan + status + retry ledger (embedded wave table)
```python
import itertools
BATCH_CAP, MAX_ATTEMPTS = 10, 2                      # (Dunn p25) RULE S5 / S4
W1 = "flow-analysis-pipeline-dunn flow-corpus-extraction flow-meta-analysis-pipeline flow-py-analysis-env flow-supervisor flow-unit-hygiene srm-confidence-intervals-comparisons srm-confidence-intervals-single srm-confounding-and-blinding-experimental srm-confounding-and-blinding-observational".split()
W2 = "srm-correlation srm-data-collection-protocols srm-distributions-normal-model srm-graphical-summaries srm-hypothesis-testing-framework srm-hypothesis-tests-categorical srm-hypothesis-tests-means srm-linear-regression srm-numerical-summaries-qualitative srm-numerical-summaries-quantitative".split()
W3 = "srm-probability-and-decision-logic srm-reporting-research srm-research-question-design srm-sampling-methods srm-sampling-variation srm-statistical-reference srm-study-design-selection srm-variable-types-encoding pda-analysis-examples pda-data-cleaning".split()
W4 = "pda-data-loading-io pda-data-wrangling-reshape pda-groupby-aggregation pda-ipython-jupyter-workflow pda-modeling-libraries pda-numpy-advanced pda-numpy-fundamentals pda-pandas-foundations pda-plotting-visualization pda-python-language-core".split()
W5 = "pda-time-series mlk-categorical-features mlk-clustering mlk-data-loading mlk-data-wrangling mlk-datetime-features mlk-dim-reduction-extraction mlk-dim-reduction-selection mlk-image-features mlk-knn".split()
W6 = "mlk-linear-regression mlk-logistic-regression mlk-model-evaluation mlk-model-persistence mlk-model-selection mlk-naive-bayes mlk-neural-networks mlk-numerical-features mlk-svm mlk-text-features".split()
W7 = "mlk-trees-forests mlk-vectors-matrices-arrays maa-effect-sizes maa-forest-plots maa-heterogeneity maa-pooling-models maa-publication-bias maa-subgroup-meta-regression maa-validation-vs-r-stata".split()
WAVES = [W1, W2, W3, W4, W5, W6, W7]
skills = list(itertools.chain(*WAVES))
assert len(skills) == 69 and all(len(w) <= BATCH_CAP for w in WAVES)  # shape check, RULE S1
results, attempts = {"srm-correlation": "built"}, {"srm-linear-regression": 2}  # pass resets; 2 fails -> blocker
built = lambda n: results.get(n) == "built"
blocked = lambda n: results.get(n) == "blocker" or attempts.get(n, 0) >= MAX_ATTEMPTS and not built(n)
pending = [n for n in skills if not built(n) and not blocked(n)]
dispatch, blockers = pending[:BATCH_CAP], [n for n in skills if blocked(n)]
print(f"DISPATCH batch={len(dispatch)} cap={BATCH_CAP} source=embedded-wave-table")
for i, n in enumerate(dispatch, 1):
    print(f"  {i:>2}. {n}  prompt: read <tasks-dir>/{n}.md and execute it")
print(f"BLOCKERS (attempts exhausted, need human): {', '.join(blockers) or 'none'}")
for k, w in enumerate(WAVES, 1):
    print(f"  wave {k}: {len(w):>2} skills  built={sum(map(built, w)):>2}")
print(f"ROLLUP {len(skills)} skills in {len(WAVES)} waves of at most {BATCH_CAP}; built={sum(map(built, skills))} on this machine; structural status only (Dunn p25).")
print(f"RESULT total_skills={len(skills)} waves={len(WAVES)} built={sum(map(built, skills))} mode=structural "
      f"dispatch_count={len(dispatch)} pending_total={len(pending)} blocker_count={len(blockers)} batch_cap={BATCH_CAP}")
```
Worked numbers (re-countable, RULE S1): `total_skills=69 waves=7`, sizes 10,10,10,10,10,10,9; with the demo ledger `built=1 blocker_count=1` (srm-linear-regression, 2/2 attempts) `dispatch_count=10`.

### Gate - structural check; live check needs a fleet project
```python
from pathlib import Path
root, found, names_ok = Path.home() / ".qoder" / "skills", 0, True
for p in sorted(root.glob("*/SKILL.md")):
    text = p.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
    found += 1
    names_ok &= text.startswith("---\nname: " + p.parent.name + "\n")
print(f"GATE installed_skills={found} frontmatter_names_ok={bool(names_ok)} "
      f"RESULT registry_ok={int(names_ok and found > 0)} live_checked=0")
print("NOOP live gate: built-state unknown - re-running the project's block-executing validator "
      "is the gate; structural check only (Dunn p25 S1/S2)")
```
A missing project still exits 0 with status labelled `mode=structural`; presenting structural status as
live validation is the cardinal reporting error `(Dunn p25)`.
### Route "<user task>" - top 5 installed skills by keyword overlap
```python
import glob, os, re
from pathlib import Path
QUERY = "funnel plot asymmetry"                        # replace with the user's task text
STOP = set("the a an and or of to in on for with by is are was be this that how what my our use "
           "using need want please can do does from as it its i you we they should would".split())
def description(t):                                   # frontmatter description: value + continuation lines
    end = t.find("\n---", 3)
    out, seen = [], False
    for ln in (t[3:end].splitlines() if t.startswith("---") and end > 0 else []):
        if re.match(r"^[a-zA-Z_-]+:", ln):
            if seen: break
            m = re.match(r"^description:\s*(.*)", ln)
            if m: seen, out = True, [m.group(1)]
        elif seen and ln.strip(): out.append(ln.strip())
    return " ".join(out)
def toks(s):                                          # lowercase, stopword-strip, naive plural stem
    o = []
    for t in re.split(r"[^a-z0-9-]+", s.lower()):
        t = t.strip("-")
        if len(t) > 3 and t.endswith("s") and not t.endswith("ss"): t = t[:-1]
        if t and t not in STOP: o.append(t)
    return o
q = set(toks(QUERY)); scored = []
for path in sorted(glob.glob(str(Path.home() / ".qoder" / "skills" / "*" / "SKILL.md"))):
    name = os.path.basename(os.path.dirname(path))
    d = description(open(path, encoding="utf-8", errors="replace").read().replace("\r\n", "\n"))
    D, N = set(toks(d)), set(toks(name))               # objective/stack now live in the description
    scored.append((3.0 * len(q & D) + 2.0 * len(q & N), name))
scored.sort(key=lambda t: (-t[0], t[1])); top = scored[:5]
gap = top[0][0] - (top[1][0] if len(top) > 1 else 0)
print(f'ROUTE query="{QUERY}" corpus={len(scored)} method=lexical-overlap weights=desc3/name2')
for i, (v, n) in enumerate(top, 1): print(f"  {i}. {n} score={v:g}")
print(f"RESULT top_skill={(top[0][1] if top[0][0] > 0 else 'none')} top_score={top[0][0]:g} gap={gap:g} "
      f"confident={gap >= 3} candidates=" + ",".join(n for _, n in top))
```
Pure lexical, no LLM calls - routing is reproducible from the log (S6). Worked example as authored:
`funnel plot asymmetry` -> `top_skill=flow-meta-analysis-pipeline score=6 gap=3`; once the maa-* wave
installs, the expected answer is `maa-publication-bias score=6`, next best 3.

## Output contract
Human lines (DISPATCH + prompts, BLOCKERS, per-wave counters, ROLLUP, GATE/NOOP, ROUTE top 5) always
end with machine-checkable tails: `RESULT total_skills= waves= built= mode= dispatch_count=
pending_total= blocker_count= batch_cap=10`, `RESULT registry_ok= live_checked=`, `RESULT top_skill=
top_score= gap= confident= candidates=`. One append-style ledger holds attempts and results; only the
supervisor writes the progress document (S3). Reports state the mode, route confidence as "score n,
gap g", attempts inside results ("passed after 1 re-queue"), blockers as "2/2 attempts + requested
decision" `(Dunn p25)`.

## Validity conditions
- Wave table shape: sizes must sum to `total_skills` and every wave stay <= 10; dispatching against a
  half-edited table is the silent-transformation failure mode - regenerate, never patch mid-run (S1).
- Routing is lexical: a synonym missing from every description scores 0; rephrase into the table's
  vocabulary, then ask the user - never run the task inside the supervisor (S6).
- The ledger is machine-local: a fresh box without the fleet project shows `built=0` - correct, not corrupt.
## Out of scope
Single-domain analyses belong to whichever skill `route` names - statistics `srm-*`, pandas/numpy
`pda-*`, machine learning `mlk-*`, meta-analysis `maa-*`, pipeline `flow-analysis-pipeline-dunn`,
extraction `flow-corpus-extraction`, environment freeze `flow-py-analysis-env`, hygiene
`flow-unit-hygiene`. This skill orchestrates and hands over, never runs a domain task itself.

## Conventions
- Stdlib only (`itertools`, `re`, `glob`, `os`, `pathlib`); this orchestrator carries no numerics, so
  no numpy/scipy imports - the `## Algorithms` heading text is fixed by the machine-parsed catalog.
- Scoring: each distinct query token, description match = 3, skill-name match = 2; top 5 sorted by
  (-score, name); ties and zero scores follow the category map, then escalate to the user.
- The skills root (`Path.home()/.qoder/skills`) is the only filesystem path read at run time; worker
  briefs are referenced only via the fixed prompt template.
- Source: Dunn, *Scientific Research and Methodology* (Springer, 2024), sec. 1.4, p25 - paraphrased.
