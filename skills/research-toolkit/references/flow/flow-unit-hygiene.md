---
name: flow-unit-hygiene
description: "Reproducibility and integrity gate for any analysis, pure Python (hashlib/json/re/runpy/subprocess + numpy/scipy). Logs RNG seeds, builds a provenance manifest (SHA-256 per file, harvested seeds, environment, git commit, stable content_hash), keeps a no-cherry-picking runpy analysis log with hypothesis_status pre/post, tests claimed-random allocation over ALL baselines (Welch t, Fisher exact, standardised differences, Fisher combined p), scores a 16-item external/internal/ecological/integrity limitations gate and a reporting checklist. Use when the user wants reproducible analysis, logged seeds, input hashes, provenance or run manifests, an audit for cherry-picked results, CONSORT or baseline-balance allocation-credibility checks, a limitations or strengths-and-limits section, or a final report gate. Triggers: reproducibility, provenance, seed, sha256, git commit, cherry-picking, CONSORT, allocation balance, limitations checklist, ecological validity."
---

# Reproducibility & Integrity Gate (Python)

## Purpose
Enforce the two duties Dunn calls ethical on any analysis: reproducible work - independent scientists repeat results from the same procedures and data, given protocol + dataset + code (Dunn p80) - and honest disclosure: practically every study has limitations, and naming their interpretive impact is important and ethical (Dunn p142).

## Inputs
- `root`: analysis directory holding `protocol.md` (pre-specified hypothesis), `data.csv`, `analysis.py`; every block below writes its own seeded demo tree inside `tempfile`.
- `group` column with exactly two levels coded 0/1, plus `k` numeric baseline columns (continuous or binary).
- `answers`: 16-item limitations map - each item needs non-empty `text` and `severity` in `low|medium|high`.
- Constants: `SEED = 42`, `ALPHA = 0.05` (raw-p flag), `STRONG = 0.01`, `SMD_CUT = 0.25`, `TOTAL = 16`.

## Algorithms (all pure numpy/scipy)
### 1. Seed logging (Dunn p80)
```python
import numpy as np
SEED = 42                                        # a seed is LOGGED, never just remembered
vals = np.random.default_rng(SEED).normal(0.0, 1.0, 8)
print(f"RESULT seed={SEED} mean={vals.mean():.6f} sd={vals.std(ddof=1):.6f} rerunnable={np.array_equal(vals, np.random.default_rng(SEED).normal(0.0, 1.0, 8))}")
```
Prints `seed=42 mean=-0.310733 sd=1.032914 rerunnable=True` - the same integer must reproduce the same numbers.

### 2. Provenance manifest: SHA-256 hashes, seeds, environment, git commit (Dunn p80-81)
```python
import hashlib as hl, json, os, platform, re, shutil, subprocess, sys, tempfile; from pathlib import Path
SEED_RE = [r"default_rng\((\d+)\)", r"(?:np\.random\.)?seed\((\d+)\)", r"random_state\s*=\s*(\d+)", r"\bseed\s*=\s*(\d+)"]
TEXT = {".py", ".ipynb", ".md", ".txt", ".csv", ".json"}; CODE = {".py", ".ipynb"}
root = Path(tempfile.mkdtemp(prefix="prov-"))
for nm, tx in (("protocol.md", "# Protocol\nHypothesis fixed before running.\n"), ("data.csv", "id,x\n1,0.49\n2,-0.31\n3,1.22\n"),
               ("analysis.py", "from numpy.random import default_rng\nrng = default_rng(42)\n")): (root / nm).write_text(tx, encoding="utf-8")
files, seeds = [], set()
for p in sorted(q for q in root.rglob("*") if q.is_file() and q.suffix.lower() in TEXT):
    files.append({"path": p.name, "bytes": p.stat().st_size, "sha256": hl.sha256(p.read_bytes()).hexdigest()})
    if p.suffix.lower() in CODE:
        for pat in SEED_RE: seeds.update(int(m) for m in re.findall(pat, p.read_text(encoding="utf-8", errors="replace")))
def git(*a, env=None): return subprocess.run(("git", "-C", str(root)) + a, capture_output=True, text=True, timeout=30, env=env).stdout.strip()
if shutil.which("git"): genv = dict(os.environ, GIT_AUTHOR_DATE="@0 +0000", GIT_COMMITTER_DATE="@0 +0000"); cfg = ("-c", "user.name=fuh", "-c", "user.email=fuh@local"); git("init", "-q"); git(*cfg, "add", "-A", env=genv); git(*cfg, "commit", "-q", "-m", "demo", env=genv)
man = {"schema": "fuh-provenance/1", "protocol": "protocol.md", "files": files, "seeds": sorted(seeds),
       "environment": {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": __import__("numpy").__version__},
       "vcs": {"available": bool(shutil.which("git")), "commit": (git("rev-parse", "HEAD") if shutil.which("git") else "") or "none"}}
man["content_hash"] = hl.sha256(("".join(f["path"] + f["sha256"] for f in files) + str(man["seeds"])).encode()).hexdigest()
(root / "provenance.json").write_text(json.dumps(man, indent=1, sort_keys=True), encoding="utf-8")
print(f"RESULT files_hashed={len(files)} seeds_found={man['seeds'] or 'none'} git_available={man['vcs']['available']} git_commit={man['vcs']['commit'][:12]} content_hash={man['content_hash'][:16]}")
print(f"RESULT unseeded_code={sum(1 for f in files if f['path'].endswith('.py') and not man['seeds'])} manifest={root.name}/provenance.json")
```
Protocol, dataset and code hashes implement the shareable minimum (Dunn p80); zero harvested seeds in a script is an unlogged-randomness failure (Dunn p81). `content_hash` covers sorted file hashes + seeds only - no clock, no git state - so two runs of one tree agree.

### 3. No-cherry-picking analysis log: runpy ledger, pre/post hypotheses (Dunn p78-79)
```python
import os, runpy, tempfile
from hashlib import sha256; from pathlib import Path
work = Path(tempfile.mkdtemp(prefix="log-"))
for i, nm in enumerate(("a1", "a2", "a3"), 1):
    (work / f"{nm}.py").write_text("from pathlib import Path\nPath('%s.json').write_text('{\"v\": %d}')" % (nm, i), encoding="utf-8")
executed, logged = ["a1", "a2", "a3"], {"a1": "pre", "a2": "post"}   # a3 was run but never written up
log = []
for nm, status in logged.items():
    cwd = os.getcwd(); os.chdir(work)                                 # the script IS the protocol transcript
    try: runpy.run_path(str(work / f"{nm}.py"), run_name="__main__")
    finally: os.chdir(cwd)
    src, out = (work / f"{nm}.py").read_bytes(), (work / f"{nm}.json").read_bytes()
    log.append({"id": nm.upper(), "script": f"{nm}.py", "script_sha256": sha256(src).hexdigest()[:12], "output_sha256": sha256(out).hexdigest()[:12],
                "hypothesis_status": status, "decision": "run", "drop_reason": ""})
gap = sorted(set(executed) - set(logged))
print(f"RESULT executed={len(executed)} logged={len(log)} unlogged={','.join(gap) or 'none'} ledger_gate={'FAIL' if gap else 'PASS'} post_hypotheses={','.join(e['id'] for e in log if e['hypothesis_status'] == 'post') or 'none'} post_disclosed={sum(e['hypothesis_status'] == 'post' for e in log)}")
```
Reporting selects from the log, it never prunes it: deciding hypotheses after the results were known was practised at often-above-50% rates by 807 ecology researchers (Dunn p78), and dismissing inconvenient runs mirrors the Challenger failure mode (Dunn p79).

### 4. Allocation credibility: baseline balance over ALL k variables (Dunn p81)
```python
import numpy as np
from scipy.stats import combine_pvalues, fisher_exact, ttest_ind
rng = np.random.default_rng(42); n = 50; group = np.r_[np.zeros(n, int), np.ones(n, int)]; ALPHA, STRONG, SMD_CUT = 0.05, 0.01, 0.25
cols = {"weight": np.r_[rng.normal(2.0, 1.0, n), rng.normal(5.5, 1.0, n)], "impairment": np.r_[rng.poisson(7.0, n), rng.poisson(6.0, n)],
        "sex": rng.integers(0, 2, 2 * n), "age": np.r_[rng.normal(40, 8, n), rng.normal(41, 8, n)]}
rows = []
for name, x in cols.items():
    a, b = x[group == 1], x[group == 0]; binary = len(set(np.round(a, 9)) | set(np.round(b, 9))) <= 2
    if binary: stat, p = fisher_exact([[int((a > 0).sum()), int((a == 0).sum())], [int((b > 0).sum()), int((b == 0).sum())]])
    else: stat, p = ttest_ind(a, b, equal_var=False)                  # Welch: no equal-variance assumption
    pooled = 1.0 if binary else np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    rows.append((name, "fisher" if binary else "welch-t", float(stat), float(p), float((a.mean() - b.mean()) / pooled)))
ps = np.array([r[3] for r in rows]); k = len(rows); chi2, p_comb = combine_pvalues(ps, method="fisher")
flag = "STRONG_SUSPICION" if ps.min() < STRONG or p_comb < STRONG else ("SUSPICIOUS_ALLOCATION" if ps.min() < ALPHA else "BALANCED")
for name, kind, stat, p, smd in rows: print(f"  {name:<11}{kind:<8}stat={stat:10.4g} p={p:10.4g} smd={smd:7.3f}")
print(f"RESULT n_groups=2 n_vars={k} min_p={ps.min():.4g} fisher_stat={chi2:.4g} fisher_df={2 * k} fisher_p={p_comb:.4g} max_abs_smd={max(abs(r[4]) for r in rows):.3f}")
print(f"RESULT imbalanced_over_cut={sum(abs(r[4]) > SMD_CUT for r in rows)} expected_false_alarms={k * ALPHA:.2f} flag={flag}")
```
Planted demo (Dunn p81 case shape): weight Welch stat 20.96, p = 5.46e-38, SMD 4.19 - the "extraordinarily unlikely" tier; impairment p = 0.033 (chance-level noise among 4 tests); sex (binary, so SMD = difference in proportions) p = 1.00; age p = 0.081; Fisher combined 183.42 on df = 8, p = 1.97e-35 -> `STRONG_SUSPICION`. The same tests on a genuine random split (`default_rng(3).integers(0, 2, 100)` over one pooled sample) give all four p > 0.16 and combined p = 0.83 -> `BALANCED`. |SMD| > 0.25 is the conventional imbalance cut, not a source threshold.

### 5. Sixteen-item limitations gate: external / internal / ecological / integrity (Dunn p141-145, p77-81)
```python
import tempfile; from pathlib import Path
BLOCKS = {"EXTERNAL": [(1, "target population = the one stated in the RQ; sample relation stated (Dunn p142-143)"), (2, "sampling mode named - random / approximately representative / convenience - with generalisability consequence (p143)"),
                        (3, "scope window: settings, times, subgroups NOT covered, e.g. night-only 22:00-07:00 (p147)")],
          "INTERNAL": [(4, "confounding: what is controlled; is the study observational-grade vulnerable (p143-144)"), (5, "carryover possible (p143)"), (6, "Hawthorne: participants aware or unblinded, consequence stated (p143-144)"),
                       (7, "observer effect and self-report social desirability (p144)"), (8, "placebo: sham or attention control (p143)")],
          "ECOLOGICAL": [(9, "methods, materials and context approximate the real situation - Definition 9.1 (p144)"), (10, "efficacy-to-effectiveness gap quantified: ~50% in study vs ~5% after mandate (p145)")],
          "INTEGRITY": [(11, "ethics approval reference (committee, date) recorded (p77, p81)"), (12, "consent, withdrawal without penalty, confidentiality, secure storage (p78)"), (13, "incentives above cost reimbursement disclosed (p78)"),
                        (14, "contributors acknowledged; no collusion, fraud, plagiarism (p78-79)"), (15, "reproducible run: protocol + dataset + code, scripted, seed logged (p80-81)"), (16, "allocation credibility: balance over ALL baselines or documented generation method (p81)")]}
TOTAL = sum(len(q) for q in BLOCKS.values()); SEV = ("low", "medium", "high")
ok = lambda a, i: bool(str(a.get(i, {}).get("text", "")).strip()) and a.get(i, {}).get("severity") in SEV
def score(a):
    ans = sum(ok(a, i) for qs in BLOCKS.values() for i, _ in qs); bl = sum(all(ok(a, i) for i, _ in qs) for qs in BLOCKS.values())
    return ans, round(ans / TOTAL * 100, 1), bl, ("PASS" if ans == TOTAL and bl == 4 else "BLOCKED")
filled = {i: {"text": "threat named + mechanism + consequence for interpretation", "severity": "low"} for qs in BLOCKS.values() for i, _ in qs}
for tag, ans in (("filled", filled), ("item7_blank", {**filled, 7: {"text": "", "severity": None}})):
    a, pct, bl, gate = score(ans); print(f"RESULT case={tag} total_items={TOTAL} answered={a} completeness_pct={pct} blocks_complete={bl} gate={gate}")
out = Path(tempfile.mkdtemp(prefix="lim-")); sk = "## Strengths and limitations\n" + "".join(f"{i}. ({b}) {q} -> [answer] [{filled[i]['severity']}]\n" for b, qs in BLOCKS.items() for i, q in qs)
(out / "limitations_section.md").write_text(sk + "In light of these the study tells us [X] and does not tell us [Y] (Dunn p141).\n", encoding="utf-8")
print(f"RESULT skeleton_items={TOTAL} section_written={out.name}/limitations_section.md")
```
`completeness_pct = answered/16*100`; blocks are items 1-3, 4-8, 9-10, 11-16; `PASS` needs 16/16 answered with text AND a valid severity and all 4 blocks complete (demo: 100.0/4/PASS; one blank item -> BLOCKED). Drafting before PASS is not allowed (Dunn p142).

### 6. Reporting checklist gate
```python
REPORT = """Analysis was scripted and reproducible: protocol, dataset and code are available
(content_hash 4f2a...), seeds [42], run from git commit 7c1e9a2. Baseline balance over all 4 variables
is reported in full (smallest p = 5.46e-38; Fisher combined p = 1.97e-35; max |SMD| 4.19); the allocation
is treated as not random until the sequence generation method is documented. Results may not generalise
beyond the stated target population; they were not intended to cover the wider group, which is not a limitation.
Participants may have behaved differently because they knew they were in a study (Hawthorne), so the effect
is an upper bound. Hypotheses defined after seeing results in the log: A2, disclosed."""
REQUIRED = ["content_hash", "seed", "git commit", "balance", "combined p", "smd", "may not generalise", "not a limitation", "hawthorne", "after seeing results"]
FORBIDDEN = ["proves", "definitely", "without limitation", "no limitations"]
missing = [t for t in REQUIRED if t not in REPORT.lower()]; bad = [w for w in FORBIDDEN if w in REPORT.lower()]
print(f"RESULT checklist_items={len(REQUIRED)} missing={missing or 'none'} certainty_words={bad or 'none'}")
print(f"RESULT report_gate={'REPORT_READY' if not missing and not bad else 'BLOCKED'}")
```

## Checklists
Gate order, each stage blocking the next: script + logged seed -> provenance manifest -> allocation balance (only if random is claimed) -> analysis log -> 16-item limitations gate -> report checklist. Ethics approval, consent, storage and incentives are committee duties; this gate only checks they are on record (Dunn p77-78, p81).

## Output contract
- `provenance.json`: `schema`, `protocol`, `files[] {path, bytes, sha256}`, `seeds[]`, `environment {python, platform, numpy}`, `vcs {available, commit}`, `content_hash`.
- `analysis_log[]`: `id`, `script`, `script_sha256`, `output_sha256`, `hypothesis_status` (`pre|post`), `decision`, `drop_reason`; `limitations_section.md` from the 16-item skeleton; flags `BALANCED|SUSPICIOUS_ALLOCATION|STRONG_SUSPICION`, `gate=PASS|BLOCKED`, `report_gate=REPORT_READY|BLOCKED`.
- Every block ends with `RESULT name=value` lines - parse those, never the prose.

## Validity conditions
- Reproducible means independent scientists repeat the result from the same procedures and data; the minimum is protocol + dataset + code, so a point-and-click or spreadsheet run is not a result (Dunn p80-81).
- k baseline tests under true randomness yield ~0.05k raw p below 0.05 by chance; weigh effect sizes and the combined p, and never test only the variable you expected to differ - that is cherry-picking in reverse (Dunn p78, p81).
- A flagged imbalance does not falsify the study; it voids the word "random" until the allocation-sequence generation method is documented (CONSORT) and downgrades the comparison to observational-grade confounding risk (Dunn p81, p143-144).
- External validity is judged only against the RQ's population - failing to cover a wider group is not a limitation (Dunn p143); a huge n cannot rescue a conflicted design (113,597 doctors surveyed by the sponsor's staff after free cartons, Dunn p142); a design that can never show an effect is a limitation, not a null result (Dunn p147).

## Reporting conventions
Fill the skeleton in this order: reproducibility sentence (`content_hash`, seeds, git commit) (Dunn p80); full balance sentence (`smallest p`, `combined p`, max |SMD|) (Dunn p81); external "may not generalise to X because ...; not intended to cover Y, which is not a limitation" (Dunn p143); each likely internal threat with its consequence ("participants may have behaved differently than usual", Dunn p143-144); the efficacy-to-effectiveness caveat (Dunn p145); close with what the study tells and does not tell us (Dunn p141). Uncertainty stays in may / most likely / unclear (Dunn p146), never in silent omission.

## Out of scope
Designing blinding, controls and confounder adjustment, the sampling design itself, stage-by-stage pipeline planning, and literature-level selective-reporting tests belong to sibling skills (Dunn p143-144).

## Conventions
- Stdlib `hashlib`, `json`, `re`, `runpy`, `subprocess`, `tempfile` for the plumbing; `numpy`/`scipy` for the numerics; each block is independent and writes only inside a `tempfile` dir.
- `git` is optional: the manifest degrades to `git_available=False`, `commit=none` with `content_hash` unchanged; demo git stamps use the POSIX epoch so hashes compare across machines.
- Seeds harvested as integers by four regexes - `default_rng(N)`, `seed(N)`, `numpy.random.seed(N)`, `random_state=N` - and written to the manifest, not to chat memory (Dunn p80).
- The source worked in R/jamovi and CONSORT checklists; both are reimplemented on this Python stack, no installs. No wall-clock or network in default paths, so artefacts stay diffable. Never invent a value - compute it, hash it, and cite the page for every rule: Dunn, *Scientific Research and Methodology* (Springer, 2024) ch.4 pp.77-82, ch.9 pp.141-148.
