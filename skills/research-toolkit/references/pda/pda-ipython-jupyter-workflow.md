---
name: pda-ipython-jupyter-workflow
description: Pure-Python reproduction of the IPython/Jupyter analysis workflow (McKinney, Python for Data Analysis ch.2 and app.B) - magic commands %timeit/%time/%run/%who/%hist via stdlib timeit+runpy+statistics, tab completion and ?/?? introspection via dir()+inspect, In[]/Out[] history bookkeeping (_ , __, _iN/_N, Ctrl-R reverse-i-search), and reproducible .ipynb notebook hygiene written as plain JSON. ipython/jupyter/nbformat are NOT installed, so each feature is reimplemented on the stdlib+numpy stack. Use when asked how to time, profile, run or introspect code the way IPython/Jupyter does, or how to structure a notebook for reproducibility.
---

# IPython / Jupyter Workflow for Reproducible Analysis (Python)

## Purpose
Give the interactive tools an analyst uses to work fast and leave a reproducible trail: timing/profiling
magics, running a script in a live namespace, tab completion, `?`/`??` introspection, input/output history,
and a well-formed notebook. The three named packages are absent, so every magic below is re-derived from
the standard library (McKinney p37-40, p528-534).

## Inputs
- `python`, `ipython`, `jupyter`, `nbformat`: the reference behaviours come from these but only `python` is
  present; reproduce with `timeit`, `runpy`, `inspect`, `statistics`, `json`, `os`, `time` + `numpy`.
- A statement or callable to time; a `.py` script to run; a namespace dict; an object to introspect; a code
  cell body to embed in a notebook.
- `SEED = 42` (every block calls `numpy.random.default_rng(42)`); timing is machine-specific and never hardcoded.

## Algorithms (all pure numpy/scipy)

### Timing magics: %timeit and %time
`%timeit` runs a statement many times and reports the mean +/- std dev over several repeat runs; `%time`
runs it once and reports wall (clock) time (McKinney p528, p541-542).
```python
import timeit, statistics, time, numpy as np
rng = np.random.default_rng(42)
a = rng.standard_normal((150, 150))
runs = timeit.repeat(lambda: a @ a, repeat=7, number=50)      # %timeit: 7 runs
per = [r / 50 * 1e6 for r in runs]                            # microseconds per loop
t0 = time.perf_counter(); a @ a; wall = (time.perf_counter() - t0) * 1e3   # %time: once
print(f"RESULT timeit_us={statistics.fmean(per):.1f} sd={statistics.stdev(per):.1f} time_wall_ms={wall:.3f}")
```
The book's 100x100 `np.dot` prints `92.5 us +/- 3.43 us` (McKinney p528) - expect a value of the same order
of magnitude here; the exact number depends on the machine, so it is measured, never asserted.

### Running a script in a live namespace: %run
`%run script.py` executes a file in a fresh namespace (identical to `python script.py`), then the file's
globals are readable in the session; `%run -i` also injects the current interactive names
(McKinney p530-531). `runpy.run_path` is the exact stdlib analogue.
```python
import runpy, tempfile, os
d = tempfile.mkdtemp(prefix="run_")
p = os.path.join(d, "script.py")
open(p, "w").write("def f(x, y, z):\n    return (x + y) / z\na = 5\nb = 6\nc = 7.5\nresult = f(a, b, c)\n")
ns = runpy.run_path(p)                                        # empty namespace; init_globals= for %run -i
print(f"RESULT run_result={ns['result']!r} c={ns['c']}")
```
For `a=5, b=6, c=7.5` this prints `run_result=1.4666666666666666 c=7.5`, matching `In[16]` in the book
(McKinney p530). To profile such a block use the stdlib `cProfile` in place of `%prun`/`%run -p` (p543-545).

### Tab completion and introspection: dir(), ?, ??
Tab completion matches a typed prefix and hides leading-underscore (magic/private) names until you type `_`;
`obj?` shows type + docstring, `obj??` shows source, and `obj.*wild*?` searches the namespace (McKinney
p41-44).
```python
import inspect, numpy as np
def add_numbers(a, b):
    """Add two numbers together."""
    return a + b
sig = str(inspect.signature(add_numbers))                     # Signature: add_numbers(a, b)
try:
    body = inspect.getsource(add_numbers).strip().splitlines()[-1]   # ?? -> source
except (OSError, TypeError):
    body = "return a + b"
table = {"an_apple": 27, "an_example": 42, "any": any, "_hidden": 0, "__dunder__": 1}
completes = sorted(k for k in table if k.startswith("an") and not k.startswith("_"))
loads = sorted(n for n in dir(np) if "load" in n and not n.startswith("_"))   # np.*load*?
print(f"RESULT sig=add_numbers{sig} doc={add_numbers.__doc__!r} body={body!r} completes={completes} np_load={loads[:4]}")
```
`completes` yields `['an_apple', 'an_example', 'any']` and drops `_hidden`/`__dunder__`, exactly the drop-down
behaviour described in the book (McKinney p41-42).

### Command history and In[]/Out[] variables
IPython keeps an on-disk input/output cache: `_` and `__` are the previous two outputs, input line `N` is the
string `_iN` and its result is `_N` (so `eval(_iN)` re-runs it); Ctrl-R does partial reverse-i-search;
`%who`/`%whos` list the namespace (McKinney p530, p533-534).
```python
class Ip:
    def __init__(self, start=25):
        self.n, self.inp, self.out, self.scope = start, {}, {}, {}
    def run(self, code):
        self.n += 1; self.inp[self.n] = code
        try:
            self.out[self.n] = eval(code, self.scope)
        except SyntaxError:
            exec(code, self.scope); self.out[self.n] = None
h = Ip()
h.run("foo = 'bar'")                                          # In[26]
h.run("foo")                                                  # In[27] -> 'bar'
rev = [c for c in h.inp.values() if "foo" in c]                # Ctrl-R reverse-i-search
whos = {k: type(v).__name__ for k, v in h.scope.items() if not k.startswith("__")}
print(f"RESULT _i27={h.inp[27]!r} _27={h.out[27]!r} _26={h.out[26]!r} eval_i27={eval(h.inp[27], h.scope)!r} rev={rev} whos={whos}")
```
This reproduces `_i27='foo'`, `_27='bar'`, `eval(_i27)='bar'` from the book (McKinney p534). `%reset` /
`%xdel` clear the cache so big objects can be garbage-collected (p534).

### Notebook hygiene: a self-contained .ipynb
Saving produces one `.ipynb` file holding code, markdown and evaluated output; the Python kernel is IPython
under the hood (McKinney p38-40). `nbformat` is missing, so build the JSON by hand and execute one cell.
```python
import json, tempfile, os
nb = {"cells": [
    {"cell_type": "markdown", "metadata": {}, "source": ["# Reproducible\n", "seed=default_rng(42)"]},
    {"cell_type": "code", "execution_count": 1, "metadata": {}, "outputs": [],
     "source": ["total = 0\n", "for i in range(100_000):\n", "    if i % 3 == 0 or i % 5 == 0:\n", "        total += i\n"]},
], "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
d = tempfile.mkdtemp(prefix="nb_"); p = os.path.join(d, "analysis.ipynb")
with open(p, "w", encoding="utf-8") as fh:
    json.dump(nb, fh)
back = json.load(open(p, encoding="utf-8"))
assert back["nbformat"] == 4 and all("metadata" in c for c in back["cells"])
scope = {}
exec("".join(back["cells"][1]["source"]), scope)              # run the one code cell
print(f"RESULT ipynb_ok={back['nbformat'] == 4} code_cells={sum(c['cell_type'] == 'code' for c in back['cells'])} cell_total={scope['total']}")
```
The cell body is the interpreter example summing multiples of 3 or 5 under 100_000, printing
`2333316668` (McKinney p63) - one seed, one file, results reproducible from the notebook alone.

## Output contract
Each block prints `RESULT name=value` lines: `timeit_us`, `sd`, `time_wall_ms` (timing); `run_result`, `c`
(%run globals); `sig`, `doc`, `body`, `completes`, `np_load` (introspection); `_i27`, `_27`, `eval_i27`,
`rev`, `whos` (history); `ipynb_ok`, `code_cells`, `cell_total` (notebook). The notebook round-trips to a
`<name>.ipynb` JSON file inside a `tempfile` dir.

## Parsing rules
- Any line starting with `%` is a magic; two `%%` marks a cell/line magic that profiles the whole block
  (McKinney p528-530, p545). Automagic lets bare `%pwd` run as `pwd` when no name collides (p529).
- A line starting with `!` is a shell command; `$var` inside it substitutes a Python value (p535).
- Tab / `?` matching is prefix-based and skips leading-underscore names until `_` is typed (p41-44).

## Validity conditions
- Timings are only meaningful relative to each other on the same machine; never compare a `%timeit` number
  across runs or systems (McKinney p542-543).
- `%run` sees only globals defined up to the first exception; a script whose results live inside `main()`
  hides them from the session - put analysis code at top level (p530, p549).

## Reporting conventions
Report the magic by name and its stdlib stand-in (e.g. `%timeit` -> `timeit.repeat`, `?` ->
`inspect.signature`), state the expected magnitude from the cited page, and give units (us / ms). Cite the
source page for any constant or behaviour claim.

## Out of scope
Language data structures (lists, dicts, comprehensions), pandas I/O and reshaping, and model fitting are
sibling skills; this file covers only the interpreter/IPython/Jupyter working environment (McKinney ch.2,
app.B).

## Conventions
- `SEED = 42` via `numpy.random.default_rng(42)`; timing blocks measure rather than assert a fixed number.
- `ipython`, `jupyter` and `nbformat` are not installed - reproduce their behaviour with stdlib
  (`timeit`/`runpy`/`inspect`/`statistics`/`json`/`os`/`time`) plus `numpy`; `line_profiler`/`%lprun` is also
  unavailable, so use stdlib `cProfile` for macro profiling (McKinney p543-547).
- Never invent a value - compute it and cite the source page; write only inside a `tempfile` dir.
- Source: McKinney, *Python for Data Analysis*, 3rd ed. (O'Reilly, 2022) ch.2 pp.35-64 and app.B pp.527-551.
