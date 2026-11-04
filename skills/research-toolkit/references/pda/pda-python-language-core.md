---
name: pda-python-language-core
description: Core Python language semantics for data analysis, pure standard library (re/itertools/collections/pathlib/os/sys/tempfile, no third-party deps) - choosing and mutating tuples/lists/dicts/sets, slicing, enumerate/zip/sorted/reversed, list-set-dict comprehensions, functions with keyword defaults, scopes and closures, lambda-as-object, generators and generator expressions, itertools grouping, try/except/finally error handling, file read/write modes, and bytes-vs-Unicode decoding with explicit encodings. Use when writing or debugging raw-Python data prep, grouping or counting with dict/set, streaming rows with a generator, or reading UTF-8 text files before pandas touches them.
---

# Python Language Core for Data Analysis (McKinney, PDA Ch. 3)

## Purpose
Built-in language tools pandas/NumPy sit on: sequence types, comprehensions, function/generator forms,
exception patterns, file-encoding discipline (McKinney p65). One job - clean Python objects pre-pandas.

## Inputs
- A `collection` (any iterable) plus optional key/filter callables; scalars `int`, `float`, `str`,
  `bool`, `None` as elements or dict keys (McKinney p77).
- A file `path`, mode `r`/`w`/`x`/`a`/`r+` (+ `b` binary, `t` default) and explicit `encoding` (McKinney p96); fixed literals, no RNG.

## Algorithms (all pure numpy/scipy)
### Tuples, lists, dicts, sets (McKinney p65-79)
```python
tup = (1, 2, 2, 2, 3, 4, 2)
a, b, *rest = (1, 2, 3, 4, 5)                # *rest plucks the tail; _ discards it (McKinney p68)
words = ["apple", "bat", "bar", "atom", "book"]
by_letter = {}
for w in words:
    by_letter.setdefault(w[0], []).append(w)  # one-line grouping (McKinney p76)
A, B = {1, 2, 3, 4, 5}, {3, 4, 5, 6, 7, 8}
print(f"RESULT count2={tup.count(2)} rest={rest} by_letter={by_letter} "
      f"union={sorted(A | B)} inter={sorted(A & B)}")
```
Prints `count2=4`, `rest=[3, 4, 5]` (McKinney p68), `by_letter={'a': ['apple', 'atom'], 'b': ['bat',
'bar', 'book']}` (McKinney p76), union 1-8, intersection `[3, 4, 5]` (McKinney p78). Keys/members must
be hashable - `hash((1, 2, [2, 3]))` raises TypeError, convert to `tuple` (McKinney p77). `insert` is
O(n) vs `append`, `in` is linear on lists but O(1) on dict/set, build with `extend` not `+`, tuple
slots are fixed though mutable elements change in place (McKinney p66-67, p70-71).
### Slicing and built-in sequence functions (McKinney p71-81)
```python
seq = [7, 2, 3, 7, 5, 6, 0, 1]
head, rev = seq[1:5], seq[::-1]              # stop excluded; step -1 reverses (McKinney p72-73)
pairs = list(zip(["foo", "bar", "baz"], ["one", "two", "three"]))  # stops at shortest (p80)
b = ["saw", "small", "He", "foxes", "six"]
b.sort(key=len)                              # in-place sort by secondary key (McKinney p71)
print(f"RESULT head={head} rev0={rev[0]} pair0={pairs[0]} shortest={b[0]} enumerated={list(enumerate('ab'))}")
```
`seq[1:5]` is `[2, 3, 7, 5]` - length `stop - start`, stop excluded, negatives from the end (p72).
`enumerate` yields `(i, value)`, `reversed` is lazy, `sorted` builds a new list (McKinney p80-81).
### List, set and dict comprehensions (McKinney p81-83)
```python
strings = ["a", "as", "bat", "car", "dove", "python"]
caps = [x.upper() for x in strings if len(x) > 2]     # [expr for value in collection if cond]
unique_lengths = {len(x) for x in strings}            # set comprehension (McKinney p82)
loc_mapping = {v: i for i, v in enumerate(strings)}   # dict comprehension (McKinney p82)
all_data = [["John", "Emily", "Michael", "Mary", "Steven"],
            ["Maria", "Juan", "Javier", "Natalia", "Pilar"]]
two_as = [n for names in all_data for n in names if n.count("a") >= 2]  # nested: order = nesting
print(f"RESULT caps={caps} lengths={sorted(unique_lengths)} nested={two_as} py_idx={loc_mapping['python']}")
```
`caps` is `['BAT', 'CAR', 'DOVE', 'PYTHON']` (McKinney p81), lengths `{1, 2, 3, 4, 6}` (p82),
`nested` is `['Maria', 'Natalia']` (McKinney p83). Nested `for` clauses read left-to-right like nested
loops with the filter last; >2-3 levels hurts readability; no `if` clause makes it a `map` (p83, p88).
### Functions, keyword defaults, scope, closures (McKinney p84-88)
```python
import re
def my_function2(x, y, z=1.5):               # keyword argument = optional default (McKinney p84)
    return z * (x + y) if z > 1 else z / (x + y)
def clean(strings, ops):                     # functions are first-class objects (McKinney p87)
    out = []
    for s in strings:
        for op in ops: s = op(s)
        out.append(s)
    return out
def counter():                               # local namespace dies at return (McKinney p85)
    n = 0
    def incr():
        nonlocal n; n += 1; return n         # nonlocal rebinds the enclosing scope (p86)
    return incr
c = counter()
ops = [str.strip, lambda v: re.sub("[!#?]", "", v), str.title]   # lambda (McKinney p88)
print(f"RESULT default={my_function2(10, 20)} kw={my_function2(5, 6, z=0.7):.5f} "
      f"pos={my_function2(3.14, 7, 3.5):.2f} cleaned={clean(['   Alabama ', 'Georgia!'], ops)[0]} count={c()}")
```
Prints `default=45.0`, `kw=0.06364`, `pos=35.49` - the three `my_function2` calls of McKinney p84-85 -
and `cleaned=Alabama`. Keyword args follow positionals in any order; no `return` gives `None`;
"multiple" returns are one tuple (or return a dict); avoid `global` (McKinney p84-86).
### Generators and itertools (McKinney p89-91)
```python
import itertools
def squares(n=10):
    for i in range(1, n + 1):
        yield i ** 2                         # nothing runs until the generator is consumed
sq5 = list(squares())[:5]
total = sum(x ** 2 for x in range(100))      # generator expression as function argument (p91)
names = ["Alan", "Adam", "Wes", "Will", "Albert", "Steven"]
groups = [(k, list(g)) for k, g in itertools.groupby(names, lambda x: x[0])]
print(f"RESULT squares={sq5} total={total} groups={groups}")
```
`sum(x**2 for x in range(100))` is **328350** (McKinney p91); squares start `[1, 4, 9, 16, 25]` (p90).
`groupby` groups *consecutive* runs - `A ['Alan', 'Adam']`, `W ['Wes', 'Will']`, `A ['Albert']`,
`S ['Steven']` - sort first for global grouping; generators avoid materializing the list, iterating a
dict yields its keys, and `chain`/`combinations`/`permutations`/`product` share the module (p89-91).
### Exception handling: try / except / finally (McKinney p92-93)
```python
def attempt_float(x):
    try:
        return float(x)
    except ValueError:                       # suppress only the expected failure (McKinney p92)
        return x                             # a TypeError means a real bug - let it surface
log = []
try:
    raise RuntimeError("bad row")
except (RuntimeError, ValueError) as e:      # tuple catches several types (McKinney p93)
    log.append(f"caught:{e}")
finally:
    log.append("closed")                     # finally always runs, e.g. to f.close()
print(f"RESULT ok={attempt_float('1.2345')} passthrough={attempt_float('something')!r} finally={log}")
```
`attempt_float("1.2345")` -> `1.2345` and `("something")` -> `'something'` unchanged (McKinney p92); `else:` runs only when the `try` succeeded (McKinney p93).
### Files and bytes vs. Unicode (McKinney p94-99)
```python
import pathlib, sys, tempfile
poem = "Sueña el rico en su riqueza,\nque más cuidados le ofrece;\n"
with tempfile.TemporaryDirectory() as td:
    p = pathlib.Path(td) / "segismundo.txt"
    p.write_text(poem, encoding="utf-8")     # default encoding varies by platform (McKinney p95)
    with p.open(encoding="utf-8") as f: chars = f.read(10)  # text: 10 Unicode characters (p96)
    with p.open("rb") as f: data = f.read(10)               # binary: exactly 10 raw bytes (p96)
    try:
        data[:4].decode("utf-8")             # cuts ñ mid-byte (McKinney p98-99)
    except UnicodeDecodeError:
        cut = "UnicodeDecodeError"
print(f"RESULT chars={chars!r} len_chars={len(chars)} nbytes={len(data)} "
      f"decoded={data.decode('utf-8')!r} cut={cut} sys_default={sys.getdefaultencoding()}")
```
Text `read(10)` gives `'Sueña el r'` (10 chars) while binary `read(10)` gives `b'Sue\xc3\xb1a el '` -
10 bytes decoding to 9 chars; UTF-8 is variable-length, text-mode `tell()` reports 11 bytes (McKinney
p96-98). Lines keep EOL markers, so `rstrip()` them; `w` truncates, `x` fails if the path exists; open
inside `with` so files close; `seek()` only in binary mode or reads raise UnicodeDecodeError
mid-character; re-encode by reading text and writing a new `encoding` (iso-8859-1) (McKinney p95, p99).

## Output contract
Return Python objects: unpacked tuples or a value-dict from multi-return functions (McKinney p86),
list/set/dict from comprehensions, a lazy generator consumed via `list(gen)`/`sum(...)` (McKinney p90),
`str` from text and `bytes` from binary reads (McKinney p98). Every block prints `RESULT name=value`;
the worked values (`45.0`, `328350`, `['BAT', 'CAR', 'DOVE', 'PYTHON']`, `'Sueña el r'`) reproduce the
source chapter.

## Validity conditions
- Dict keys and set members must be immutable/hashable; a list key raises TypeError (McKinney p77).
- Slice result is `stop - start` with stop excluded; `zip` length is the shortest input (McKinney p72, p80).
- Text-mode `read(n)` counts characters, binary `read(n)` counts bytes - never mix (McKinney p98).
- `groupby` groups only *consecutive* equal keys (McKinney p91).

## Conventions
- Standard library only (`re`, `itertools`, `collections`, `pathlib`, `os`, `sys`, `tempfile`) - language semantics need no numpy/scipy (McKinney p65).
- Always pass `encoding="utf-8"` on text files; `_` is the convention for discarded unpack targets
  (McKinney p68, p95).
- Prefer `defaultdict(list)` over `setdefault` in hot loops, comprehensions over `map` when filtering,
  and at most 2-3 nesting levels (McKinney p76, p83, p88).
- Never invent a value - always compute it and print the `RESULT` lines, citing the source page.
## Out of scope
NumPy arrays and pandas Series/DataFrame (McKinney p100 hands off to array computing), the CSV/JSON/Excel/HDF5 readers and IPython traceback magics (McKinney p94) - all owned by sibling skills.
