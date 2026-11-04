---
name: pda-numpy-advanced
description: Advanced NumPy in pure Python (numpy) - ndarray internals (shape/strides/flags, zero-copy strided views, dtype hierarchy via np.issubdtype), the broadcasting rule with shape-checked np.newaxis reshaping, ufunc methods (reduce/accumulate/outer/reduceat), fancy-indexing idioms (take/put, argsort/lexsort, searchsorted, partition), structured arrays and np.memmap disk roundtrips, plus C- vs F-contiguous performance notes. Use when fixing "operands could not be broadcast together" errors, handling arrays too large for RAM, reading/writing binary fixed-record files, building indirect or multi-key sorts, or tuning numpy hot paths.
---

# Advanced NumPy: Internals, Broadcasting, Structured & Memory-Mapped Arrays (Python)

## Purpose
One job: work on ndarray memory, shape and dtype below the pandas layer - zero-copy views, the broadcasting rule, structured/memmap storage of tabular or too-big-for-RAM data (McKinney p491-526).

## Inputs
- `arr`: a numpy `ndarray` (or `np.memmap`); `axis`: int axis for reductions/repeats; `inds`: int index array.
- `dtype`: numpy dtype, or structured spec `[(field_name, field_type[, shape]), ...]`.
- Demo seeds: `numpy.random.default_rng(42)` (book uses seed 12345, McKinney p491); all worked numbers are seed-free book values.

## Algorithms (all pure numpy/scipy)

### Array attributes: shape, strides, flags, dtype hierarchy
```python
import numpy as np
strides = np.ones((3, 4, 5), dtype=np.float64).strides          # C-order 8-byte values (McKinney p492)
assert strides == (160, 40, 8) and np.ones((10, 5)).shape == (10, 5)
base = np.zeros((6, 4)); view = base[::2, ::-1]     # strided view: copies nothing; negative stride walks memory backward
assert view.base is base and view.strides[1] == -8
assert np.issubdtype(np.uint16, np.integer) and np.issubdtype(np.float64, np.floating)
assert np.float64.mro()[:3] == [np.float64, np.floating, np.inexact]   # dtype superclass chain (McKinney p492-493)
arr_f = np.ones((100, 10000), order="F")   # column-major: contiguous down columns, not along rows
assert arr_f.flags.f_contiguous and not arr_f.flags.c_contiguous       # flags check the memory layout (McKinney p523-524)
print(f"RESULT strides={strides} view_copy=False neg_stride={view.strides[1]} f_contig={bool(arr_f.flags.f_contiguous)}")
```
Prints the book's `strides=(160, 40, 8)` - bytes stepped per element per axis; larger strides make computation along that axis costlier (McKinney p492).
`ravel()` views when contiguous, `flatten()` always copies; `order` (`'C'` walks higher axes first, `'F'` last) changes the sequence (McKinney p495-496).

### The broadcasting rule, with shape assertions
```python
import numpy as np
arr = np.random.default_rng(42).standard_normal((4, 3))
d = arr - arr.mean(0)                                # (4,3) with (3,): trailing dimensions match (McKinney p502-503)
assert d.shape == (4, 3) and np.allclose(d.mean(0), 0)
try:
    arr - arr.mean(1)                                # (4,3) with (4,): trailing 3 != 4 and neither length is 1
    raise AssertionError("expected a broadcast error")
except ValueError as e:
    assert "broadcast" in str(e)                     # the book's ValueError, shapes (4,3) and (4,) (McKinney p505)
d = arr - arr.mean(1).reshape((4, 1))                # to broadcast over axis 1 the small array needs shape (4,1) (McKinney p504)
assert d.shape == (4, 3) and np.allclose(d.mean(1), 0)
def demean_axis(a, axis):                            # generalizes [:, :, np.newaxis] to N dimensions (McKinney p507)
    indexer = [slice(None)] * a.ndim
    indexer[axis] = np.newaxis
    return a - a.mean(axis)[tuple(indexer)]
d3 = demean_axis(np.random.default_rng(42).standard_normal((3, 4, 5)), 2)
assert d3.shape == (3, 4, 5) and np.allclose(d3.mean(2), 0)
target = np.zeros((4, 3))
target[:] = np.array([1.28, -0.42, 0.44, 1.6])[:, np.newaxis]  # same rule governs setting values (McKinney p507-508)
assert np.allclose(target[:, 0], [1.28, -0.42, 0.44, 1.6])
print(f"RESULT bcast_ok=True shape_d={d.shape} row_demean_maxerr={np.abs(d.mean(1)).max():.1e}")
```
Rule: compatible when each trailing dimension's lengths match or either is 1; broadcasting fills the length-1/missing dims (McKinney p503).
Fix shapes by reshaping the small operand with newaxis, never by tiling (McKinney p504-505).

### ufunc instance methods: reduce, accumulate, outer, reduceat
```python
import numpy as np
assert np.add.reduce(np.arange(10)) == 45 == np.arange(10).sum()   # reduce = successive binary application, optional axis (McKinney p508)
assert np.add.accumulate(np.arange(15).reshape((3, 5)), axis=1)[2, 4] == 60  # keeps every partial aggregate (McKinney p509)
grid = np.multiply.outer(np.repeat(np.arange(3), [1, 2, 2]), np.arange(5))
assert grid.shape == (5, 5) and grid[4, 4] == 8      # all pairwise products (McKinney p509)
assert np.subtract.outer(np.zeros((3, 4)), np.zeros(5)).shape == (3, 4, 5)  # out ndim = sum of input ndims (McKinney p510)
reduceat = np.add.reduceat(np.arange(10), [0, 5, 8]) # "group by" sums over arr[0:5], arr[5:8], arr[8:] (McKinney p510)
assert [int(v) for v in reduceat] == [10, 18, 17]
rows = np.array([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]])
assert list(np.logical_and.reduce(rows[:, :-1] < rows[:, 1:], axis=1)) == [True, False]  # loop-free row-sorted test; == .all() (McKinney p509)
print(f"RESULT reduce={np.add.reduce(np.arange(10))} reduceat={[int(v) for v in reduceat]}")
```
`reduceat=[10, 18, 17]` is the book's printed value (McKinney p510); prefer broadcasting over tile/repeat when replication only aligns shapes (McKinney p500-501).

### Fancy-indexing idioms: take/put, argsort/lexsort, searchsorted, partition
```python
import numpy as np
arr = np.arange(10) * 100
inds = [7, 1, 2, 6]
assert list(arr.take(inds)) == [700, 100, 200, 600]  # take = fancy index on one axis; pass axis= for others (McKinney p501)
arr.put(inds, [40, 41, 42, 43])                      # put flattens (C order), maps positionally, has no axis arg (McKinney p501-502)
assert list(arr) == [0, 41, 42, 300, 400, 500, 43, 40, 800, 900]
values = np.array([5, 0, 1, 3, 2]); indexer = values.argsort()
assert list(indexer) == [1, 2, 4, 3, 0] and list(values[indexer]) == [0, 1, 2, 3, 5]  # indexer reorders to sorted (McKinney p515)
first = np.array(["Bob", "Jane", "Steve", "Bill", "Barbara"])
last = np.array(["Jones", "Arnold", "Arnold", "Jones", "Walters"])
sorter = np.lexsort((first, last))                   # last key is primary: sorts last_name, then first_name (McKinney p516)
assert [int(v) for v in sorter] == [1, 2, 3, 0, 4]
assert [int(v) for v in np.array([2, 2, 1, 1, 1]).argsort(kind="mergesort")] == [2, 3, 4, 0, 1]  # stable ties (McKinney p516-517)
sa = np.array([0, 1, 7, 12, 15])
assert sa.searchsorted(9) == 3 and [int(v) for v in sa.searchsorted([0, 8, 11, 16])] == [0, 3, 3, 5]  # binary insert points (McKinney p518)
assert [int(v) for v in np.array([0, 0, 0, 1, 1, 1, 1]).searchsorted([0, 1], side="right")] == [3, 7]
r = np.random.default_rng(42).standard_normal(20); p = np.partition(r, 3)  # first k slots = k smallest, unordered (McKinney p517-518)
assert p[:3].max() <= p[3:].min() and list(np.sort(p[:3])) == list(np.sort(r)[:3])
print(f"RESULT argsort={[int(v) for v in indexer]} searchsorted9={sa.searchsorted(9)}")
```
`argsort=[1, 2, 4, 3, 0]` and `searchsorted(9)=3` are the book's values (McKinney p515, p518); `bins.searchsorted(data)` bins in one vectorized call (McKinney p518).

### Structured arrays and a memmap roundtrip in a temp file
```python
import os, shutil, tempfile
import numpy as np
dtype = [("x", np.float64), ("y", np.int32)]         # list of (field_name, field_type) tuples (McKinney p512)
sarr = np.array([(1.5, 6), (np.pi, -2)], dtype=dtype)  # each element is a C-struct-like record (McKinney p511-512)
assert sarr[0]["y"] == 6 and sarr.dtype.names == ("x", "y")
assert np.allclose(sarr["x"], [1.5, np.pi])          # field access returns a strided view, copying nothing
nested = np.zeros(4, dtype=[("x", np.int64, 3), ("y", np.int32)])
assert nested["x"].shape == (4, 3)                   # a field shape turns the column 2-D (McKinney p512)
tmpdir = tempfile.mkdtemp(prefix="pda-memmap-")
try:
    mmap = np.memmap(os.path.join(tmpdir, "mymmap"), dtype=dtype, mode="w+", shape=(1000,))  # book: 10000x10000 float64 (McKinney p521)
    mmap["x"] = np.arange(1000) * 0.5                # slices of a memmap are views onto the on-disk data
    mmap.flush()                                     # writes are buffered; del/garbage-collection also flushes (McKinney p521-522)
    del mmap
    mmap2 = np.memmap(os.path.join(tmpdir, "mymmap"), dtype=dtype, mode="c", shape=(1000,))  # raw bytes: dtype+shape mandatory
    got = float(mmap2[500]["x"]); del mmap2
    assert got == 250.0
finally:
    shutil.rmtree(tmpdir, ignore_errors=True)        # nothing written outside the temp dir
print(f"RESULT structured=ok sarr_y0={sarr[0]['y']} memmap_x500={got}")
```
`sarr_y0=6` reproduces the book's `sarr[0]['y']` (McKinney p512); the reopened memmap returns record 500 as `x=250.0`, proving flush/reopen (McKinney p521-522).
Fixed byte layout (C `struct` style) makes structured arrays the low-level disk/network record store - `np.fromfile` loads legacy fixed-length files -
and pandas has no equivalent of nested fields (McKinney p513-514).

## Output contract
Every block prints exactly one `RESULT name=value ...` line: `strides`/`view_copy`/`neg_stride`/`f_contig`,
`bcast_ok`/`shape_d`/`row_demean_maxerr`, `reduce`/`reduceat`, `argsort`/`searchsorted9`, `structured`/`sarr_y0`/`memmap_x500`.
Functions otherwise return plain ndarrays; the memmap demo writes one raw binary file into a `tempfile` dir it
removes. Passing shape/strides/index assertions are the deliverable.

## Validity conditions
- Broadcasting: for every trailing dimension, lengths must match or one must be 1, else ValueError (McKinney p503).
- `searchsorted` is only correct on a sorted array; `partition` guarantees the k smallest occupy the first k slots but not their order (McKinney p517-518).
- `lexsort` takes keys last-to-first: the final array passed is the primary key (McKinney p516).
- Stable ties need `kind="mergesort"` (O(n log n), n/2 workspace); quicksort is fastest but O(n^2) worst case; heapsort is O(n log n) yet unstable (McKinney p516-517).
- memmap files carry no metadata: reopen with identical dtype and shape; unflushed writes stay invisible to other applications until `flush()` or garbage collection (McKinney p521-522).
- A view is not guaranteed contiguous: `arr_c[:, :50]` of a C-order array is neither C- nor F-contiguous (McKinney p525).

## Reporting conventions
- Print one `RESULT name=value` per quantity; quote shapes/strides as exact tuples; round only reported numbers, never the working.
  Never invent a value - compute it and cite the page; when explaining a broadcast error, quote the failing shapes, e.g. `(4,3)` vs `(4,)` (McKinney p505).

## Conventions
- SEED = `default_rng(42)` for demo draws; worked numbers are deterministic. Mirror the book's display
  (`np.set_printoptions(precision=4, suppress=True)`) when quoting arrays (McKinney p491).
- Performance: loops -> array ops and ufunc methods; broadcast instead of tile/repeat; slice views instead of copies;
  reduce along the contiguous axis - C-order rows summed in 444 us vs 581 us on a 100x10000 float64 array; fix layout
  with `copy("C")`/`copy("F")`; final resort is C/Fortran/Cython (McKinney p523-526).
- No descending sort exists: sort ascending, then take the zero-copy reverse view `arr[:, ::-1]` (McKinney p515).
- The source's Numba section is not executed (`numba` not installed): its 10M-element loop mean-distance took 2 s vs
  14.7 ms vectorized, ~10.3 ms jit; `frompyfunc`/`vectorize` run ~2.43 ms vs 2.88 us for `np.add`, one Python call per
  element (McKinney p511, p519-520). HDF5/PyTables likewise noted, not run.

## Out of scope
Basic ndarray creation/dtypes, simple slicing, boolean masks, one-axis fancy indexing, set logic, linalg and random
sampling belong to the fundamentals skill; DataFrames/reshape/groupby to the pandas skills; figures to the plotting
skill. This file covers only array internals, broadcasting, structured/memmap storage and performance notes.
