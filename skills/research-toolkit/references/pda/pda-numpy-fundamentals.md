---
name: pda-numpy-fundamentals
description: The NumPy ndarray toolkit for array-oriented data work, pure Python (numpy/numpy.random). Creates arrays and casts dtypes, runs vectorised arithmetic, basic/boolean/fancy indexing, transpose and swapaxes, universal functions (ufuncs), array-oriented stats with axis reductions, in-place sorting and set logic, numpy.linalg products/inverses, and vectorised random walks with a seeded generator. Use when asked to build/subset/aggregate a NumPy array, select rows by a boolean mask or integer index array, compute a sum/mean along an axis, sort or unique an array, or simulate random walks.
---

# NumPy Fundamentals - ndarrays and Vectorized Computation (Python)

## Purpose
Express batch operations on whole arrays instead of Python loops (vectorization); NumPy's C algorithms
run 10-100x faster and leaner on large data (McKinney p103). One `ndarray` holds homogeneous data with a
`shape` tuple and a `dtype` (McKinney p104).
## Inputs
- A sequence/nested sequence/array, or a `shape` tuple for `zeros`/`empty`/`arange`.
- An axis (0 = rows/"down", 1 = columns/"across") for reductions, sorts and `cumsum` (McKinney p130).
- A boolean mask the same length as the indexed axis; an integer index array for fancy picks.
- `SEED = 12345` with `numpy.random.default_rng(SEED)` so every printed number is reproducible.
## Algorithms (all pure numpy/scipy)
### Create arrays, read shape/dtype, cast
```python
import numpy as np
arr1 = np.array([6, 7.5, 8, 0, 1])                   # dtype inferred
arr2 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])        # nested -> 2D
trunc = np.array([3.7, -1.2, -2.6, 0.5, 12.9, 10.1]).astype(np.int32)
nums = np.array(["1.25", "-9.6", "42"], dtype=np.bytes_).astype(float)
print("RESULT dtype=%s shape=%s ndim=%d trunc=%s nums=%s"
      % (arr1.dtype, arr2.shape, arr2.ndim, list(trunc), list(nums)))
```
`astype` float->int truncates to `[3,-1,-2,0,12,10]` and always copies; `np.empty` returns uninitialized memory until filled (McKinney p106, p108).
### Vectorised arithmetic and comparison
```python
import numpy as np
arr = np.array([[1., 2., 3.], [4., 5., 6.]])
arr2 = np.array([[0., 4., 1.], [7., 2., 12.]])
print("RESULT square=%s recip00=%.4f gt0=%s"
      % (list((arr*arr)[1]), (1/arr)[0,0], list((arr2>arr)[0])))
```
Equal-size arrays act element-wise (`[16,25,36]`), scalars broadcast, comparisons yield `[False,True,False]`; differing sizes need broadcasting (McKinney p109-110).
### Basic indexing, slicing and view semantics
```python
import numpy as np
arr = np.arange(10); arr[5:8] = 12
sl = arr[5:8]; sl[1] = 12345                          # slice is a VIEW of arr
arr3d = np.array([[[1,2,3],[4,5,6]], [[7,8,9],[10,11,12]]])
print("RESULT slice=%s is_view=%s arr3d_1_0=%s"
      % (list(arr[5:8]), bool(arr[6]==12345), list(arr3d[1,0])))
```
Slicing never copies, so `sl[1]=12345` mutates `arr`; `.copy()` is needed to isolate a slice (McKinney p110-113).
### Boolean indexing (label / value masks)
```python
import numpy as np
names = np.array(["Bob","Joe","Will","Bob","Will","Joe","Joe"])
data = np.array([[4,7],[0,2],[-5,6],[0,0],[1,2],[-12,-4],[3,4]])
print("RESULT bob0=%s not_bob_rows=%d mask_rows=%d"
      % (list(data[names=="Bob"][0]), len(data[~(names=="Bob")]),
         len(data[(names=="Bob") | (names=="Will")])))
```
`~` inverts a mask, `&`/`|` combine them (never `and`/`or`); mask length equals the axis and the result is a copy; `data[data<0]=0` zeroes negatives (McKinney p115-118).
### Fancy indexing with integer arrays
```python
import numpy as np
arr = np.arange(32).reshape((8, 4))
picked = arr[[1, 5, 7, 2], [0, 3, 1, 2]]              # element pairs -> 1D
rect = arr[[1, 5, 7, 2]][:, [0, 3, 1, 2]]             # rows then cols -> 2D block
print("RESULT picked=%s rect_shape=%s" % (list(picked), rect.shape))
```
Two index arrays pair element-wise giving `[4,23,29,10]`; the chained `[:, ...]` form selects a rectangular block; negative indices count from the end and fancy indexing always copies (McKinney p118-120).
### Transpose and inner product
```python
import numpy as np
arr = np.array([[0,1,0],[1,2,-2],[6,3,2],[-1,0,-1],[1,0,1]])
gram = np.dot(arr.T, arr)                             # == arr.T @ arr
print("RESULT gram_diag=%s swapped_shape=%s" % (list(np.diagonal(gram)), arr.swapaxes(0,1).shape))
```
The Gram matrix diagonal is `[39,14,10]` (McKinney p120-121). `.T`, `dot` and `swapaxes` return views, not copies.
### Universal functions (ufuncs)
```python
import numpy as np
rng = np.random.default_rng(12345)
rng.standard_normal((2,3)); x = rng.standard_normal(8); y = rng.standard_normal(8)
mx = np.maximum(x, y); rem, whole = np.modf(rng.standard_normal(7)*5)
out = np.zeros_like(x); np.add(x, 1, out=out)         # out= writes in place
print("RESULT max0=%.4f sqrt9=%.4f exp1=%.4f rem0=%.4f whole0=%.1f out0=%.4f"
      % (mx[0], np.sqrt(9), np.exp(1), rem[0], whole[0], out[0]))
```
Unary ufuncs map one array (`sqrt`,`exp`), binary ufuncs combine two (`add`,`maximum`), `modf` returns two arrays; with the book's `seed=12345` stream `max0` matches its printed maximum (McKinney p123-125).
### Array-oriented stats, where, and grid evaluation
```python
import numpy as np
rng = np.random.default_rng(12345)
for s in [(2,3),(8,),(8,),(7,),(4,4),(5,4)]: rng.standard_normal(s)
pos = (rng.standard_normal(100) > 0).sum()            # bool coerced to 0/1, so sum counts True
cond = np.array([True, False, True, True, False])
xa = np.array([1.1,1.2,1.3,1.4,1.5]); ya = np.array([2.1,2.2,2.3,2.4,2.5])
pts = np.arange(-5, 5, 0.01); xs, ys = np.meshgrid(pts, pts)
z = np.sqrt(xs**2 + ys**2)
print("RESULT pos=%d where=%s corner=%.4f cumsum_last=%d"
      % (pos, list(np.where(cond, xa, ya)), z[0,0], np.arange(8).cumsum()[-1]))
```
`np.where(cond,a,b)` is a vectorised ternary (`sum`/`mean`/`std` take an `axis`; `any`/`all` test a bool array); seeded, 48 of 100 are positive and `meshgrid` gives corner `7.0711` = sqrt(50) (McKinney p126-131).
### Sorting and set logic
```python
import numpy as np
arr = np.array([5, -10, 7, 1, 0, -3])
names = np.array(["Bob","Will","Joe","Bob","Will","Joe","Joe"])
vals = np.array([6, 0, 0, 3, 2, 5, 6])
print("RESULT sorted=%s unique=%s isin=%s"
      % (list(np.sort(arr)), list(np.unique(names)), list(np.isin(vals, [2,3,6]))))
```
`arr.sort()` sorts in place (optionally by `axis`), `np.sort` returns a copy; `unique` gives sorted distinct
values `['Bob','Joe','Will']`, `isin` a membership bool array (McKinney p132-134).
### Linear algebra: dot, @ and inverse
```python
import numpy as np
from numpy.linalg import inv
x = np.array([[1.,2.,3.],[4.,5.,6.]]); y = np.array([[6.,23.],[-1,7],[8,9]])
prod = x.dot(y)
rng = np.random.default_rng(12345)
for s in [(2,3),(8,),(8,),(7,),(4,4),(5,4),(100,),(6,),(5,3)]: rng.standard_normal(s)
X = rng.standard_normal((5,5)); mat = X.T @ X
print("RESULT xy00=%.0f xy11=%.0f inv00=%.4f is_identity=%s"
      % (prod[0,0], prod[1,1], inv(mat)[0,0], bool(np.allclose(mat@inv(mat), np.eye(5)))))
```
`x.dot(y)` gives `[28,64,67,181]` (McKinney p135); on the book's seeded `X.T@X` the printed inverse corner is
`3.4993` and `mat@inv(mat)` is the identity (McKinney p135-136).
### Random walks (vectorised simulation)
```python
import numpy as np
nsteps, nwalks = 1000, 5000
rng = np.random.default_rng(12345)                    # fresh generator per the book
draws = rng.integers(0, 2, size=nsteps)
walk = np.where(draws == 0, 1, -1).cumsum()
draws2 = rng.integers(0, 2, size=(nwalks, nsteps))
walks = np.where(draws2 > 0, 1, -1).cumsum(axis=1)
hits30 = (np.abs(walks) >= 30).any(axis=1)
cross = (np.abs(walks[hits30]) >= 30).argmax(axis=1)
print("RESULT min=%d max=%d cross10=%d hits30=%d mean_cross=%.4f"
      % (walk.min(), walk.max(), (np.abs(walk)>=10).argmax(), hits30.sum(), cross.mean()))
```
A walk is the cumsum of +/-1 steps; `argmax` on a bool array returns the first True (the crossing step). The
book's `seed=12345` gives `-8, 50, 155`, `3395` of 5000 walks hitting +/-30, mean crossing `500.5700` (McKinney p136-139).
## Output contract
Each `RESULT name=value` line is the computed answer: array attributes (`ndim`/`shape`/`dtype`) and a cast
array, element-wise arithmetic and comparison arrays, selected rows (boolean or fancy index), the transpose
Gram diagonal, ufunc outputs, reductions (`sum`/`mean`/`std` with `axis`) and `np.where`, sorted/unique/isin
arrays, a matrix product and inverse, and random-walk statistics (`min`, `max`, first-crossing step, hit
count, mean crossing time). Print one `RESULT` line per block; never hand-write a number you did not compute.
## Validity conditions
- One dtype per array; mixed input upcasts or becomes `object` (McKinney p104).
- A boolean mask length must equal the axis it indexes; the fancy-index result shape follows the index arrays (McKinney p116, p119).
- Vectorised multi-walk simulation allocates `nwalks * nsteps` elements; large grids can exhaust memory (McKinney p139).
## Conventions
- Always `import numpy as np`; the namespace shadows builtins `min`/`max` (McKinney p104).
- Seeds: `default_rng(12345)` (the book's generator); the random-walk block re-seeds fresh.
- Axis model: axis 0 = rows, axis 1 = columns; `x if cond else y` becomes `np.where` (McKinney p128).
- NumPy 2.x removed `np.string_`/`np.in1d`: the blocks use the drop-in `np.bytes_` and `np.isin`; the print
  convention `RESULT name=value` matches the reference skills.
## Out of scope
Broadcasting rules, reshape/memory layout (the book's appendix), `.npy`/`.npz` file IO, and pandas group-by
are not covered here; statistical inference on arrays belongs to the sister skills.
