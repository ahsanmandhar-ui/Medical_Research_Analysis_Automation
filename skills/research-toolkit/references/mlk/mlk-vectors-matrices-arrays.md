---
name: mlk-vectors-matrices-arrays
description: "Albon's linear-algebra-for-ML slice - vectors/matrices/tensors as the numeric substrate every ML recipe sits on, done in pure numpy/scipy. Builds row/column vectors and design matrices, sparse CSR matrices for mostly-zero data, then runs the algebra that models need: dot products, L1/L2/inf norms and cosine angles, vector projection, matrix products, inverse and linear solves, determinant/rank/trace/diagonal, eigendecomposition (Av=lambda v), SVD with energy fractions, and the least-squares gradient plus normal-equations closed form. NumPy arrays (not the deprecated np.mat), modern API, seeded or literal data, every number computed and printed with asserts. Use when asked for the linear algebra behind a model - dot/inner products, norms, projections, matrix multiply/inverse/solve, eigenvalues, SVD/PCA mechanics, or a loss gradient - as opposed to general ndarray slicing or ufunc tutorials (those belong to the NumPy fundamentals skills)."
---

# Vectors, Matrices and Arrays as ML Data (Python)

## Purpose
Give the linear-algebra layer that every model recipe rests on: data as vectors/matrices/tensors, and the
products, norms, decompositions and gradients computed on them (Albon ch.1 p16). This is the algebra slice,
not a NumPy primer - it assumes the ndarray/broadcasting toolkit already.

## Inputs
- Literal arrays or `numpy.random.default_rng(42)` draws; never a file read.
- Vectors as 1D `np.array` (row) or `np.array([[1],[2],[3]])` (column); a feature/design matrix is 2D.
- Square matrices for determinant/inverse/eigen/SVD; `axis` 0 = rows, 1 = columns.

## Algorithms (all pure numpy/scipy)
### Vectors and matrices as data (Albon ch.1 p16-17, p21)
```python
import numpy as np
row = np.array([1, 2, 3])                      # 1D vector, no orientation (p16)
col = np.array([[1], [2], [3]])                # column vector = 2D (3x1) (p16)
X = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])   # design matrix (p21)
assert X.shape == (3, 4) and X.size == 12 and X.ndim == 2
print("RESULT row_shape=%s col_shape=%s X_shape=%s size=%d ndim=%d" % (row.shape, col.shape, X.shape, X.size, X.ndim))
```
NumPy arrays are the de facto structure; `np.mat` is deprecated and most ops return arrays, so use `np.array`
(Albon ch.1 p17). `shape`/`size`/`ndim` are the routine gut-check after every operation (p21).

### Sparse matrix for mostly-zero data (Albon ch.1 p18-19)
```python
import numpy as np
from scipy import sparse
M = np.array([[0, 0], [0, 1], [3, 0]])
S = sparse.csr_matrix(M)
nzc = [tuple(int(i) for i in c) for c in zip(*S.nonzero())]   # nonzero (row, col) pairs
assert S.nnz == 2 and S.shape == (3, 2) and nzc == [(1, 1), (2, 0)]
print("RESULT nnz=%d stored=%s shape=%s density=%.4f" % (S.nnz, nzc, S.shape, S.nnz / (S.shape[0] * S.shape[1])))
```
CSR stores only the two nonzeros at zero-indexed (1,1)=1 and (2,0)=3; adding thousands of zero columns does
not grow the sparse footprint (Albon ch.1 p18-19). No "best" format - CSR/CSC/LIL/COO differ by access pattern.

### Dot product, norms and cosine angle (Albon ch.1 p31)
```python
import numpy as np
a = np.array([1, 2, 3]); b = np.array([4, 5, 6]); v = np.array([3.0, 4.0])
dot = a @ b                                     # sum_i a_i b_i = 32 (also np.dot)
cos = dot / (np.linalg.norm(a) * np.linalg.norm(b))
assert dot == 32
print("RESULT dot=%d l2=%.1f l1=%.1f linf=%.1f cos=%.4f" % (dot, np.linalg.norm(v), np.linalg.norm(v, 1), np.linalg.norm(v, np.inf), cos))
```
The dot product `sum a_i b_i` gives 32; the L2 norm is the Euclidean length, L1 the absolute sum, Linf the max
(Albon ch.1 p31). Cosine = normalised dot, the similarity most distance/attention models reduce to.

### Vector projection (Albon ch.1 p31)
```python
import numpy as np
a = np.array([1., 2., 3.]); b = np.array([4., 5., 6.])
scalar = (a @ b) / (a @ a)                       # signed length of b along a
proj = scalar * a                                # component of b in a's direction
assert np.allclose((b - proj) @ a, 0.0)          # residual is orthogonal to a
print("RESULT scalar=%.6f proj=%s" % (scalar, np.round(proj, 6).tolist()))
```
Projection `(a.b)/(a.a) a` = 2.285714 times a; the leftover `b-proj` is orthogonal to a (dot 0). It is the
least-squares fit of b on one direction and the geometric engine inside Gram-Schmidt and PCA (from p31).

### Matrix product, inverse and solve (Albon ch.1 p33-34)
```python
import numpy as np
A = np.array([[1, 1], [1, 2]]); B = np.array([[1, 3], [1, 2]])
assert (A @ B).tolist() == [[2, 5], [3, 7]]       # rows-by-columns, not element-wise
S = np.array([[1., 4.], [2., 5.]])
inv = np.linalg.inv(S); X = np.array([1., 1.])
assert np.allclose(S @ inv, np.eye(2))
print("RESULT AB=%s detS=%.1f inv=%s solve=%s" % ((A @ B).tolist(), np.linalg.det(S), np.round(inv, 6).ravel().tolist(), np.round(np.linalg.solve(S, X), 6).tolist()))
```
`@` multiplies matrices (`*` is element-wise) giving [[2,5],[3,7]] (Albon ch.1 p33). The inverse satisfies
`A A^-1 = I`, corner -1.666667 (p34); prefer `solve(A,b)` over `inv(A)@b` for stability.

### Determinant, rank, trace and diagonal (Albon ch.1 p27-29)
```python
import numpy as np
D = np.array([[1, 2, 3], [2, 4, 6], [3, 8, 9]])
R = np.array([[1, 1, 1], [1, 1, 10], [1, 1, 15]])
assert np.isclose(np.linalg.det(D), 0.0)          # rows linearly dependent
assert np.linalg.matrix_rank(R) == 2
assert D.trace() == sum(D.diagonal()) == 14
print("RESULT detD=%.1f rankR=%d trace=%d diag=%s off=%s" % (np.linalg.det(D), np.linalg.matrix_rank(R), D.trace(), D.diagonal().tolist(), D.diagonal(offset=1).tolist()))
```
`det`=0.0 flags a singular matrix (p28); `rank`=2 is the dimension of the spanned space (p27); trace sums the
diagonal 1+4+9=14 (p29); `diagonal(offset=1)` reads the super-diagonal [2,6] (p29).

### Eigendecomposition (Albon ch.1 p30)
```python
import numpy as np
A = np.array([[1, -1, 3], [1, 1, 6], [3, 8, 9]])
w, V = np.linalg.eig(A)
assert np.allclose(A @ V[:, 0], w[0] * V[:, 0])   # A v = lambda v
print("RESULT eigenvalues=%s check_all=%s" % (np.round(w.real, 8).tolist(), bool(np.allclose(A @ V, V @ np.diag(w)))))
```
Eigenvectors change only in scale under A; values `13.55075847, 0.74003145, -3.29078992` match the book
(Albon ch.1 p30). Column `V[:,i]` pairs with `w[i]`; this is the core of PCA and spectral clustering.

### SVD and explained energy (Albon ch.1 p30)
```python
import numpy as np
X = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [2., 4., 1.]])
U, s, Vt = np.linalg.svd(X, full_matrices=False)
assert np.allclose(U @ np.diag(s) @ Vt, X)
energy = (s**2 / np.sum(s**2)).round(4)
print("RESULT singular=%s top_energy=%.4f rank=%d" % (np.round(s, 4).tolist(), energy[0], np.linalg.matrix_rank(X)))
```
`X = U s Vt` factors any matrix into orthogonal directions with singular values; the squared-value share of the
top component is the energy a k-component truncation keeps (from p30; rank from p27). Basis of PCA/LSA.

### Matrix calculus: gradient and normal equations (Albon ch.1 p34)
```python
import numpy as np
X = np.array([[1., 1.], [1., 2.], [1., 3.]]); y = np.array([2., 4., 6.])
beta = np.linalg.inv(X.T @ X) @ X.T @ y           # closed-form least squares
g0 = X.T @ (X @ np.zeros(2) - y)                  # gradient of 0.5*||Xb-y||^2 at b=0
b = np.zeros(2)
for _ in range(2000):
    b = b - 0.05 * (X.T @ (X @ b - y))            # gradient descent step
assert np.allclose(beta, [0, 2], atol=1e-6) and np.allclose(b, [0, 2], atol=1e-3)
print("RESULT beta=%s g0=%s gd=%s" % (np.round(beta, 4).tolist(), g0.tolist(), np.round(b, 3).tolist()))
```
The normal-equation solution is `[0, 2]`; the loss gradient `X^T(Xb-y)` starts at `[-12, -28]` and descent
recovers the same beta (inverse from p34). This gradient is the atom every trained model iterates on.

## Output contract
Each block prints `RESULT name=value` lines: array shapes/size, sparse `nnz`/density, `dot`/norms/cosine, a
projection vector, `AB`/`det`/`inv`/`solve`, rank/trace/diagonal, eigenvalues, SVD singular values plus top
energy, and least-squares `beta`/`g0`/`gd`. Compute and print every value - never hand-write a result.

## Validity conditions
- determinant, inverse, eigen and (square) SVD need a square matrix; singular matrices (`det`=0) have no
  inverse (Albon ch.1 p28, p34).
- Projection/normal-equation solve presume `a.a` and `X^T X` non-singular; rank-deficient X needs pinv/ridge.
- Sparse CSR pays off only when zeros dominate; dense data wastes memory and time (p18-19).

## Conventions
- Use `np.array` never the deprecated `np.mat`; `@` for products, `np.linalg.*` for algebra, `np.isclose` for
  float det/inverse checks; modern `scipy.sparse.csr_matrix` (Albon ch.1 p17-18).
- Seeds via `default_rng(42)` for draws; algebra demos use literal arrays so the printed numbers are exact.
- Eigenvalues/SVD ordering is NumPy's, not the book's sort; assert the invariant (Av=lambda v, reconstruct)
  instead of assuming order. Source: Albon, *ML with Python Cookbook* 2e, Ch.1 "Vectors, Matrices, and Arrays".

## Out of scope
Broadcasting rules, ufunc/stream mechanics, indexing/slicing and set logic (NumPy fundamentals skills);
feature encodings, dimensionality-reduction model fitting and regression training are sibling skills - this
one supplies only the underlying array algebra.
