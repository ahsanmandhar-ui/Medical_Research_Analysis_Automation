---
name: mlk-numerical-features
description: Build ML-ready numeric features from quantitative columns in pure Python (sklearn.preprocessing + numpy/pandas). Min-max rescale to [0,1], z-score standardisation (StandardScaler) and outlier-robust scaling (RobustScaler), per-observation L1/L2 normalisation (Normalizer), polynomial and interaction features (PolynomialFeatures), custom / log / sqrt transforms (FunctionTransformer, np.log, df.apply), IQR and EllipticEnvelope outlier detection plus handling (drop / flag / log-dampen), threshold discretisation (Binarizer, np.digitize) and k-means cluster-membership features. Use when asked to scale, normalise or standardise a numeric feature, add polynomial or interaction terms, apply a log/sqrt transform, detect or dampen outliers, bin or discretise a continuous feature, or turn a numeric column into bins or cluster groups for a model.
---

# Numerical Feature Engineering (Python)

## Purpose
Turn raw quantitative columns into features built for a learning algorithm: rescale, standardise,
normalise, expand with polynomial/interaction terms, apply custom log/sqrt transforms, handle outliers,
discretise into bins, or add a cluster-membership column (Albon ch.4 p76-90).

## Inputs
- A numeric feature column `f`/`x` `[n,1]`, a feature matrix `X` `[n,k]`, or a pandas numeric column.
- A scalar `threshold` / `bins` list for discretisation; a `degree` for polynomial expansion; a `k` for clustering.
- `random_state=1` for any simulated demo data (matches Albon's `make_blobs`); every result is computed and printed.

## Algorithms (all pure numpy/scipy)
### Rescale and standardise (Albon ch.4 p76-79)
```python
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
f = np.array([[-500.5],[-100.1],[0.0],[100.1],[900.9]])              # Albon ch.4 p76
mm = MinMaxScaler(feature_range=(0, 1)).fit_transform(f).ravel()     # x' = (x-min)/(max-min)
assert abs(mm[0]) < 1e-12 and abs(mm[-1] - 1) < 1e-6                 # pinned to 0..1
print("RESULT minmax =", np.round(mm, 8).tolist())                   # [0, 0.28571429, 0.35714286, 0.42857143, 1.0]
x = np.array([[-1000.1],[-200.2],[500.5],[600.6],[9000.9]])          # Albon ch.4 p78
z = StandardScaler().fit_transform(x).ravel()                        # x' = (x-mean)/sd  -> z-score
rb = RobustScaler().fit_transform(x).ravel()                         # (x-median)/IQR, outlier-safe
assert abs(z.mean()) < 1e-9 and abs(z.std() - 1) < 1e-9              # mean 0, sd 1
print("RESULT standard =", np.round(z, 8).tolist())                  # [-0.76058269,-0.54177196,-0.35009716,-0.32271504,1.97516685]
print("RESULT robust =", np.round(rb, 8).tolist())                   # [-1.87387612,-0.875,0,0.125,10.61488511]
```
Standardisation is the default go-to scale; use `RobustScaler` when significant outliers distort the
mean/variance (Albon ch.4 p79, p87).

### Normalise observations to unit norm (Albon ch.4 p80-81)
```python
import numpy as np
from sklearn.preprocessing import Normalizer
X = np.array([[0.5,0.5],[1.1,3.4],[1.5,20.2],[1.63,34.4],[10.9,3.3]])   # Albon ch.4 p80
l2 = Normalizer(norm="l2").transform(X)     # each ROW to unit Euclidean length sqrt(sum x^2)
l1 = Normalizer(norm="l1").transform(X)     # Manhattan/taxicab: each ROW sums to 1
assert np.allclose(np.linalg.norm(l2, axis=1), 1) and np.allclose(l1.sum(axis=1), 1)
print("RESULT l2_row0 =", np.round(l2[0], 8).tolist())     # [0.70710678, 0.70710678]
print("RESULT l1_row0 =", np.round(l1[0], 8).tolist(), "sum =", float(l1[0].sum()))  # [0.5,0.5] 1.0
```
Unlike the scalers above, `Normalizer` rescales across observations (rows), not features, and is
stateless (transform without fit); use it with many equivalent features such as text counts (Albon ch.4 p80).

### Polynomial and interaction features (Albon ch.4 p82)
```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
F = np.array([[2., 3.], [2., 3.], [2., 3.]])                                   # Albon ch.4 p81
d2 = PolynomialFeatures(degree=2, include_bias=False).fit_transform(F)         # x1,x2,x1^2,x1x2,x2^2
io = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False).fit_transform(F)
assert d2.shape[1] == 5 and io.shape[1] == 3
print("RESULT poly_deg2_row0 =", d2[0].tolist())        # [2,3,4,6,9]  (4=2^2, 6=2*3, 9=3^2)
print("RESULT interaction_only_row0 =", io[0].tolist())  # [2,3,6]  only x1*x2 added
```
Encodes a nonlinear feature-target effect (higher powers) and a dependency between two features
(their product) (Albon ch.4 p82-83).

### Custom, log and sqrt transforms (Albon ch.4 p83, p87)
```python
import numpy as np, pandas as pd
from sklearn.preprocessing import FunctionTransformer
F = np.array([[2, 3], [2, 3], [2, 3]])
def add_ten(v): return v + 10
ft = FunctionTransformer(add_ten).transform(F)                                  # Albon ch.4 p83
assert np.array_equal(ft, np.array([[12, 13]] * 3))
print("RESULT add_ten_row0 =", ft[0].tolist())                                   # [12,13]
sqft = pd.DataFrame({"Square_Feet": [1500, 2500, 1500, 48000]})                  # 48000 is an outlier
sqft["Log"] = np.log(sqft["Square_Feet"])                                        # dampen outliers
print("RESULT log_sqft =", ", ".join(f"{v:.6f}" for v in sqft["Log"]))           # 7.313220,7.824046,7.313220,10.778956
assert abs(sqft["Log"].iloc[-1] - 10.778956) < 1e-5
```
`FunctionTransformer` wraps any callable as a pipeline step; the same map is available as
`df.apply(add_ten)` in pandas. Log/sqrt compress a heavy right tail so one extreme value stops dominating scale (Albon ch.4 p87).

### Detect and handle outliers (Albon ch.4 p84-86)
```python
import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.datasets import make_blobs
X, _ = make_blobs(n_samples=10, n_features=2, centers=1, random_state=1)        # Albon ch.4 p84
X[0, 0] = X[0, 1] = 10000
lab = EllipticEnvelope(contamination=0.1).fit_predict(X)                         # -1 outlier, 1 inlier
assert lab[0] == -1 and set(lab[1:]) == {1}
print("RESULT elliptic =", lab.tolist())                                         # [-1,1,1,1,1,1,1,1,1,1]
def iqr_out(v):                                                                  # per-feature 1.5*IQR rule
    q1, q3 = np.percentile(v, [25, 75]); iqr = q3 - q1
    return np.where((v > q3 + 1.5 * iqr) | (v < q1 - 1.5 * iqr))[0]
print("RESULT iqr_outliers =", iqr_out(X[:, 0]).tolist())                        # [0]
```
`contamination` is your estimate of the outlier fraction (an art, not a science). Handling: drop the
observation, flag it with a binary `Outlier` feature, or log-transform its value - choose by whether the
extreme is an error or a genuine mansion (Albon ch.4 p86-87).

### Discretise a feature (Albon ch.4 p88-89)
```python
import numpy as np
from sklearn.preprocessing import Binarizer
age = np.array([[6], [12], [20], [36], [65]])                                    # Albon ch.4 p88
bn = Binarizer(threshold=18).fit_transform(age).ravel()                          # 2 bins: can drink?
assert bn.tolist() == [0, 0, 1, 1, 1]
print("RESULT binarize =", bn.tolist())                                          # [0,0,1,1,1]
d = np.digitize(age.ravel(), bins=[20, 30, 64]).tolist()                         # bins are LEFT edges
dr = np.digitize(age.ravel(), bins=[20, 30, 64], right=True).tolist()            # right=True -> RIGHT edges
assert d == [0, 0, 1, 2, 3] and dr == [0, 0, 0, 2, 3]
print("RESULT digitize =", d, "right =", dr)
```
Discretise when a numeric feature should behave categorically (a 20 vs 21 birthday matters, a 19 vs 20
does not); `Binarizer` for two bins, `np.digitize` for three or more (Albon ch.4 p88-89).

### Cluster-membership feature (Albon ch.4 p90)
```python
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
X, _ = make_blobs(n_samples=50, n_features=2, centers=3, random_state=1)         # Albon ch.4 p90
g = KMeans(n_clusters=3, random_state=0).fit_predict(X)                          # add as a categorical column
assert g[0] != g[1] and len({g[1], g[2], g[3], g[4]}) == 1                       # row0 splits from rows1-4
print("RESULT group_first5 =", g[:5].tolist(), "n_groups =", len(set(g.tolist())))
```
K-means turns a feature matrix into a group-id column of similar observations. The integer labels are
arbitrary and permute across sklearn versions: this build returns the same partition as the printed
example but as `[2,0,0,0,0]` where Albon showed `[0,2,2,2,2]`, so treat the group id as nominal, never ordered.

## Output contract
Each block prints `RESULT name=value` lines. Downstream the model receives: min-max/z-score/robust = a
`[n,1]` or `[n,k]` float matrix in the new scale; Normalizer = a `[n,k]` matrix with unit-norm rows;
PolynomialFeatures = `[n, m]` with m growing combinatorially in degree; custom/log/sqrt = the same shape,
transformed; discretisation = a `[n]` integer bin vector; KMeans = a `[n]` integer group vector. Keep the
fitted scaler/encoder so `transform` and `inverse_transform` stay aligned with the training set.

## Validity conditions
- Fit every scaler on the TRAINING split only, then transform test - fitting on all data leaks (no page, standard practice).
- Min-max is destroyed by outliers (one extreme pins everything else near 0); switch to RobustScaler (Albon ch.4 p79).
- Polynomial degree d explodes the column count - scale after, and prefer `interaction_only` when only products matter (Albon ch.4 p82).
- `np.digitize` bins are LEFT edges by default (`right=False`); an observation equal to a bin falls to the next group (Albon ch.4 p89).
- Cluster labels carry no order and are not stable across versions - encode them as a nominal category (Albon ch.4 p90).

## Out of scope
Deleting (Albon ch.4 p91-92) and imputing (Albon ch.4 p93-94) missing values - the fancyimpute.KNN / removed
`preprocessing.Imputer` recipes - belong to the data-wrangling skill; full clustering modelling (ch.19) to
mlk-clustering; encoding the resulting bins/group ids into model inputs to mlk-categorical-features.

## Conventions
- sklearn 1.9: pass keyword args (`Binarizer(threshold=18)`, `KMeans(n_clusters=3)`) - positional args were removed; use `RobustScaler` not a hand-rolled median/IQR split.
- Simulated demo data via `make_blobs(..., random_state=1)` to match the book; `np.random.default_rng(42)` for any other sampling. Compute and print every value - never hardcode a result.
- The book's printed `Imputer` class and `fancyimpute` are absent on this build; imputation is declared out of scope rather than silently faked.
- Source: Albon, *Machine Learning with Python Cookbook* 2e, Ch.4 "Handling Numerical Data" (Albon ch.4 p76-95).
