---
name: pda-modeling-libraries
description: "Fit statistical and ML models from a pandas frame in pure Python - Patsy formula design matrices (dmatrices, standardize/center, C(), I(), key1:key2 interactions), statsmodels OLS on the array and formula interfaces, AR(p) time-series estimation, and scikit-learn LogisticRegression with train-median imputation and cv=4 cross-validation. Use when asked to build or estimate a regression, logistic or autoregressive model with statsmodels/patsy/sklearn, to design a model matrix from a formula string, or to cross-validate a classifier. Source: McKinney, Python for Data Analysis 3rd ed., ch.12 (PDF pp.423-442)."
---

# Modeling Libraries - Patsy, statsmodels & scikit-learn (Python)

## Purpose
Cross from pandas data-wrangling into model fitting: hand off NumPy arrays, build designs with Patsy
formulas, estimate OLS/AR models with statsmodels, and train and cross-validate a logistic classifier
with scikit-learn (McKinney p423-424).

## Inputs
- A homogeneous numeric `pandas.DataFrame` (or matching NumPy arrays) with response and predictors; a
  formula like `y ~ x0 + x1`, where `+` joins design-matrix terms, not arithmetic (McKinney p426-427).
- Book demos use the author's seed `numpy.random.default_rng(12345)` (McKinney p434); generic demos use
  `default_rng(42)`. Models cannot be fed missing data (McKinney p439-440).

## Algorithms (all pure numpy/scipy)
### pandas -> NumPy handoff and the Patsy design matrix (McKinney p424-428)
```python
import numpy as np, pandas as pd, patsy
data = pd.DataFrame({"x0": [1, 2, 3, 4, 5], "x1": [0.01, -0.01, 0.25, -4.1, 0.],
                     "y": [-1.5, 0., 3.6, 1.3, -2.]})
Xsub = data.loc[:, ["x0", "x1"]].to_numpy()        # subset via loc, then to_numpy (p425)
obj = data.assign(strings=list("abcde")).to_numpy()  # heterogeneous -> dtype=object (p425)
y, X = patsy.dmatrices("y ~ x0 + x1", data)         # DesignMatrix = ndarray + metadata (p427)
Xd = patsy.dmatrices("y ~ x0 + x1 + 0", data)[1]    # "+ 0" suppresses the intercept (p428)
coef = pd.Series(np.linalg.lstsq(X, y, rcond=None)[0].squeeze(),
                 index=X.design_info.column_names)  # reattach parameter names (p428)
dummies = pd.get_dummies(pd.Categorical(["a","b","a","a","b"], categories=["a","b"]),
                         prefix="category")          # manual dummy path (p426)
print(f"RESULT subset={Xsub.shape} obj_dtype={obj.dtype} cols={X.design_info.column_names} "
      f"no_const_shape={Xd.shape} dummies={list(dummies.columns)}")
print("RESULT coef=", {k: round(float(v), 6) for k, v in coef.items()})
assert np.allclose(coef.values, [0.312910, -0.079106, -0.265464], atol=1e-6)
```
Coefficients reproduce the book (Intercept 0.312910, x0 -0.079106, x1 -0.265464, McKinney p428); the
intercept is an OLS convention. Prefer Patsy over manual `get_dummies` joins once nonnumeric columns appear (McKinney p426).

### Formula transforms and categorical encoding (McKinney p429-433)
```python
import numpy as np, pandas as pd, patsy
data = pd.DataFrame({"x0": [1, 2, 3, 4, 5], "x1": [0.01, -0.01, 0.25, -4.1, 0.],
                     "y": [-1.5, 0., 3.6, 1.3, -2.]})
y, X = patsy.dmatrices("y ~ standardize(x0) + center(x1)", data)   # built-in transforms (p429)
new = pd.DataFrame({"x0": [6, 7, 8, 9], "x1": [3.1, -0.5, 0, 2.3], "y": [1, 2, 3, 4]})
new_X, = patsy.build_design_matrices([X.design_info], new)  # reuse TRAIN mean/sd (p429-430)
v = round(float(np.asarray(new_X)[0, 1]), 5)                # book: 2.12132 (McKinney p430)
assert abs(v - 2.12132) < 1e-4
_, Xa = patsy.dmatrices("y ~ I(x0 + x1)", data)             # I() = real column addition (p430)
cat = pd.DataFrame({"key1": list("aabababb"), "key2": [0, 1, 0, 1, 0, 1, 0, 0],
                    "v2": [-1, 0, 2.5, -0.5, 4.0, -1.2, 0.2, -1.7]})
_, Xc = patsy.dmatrices("v2 ~ key1", cat)                   # one level dropped (p430)
_, Xc0 = patsy.dmatrices("v2 ~ key1 + 0", cat)              # all levels, no intercept (p431)
_, Xn = patsy.dmatrices("v2 ~ C(key2)", cat)                # numeric read as categorical (p431)
cat["key2"] = cat["key2"].map({0: "zero", 1: "one"})
_, Xi = patsy.dmatrices("v2 ~ key1 + key2 + key1:key2", cat)  # ANOVA interaction (p432)
print(f"RESULT standardize_x0_6={v} I_sum_cols={Xa.design_info.column_names}")
print("RESULT dropped_level=", Xc.design_info.column_names, "all_levels=", Xc0.design_info.column_names)
print("RESULT C_numeric=", Xn.design_info.column_names, "interaction=", Xi.design_info.column_names)
```

### statsmodels OLS and AR(p), array vs formula interfaces (McKinney p433-437)
```python
import numpy as np, pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.ar_model import AutoReg
rng = np.random.default_rng(12345)                 # the book's reproducibility seed (p434)
def dnorm(mean, variance, size=1):
    if isinstance(size, int):
        size = size,
    return mean + np.sqrt(variance) * rng.standard_normal(*size)
N = 100
X = np.c_[dnorm(0, 0.4, size=N), dnorm(0, 0.6, size=N), dnorm(0, 0.2, size=N)]
eps = dnorm(0, 0.1, size=N)
y = np.dot(X, [0.1, 0.3, 0.5]) + eps               # true beta written down first (p434)
res_arr = sm.OLS(y, X).fit()                       # array route: no constant added for you
print("RESULT x_row0=", np.round(X[0], 4), "noconst_params=", np.round(res_arr.params, 4),
      f"uncentered_R2={res_arr.rsquared:.3f} F={res_arr.fvalue:.2f}")
data = pd.DataFrame(X, columns=["col0", "col1", "col2"]); data["y"] = y
res = smf.ols("y ~ col0 + col1 + col2", data=data).fit()   # formula route: auto intercept (p436)
print("RESULT formula_params=", {k: round(float(v), 6) for k, v in res.params.items()})
print("RESULT formula_tvalues=", {k: round(float(v), 4) for k, v in res.tvalues.items()})
print("RESULT first_predict=", round(float(res.predict(data[:5]).iloc[0]), 6))
init_x, N, b0, b1 = 4, 1000, 0.8, -0.4             # truth: AR(2) with coefficients 0.8 and -0.4
values, noise = [init_x, init_x], dnorm(0, 0.1, N)  # continuing the session keeps the book's draws
for i in range(N):
    values.append(values[-1] * b0 + values[-2] * b1 + noise[i])
ar = AutoReg(values, lags=5).fit()                  # unknown lag count -> fit more, then inspect
print("RESULT ar_params=", np.round(ar.params, 4))  # intercept first, then lag 1..5 (p437)
assert abs(ar.params[1] - 0.8) < 0.05 and abs(ar.params[2] + 0.4) < 0.05
```
Seed 12345 reproduces every book number: x row 0 [-0.9005, -0.1894, -1.0279]; no-constant params
[0.0668, 0.2680, 0.4505], uncentered R2 0.469, F 28.51 (McKinney p435-436); formula params
-0.020799/0.065813/0.268970/0.449419, t-values -0.6525/1.2198/6.3124/6.5674, first prediction
-0.592959 (McKinney p436); AR estimates [0.0235, 0.8097, -0.4287, -0.0334, 0.0427, -0.0567] recover
0.8/-0.4 at lags 1-2, extras near zero (McKinney p437). Formula results are labelled with DataFrame
column names; use `sm.add_constant` only on the array route; `summary()` adds further diagnostics.

### scikit-learn logistic model, imputation and cross-validation (McKinney p438-441)
```python
import warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import cross_val_score
rng = np.random.default_rng(42)                    # Titanic CSV unavailable offline -> seeded analog
n = 800
pclass, female, age = rng.integers(1, 4, n), rng.random(n) < 0.5, rng.normal(30, 13, n)
logit = -1.0 + 0.9 * (pclass == 3) + 2.3 * female - 0.02 * age
train = pd.DataFrame({"Pclass": pclass, "Sex": np.where(female, "female", "male"),
                      "Age": np.where(rng.random(n) < 0.2, np.nan, age),
                      "Survived": (rng.random(n) < 1/(1+np.exp(-logit))).astype(int)})
train["Age"] = train["Age"].fillna(train["Age"].median())    # median of TRAIN fills nulls (p440)
train["IsFemale"] = (train["Sex"] == "female").astype(int)   # binary encoding (p440)
X_train = train.loc[:, ["Pclass", "IsFemale", "Age"]].to_numpy()  # book's predictors (p440)
y_train = train["Survived"].to_numpy()
model = LogisticRegression(max_iter=1000).fit(X_train, y_train)    # fit/predict API style (p440)
scores = cross_val_score(LogisticRegression(max_iter=1000), X_train, y_train, cv=4)
print(f"RESULT median_age={train['Age'].median():.1f} train_acc={np.mean(y_train == model.predict(X_train)):.4f}")
print("RESULT cv4_scores=", np.round(scores, 4), "mean=", round(float(scores.mean()), 4))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")                # sklearn 1.9 deprecation noise only
    C = LogisticRegressionCV(Cs=4, max_iter=1000).fit(X_train, y_train).C_[0]  # grid on C (p441)
print("RESULT cv_C_selected=", round(float(C), 4))
```
Accuracy is `(y_true == y_predict).mean()` (McKinney p440). On the real Kaggle train split the book's
four nonoverlapping folds score [0.7758, 0.7982, 0.7758, 0.7883] (McKinney p441); the seeded analog
lands in the same band. CV simulates out-of-sample prediction to avoid overfitting the training data -
slower to train, usually better and more robust performance (McKinney p440-441).

## Output contract
One `RESULT name=value` line per quantity. Patsy returns `(y, X)` DesignMatrices carrying
`design_info.column_names`; a statsmodels `.fit()` returns results with `params`, `tvalues`, `rsquared`,
`fvalue`, `predict(new_df)`, `summary()`; sklearn models expose `fit`/`predict`/`score`, `coef_` and
`LogisticRegressionCV.C_`. Attach column names to every coefficient vector before reporting it.

## Validity conditions
- No missing values in the design: run `isna().sum()` per column; impute numeric nulls with the training
  median only - in the book Age had 177 nulls in train and 86 in test (McKinney p439-440).
- `to_numpy` on a heterogeneous frame yields a `dtype=object` array, not numbers (McKinney p425).
- With an intercept Patsy drops one categorical level to avoid collinearity; keeping all levels needs
  `+ 0` (McKinney p430-431).
- `standardize`/`center` are stateful - transform new data with `build_design_matrices` using the saved
  in-sample statistics, not the new data's own (McKinney p429-430).
- statsmodels standard errors assume the error covariance is correctly specified (McKinney p435); the
  default `cross_val_score` metric is model-dependent - set `scoring` explicitly (McKinney p441).

## Conventions
- Cite PDF pages of McKinney 3rd ed.; ch.12 is pp.423-442. Formula API is
  `smf.ols("y ~ a + b", data=df).fit()`; the array API needs `sm.add_constant` (McKinney p434-436).
- `np.linalg.lstsq(..., rcond=None)` - the kwarg is now required. Seeded demos only; never hardcode a
  result, compute and print `RESULT`.

## Out of scope
File loading (CSV/HDF5/Parquet) is the data-loading skill's job; NA policy, sentinel codes and outlier
capping belong to the data-cleaning skill; interpreting one fitted line belongs to the linear-regression
skill; trees, clustering and Bayesian models sit outside this chapter's toolkit (McKinney p433, p438).
