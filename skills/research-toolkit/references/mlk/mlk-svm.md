---
name: mlk-svm
description: Train, inspect and tune support vector machines with sklearn.svm (LinearSVC, SVC, SVR) in pure Python per Albon ch.17 - the max-margin hyperplane on standardised features, linear/polynomial/RBF kernels for linearly inseparable classes, the C misclassification penalty with gamma and degree, support_vectors_/support_/n_support_ inspection, Platt-scaled predict_proba, class_weight="balanced" (Ck = C*wj, wj = n/(k*nj)) for imbalanced labels, and an epsilon-insensitive support vector regressor. Use when asked for an SVM/SVC/SVR, the kernel trick, margins or support vectors, C/gamma values, class probabilities from an SVC, or penalising misclassified minority cases.
---

# Support Vector Machines (Python)

## Purpose
An SVM places a hyperplane - an `n-1` subspace cutting an `n`-dimensional space - so the margin between the classes of the training data is maximal
(Albon p282); kernels bend it so linearly inseparable classes still split, and `C` trades margin width against misclassification (Albon p284-285). The
iris recipes z-score the features first (Albon p283) - a margin and one shared penalty only compare sensibly on a single scale; the XOR draw is already standard-normal.

## Inputs
- `X`: numeric `(n_samples, p_features)` matrix; `y`: 0/1 or multi-class labels (a numeric target for SVR); a `StandardScaler` fit on training rows, applied before every fit (Albon p283, p290).
- Hyperparameters tuned *as a set*: `C` (misclassification penalty, default 1.0), `kernel` (`linear`/`poly`/`rbf`), `gamma` (RBF, > 0), `degree` (poly), `class_weight` (`None`/`balanced`), `epsilon` (Albon p285-286, p289).
- A new observation is always 2-D: `svc.predict([[-2, 3]])`, never `[-2, 3]` (Albon p284).

## Algorithms (all pure numpy/scipy)

### Linear classifier, its hyperplane, its support vectors (Albon p282-284, p291-292)
With two features the boundary is the line `y = a*x - b/w1`, `a = -w0/w1`: above it reads class 0, below class 1 (Albon p284), band width `2/||w||`.
`support_vectors_`/`support_`/`n_support_` name the few observations carrying that boundary - remove one and the maximum margin is lost (Albon p292).
```python
import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
iris = datasets.load_iris(); y = iris.target[:100]
X2 = StandardScaler().fit_transform(iris.data[:100, :2])          # 2 classes, 2 features: plottable
svc = LinearSVC(C=1.0, max_iter=20000, tol=1e-4).fit(X2, y)
w, b = svc.coef_[0], svc.intercept_[0]
edge = (-w[0] / w[1]) * (-2.0) - b / w[1]                         # hyperplane height at x = -2
above, below = [[-2.0, edge + 1.0]], [[-2.0, edge - 1.0]]
pred = int(svc.predict([[-2, 3]])[0])                             # the book's array([0]), Albon p284
m = SVC(kernel="linear", C=1.0, random_state=0, max_iter=5000, tol=1e-3).fit(StandardScaler().fit_transform(iris.data[:100, :]), y)
sv = m.support_vectors_; wm = (m.dual_coef_ @ sv)[0]; kkt = sv @ wm + m.intercept_[0]
print(f"RESULT slope_a={-w[0]/w[1]:.4f} intercept={b:.4f} margin_width={2/np.linalg.norm(w):.4f} n_iter={svc.n_iter_} pred_top_left={pred}")
print(f"RESULT above_line={int(svc.predict(above)[0])} below_line={int(svc.predict(below)[0])} support_={m.support_} n_support_={m.n_support_} sv0={np.round(sv[0], 7)} on_margin={np.round(kkt, 4)}")
assert pred == 0 and int(svc.predict(above)[0]) == 0 and int(svc.predict(below)[0]) == 1 and svc.n_iter_ < 20000
assert list(m.support_) == [23, 41, 57, 98] and list(m.n_support_) == [2, 2] and np.allclose(kkt, 2 * y[m.support_] - 1, atol=1e-3)
```
Reproduces `array([0])` for the top-left point (Albon p284) and `support_=[23 41 57 98]`, `n_support_=[2 2]` exactly (Albon p292), margin_width=0.9540,
with every support vector on `w.x+b = +/-1`. The book's `support_vectors_` digits differ from column 2 onward (iris data was corrected since printing).

### Kernels and C for linearly inseparable classes (Albon p285-289)
The trained machine is `f(x) = b0 + sum_i alpha_i K(x_i, x')` summed over the support vectors only, so the kernel - not the raw features - sets the
boundary shape (Albon p286): linear `K = sum_j x_ij*x'_ij`, poly `K = (1 + sum_j x_ij*x'_ij)^d`, RBF `K = exp(-gamma * sum_j (x_ij - x'_ij)^2)`, gamma > 0.
```python
import numpy as np
from sklearn.svm import SVC
np.random.seed(0)                                                 # the book's legacy seed for its XOR draw
features = np.random.randn(200, 2)
target = np.where(np.logical_xor(features[:, 0] > 0, features[:, 1] > 0), 0, 1)
models = {tag: SVC(C=1.0, random_state=0, max_iter=5000, tol=1e-3, **kw).fit(features, target) for tag, kw in [("linear", dict(kernel="linear")), ("poly", dict(kernel="poly", degree=3)), ("rbf", dict(kernel="rbf", gamma=1))]}
cs = {C: SVC(kernel="rbf", gamma=1, C=C, random_state=0, max_iter=5000, tol=1e-3).fit(features, target) for C in (0.01, 1.0, 100.0)}
print(f"RESULT linear_acc={models['linear'].score(features, target):.3f} nSV={int(models['linear'].n_support_.sum())} poly_acc={models['poly'].score(features, target):.3f} rbf_acc={models['rbf'].score(features, target):.3f} nSV={int(models['rbf'].n_support_.sum())}")
print("RESULT " + " ".join(f"C{c}_acc={m.score(features, target):.3f}_mis={int((m.predict(features) != target).sum())}" for c, m in cs.items()))
print("RESULT " + " ".join(f"g{g}_C{c}_acc={SVC(kernel='rbf', gamma=g, C=c, random_state=0, max_iter=5000, tol=1e-3).fit(features, target).score(features, target):.3f}" for g, c in [(0.1, 0.1), (1.0, 1.0), (10.0, 100.0)]))
assert models["rbf"].score(features, target) > models["linear"].score(features, target) and int(models["rbf"].n_support_.sum()) < int(models["linear"].n_support_.sum())
assert cs[0.01].score(features, target) < cs[100.0].score(features, target)
```
Linear reaches only 0.670 on this XOR grid, poly(d=3) 0.520, RBF(gamma=1, C=1) 0.965 with 82 support vectors instead of 187 - the book's "much better" boundary (Albon p288-289).
Small C tolerates errors (high bias, low variance), large C avoids them (low bias, high variance): misclassified 99 -> 7 -> 1 as C goes 0.01 -> 1 -> 100 (Albon p285), and gamma bites as
hard (0.730 at 0.1/0.1 vs 1.000 at 10/100), so tune all three together by cross-validation (Albon p289).

### Predicted probabilities - Platt scaling (Albon p289-291)
A hyperplane emits no probability, so `probability=True` trains the SVC then fits a separate cross-validated logistic regression mapping the signed
distance f(x) onto `P(y=1|x) = 1 / (1 + exp(A*f(x) + B))`; more than two classes use an extension of it, and that inner CV is why it can be slow (Albon p290-291).
```python
import warnings
import numpy as np
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(datasets.load_iris().data); y = datasets.load_iris().target
new = [[.4, .4, .4, .4]]
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)                # probability=True deprecated in sklearn 1.9, gone in 1.11
    model = SVC(kernel="linear", probability=True, random_state=0, max_iter=5000, tol=1e-3).fit(X, y)
p = model.predict_proba(new)[0]
mod = CalibratedClassifierCV(SVC(kernel="linear", random_state=0, max_iter=5000, tol=1e-3), ensemble=False).fit(X, y)
print(f"RESULT proba={np.round(p, 6)} argmax={int(np.argmax(p))} predict={int(model.predict(new)[0])} sum={p.sum():.6f}")
print(f"RESULT classes={model.classes_} n_iter_={model.n_iter_} sklearn_1_9_route={np.round(mod.predict_proba(new)[0], 6)}")
assert int(np.argmax(p)) == int(model.predict(new)[0]) and abs(p.sum() - 1) < 1e-9 and int(np.argmax(mod.predict_proba(new)[0])) == 1
```
The book prints `[0.00588822, 0.96874828, 0.0253635]` for `[.4,.4,.4,.4]`; this build gives `[0.005418, 0.973488, 0.021094]` - same class-1 call,
different digits, because Platt's inner CV and solver defaults changed since publication. Recompute; never copy the printed probabilities.

### Imbalanced classes - class-weighted C (Albon p292-293)
Weight the penalty per class, `C_k = C * w_j`, with `w_j = n / (k * n_j)` (n observations, k classes, n_j in class j), so a minority error costs more
and the small class is not overwhelmed (Albon p293).
```python
import numpy as np
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
iris = datasets.load_iris()
f, t = iris.data[40:100, :], np.where(iris.target[40:100] == 0, 0, 1)    # trim 40 rows -> 10 vs 50 (Albon p293)
X, n, k = StandardScaler().fit_transform(f), 60, 2
n0, n1 = int((t == 0).sum()), int((t == 1).sum()); w0, w1 = n / (k * n0), n / (k * n1)
bal = SVC(kernel="linear", class_weight="balanced", C=1.0, random_state=0, max_iter=5000, tol=1e-3).fit(X, t)
rng = np.random.default_rng(42)                                          # overlapping 20 vs 200 shows the real cost
O = StandardScaler().fit_transform(np.vstack([rng.normal(1., 1.2, (20, 1)), rng.normal(2.6, 1.2, (200, 1))]))
Oy = np.array([0] * 20 + [1] * 200)
res = {cw: SVC(kernel="linear", C=1.0, class_weight=cw, random_state=0, max_iter=5000, tol=1e-3).fit(O, Oy) for cw in (None, "balanced")}
print(f"RESULT n={n} n0={n0} n1={n1} w_minority={w0} w_majority={w1} Ck={1.0*w0}/{1.0*w1} balanced_acc_book_subset={bal.score(X, t):.4f}")
print(" ".join(f"RESULT cw={cw}: minority_recall={float((m.predict(O)[Oy == 0] == 0).mean()):.4f} acc={m.score(O, Oy):.4f}" for cw, m in res.items()))
assert (w0, w1) == (3.0, 0.6) and bal.score(X, t) == 1.0 and res["balanced"].score(O, Oy) < res[None].score(O, Oy)
assert float((res["balanced"].predict(O)[Oy == 0] == 0).mean()) > float((res[None].predict(O)[Oy == 0] == 0).mean())
```
w = 3.0 vs 0.6 (a 5:1 penalty ratio); the balanced fit scores 1.0000 on the trimmed iris because it is separable. On the seeded overlap the unweighted model never
predicts the 20-case minority (recall 0.0, accuracy 0.9091) while `balanced` lifts it to 0.70 at accuracy 0.7318 - recall bought with overall accuracy.

### Support vector regression - the epsilon tube (beyond recipes 17.0-17.5)
The chapter ships classifiers only; `SVR`/`LinearSVR` reuse the same kernel/C vocabulary with an `epsilon`-insensitive loss - residuals inside the tube
cost nothing, so noise cannot drag the curve, and C still prices the escapes.
```python
import numpy as np
from sklearn.svm import SVR, LinearSVR
from sklearn.preprocessing import StandardScaler
rng = np.random.default_rng(42)
x = np.sort(rng.uniform(0, 6, 120)).reshape(-1, 1)
y = np.sin(x).ravel() + rng.normal(0, .1, 120)
X = StandardScaler().fit_transform(x)
mods = {"SVR_C1_eps0.1": SVR(C=1.0, max_iter=50000, tol=1e-3), "SVR_C100_eps0.05": SVR(C=100.0, epsilon=0.05, max_iter=50000, tol=1e-3), "LinearSVR_C1": LinearSVR(C=1.0, max_iter=50000, tol=1e-4)}
out = {k: m.fit(X, y) for k, m in mods.items()}
print(" ".join(f"RESULT {k}: R2={v.score(X, y):.4f} mae={float(np.abs(v.predict(X) - y).mean()):.4f} nSV={getattr(v, 'n_support_', 'n/a')}" for k, v in out.items()))
assert out["SVR_C1_eps0.1"].score(X, y) > 0.9 > out["LinearSVR_C1"].score(X, y) and all(v.n_iter_ is None or int(np.max(v.n_iter_)) < 50000 for v in out.values())
assert int(out["SVR_C100_eps0.05"].n_support_[0]) > int(out["SVR_C1_eps0.1"].n_support_[0])
```
RBF SVR explains R2=0.9824 of the noisy sine where the linear variant reaches 0.6943, and tightening the tube (0.1 -> 0.05) raises support vectors 43 -> 75 - the regression face of a narrower margin.

## Output contract
Report the estimator with its active `C`/`kernel`/`gamma`/`degree`/`class_weight`, then `classes_` and the `predict` label, `decision_function` (signed
distance to the hyperplane), `predict_proba` per class only when calibrated, `support_`/`support_vectors_`/`n_support_`, the per-class weights `w_j`,
`margin_width = 2/||w||` with `||w|| = ||dual_coef_ @ support_vectors_||`, `n_iter_` against the `max_iter` ceiling, and for SVR `R2`/MAE - as `RESULT name=value` lines, every one computed here rather than transcribed from the book.

## Validity conditions
- Standardise first: distance-scaled kernels and one shared C across columns are meaningless on raw scales, and `StandardScaler` opens every iris recipe (Albon p283, p290).
- A linear kernel/`LinearSVC` is only defensible when the classes are linearly separable - rarely true of real data, which is the whole motivation for kernels (Albon p284-285).
- `gamma` must be > 0 for RBF; `degree` applies to poly only; C, gamma and degree move together, so read them as one tuned set, not three separate knobs (Albon p286, p289).
- Calibrated probabilities come from a cross-validated second model, so they can disagree with `predict` and they slow training (Albon p291).
- `class_weight="balanced"` re-prices the loss only; it cannot supply minority cases the data does not contain, and it lowers overall accuracy by design.

## Out of scope
Cross-validation and search harnesses belong to `mlk-model-selection`; accuracy/precision/recall/ROC-AUC and calibration plots to `mlk-model-evaluation`; the
scaler to `mlk-numerical-features`; the log-loss competitor to `mlk-logistic-regression` (a linear-kernel SVM scores the same `b0 + w.x`, priced by the margin
instead of the likelihood) and `mlk-naive-bayes`. `SGDClassifier(loss="hinge")` for huge data and One-Class SVM anomaly detection are not covered.

## Conventions
- `random_state=0` (as in Albon) plus an explicit `max_iter` and `tol=1e-3` (SVC/SVR) or `1e-4` (Linear*) on every fit, `n_iter_` asserted below the ceiling so
  no block can hang; blocks are deterministic and self-contained (own imports, own seeded data, nothing read from disk, timings never asserted - too machine-bound).
- The XOR demo keeps the book's legacy `np.random.seed(0)` + `randn(200, 2)` so its accuracies describe the book's data; other synthetic draws use `default_rng(42)`.
- Figures stay headless if asked for: `matplotlib.use("Agg")`, `contourf` of a prediction meshgrid (the Raschka-style decision regions, Albon p287), written to `tempfile`.
- sklearn 1.9 deprecates `SVC(probability=True)` (removed 1.11) - use `CalibratedClassifierCV(SVC(...), ensemble=False)`; `LinearSVC`/`LinearSVR` expose neither
  `predict_proba` nor `support_vectors_`, so use `SVC`/`SVR` when those attributes are needed.
- Source: Albon, *Machine Learning with Python Cookbook*, 2nd ed. (O'Reilly) Ch.17, PDF pp.282-293 (printed pp.267-278), recipes 17.0-17.5.
