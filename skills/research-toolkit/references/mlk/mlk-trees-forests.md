---
name: mlk-trees-forests
description: Train, prune, export and interpret decision trees, random forests and boosted trees in pure Python (sklearn.tree, sklearn.ensemble, numpy, matplotlib) per Albon ch.14 - Gini/entropy split criteria, MSE/MAE regression, DOT and text export, tree-size parameters, feature_importances_, SelectFromModel importance thresholds, class_weight="balanced", out-of-bag scoring, AdaBoost and HistGradientBoosting. Use when asked for a tree or forest model, "which features matter", how to limit tree size or overfitting, an OOB score, gradient boosting without xgboost, or a picture of a decision tree.
---

# Trees and Forests (Python)

## Purpose
A decision tree chains if-then rules (node -> branch -> leaf) into a drawable model; a forest either *decorrelates*
bootstrapped trees by voting/averaging, or *boosts* weak trees onto the hardest observations (Albon p248, p254, p263).

## Inputs
- `features`: float matrix (n, d); trees need no scaling, but nominal categories must be one-hot encoded first
  (Albon p258). `target`: class labels (classifier) or numeric values (regressor).
- Knobs: `criterion`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_leaf_nodes`, `min_impurity_decrease`, `n_estimators`, `max_features`, `bootstrap`, `oob_score`, `class_weight`, `learning_rate`, AdaBoost `estimator`.
- Data: `load_iris()` (150x4) / `load_diabetes()` (442x10) or seeded `default_rng(42)`; `TEST_SIZE = 0.25`, `N_ESTIMATORS = 100`, `random_state = 0` on every estimator.

## Algorithms (all pure numpy/scipy)

### Decision tree - impurity, prediction and size controls (Albon p248-250, p261-262)
The learner keeps the rule with the greatest impurity decrease and recurses until leaves are pure or a cut-off stops
it: Gini `G(t) = 1 - sum_i p_i^2` (classifier default), entropy `H(t) = - sum_i p_i log2 p_i`.
```python
import numpy as np; from sklearn import datasets
from sklearn.model_selection import train_test_split; from sklearn.tree import DecisionTreeClassifier
iris = datasets.load_iris(); X, y = iris.data, iris.target; p = np.bincount(y) / len(y)   # root shares 1/3 each
model = DecisionTreeClassifier(random_state=0).fit(X, y); ent = DecisionTreeClassifier(criterion="entropy", random_state=0).fit(X, y)
print(f"RESULT root_gini={1 - (p**2).sum():.4f} root_entropy={-(p * np.log2(p)).sum():.4f} pred={model.predict([[5, 4, 3, 2]])[0]} proba={np.round(model.predict_proba([[5, 4, 3, 2]])[0], 3)} nodes={model.tree_.node_count} leaves={model.get_n_leaves()} depth={model.get_depth()} entropy_leaves={ent.get_n_leaves()}")
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
for kw in [{}, {"max_depth": 2}, {"min_samples_split": 40}, {"min_samples_leaf": 5}, {"max_leaf_nodes": 4}, {"min_impurity_decrease": 0.01}]:
    pruned = DecisionTreeClassifier(random_state=0, **kw).fit(Xtr, ytr)
    print(f"RESULT {str(kw):31s} leaves={pruned.get_n_leaves():2d} depth={pruned.get_depth()} train={pruned.score(Xtr, ytr):.4f} test={pruned.score(Xte, yte):.4f}")
```
Prints `pred=1 proba=[0. 1. 0.]` - the book's `array([1])` / `array([[0., 1., 0.]])` for `[[5,4,3,2]]` (Albon p249) -
with root impurity 0.6667 Gini / 1.5850 bits, and a 17-node, 9-leaf, depth-5 unpruned tree on all 150 rows.
`max_depth=None` grows until pure, an int prunes; `min_samples_split`/`min_samples_leaf` take an int (count) or a
float (fraction of n); `min_impurity_decrease` replaces the removed `min_impurity_split`. All five constraints shrink
the 7-leaf unpruned tree to 3-5 leaves; only `max_leaf_nodes=4` and `min_impurity_decrease` keep test 0.9737 on this
112-row split (the rest fall to 0.8947) while the unpruned tree memorises 100% of train - stumps cut variance (Albon p262).

### Decision tree regressor - MSE vs MAE splits (Albon p250-251)
Regression trees rank splits by the drop in `MSE = (1/n) sum_i (yi - yhat_i)^2`; `criterion="absolute_error"` ranks
them by mean absolute error and predicts the leaf median instead of the leaf mean.
```python
import numpy as np; from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor
dia = datasets.load_diabetes(); X, y = dia.data[:, 0:2], dia.target   # first two columns only
mean_tree = DecisionTreeRegressor(random_state=0).fit(X, y)            # split score = drop in MSE
med_tree = DecisionTreeRegressor(criterion="absolute_error", random_state=0).fit(X, y)
print(f"RESULT pred_first_obs={mean_tree.predict(X[:1])[0]:.1f} actual={y[0]:.1f} train_R2={mean_tree.score(X, y):.4f} leaves={mean_tree.get_n_leaves()}")
print(f"RESULT mse_of_mean_tree={np.mean((y - mean_tree.predict(X))**2):.1f} mae_of_median_tree={np.mean(np.abs(y - med_tree.predict(X))):.1f}")
```
Prints pred_first_obs=149.0 (actual 151.0), train_R2=0.2709, MSE 4323.7, MAE 50.9. The book's regression output is
unreproducible: `load_boston` was deleted in sklearn 1.2, so `predict([[0.02, 16]]) = 33.` (Albon p251) cannot run.

### Exporting and reading a tree (Albon p252-253)
DOT is a graph description language; each node line carries its rule, impurity, sample count, per-class counts
and winning class. `pydotplus`/`graphviz` are absent, so assert on the DOT text and rasterise with `plot_tree`.
```python
import os, tempfile; import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn import datasets; from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text, plot_tree
iris = datasets.load_iris(); X, y = iris.data, iris.target; model = DecisionTreeClassifier(random_state=0).fit(X, y)
dot = export_graphviz(model, out_file=None, feature_names=iris.feature_names, class_names=iris.target_names)
assert "petal width (cm) <= 0.8" in dot and dot.count('[label="') == model.tree_.node_count, "unexpected DOT body"
two = DecisionTreeClassifier(random_state=0, max_depth=2).fit(X, y); t = two.tree_
print(f"RESULT root_rule={export_text(two, feature_names=iris.feature_names).splitlines()[0].strip()} gini={t.impurity[0]:.4f} n={t.n_node_samples[0]} counts={(t.value[0][0] * t.n_node_samples[0]).astype(int)} class={iris.target_names[t.value[0][0].argmax()]}")
print(f"RESULT two_rules_correct={round(two.score(X, y) * 150)}/150")
fig, ax = plt.subplots(figsize=(6, 4)); plot_tree(two, feature_names=iris.feature_names, class_names=iris.target_names, ax=ax, filled=True, rounded=True)
png = os.path.join(tempfile.mkdtemp(), "iris_tree.png"); fig.savefig(png); plt.close(fig); print(f"RESULT png_bytes={os.path.getsize(png)}")
```
`two_rules_correct=144/150` is Albon's own reading of the diagram: petal width <= 0.8 isolates setosa perfectly and
one more petal-width rule classifies 144 of 150 observations, making petal width a very important feature (Albon
p253); the root line reports Gini 0.6667, n=150, counts [50,50,50], class setosa.

### Random forest, feature importance and threshold selection (Albon p254-260)
Each tree trains on a bootstrapped sample (`bootstrap=True` = with replacement) and each node considers a random
feature subset: `max_features` takes an int, a float fraction or `"sqrt"`, default `sqrt` for classifiers and all d
for regressors (Albon p255-256). Importance is the mean decrease in impurity over all trees, normalised to sum 1;
`SelectFromModel` keeps columns clearing a threshold and a second forest is refit on them (Albon p258-259).
```python
import numpy as np; from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier; from sklearn.feature_selection import SelectFromModel
iris = datasets.load_iris(); X, y = iris.data, iris.target
forest = RandomForestClassifier(random_state=0, n_estimators=100, max_features="sqrt"); imp = forest.fit(X, y).feature_importances_
assert np.isclose(imp.sum(), 1.0), "feature importances must sum to 1"
selector = SelectFromModel(forest, threshold=0.3).fit(X, y)
X_keep = selector.transform(X); keep = [iris.feature_names[i] for i in np.where(selector.get_support())[0]]
small = RandomForestClassifier(random_state=0, n_estimators=100, max_features="sqrt").fit(X_keep, y)
print(f"RESULT importances={np.round(imp, 4)} sum={imp.sum():.6f} ranked={[iris.feature_names[i] for i in np.argsort(imp)[::-1]]} train={forest.score(X, y):.4f}")
print(f"RESULT kept={keep} shape={X_keep.shape} retrained_train={small.score(X_keep, y):.4f}")
```
Petal length (0.4604) then petal width (0.4241) lead; the book printed `[0.11896532, 0.0231668, 0.36804744,
0.48982043]` with petal width first (Albon p258) - same top-two set, but the numbers are sklearn-version
dependent, so recompute them. `threshold=0.3` keeps 2 of 4 columns and the refit forest scores 0.9933 on it.

### Imbalanced classes and the out-of-bag score (Albon p260-261, p264-265)
`class_weight="balanced"` weights class j by `w_j = n / (k * n_j)`; a dict such as `{"male": 0.2, "female": 0.8}`
sets weights by hand. Because each tree sees only a bootstrap sample, the left-out rows are a free test set: set
`oob_score=True` and read `oob_score_` instead of running cross-validation.
```python
import numpy as np; from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
iris = datasets.load_iris()
X, y = iris.data[40:, :], np.where(iris.target[40:] == 0, 0, 1); n, k = len(y), 2   # 10 of class 0 vs 100 of class 1
balanced = RandomForestClassifier(random_state=0, n_estimators=100, max_features="sqrt", class_weight="balanced").fit(X, y)
print(f"RESULT n={n} n0={int((y == 0).sum())} n1={int((y == 1).sum())} weight_minority={n / (k * int((y == 0).sum()))} weight_majority={n / (k * int((y == 1).sum()))} train={balanced.score(X, y):.4f} proba_5_4_3_2={np.round(balanced.predict_proba([[5, 4, 3, 2]])[0], 2)}")
oob = RandomForestClassifier(random_state=0, n_estimators=1000, max_features="sqrt", oob_score=True).fit(iris.data, iris.target)
print(f"RESULT oob_score={oob.oob_score_} in_sample_train={oob.score(iris.data, iris.target):.4f}")
```
`weight_minority=5.5` and `weight_majority=0.55` are exactly `110/(2*10)` and `110/(2*100)` (Albon p261);
`oob_score=0.9533333333333334` reproduces the book's 0.95333333333333337 (Albon p265) and is the score to report,
since the in-sample 1.0 only says each tree voted on rows it had memorised.

### Boosting with trees (Albon p262-264)
AdaBoost assigns every observation weight `1/n`, trains a weak model (a depth-1 tree by default), raises the weight
of the observations it missed and repeats `n_estimators` times; `learning_rate` shrinks each model's contribution,
slowing training but often scoring better (`AdaBoostRegressor` also has `loss`: linear/square/exponential).
```python
from sklearn import datasets; from sklearn.model_selection import train_test_split; from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
iris = datasets.load_iris(); Xtr, Xte, ytr, yte = train_test_split(iris.data, iris.target, test_size=0.25, random_state=0)
models = {"adaboost_default": AdaBoostClassifier(random_state=0, n_estimators=50),
          "adaboost_stump_lr0.5": AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), random_state=0, n_estimators=50, learning_rate=0.5),
          "random_forest": RandomForestClassifier(random_state=0, n_estimators=200, max_features="sqrt"),
          "grad_boost": GradientBoostingClassifier(random_state=0, n_estimators=100), "hist_grad_boost": HistGradientBoostingClassifier(random_state=0, max_iter=100)}
for name, model in models.items():
    model.fit(Xtr, ytr); print(f"RESULT {name:21s} train={model.score(Xtr, ytr):.4f} test={model.score(Xte, yte):.4f}")
print(f"RESULT initial_obs_weight={1 / Xtr.shape[0]:.4f} ada_rounds={len(models['adaboost_default'].estimators_)}")
```
Boosting takes the held-out quarter here: AdaBoost 0.9474 at the default `learning_rate=1`, 0.9737 at 0.5. ExtraTrees
is the forest loop with fully random thresholds; `HistGradientBoostingClassifier` is the installed stand-in for
xgboost/lightgbm histogram boosting.

## Output contract
Every estimator returns `predict`, `predict_proba` (forests: vote fraction per class) and `score(X, y)`. Trees add
`tree_` (`feature`, `threshold`, `impurity`, `n_node_samples`, `value` as per-class proportions), `get_n_leaves()`,
`get_depth()` and the DOT/text of `export_graphviz`/`export_text`; forests and boosted models add
`feature_importances_` (sums to 1), `estimators_` and, with `oob_score=True`, `oob_score_`. Print one `RESULT name=value` line per quantity; figures go into a `tempfile` dir.

## Validity conditions
- Nominal features must be split into binary columns first, which dilutes their importance over the dummies; two correlated features split one signal unevenly - never read the smaller share as unimportant (Albon p258, p260).
- `oob_score_` needs `bootstrap=True` and enough trees that most rows are out-of-bag somewhere (Albon p264-265).
- `class_weight="balanced"` reweights the vote only; a class with almost no real cases still needs more data (Albon p261). Judge generalisation on held-out or OOB data - an unpruned tree fits 100% of its own training rows.

## Out of scope
Metrics and CV loops belong to model-evaluation; grid/random/nested search over these same knobs belongs to
model-selection; RFE, SelectKBest and permutation importance belong to feature-selection; encoding, k-NN, logistic
regression, clustering and neural nets are separate skills; stacking/blending is out of scope (Albon p262-265).
## Conventions
- `xgboost`, `lightgbm` and `catboost` are NOT installed and must not be pip-installed: gradient boosting is
  `sklearn.ensemble.HistGradientBoostingClassifier/Regressor` (or the slower `GradientBoosting*`).
- Graphviz routes (`pydotplus`, `graphviz`, `IPython.display.Image`) are absent, so trees are exported as DOT text
  and drawn headless with `plot_tree`/`plt.bar` into a `tempfile` dir; always `matplotlib.use("Agg")`.
- API drift vs the book: `criterion="mae"` is now `"absolute_error"`; AdaBoost's `base_estimator` is now `estimator`;
  `n_features_` and `tree_.class_idx` are gone (use `n_features_in_`, `value[0].argmax()`); forests default to
  `n_estimators=100` (book: 10) and `max_features="sqrt"`/`1.0` (book: `auto`); `load_boston` is deleted.
- Fix `random_state=0` on every estimator and split so importances, DOT rules and `oob_score_` reproduce; `default_rng(42)`
  for synthetic data; `n_jobs=-1` is optional on forests but omitted here for stability; keep 100-200 trees (only the
  OOB check uses 1000), at most 442 rows, never a 60k-row frame.
