---
name: mlk-neural-networks
description: Build, train and diagnose feedforward neural networks in pure Python - a hand-written one-hidden-layer numpy MLP (forward propagation, backpropagation, SGD, early stopping) plus sklearn MLPClassifier/MLPRegressor for k-fold CV, small-grid tuning and regression MSE, plus minimal numpy conv2d/max-pool and one-step LSTM-cell kernels - following Albon, ML with Python Cookbook ch.20. tensorflow/keras/torch are not used. Use when asked for Keras-style DNN/MLP recipes, CNN or RNN/LSTM fundamentals, NN overfitting/training-history diagnostics, or "train a neural network" on a sklearn-only machine.
---

# Neural Networks - MLP, CNN and RNN kernels (Python)

## Purpose
Reproduce Albon ch.20's neural-network recipes without Keras: every recipe's numerics (unit,
forward/backward propagation, epoch, loss) runs in numpy or sklearn, keeping the book's design rules and numbers.

## Inputs
- `X` `(n_samples, n_features)` float matrix, `y` 0/1 labels (binary), class ints (multiclass) or floats (regression).
- Standardize every feature to mean 0 / sd 1 first (Albon p313-314); skip only for all-binary features (Albon p314).
- Hyperparameters below: `H=8` hidden units, `LR=0.1`, `PATIENCE=2`, `MAX_EPOCH=200`, seed 42; book nets use 16 units for binary and 100 for the 46-class task (Albon p315, p320-321).
- Output/loss pattern: binary = 1 sigmoid unit + binary cross-entropy; multiclass = k softmax units + categorical cross-entropy; regression = 1 unit, no activation + MSE (Albon p316-317).

## Algorithms (all pure numpy/scipy)
### Preprocess: standardize (Albon p313-314)
```python
import numpy as np
from sklearn import preprocessing
features = np.array([[-100.1, 3240.1], [-200.2, -234.1], [5000.5, 150.1],
                     [6000.6, -125.1], [9000.9, -673.1]])              # Albon p314 matrix
zs = preprocessing.StandardScaler().fit_transform(features)
print(f"RESULT row0_0={zs[0,0]:.8f} row0_1={zs[0,1]:.8f} mean={zs[:,0].mean():.1f} std={zs[:,0].std():.1f}")
```
The printed first row must be `[-1.12541308, 1.96429418]`, exactly the book's output, with mean 0.0 and sd 1.0
(Albon p314). Nets initialize weights as small random values, so raw features far larger destabilize training (Albon p313-314).

### Hand-written MLP: forward, backprop, early stopping
```python
import numpy as np
rng = np.random.default_rng(42)
X = np.vstack([rng.normal(-1.5, 1.0, (150, 2)), rng.normal(1.5, 1.0, (150, 2))])
y = np.r_[np.zeros(150), np.ones(150)]
X = (X - X.mean(0)) / X.std(0)
idx = rng.permutation(len(y)); Xtr, ytr = X[idx[:240]], y[idx[:240]]; Xva, yva = X[idx[240:]], y[idx[240:]]
H, LR, PAT, MAX_EPOCH = 8, 0.1, 2, 200
W1, b1 = rng.normal(0, .5, (2, H)), np.zeros(H)
W2, b2 = rng.normal(0, .5, (H, 1)), np.zeros(1)
def fwd(Z):
    h = np.maximum(0.0, Z @ W1 + b1)                                    # hidden layer, ReLU f(z)=max(0,z) (Albon p316)
    return h, 1.0 / (1.0 + np.exp(-(h @ W2 + b2)))[:, 0]                # sigmoid output = P(class 1) (Albon p312)
def bce(p, t): return -np.mean(t * np.log(p) + (1 - t) * np.log(1 - p)) # binary cross-entropy (Albon p317)
loss0 = bce(fwd(Xtr)[1], ytr); hist, best_val, best_ep, bad = [], np.inf, 0, 0
for ep in range(1, MAX_EPOCH + 1):                                      # one pass over all data = one epoch (Albon p313)
    h, p = fwd(Xtr); d = (p - ytr)[:, None] / len(ytr)                  # backprop: per-parameter error contribution (Albon p313)
    dh = (d @ W2.T) * (h > 0)
    W2 -= LR * (h.T @ d); b2 -= LR * d.sum(); W1 -= LR * (Xtr.T @ dh); b1 -= LR * dh.sum(0)
    vl = bce(fwd(Xva)[1], yva); hist.append(vl)                         # monitor test loss per epoch (Albon p331)
    if vl < best_val - 1e-6: best_val, best_ep, bad = vl, ep, 0; Wb = (W1.copy(), b1.copy(), W2.copy(), b2.copy())
    else:
        bad += 1
        if bad >= PAT: break                                            # early stopping, patience=2 (Albon p331)
W1, b1, W2, b2 = Wb                                                     # restore best model (ModelCheckpoint save_best_only, Albon p332)
loss_end = bce(fwd(Xtr)[1], ytr); va = ((fwd(Xva)[1] > .5) == (yva == 1)).mean()
print(f"RESULT loss_start={loss0:.4f} loss_end={loss_end:.4f} best_val={best_val:.4f} best_epoch={best_ep} epochs_run={len(hist)} val_acc={va:.4f}")
assert loss_end < loss0 and va > 0.8                                    # smoke gate: the net must learn a separable set
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, tempfile, pathlib
plt.plot(range(1, len(hist) + 1), hist, label="test loss"); plt.legend(); plt.xlabel("Epoch"); plt.ylabel("Loss")
out = pathlib.Path(tempfile.mkdtemp()) / "history.png"; plt.savefig(out); plt.close()
print(f"RESULT figure_written={out.exists()}")
```
Dropout, the third anti-overfit lever: during training only, `h = h * rng.binomial(1, 1-p, h.shape) / (1-p)` drops units and rescales survivors; book rates p=0.2 input, p=0.5 hidden (Albon p332-334). L2 weight decay adds `lambda*||W||^2` to the loss, lambda=0.01, driving weights small (Albon p329-330).

### sklearn MLP: k-fold CV, small grid, regressor MSE
```python
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline; from sklearn.preprocessing import StandardScaler
X, y = make_classification(n_samples=600, n_features=20, n_informative=5, n_redundant=0,
                           weights=[.5, .5], random_state=0)            # generator, Albon p336
clf = make_pipeline(StandardScaler(), MLPClassifier(solver="lbfgs", hidden_layer_sizes=(16, 16),
                    activation="relu", random_state=0))                # two dense 16-unit hidden layers (Albon p315)
scores = cross_val_score(clf, X, y, cv=3)                               # three-fold CV (Albon p336-337)
print(f"RESULT cv_scores={np.round(scores, 4).tolist()}")
grid = GridSearchCV(make_pipeline(StandardScaler(), MLPClassifier(solver="lbfgs", random_state=0)),
                    {"mlpclassifier__hidden_layer_sizes": [(8,), (16, 8)],
                     "mlpclassifier__alpha": [1e-4, 1e-2]}, cv=3)       # alpha = L2 weight decay, cf Albon p329-330
best = grid.fit(X, y).best_params_
print(f"RESULT best_params={ {k.split('__')[1]: v for k, v in best.items()} }")
Xr, yr = make_regression(n_samples=300, n_features=3, n_informative=3, n_targets=1,
                         noise=0.0, random_state=0)                     # Albon p322 generator
Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=0.33, random_state=0)
reg = make_pipeline(StandardScaler(), MLPRegressor(solver="lbfgs", hidden_layer_sizes=(16,), activation="relu",
                    max_iter=2000, random_state=0)).fit(Xtr, ytr)
mse = float(np.mean((yte - reg.predict(Xte)) ** 2))                     # MSE = (1/n) sum (y_i - yhat_i)^2 (Albon p322-323)
print(f"RESULT reg_mse={mse:.4f} target_var={float(np.var(yte)):.1f}")
```
The book's Keras three-fold run scored `[0.9046, 0.7744, 0.8707]` and its grid picked
`batch_size=10, epochs=5, optimizer=adam` (Albon p337, p339); sklearn numbers differ but the pattern is identical.
A noise-free `make_regression` target should drive `reg_mse` far below `target_var`.

### CNN kernel: 2D convolution + max pooling (Albon p343-345)
```python
import numpy as np
rng = np.random.default_rng(42)
img = rng.integers(0, 256, (8, 8)) / 255.0                              # pixel intensities rescaled to 0-1 (Albon p343)
kern = np.array([[1.0, -1.0], [-1.0, 1.0]])                             # one filter (book: 64 filters of 5x5, Albon p343)
kh, kw = kern.shape
feat = np.array([[(img[i:i+kh, j:j+kw] * kern).sum() for j in range(img.shape[1]-kw+1)]
                 for i in range(img.shape[0]-kh+1)])                    # valid conv: window slides over neighbors (Albon p344)
pool = feat[:6, :6].reshape(3, 2, 3, 2).max(axis=(1, 3))                # 2x2 max pooling = zoom out (Albon p345)
print(f"RESULT conv_shape={feat.shape} pooled_shape={pool.shape} pixel00={img[0,0]:.3f} feat00={feat[0,0]:.4f}")
```
Conv nets exist because feedforward nets ignore pixel spatial structure and learn only global patterns (Albon p344).
The book's MNIST net (70,000 28x28 digits, 60k/10k split) then flattens into 128-unit and 10-unit softmax layers (Albon p342-343, p345-346).

### Sequence kernel: pad_sequences + one LSTM step (Albon p348-350)
```python
import numpy as np
rng = np.random.default_rng(42)
seqs = [list(rng.integers(1, 100, n)) for n in (3, 5, 4)]               # word-index lists of unequal length (Albon p349)
MAXLEN = 8
pad = np.zeros((len(seqs), MAXLEN), dtype=int)
for r, s in enumerate(seqs): pad[r, -len(s):] = s                       # left-pad with 0s like pad_sequences(maxlen) (Albon p348, p350)
dim = 4
x = rng.normal(0, 1, dim)                                               # word vector from an Embedding layer (Albon p348)
h0 = c0 = np.zeros(dim)
Wx, Wh = rng.normal(0, .5, (4 * dim, dim)), rng.normal(0, .5, (4 * dim, dim))
g = Wx @ x + Wh @ h0
sig = lambda a: 1.0 / (1.0 + np.exp(-a))
f, i, cd, o = sig(g[:dim]), sig(g[dim:2*dim]), np.tanh(g[2*dim:3*dim]), sig(g[3*dim:])
c1 = f * c0 + i * cd
h1 = o * np.tanh(c1)                                                    # forget/input/candidate/output gates (Albon p348-350)
print(f"RESULT pad_shape={pad.shape} row0={pad[0].tolist()} c1={np.round(c1, 4).tolist()} h1={np.round(h1, 4).tolist()}")
```
Recurrent nets loop information backward, giving memory for sequential data; the book pads reviews to 400 word slots and stacks Embedding -> LSTM(128 units) -> sigmoid (Albon p348-350).

## Output contract
Each block prints `RESULT name=value` lines: `row0_0,row0_1,mean,std`;
`loss_start,loss_end,best_val,best_epoch,epochs_run,val_acc,figure_written`;
`cv_scores,best_params,reg_mse,target_var`; `conv_shape,pooled_shape,pixel00,feat00`; `pad_shape,row0,c1,h1`.
Report best validation loss and best epoch, not just the final one.

## Validity conditions
- Features standardized before training, else small random weights are swamped (Albon p313-314); pixels /255 (Albon p343).
- Overfitting signature: train loss keeps falling while test loss turns up; the book's sweet spot was ~epoch 5 (Albon p327-328). Counter it with early stopping patience=2, L2 lambda=0.01 or dropout 0.2/0.5 (Albon p328-334).
- k-fold CV and grid search on nets are "neither necessary nor advisable" when training is slow - a 1-day model costs 10 days at 10 folds; on big data use one held-out test set instead (Albon p336-337, p339).
- RNN/LSTM input rows must all have equal length (pad or truncate) before feeding (Albon p348-350); more units and more layers learn more complex patterns but raise overfit and compute cost (Albon p316).

## Conventions
- Seed with `numpy.random.default_rng(42)` and `random_state=0`; the book fixes `np.random.seed(0)` (Albon p318).
- Headless matplotlib only (`matplotlib.use("Agg")`); figures go to a `tempfile` directory, never to the project.
- Missing package: the source recipes are Keras/TensorFlow code; tensorflow, keras and torch are NOT imported here - every runnable block is numpy/sklearn, and Keras-only APIs are named in Out of scope.
- Batch sizes in the book: 100 for dense nets, 1000 for MNIST, epochs 3-20 (Albon p319, p335, p343).
- Never invent a value - always compute it and cite the source page for the method.

## Out of scope
- GANs, word2vec, seq2seq and transformers: no recipe in Albon ch.20 (PDF p312-351), so none are shipped here.
- Keras-only APIs (Sequential/compile/fit, History, Dropout and Embedding layers, ImageDataGenerator augmentation,
  ModelCheckpoint .h5 files, KerasClassifier wrapper, GPU and end-to-end LSTM/CNN training) are optional-dependency
  prose only; the kernels above re-implement their forward numerics. PyTorch likewise.
- Encoding categorical features and general model selection are owned by sibling skills; only MLP hyperparameters are tuned here.
Source: Albon, Machine Learning with Python Cookbook, 2nd ed., ch.20 Neural Networks, PDF pages 312-351.
