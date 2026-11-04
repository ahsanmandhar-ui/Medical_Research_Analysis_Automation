---
name: mlk-dim-reduction-extraction
description: Reduce p_original to a smaller p_new by projecting features into new, non-interpretable components with scikit-learn - PCA with a variance target, KernelPCA for curved data, LDA for class separability, NMF for nonnegative data, TruncatedSVD for sparse matrices, FactorAnalysis and FastICA for latent factors and sources, t-SNE for 2-D projections - plus reconstruction diagnostics (fit/inverse MSE falling as components rise, whitening checks, 95%/99% explained-variance cut-offs, neighbour-preservation scores). Use when asked for "dimensionality reduction by feature extraction", PCA/ICA/NMF/kernel PCA/t-SNE/TruncatedSVD, "how many components keep 95% or 99% of the variance", latent features before a model, or reconstruction error.
---

# Dimensionality Reduction Using Feature Extraction (Python)

## Purpose
Project `p_original` features into fewer `p_new` columns that keep nearly all the information a model needs,
accepting that the new columns are unreadable mixtures (Albon p172); only LDA is supervised (Albon p173, p178).
## Inputs
- `X`: numeric `(n_samples, n_features)` matrix; standardise (`StandardScaler`) before PCA/TSVD/FA/ICA
  (Albon p173), hand NMF the raw nonnegative matrix instead (Albon p180); `y` is needed only by LDA (Albon p177).
- `GOAL_VAR = 0.95` (or `0.99`) retained-variance target (Albon p174, p179, p182); demos `load_digits()`
  (1797 x 64 pixel features), `load_iris()`, `make_circles` (Albon p173-183).
## Algorithms (all pure numpy/scipy)
### PCA with a variance target (Albon p173-174)
`n_components` in (0,1) returns the *minimum* features holding that variance share, `>1` means "exactly that
many"; `whiten=True` gives zero mean/unit variance per component, `svd_solver="randomized"` is the fast path (p174).
```python
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
F = StandardScaler().fit_transform(datasets.load_digits().data)        # 1797 x 64 (Albon p173)
pca = PCA(n_components=0.99, whiten=True).fit(F)
Z = pca.transform(F)
print(f"RESULT orig_features={F.shape[1]} reduced_features={Z.shape[1]} variance_retained={pca.explained_variance_ratio_.sum():.5f} pc1_share={pca.explained_variance_ratio_[0]:.4f} whitened_abs_mean_max={abs(Z.mean(0)).max():.1e} whitened_std_min={Z.std(0, ddof=1).min():.4f} whitened_std_max={Z.std(0, ddof=1).max():.4f}")
```
The book's 64 -> 54 features at 99% variance reproduce: ten columns dropped "while still retaining 99% of the
information" (Albon p173-174); retained share 0.99077, PC1 alone 12.03%, whitened component std 1.0000.

### Kernel PCA and LDA: projections aimed at separability (Albon p175-179)
A kernel turns a curved boundary straight (`rbf`, also `poly`, `sigmoid`, `linear`) but has no variance target -
`n_components` and `gamma` come from trial and error (Albon p176-177); LDA maximises class separation instead (p178).
```python
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
X, y = datasets.make_circles(n_samples=1000, random_state=1, noise=0.1, factor=0.1)   # Albon p175
Zk, Zp = KernelPCA(kernel="rbf", gamma=15, n_components=1).fit_transform(X), PCA(n_components=1).fit_transform(X)
def split_acc(z, labels):            # best single threshold on the projected axis (in-sample)
    s = labels[np.argsort(z)]; i = np.arange(1, len(s)); l1 = np.cumsum(s)[:len(s) - 1]; r1 = s.sum() - l1
    return float(np.maximum((i - l1 + r1) / len(s), (l1 + len(s) - i - r1) / len(s)).max())
iris = datasets.load_iris(); lda = LinearDiscriminantAnalysis(n_components=1).fit(iris.data, iris.target)
full = LinearDiscriminantAnalysis(n_components=None).fit(iris.data, iris.target)
n95 = int(np.searchsorted(np.cumsum(full.explained_variance_ratio_), 0.95) + 1)   # source loop (p179)
print(f"RESULT kpca_orig_features={X.shape[1]} kpca_reduced_features={Zk.shape[1]} kpca1d_split_acc={split_acc(Zk[:, 0], y):.3f} linear_pca1d_split_acc={split_acc(Zp[:, 0], y):.3f}")
print(f"RESULT lda_orig_features={iris.data.shape[1]} lda_reduced_features={lda.transform(iris.data).shape[1]} lda_evr={lda.explained_variance_ratio_[0]:.7f} lda_n_for_95pct={n95}")
```
2 -> 1 feature as on Albon p175: the RBF axis splits the rings cleanly (1.000) where the linear axis intertwines them
(0.710) - the p176 failure mode. LDA is 4 -> 1 on the axis over 99% of the variance, book 0.99147248 versus 0.9912126
here (equal to 3 dp), and that one axis already clears the 95% goal (Albon p178-179).

### NMF and TruncatedSVD: nonnegative and sparse factorization (Albon p180-183)
NMF fits `V ~= WH` (`V` d x n, `W` d x r, `H` r x n) so `r` is the reduction, with no explained variance at all;
sklearn returns `H` row-wise and `W` transposed in `components_` (Albon p180-181). TSVD is PCA-like, takes sparse
input, publishes variance ratios, and can flip signs between fittings - fit once, transform many (Albon p182).
```python
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import NMF, TruncatedSVD
from scipy.sparse import csr_matrix
X0 = datasets.load_digits().data     # pixel counts, all >= 0 (Albon p180)
nmf = NMF(n_components=10, random_state=1, max_iter=800).fit(X0)
F = StandardScaler().fit_transform(X0); S = csr_matrix(F)
tsvd = TruncatedSVD(n_components=10).fit(S)          # one fit, many transforms (Albon p182)
cum = np.cumsum(TruncatedSVD(n_components=S.shape[1] - 1).fit(F).explained_variance_ratio_)
n95 = int(np.searchsorted(cum, 0.95) + 1)
print(f"RESULT nmf_orig_features={X0.shape[1]} nmf_reduced_features={nmf.transform(X0).shape[1]} any_negative_input={bool((X0 < 0).any())} components_min={nmf.components_.min():.3f} nmf_recon_mse={np.mean((X0 - nmf.transform(X0) @ nmf.components_) ** 2):.4f}")
print(f"RESULT tsvd_orig_features={S.shape[1]} tsvd_reduced_features={tsvd.transform(S).shape[1]} tsvd_first3_evr_sum={tsvd.explained_variance_ratio_[0:3].sum():.8f} tsvd_n_for_95pct={n95} tsvd_cum_evr_at_n={cum[n95 - 1]:.6f}")
```
Both reduce 64 -> 10 features (Albon p180, p181-182). The first three TSVD components sum to 0.30039385 against
the printed 0.30039385386597783 (Albon p182), and 40 components are needed to reach 95% (Albon p183).

### Reconstruction diagnostics: MSE must fall as components rise (FA and ICA benchmarks)
Sweep `k`, reconstruct with `inverse_transform` (or `transform() @ components_` where it is missing) and assert a
strictly decreasing curve; FA adds `noise_variance_` plus in-sample log-likelihood, ICA is judged by excess
kurtosis because it optimises independence, not squared error.
```python
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD, NMF, FactorAnalysis, FastICA
from scipy.sparse import csr_matrix
X0 = datasets.load_digits().data; F = StandardScaler().fit_transform(X0)
def recon_mse(e, D, linear=True):
    Z = e.transform(D); R = e.inverse_transform(Z) if linear else Z @ e.components_ + getattr(e, "mean_", 0.0)
    return float(np.mean((D - R) ** 2))
c = {n: [] for n in ("pca_mse", "tsvd_mse", "nmf_mse", "fa_mse")}
for k in (2, 5, 10, 20, 40):
    c["pca_mse"].append(recon_mse(PCA(n_components=k).fit(F), F)); c["tsvd_mse"].append(recon_mse(TruncatedSVD(n_components=k).fit(csr_matrix(F)), F))
    c["nmf_mse"].append(recon_mse(NMF(n_components=k, random_state=1, max_iter=800).fit(X0), X0, False)); c["fa_mse"].append(recon_mse(FactorAnalysis(n_components=k, random_state=42).fit(F), F, False))
assert all(np.all(np.diff(v) < 0) for v in c.values()), c
print(" ".join(f"RESULT {n}={[round(x, 5) for x in v]}" for n, v in c.items()), "all_monotone=True")
fa = FactorAnalysis(n_components=10, random_state=42).fit(F); ica = FastICA(n_components=10, random_state=42, max_iter=5000, tol=1e-3).fit(F)
print(f"RESULT fa_shared_var_share={1 - fa.noise_variance_.mean() / F.var(ddof=0).mean():.4f} fa_mean_loglik={[round(float(FactorAnalysis(n_components=k, random_state=42).fit(F).score_samples(F).mean()), 2) for k in (2, 5, 10, 20)]} ica_excess_kurtosis={float(np.mean([np.mean((z - z.mean()) ** 4) / z.var() ** 2 - 3 for z in ica.transform(F).T])):.4f} pca_mse_k10={recon_mse(PCA(n_components=10).fit(F), F):.5f} ica_mse_k10={float(np.mean((F - (ica.transform(F) @ ica.mixing_.T + ica.mean_)) ** 2)):.5f}")
```
PCA runs 0.74730, 0.55855, 0.39198, 0.19717, 0.04691 for k = 2, 5, 10, 20, 40 (MSE per standardised feature), TSVD
almost identical - the "TSVD is similar to PCA" note (Albon p182). At k = 10 PCA (0.39198) beats FA (0.42220) and ICA
(0.45392) - the optimal linear rank-k projector wins on squared error, and FA's log-likelihood still climbs with k.

### t-SNE: nonlinear projection scored by neighbour preservation (sklearn.manifold)
t-SNE has no `inverse_transform`, so reconstruction means neighbourhood preservation: trustworthiness penalises
embedding neighbours that were far in the input space, continuity is its mirror - both rebuilt from rank matrices,
since this build keeps `trustworthiness` in `sklearn.manifold` and has no `continuity`.
```python
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
rng = np.random.default_rng(42); F = StandardScaler().fit_transform(datasets.load_digits().data)
Xs = np.ascontiguousarray(F[rng.choice(F.shape[0], 400, replace=False)]); K = 5; rows = np.arange(Xs.shape[0])
def ranks(P):
    d = pairwise_distances(P); np.fill_diagonal(d, np.inf); o = np.argsort(d, axis=1); r = np.empty(o.shape, np.int64)
    np.put_along_axis(r, o, np.arange(1, o.shape[1] + 1)[None, :], axis=1); return r, o[:, :K]
u, nu = ranks(Xs); norm = 2.0 / (Xs.shape[0] * K * (2.0 * Xs.shape[0] - 3.0 * K - 1.0))
def scores(Yh):
    v, nv = ranks(Yh); T = float(1.0 - norm * np.maximum(u[rows[:, None], nv] - K, 0).sum()); Q = float(1.0 - norm * np.maximum(v[rows[:, None], nu] - K, 0).sum())
    return T, Q, float(np.mean([len(set(nv[i].tolist()) & set(nu[i].tolist())) / K for i in rows]))
Zt = TSNE(n_components=2, learning_rate="auto", init="pca", perplexity=30, random_state=42).fit_transform(Xs)
Tp, Qp, Op = scores(Zt); Td, _, Od = scores(PCA(n_components=2).fit_transform(Xs)); assert abs(Tp - float(trustworthiness(Xs, Zt, n_neighbors=K))) < 1e-6
print(f"RESULT orig_features={Xs.shape[1]} tsne_reduced_features={Zt.shape[1]} tsne_trust={Tp:.5f} tsne_continuity={Qp:.5f} tsne_nbr_overlap={Op:.3f} pca2_trust={Td:.5f} pca2_nbr_overlap={Od:.3f} matches_sklearn_trustworthiness=True")
```
On the seeded 400-observation subset t-SNE scores trust 0.98372 / continuity 0.97780 / neighbour overlap 0.640 versus
0.80827 / 0.158 for 2-component PCA: local structure kept, global distance not - never read distances or variance shares off a t-SNE plot.

## Output contract
Report per estimator: `orig_features` (`p_original`), `reduced_features` (`p_new`), `explained_variance_ratio_` /
`variance_retained`, the `n_components` meeting `GOAL_VAR`, `recon_mse`, FA `noise_variance_`/`score_samples`, ICA
excess kurtosis, and for projections `trust`, `continuity`, `nbr_overlap` - one `RESULT name=value` line per quantity, naming the MSE space (standardised features = z-score-squared units; raw counts for NMF).

## Validity conditions
- Extraction buys size with interpretability; if columns must stay readable, reduce by selection instead (Albon p172).
- NMF is invalid as soon as any value is negative and publishes no variance measure (Albon p181); kernel PCA has no variance target either, so `n_components` and the kernel hyperparameters must be pre-set (Albon p177).
- Standardise for PCA/TSVD/FA/ICA - linear methods whose variance shares are meaningless on raw scales (Albon p173) - and fit TruncatedSVD once per pipeline because its signs can flip between fittings (Albon p182).
- LDA reads the target and returns at most `n_classes - 1` axes; a 99% variance share is not proof of separation.
- t-SNE is stochastic and perplexity-dependent: seeds and neighbour scores reproduce, coordinates do not.

## Workflow
Standardise (or confirm nonnegativity) -> fit the extractor inside the pipeline, once, on train only -> choose
`n_components` by the 95%/99% rule where ratios exist (PCA, LDA, TSVD) and by end-model search where they do not (NMF, kernel PCA) (Albon p177, p181-183) -> assert the MSE curve falls -> hand `p_new` to the model.

## Conventions
- Seeds: `numpy.random.default_rng(42)` for subsets, `random_state=1` where the source uses it (NMF, `make_circles`), `random_state=42` for t-SNE/FA/ICA; `NMF(max_iter=800)` and `FastICA(max_iter=5000, tol=1e-3)` keep the digits fits converged and warning-free.
- Libs: `sklearn.decomposition`, `sklearn.discriminant_analysis`, `sklearn.manifold`, `sklearn.metrics.pairwise_distances`, `scipy.sparse`. Missing package, named honestly: `umap-learn` is not installed, so no UMAP run or number appears here.
- Version drift: import `trustworthiness` from `sklearn.manifold` (not `sklearn.metrics`); `KernelPCA.lambdas_` exists only for a precomputed kernel; `FactorAnalysis` has no `inverse_transform`. A 4th-decimal gap to a printed book number is reported as drift, never as a new value.
- Never invent a value - compute it, print `RESULT`, cite the page (Albon ch.9 p172-183).

## Out of scope
Feature *selection* (variance threshold, `SelectKBest`, RFE, importance, mutual information, synthetic features)
is Albon ch.10 and belongs to the sibling `mlk-dim-reduction-selection` skill; this skill only projects or factorises
columns into new ones. UMAP is excluded for the missing-package reason above (`Isomap`/`SpectralEmbedding` are the
installed graph embeddings the neighbour block can judge instead), and kernel/`gamma`/`n_components`/perplexity
tuning by cross-validation is model selection (Albon p177).
