---
name: mlk-clustering
description: Group a feature matrix into latent clusters with scikit-learn (sklearn.cluster + sklearn.metrics), following Albon, ML with Python Cookbook ch.19 - K-Means, mini-batch K-Means, Mean-Shift, DBSCAN and agglomerative/hierarchical merging, plus silhouette-based selection of k. Use when the user wants to partition observations with no target ("unsupervised"/"cluster"), asks "how many clusters", names k-means/DBSCAN/mean-shift/hierarchical, or wants density outliers.
---

# Clustering (Python)

## Purpose
Assign every observation to a latent group when only features (no target) exist: within-group
differences are smaller than between-group differences (Albon p302). One distance/similarity model
per algorithm; `labels_` is the per-observation assignment.

## Inputs
- `X`: numeric feature matrix `(n_samples, n_features)`. Standardise first - distance methods weight
  each raw feature, so equal scale is assumed (Albon p303).
- `n_clusters=k` for the centroid/agglomerative methods; DBSCAN uses `eps`/`min_samples` instead;
  Mean-Shift needs neither (bandwidth is auto-estimated, Albon p306-307).
- Demo here uses `datasets.load_iris()` (150 x 4, true 3 classes; Albon uses it to check against, p304).

## Algorithms (all pure numpy/scipy)

### K-Means (Albon p303-305)
Random centres -> assign each point to the nearest -> recompute centres as means -> repeat until no
point changes cluster. Assumes convex, equally-scaled, roughly balanced clusters (p303).
```python
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
F = StandardScaler().fit_transform(datasets.load_iris().data)
m = KMeans(n_clusters=3, random_state=0, n_init=10).fit(F)   # k=3, Iris (Albon p303)
new = m.predict([[0.8, 0.8, 0.8, 0.8]])[0]                   # assigned to nearest centre (p304)
print(f"RESULT n_clusters={m.n_clusters} inertia={m.inertia_:.3f} centres_shape={m.cluster_centers_.shape} new_obs_cluster={new}")
```
Inertia (sum of squared distance to the centre) lands at ~139.8 on standardised Iris; the integer a
new point gets is an arbitrary label - Albon p304 prints `array([0])` for a differently-labelled fit.

### Mini-batch K-Means (Albon p305-306)
Same steps, but each iteration updates centres on a random `batch_size` subset - far faster, tiny
quality cost (p306).
```python
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
F = StandardScaler().fit_transform(datasets.load_iris().data)
m = MiniBatchKMeans(n_clusters=3, random_state=0, batch_size=100).fit(F)   # batch_size is the only new knob (p306)
print(f"RESULT mbk_n_clusters={m.n_clusters} mbk_inertia={m.inertia_:.3f}")
```
`mbk_inertia ~ 140.0` vs full k-means `139.8` - the "small cost in quality" the source describes (p306).

### Mean-Shift (Albon p306-307)
Each point climbs toward the densest region within a `bandwidth` kernel until it settles on a mode; no
`k` or shape assumed. Orphan points join the nearest kernel unless `cluster_all=False` (label -1).
```python
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MeanShift
import numpy as np
F = StandardScaler().fit_transform(datasets.load_iris().data)
m = MeanShift().fit(F)                        # bandwidth auto-estimated by default (p307)
print(f"RESULT ms_estimated_k={len(np.unique(m.labels_))} ms_orphan_policy='cluster_all=True (default)'")
```

### DBSCAN (Albon p307-309)
Density-based: a point is core if it has `min_samples` neighbours within `eps`; core regions grow into
clusters, border points join, the rest are outliers labelled -1. No cluster-shape assumption (p308).
```python
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
F = StandardScaler().fit_transform(datasets.load_iris().data)
lab = DBSCAN().fit(F).labels_                 # defaults eps=0.5, min_samples=5 (Albon p308)
clusters = sorted(set(lab.tolist()) - {-1})
print(f"RESULT dbscan_n_clusters={len(clusters)} cluster_labels={clusters} dbscan_n_outliers={int((lab == -1).sum())}")
```
Worked number: DBSCAN finds **2 clusters** on standardised Iris, the split Albon p308-309 shows as
"two clusters ... 0 and 1, while outlier observations are labeled -1" (the exact -1 count, ~35 there,
moves by one across scikit-learn versions - one borderline point flips between core and outlier).

### Agglomerative / hierarchical merging (Albon p309-310)
Every observation starts as its own cluster; the closest pair merges repeatedly until `n_clusters`
remain. `linkage` sets the merge criterion: `ward` (min merged variance), `average`, or `complete`
(max pairwise distance) (p310).
```python
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import numpy as np
F = StandardScaler().fit_transform(datasets.load_iris().data)
m = AgglomerativeClustering(n_clusters=3, linkage="ward").fit(F)   # ward minimises merged variance (p310)
print(f"RESULT agg_n_clusters={len(np.unique(m.labels_))} agg_linkage=ward")
```

### Selecting k with the silhouette (Albon p304)
When `k` is unknown, pick the value whose silhouette - within-cluster similarity vs between-cluster
similarity - is highest; a wrong k hurts k-means "critically" (p304).
```python
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
F = StandardScaler().fit_transform(datasets.load_iris().data)
sc = {k: silhouette_score(F, KMeans(n_clusters=k, random_state=0, n_init=10).fit_predict(F)) for k in range(2, 7)}
best = max(sc, key=sc.get)
print(f"RESULT best_k={best} best_silhouette={sc[best]:.4f} curve={ {k: round(v, 4) for k, v in sorted(sc.items())} }")
```
On Iris the curve peaks at k=2 (~0.58) even though three species exist - silhouette rewards the two
well-separated groups over the touching pair; treat it as a criterion, not the ground truth (p304).

## Output contract
Return/report: `labels_` (per-observation cluster id; -1 = outlier/unassigned for DBSCAN and
Mean-Shift `cluster_all=False`), `cluster_centers_` (centroid methods), `n_clusters`/`inertia_`,
`dbscan_n_outliers`, and for selection the `best_k` + `silhouette` curve. Print one `RESULT name=value`
line per quantity; state units (features are standardised, so inertia is in z-score-squared units).

## Validity conditions
- Standardise before any distance method - unequal scales break k-means' equal-feature assumption (p303).
- k-means expects convex, balanced, equal-variance clusters; if that is doubtful use DBSCAN/Mean-Shift (p303).
- A wrong `k` degrades k-means critically - choose it by silhouette or domain knowledge (p304).
- DBSCAN is meaningless without a sensible `eps`/`min_samples`; sparse settings label almost everything -1 (p308).
- Integer labels are arbitrary - they are not comparable across runs, models, or versions (p304).

## Conventions
- `StandardScaler` first; `random_state=0` and `n_init=10` for the k-means family; demo `datasets.load_iris()`.
- Libs: `sklearn.cluster` (KMeans, MiniBatchKMeans, MeanShift, DBSCAN, AgglomerativeClustering) +
  `sklearn.metrics.silhouette_score`; no third-party package is required by these ch.19 recipes.
- In recent scikit-learn, AgglomerativeClustering's `affinity` is now `metric` and `n_jobs` has dropped
  from KMeans - omit the removed args or the call warns/fails.
- Never invent a value - compute and print `RESULT`, and cite the page for each method (Albon ch.19 p302-310).

## Out of scope
Gaussian-mixture (soft/probabilistic) and spectral (graph-cut) clustering, and the full silhouette
derivation, are NOT in the grounded pages here - Albon covers them as separate ch.19 recipes and
cross-references the silhouette to Recipe 11.9 (ch.11, Albon p304), so they are left to sibling skills.
Supervised classification, dimensionality reduction (PCA) before clustering, and DBSCAN tuning by
k-distance plot are also excluded; this skill fits the five distance/density/hierarchical models above.
