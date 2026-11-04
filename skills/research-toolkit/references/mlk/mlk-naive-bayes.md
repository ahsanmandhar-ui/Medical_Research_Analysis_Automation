---
name: mlk-naive-bayes
description: Train and use Gaussian, multinomial and Bernoulli naive Bayes classifiers in scikit-learn (sklearn.naive_bayes with numpy), per Albon ch.18 - compare the per-class posterior numerator likelihood x prior and take the argmax, set class priors (priors / class_prior) or a uniform prior with fit_prior=False, tune additive smoothing alpha, classify documents fed as count/presence vectors, and recalibrate extreme NB probabilities with CalibratedClassifierCV. Use when the user wants naive Bayes / an NB classifier, a Gaussian NB on continuous features, multinomial or Bernoulli NB on counts or binary text vectors, class priors for NB, smoothing for NB, or calibrated predict_proba from an NB model.
---

# Naive Bayes Classifiers (Python)

## Purpose
Fit and predict with the three naive Bayes (NB) classifiers scikit-learn ships. Core rule: NB compares
the *numerators* of the posterior, P(x1..xj | y) x P(y), across classes - the marginal P(x1..xj) is the
same constant for all - and the class with the greatest posterior numerator becomes y-hat
(Albon p294-295). Bayes' theorem itself: P(A|B) = P(B|A)P(A) / P(B) (Albon p294).

## Inputs
- `X`: (n_samples, n_features) numeric array; `y`: class labels. Which NB to use is decided by the
  *nature of the features*: continuous -> Gaussian, discrete/count -> multinomial, binary -> Bernoulli
  (Albon p295).
- Optional priors: an explicit list of per-class probabilities (`priors` on GaussianNB, `class_prior`
  on the discrete ones); omitted, the prior is learned from the data's class frequencies (Albon p296-298).
- Second assumption baked into every variant: each feature and its likelihood is conditionally
  independent of the others given the class - the "naive" part, frequently wrong yet rarely fatal to
  classifier quality (Albon p295).

## Algorithms (all pure numpy/scipy)

### Gaussian NB - continuous features (Albon p296)
- Likelihood is a normal density per class: p(x_j | y) = N(x_j; mu_y, sigma_y^2), the class mean and
  variance of feature x_j; best used when *all* features are continuous (Albon p296).
```python
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
iris = datasets.load_iris()
features, target = iris.data, iris.target
model = GaussianNB().fit(features, target)          # prior learned from data
pred = int(model.predict([[4, 4, 4, 0.4]])[0])
print(f"RESULT gaussian_pred={pred}")               # book prints array([1])
gm = GaussianNB(priors=[0.25, 0.25, 0.5]).fit(features, target)
print(f"RESULT priors_used={gm.class_prior_.round(4).tolist()}")
```
The book predicts class `1` (versicolor) for the observation [4, 4, 4, 0.4]; this block prints the same
`gaussian_pred=1` (Albon p296). `priors=[...]` is the belief about each class *before* seeing the data,
listed in sorted `classes_` order (Albon p296).

### Multinomial NB - discrete and count features (Albon p297-298)
```python
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
text_data = np.array(['I love Brazil. Brazil!', 'Brazil is best', 'Germany beats both'])
features = CountVectorizer().fit_transform(text_data).toarray()   # bag-of-words count matrix
model = MultinomialNB(class_prior=[0.25, 0.5]).fit(features, np.array([0, 0, 1]))
pred = int(model.predict([[0, 0, 0, 1, 0, 1, 0]])[0])
print(f"RESULT multinomial_pred={pred}")            # book prints array([0])
print(f"RESULT class_log_prior={model.class_log_prior_.round(4).tolist()}")
```
Likelihoods are multinomial instead of normal - the usual pick for discrete data (e.g. 1-5 star
ratings) and, most commonly, text classification over bag-of-words or tf-idf vectors (Albon p297-298).
If `class_prior` is omitted the priors are learned; `fit_prior=False` forces a uniform prior;
`alpha` is additive (Laplace) smoothing, default 1.0, 0.0 = no smoothing, and should be tuned
(Albon p298). Class `0` = pro-Brazil, matching the book (Albon p298).

### Bernoulli NB - binary features (Albon p298-299)
```python
import numpy as np
from sklearn.naive_bayes import BernoulliNB
rng = np.random.default_rng(42)                     # book used an unseeded np.random.randint
features = rng.integers(2, size=(100, 3))           # three binary features
target = rng.integers(2, size=100)
model = BernoulliNB(class_prior=[0.25, 0.5]).fit(features, target)
proba = model.predict_proba([[1, 0, 1]])[0]
print(f"RESULT bernoulli_pred={int(proba.argmax())} proba={proba.round(4).tolist()}")
```
All features must be binary (e.g. a one-hot nominal, or word presence/absence in a document - why this
variant also sees heavy text-classification use). Same `alpha` smoothing and prior knobs as
MultinomialNB: `class_prior=[...]` for beliefs, `fit_prior=False` for a uniform prior (Albon p298-299).

### Calibrating NB predicted probabilities (Albon p299-301)
```python
import numpy as np
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
iris = datasets.load_iris()
features, target = iris.data, iris.target
obs = [[2.6, 2.6, 2.6, 0.4]]
raw = GaussianNB().fit(features, target)
cal = CalibratedClassifierCV(GaussianNB(), cv=2, method="sigmoid").fit(features, target)
print(f"RESULT raw_proba={raw.predict_proba(obs)[0].round(6).tolist()}")
print(f"RESULT calib_proba={cal.predict_proba(obs)[0].round(6).tolist()}")
```
Raw NB `predict_proba` ranks classes validly but the values run to extremes near 0 and 1 and "should
not be believed" (Albon p296, p300). This block prints a raw vector of about
[0.000232, 0.999768, 3.2e-07] (book: 2.58e-04, 0.99974, 3.24e-07) and a calibrated vector of
[0.3186, 0.6366, 0.0448] - the book's calibrated [0.31859969, 0.63663466, 0.04476565] reproduced to
3 dp (Albon p300). `CalibratedClassifierCV` trains on the k-fold training sets, calibrates on each
test fold, and averages the k folds; `method` is Platt's sigmoid or isotonic regression, and isotonic
tends to overfit below ~100 observations, hence sigmoid for the 150-row iris set (Albon p300-301).

## Output contract
Fitted objects expose `classes_`, `class_prior_`/`class_log_prior_`, and the class-conditional feature
parameters (`theta_`, `var_` for GaussianNB; `feature_count_`, `feature_log_prob_` for the discrete
ones). `predict` returns labels; `predict_proba` returns an (n_samples, n_classes) row-normalised
posterior matrix; the calibrated estimator swaps raw `predict_proba` for calibrated values. Print one
`RESULT name=value` line per quantity.

## Validity conditions
- Feature-independence-given-class holds "by assumption", not by test - verify the distribution fit
  instead: Gaussian needs continuous features, multinomial needs non-negative counts, Bernoulli needs
  0/1 values (Albon p295-299).
- Priors must be a probability-style list (one entry per class, in sorted-class order; sklearn
  normalises it, which is why [0.25, 0.5] works for two classes) (Albon p296-298).
- NB probabilities are decision-grade only after calibration; without it, use ranks/thresholds with
  caution (Albon p300).

## Conventions
- Smoothing ALPHA = 1.0 (library default), tune with model selection; 0.0 disables it (Albon p298).
- Calibration: `cv=2, method="sigmoid"` as in the book; prefer sigmoid over isotonic for small n
  (Albon p300-301).
- All demo randomness via `numpy.random.default_rng(42)` so printed numbers reproduce; toy data only
  (iris, the Brazil/Germany sentences, seeded 0/1 arrays) - never read files.
- Validated on sklearn 1.9 where `class_prior` still fits; if a newer build rejects it, omit it
  (prior is then learned) or use `fit_prior=False`, and note the drift rather than inventing values.
- Source: Albon, *Machine Learning with Python Cookbook*, 2nd ed., Ch.18 Naive Bayes, pp.294-301
  (PDF pagination).

## Out of scope
Text preprocessing and vectorisation - tokenizing, stemming, stop-words, CountVectorizer/TF-IDF tuning
- belong to the sibling `mlk-text-features` skill; this skill starts from a ready count/presence matrix
(the toy vectorizer above only reproduces the book example). Hyper-parameter search over `alpha`,
calibrating non-NB models, and the other classifiers (logistic regression, kNN) are sibling skills.
