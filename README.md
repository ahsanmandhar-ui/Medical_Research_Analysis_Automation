# Research Automation Toolkit

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![TimesFM](https://img.shields.io/badge/TimesFM-1.0--200M-purple.svg)](https://github.com/google-research/timesfm)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: Active](https://img.shields.io/badge/Status-Active-brightgreen.svg)](https://github.com/ahsanmandhar-ui/research-automation-toolkit)

> **Narrative:** Data analysis will be automated in the future.
> This repository bundles a complete research-and-data-analysis toolkit with 61 specialist method guides, a TimesFM-1.0-200M zero-shot forecasting pipeline, and Mermaid skill-graph diagrams — all structured for reproducible, AI-augmented research automation.
>
> **Copyright Ahsan Mandhar**

---

## 📊 TimesFM Forecast: Meta-Analysis Publications (2015–2031)

Zero-shot foundation forecast under AI-automation structural shift, generated with **TimesFM-1.0-200M** (CPU-only, bfloat16, <2GB memory).

![Meta-Analysis Publications Forecast](data/meta_analysis_forecast_2027_2031.png)

### Forecast Table

| Year | Point Forecast (mean) | 10th pct (worst-case decline) | 90th pct (best-case stability) |
|------|----------------------|-------------------------------|--------------------------------|
| 2027 | — | — | — |
| 2028 | — | — | — |
| 2029 | — | — | — |
| 2030 | — | — | — |
| 2031 | — | — | — |

> Run `python analysis/timesfm_forecast_v2.py` to generate the forecast CSV, table, and PNG. Outputs land in `data/`.
### SRM — Statistical Research Methods (22 skills)

```mermaid
flowchart TD
    classDef default fill:#e1f5fe,stroke:#01579b,stroke-width:2px,color:#01579b,font-size:11px
    classDef group fill:#c5e1a5,stroke:#33691e,stroke-width:3px,color:#33691e,font-size:12px,font-weight:bold

    subgraph SRM["SRM - Statistical Research Methods (22 skills) - Copyright Ahsan Mandhar"]
        direction TB

        s1[["srm-research-question-design<br/>Turn a topic into an answerable quantitative research question<br/>Extract POCI elements: Population, Outcome, Comparison, Intervention<br/>Identify response and explanatory variables"]]
        s2[["srm-study-design-selection<br/>Pick and encode the right quantitative design from POCI components<br/>Descriptive vs analytical vs experimental design<br/>Draft a study-design protocol"]]
        s3[["srm-sampling-methods<br/>Draw and audit research samples using numpy.random and pandas.sample<br/>Simple random, systematic, stratified, cluster sampling<br/>Assess whether a sample can represent its population"]]
        s4[["srm-variable-types-encoding<br/>Classify every variable as quantitative or qualitative<br/>Nominal vs ordinal, discrete vs continuous<br/>Encode into correct pandas dtype"]]
        s5[["srm-data-collection-protocols<br/>Study measurement and data capture protocols<br/>Data collection planning and execution"]]
        s6[["srm-numerical-summaries-quantitative<br/>Numerically describe one quantitative variable or many by group<br/>Mean vs median, range, sample standard deviation, variance<br/>IQR, percentiles, standardise to z-score, flag outliers"]]
        s7[["srm-numerical-summaries-qualitative<br/>Numerically summarise qualitative categorical data<br/>Overall, row and column proportions and percentages<br/>Odds or odds ratio from a contingency table"]]
        s8[["srm-graphical-summaries<br/>Visualise distributions and relationships graphically<br/>Publication-quality figures from pandas frames"]]
        s9[["srm-distributions-normal-model<br/>Normal distribution model in pure Python<br/>Probability density, z-scores, standard normal table"]]
        s10[["srm-confidence-intervals-single<br/>Confidence interval for ONE population proportion or mean<br/>90/95/99% CI, margin of error, standard error<br/>Sample size determination"]]
        s11[["srm-confidence-intervals-comparisons<br/>Confidence intervals for comparisons between two groups<br/>Paired before/after mean differences<br/>Two independent means: Welch vs pooled<br/>Odds ratio from four cell counts"]]
        s12[["srm-hypothesis-testing-framework<br/>Hypothesis testing framework: null vs alternative<br/>P-values, Type I and Type II errors, statistical power<br/>Decision logic for accepting or rejecting hypotheses"]]
        s13[["srm-hypothesis-tests-means<br/>t-tests: one-sample, paired/dependent, two-sample/Welch<br/>Equal vs unequal variance assumptions<br/>Turn a t-statistic into a P-value"]]
        s14[["srm-hypothesis-tests-categorical<br/>Chi-square tests for categorical data<br/>Independence and goodness-of-fit tests<br/>Expected counts below 5 rule"]]
        s15[["srm-correlation<br/>Scatterplot reading: form, direction, variation, outliers<br/>Pearson correlation coefficient r and R-squared<br/>Influential points detection"]]
        s16[["srm-linear-regression<br/>Fit simple linear regression y_hat = b0 + b1*x<br/>Least-squares OLS via pure Python<br/>Interpret slope and intercept, test slope significance"]]
        s17[["srm-probability-and-decision-logic<br/>Probability from three approaches: classical, frequentist, subjective<br/>Independence of events, conditional probability<br/>Expected number of outcomes"]]
        s18[["srm-confounding-and-blinding-experimental<br/>Protect internal validity of an experiment treatment vs control<br/>Balance groups, check randomisation, SRM check<br/>A/B test and clinical trial design audit"]]
        s19[["srm-confounding-and-blinding-observational<br/>Confounding and blinding for observational non-randomised studies<br/>Cohort, cross-sectional, case-control designs<br/>Baseline imbalance tables, crude vs adjusted numbers"]]
        s20[["srm-statistical-reference<br/>Lookup reference layer for Dunn Scientific Research and Methodology<br/>Notation dictionary, parameter vs statistic vs standard error<br/>Verify code against book printed answers"]]
        s21[["srm-reporting-research<br/>Build and QC the written scientific report following Dunn structure<br/>AIMRaD and STARD-compliant report assembly<br/>Results section, tables, figures, critical reading"]]
        s22[["srm-correlation-linear-integration<br/>Integrate correlation and regression results<br/>Combine r, R-squared, and regression output<br/>Report strength of evidence and effect size"]]

        s1 --> s2 --> s3 --> s4 --> s5 --> s6 --> s7 --> s8 --> s9 --> s10 --> s11 --> s12 --> s13 --> s14 --> s15 --> s16 --> s17 --> s18 --> s19 --> s20 --> s21 --> s22
### MAA — Meta-Analysis (7 skills)

```mermaid
flowchart TD
    classDef default fill:#f3e5f5,stroke:#4a148c,stroke-width:2px,color:#4a148c,font-size:11px
    classDef group fill:#c5e1a5,stroke:#33691e,stroke-width:3px,color:#33691e,font-size:12px,font-weight:bold

    subgraph MAA["MAA - Meta-Analysis (7 skills) - Copyright Ahsan Mandhar"]
        direction TB

        m1[["maa-effect-sizes<br/>Compute study-level effect sizes for meta-analytic input table<br/>Risk ratio RR, odds ratio OR on log-OR scale, risk difference RD<br/>Variance, standard error, CI behind one effect size<br/>Handle zero cell or dropouts"]]
        m2[["maa-pooling-models<br/>Fixed- and random-effects pooling of meta-analytic evidence<br/>Inverse-variance fixed-effect weights<br/>Mantel-Haenszel and Peto methods<br/>Study weights and pooled result"]]
        m3[["maa-heterogeneity<br/>Heterogeneity statistics computed from first formulas<br/>Cochran's Q with chi-square P value<br/>I-squared, tau-squared, H-squared<br/>Prediction interval"]]
        m4[["maa-forest-plots<br/>Draw publication-quality meta-analysis forest plots<br/>Replace unavailable PythonMeta and zEpid ForestPlot<br/>Log versus linear axis, null-line placement<br/>Size study marks by precision"]]
        m5[["maa-publication-bias<br/>Detect publication bias and small-study effects<br/>Contour-enhanced funnel plot with matplotlib zones<br/>Egger and Begg P-values<br/>Assess asymmetry in funnel plot"]]
        m6[["maa-subgroup-meta-regression<br/>Subgroup contrasts and meta-regression<br/>Sensitivity and robustness analysis<br/>Test whether study characteristics explain pooled effect<br/>Drop-one sensitivity analysis"]]
        m7[["maa-validation-vs-r-stata<br/>Cross-validate Python meta-analysis against R meta/metafor or Stata<br/>Write reference path for comparison<br/>Choose cross-tool tolerance<br/>Explain small discrepancies between tools"]]

        m1 --> m2 --> m3 --> m4 --> m5 --> m6 --> m7
### PDA — Python Data Analysis (13 skills)

```mermaid
flowchart TD
    classDef default fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px,color:#1b5e20,font-size:11px
    classDef group fill:#c5e1a5,stroke:#33691e,stroke-width:3px,color:#33691e,font-size:12px,font-weight:bold

    subgraph PDA["PDA - Python Data Analysis (13 skills) - Copyright Ahsan Mandhar"]
        direction TB

        p1[["pda-python-language-core<br/>Core Python language semantics for data analysis<br/>Standard library: re, itertools, collections, pathlib, os, sys, tempfile<br/>No third-party dependencies<br/>Raw-Python data prep, grouping, counting with dict/set"]]
        p2[["pda-numpy-fundamentals<br/>NumPy ndarray toolkit for array-oriented data work<br/>Build, subset, aggregate NumPy arrays<br/>Select rows by boolean mask or integer index array<br/>Compute sum/mean along axis, sort, unique"]]
        p3[["pda-numpy-advanced<br/>Advanced NumPy internals: shape, strides, flags, dtype hierarchy<br/>Zero-copy strided views, np.issubdtype<br/>Fix operands could not be broadcast together errors<br/>Handle arrays too large for RAM"]]
        p4[["pda-pandas-foundations<br/>Build and inspect core pandas objects: Series, DataFrame, Index<br/>Creation from arrays, lists, dicts<br/>Index and name attributes<br/>Select rows/columns with loc/iloc, filter with mask"]]
        p5[["pda-data-loading-io<br/>Load data into pandas DataFrame from every source<br/>CSV, Excel, JSON, database query into DataFrame<br/>scikit-learn toy datasets, seeded simulated generators"]]
        p6[["pda-data-cleaning<br/>Clean data end to end in pure Python pandas/numpy<br/>Drop duplicate rows, filter rows on condition<br/>Handle NaN values, join two frames"]]
        p7[["pda-data-wrangling-reshape<br/>Wrangle pandas DataFrame end to end<br/>Creation, description head/tail/shape/describe<br/>loc vs iloc and conditional selection<br/>Count unique classes, group by category"]]
        p8[["pda-groupby-aggregation<br/>Split-apply-combine in pure pandas/numpy<br/>Group by array, column, dict, Series, callable, MultiIndex level<br/>Optimized or custom aggregation functions<br/>Per-group stats, top-n per group, pivot tables"]]
        p9[["pda-time-series<br/>Date/timestamp APIs, date_range and frequency aliases<br/>Shifting, resampling down/up-sample, OHLC<br/>Rolling/expanding windows with min_periods, EWMA<br/>Parse dates, handle time zones"]]
        p10[["pda-plotting-visualization<br/>Build publication figures from pandas frames with matplotlib and seaborn<br/>Plot Series or DataFrame, histogram, KDE<br/>Subplots/panels, legend, error bars<br/>Save figure headless"]]
        p11[["pda-modeling-libraries<br/>Fit statistical and ML models from pandas frame<br/>Patsy formula design matrices: dmatrices, standardize/center<br/>C(), I(), key1:key2 interactions<br/>statsmodels, patsy, sklearn integration"]]
        p12[["pda-ipython-jupyter-workflow<br/>Pure-Python reproduction of IPython/Jupyter analysis workflow<br/>Magic commands, timing, profiling, code introspection<br/>Structure notebook for reproducibility"]]
        p13[["pda-analysis-examples<br/>Analysis examples and worked demonstrations<br/>End-to-end analysis walkthroughs<br/>Reference implementations"]]]

        p1 --> p2 --> p3 --> p4 --> p5 --> p6 --> p7 --> p8 --> p9 --> p10 --> p11 --> p12 --> p13
### MLK — Machine Learning (13 skills)

```mermaid
flowchart TD
    classDef default fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#e65100,font-size:11px
    classDef group fill:#c5e1a5,stroke:#33691e,stroke-width:3px,color:#33691e,font-size:12px,font-weight:bold

    subgraph MLK["MLK - Machine Learning (13 skills) - Copyright Ahsan Mandhar"]
        direction TB

        l1[["mlk-vectors-matrices-arrays<br/>Linear algebra for ML: vectors, matrices, tensors<br/>Dot/inner products, norms, projections<br/>Matrix multiply, inverse, solve<br/>Eigenvalues, SVD, PCA mechanics, loss functions"]]
        l2[["mlk-data-loading<br/>Load data into training-ready pandas/sklearn frame<br/>scikit-learn toy datasets, seeded simulated generators<br/>Import CSV, Excel, JSON<br/>Query database into DataFrame"]]
        l3[["mlk-data-wrangling<br/>Wrangle pandas DataFrame end to end in pure Python<br/>Creation, description, head/tail/shape/describe<br/>Drop duplicates, filter rows, group by category<br/>Handle NaN, join two frames"]]
        l4[["mlk-datetime-features<br/>Parse date strings and build ML-ready temporal features<br/>to_datetime with explicit format<br/>Break timestamp into component features<br/>Encode day-of-week, compute time differences"]]
        l5[["mlk-categorical-features<br/>Encode categorical string features into numeric ML inputs<br/>Label and ordinal reduction, one-hot/dummy encoding<br/>Target-encode a group, handle high-cardinality features"]]
        l6[["mlk-text-features<br/>Text features for machine learning in pure Python<br/>Strip whitespace and punctuation<br/>CountVectorizer, TfidfVectorizer pipelines<br/>Stemming, stop-word lists"]]
        l7[["mlk-numerical-features<br/>Build ML-ready numeric features from quantitative columns<br/>Scale, normalise, standardise numeric features<br/>Polynomial and interaction terms<br/>Log/sqrt transform, outlier detection, binning"]]
        l8[["mlk-dim-reduction-extraction<br/>Reduce original features to smaller projected components<br/>PCA with variance target, ICA, NMF, kernel PCA<br/>t-SNE, TruncatedSVD<br/>How many components keep 95% or 99% variance"]]
        l9[["mlk-naive-bayes<br/>Train Gaussian, multinomial, Bernoulli naive Bayes classifiers<br/>sklearn.naive_bayes with numpy<br/>Compare NB variants for different data types"]]
        l10[["mlk-svm<br/>Train, inspect and tune support vector machines<br/>sklearn.svm LinearSVC, SVC, SVR<br/>Max-margin hyperplane, kernel trick<br/>C/gamma values, class probabilities"]]
        l11[["mlk-trees-forests<br/>Train, prune, export and interpret decision trees, random forests, boosted trees<br/>sklearn.tree, sklearn.ensemble, numpy<br/>Feature importance, OOB score, gradient boosting"]]
        l12[["mlk-neural-networks<br/>Build, train and diagnose feedforward neural networks<br/>Hand-written one-hidden-layer numpy MLP<br/>Forward propagation, overfitting diagnostics<br/>Keras-style DNN/MLP fundamentals"]]
        l13[["mlk-clustering<br/>Partition observations with no target unsupervised clustering<br/>sklearn.cluster + sklearn.metrics<br/>How many clusters, silhouette analysis<br/>K-means, hierarchical, DBSCAN"]]]

        l1 --> l2 --> l3 --> l4 --> l5 --> l6 --> l7 --> l8 --> l9 --> l10 --> l11 --> l12 --> l13
### FLOW — Workflow (6 skills)

```mermaid
flowchart TD
    classDef default fill:#fce4ec,stroke:#880e4f,stroke-width:2px,color:#880e4f,font-size:11px
    classDef group fill:#c5e1a5,stroke:#33691e,stroke-width:3px,color:#33691e,font-size:12px,font-weight:bold

    subgraph FLOW["FLOW - Workflow (6 skills) - Copyright Ahsan Mandhar"]
        direction TB

        f1[["flow-py-analysis-env<br/>Python analysis environment setup<br/>Environment pinning, reproducibility gates"]]
        f2[["flow-unit-hygiene<br/>Unit hygiene and provenance gates<br/>Audit trails, reproducibility checks"]]
        f3[["flow-supervisor<br/>Organise parallel work across many skills<br/>Fleet supervision, task coordination"]]
        f4[["flow-analysis-pipeline-dunn<br/>Run and gate a whole quantitative study in Python<br/>Research question -> design -> describe -> infer -> report<br/>Following Dunn's six basic steps"]]
        f5[["flow-meta-analysis-pipeline<br/>End-to-end meta-analysis / evidence pooling<br/>Study-level intake, per-study effect sizes<br/>Reproduce Cochrane pooled estimate"]]
        f6[["flow-corpus-extraction<br/>Re-extract or extend PDF document corpus with PyMuPDF<br/>Inclusive page-range Markdown rendering<br/>Monospace code detection by per-line font mix<br/>Map printed folios to PDF pages"]]

        f1 --> f2 --> f3 --> f4 --> f5 --> f6
### Full Skill Graph — All 61 Skills

```mermaid
flowchart TB
    classDef family fill:#e1f5fe,stroke:#01579b,stroke-width:3px,color:#01579b,font-size:13px,font-weight:bold

    subgraph SRM["SRM - Statistical Research Methods (22 skills) - Copyright Ahsan Mandhar"]
        direction TB
        s1[Study Design & Question] --> s2[Sampling Methods] --> s3[Variable Types & Encoding] --> s4[Data Collection] --> s5[Distributions] --> s6[Graphical Summaries] --> s7[CI Single] --> s8[CI Comparisons] --> s9[Hypothesis Framework] --> s10[t-Tests Means] --> s11[Chi-Square Categorical] --> s12[Correlation] --> s13[Linear Regression] --> s14[Probability & Decision] --> s15[Confounding Experimental] --> s16[Confounding Observational] --> s17[Statistical Reference] --> s18[Reporting Research] --> s19[Correlation-Linear Integration]
    end

    subgraph MAA["MAA - Meta-Analysis (7 skills) - Copyright Ahsan Mandhar"]
        direction TB
        m1[Effect Sizes] --> m2[Pooling Models] --> m3[Heterogeneity] --> m4[Forest Plots] --> m5[Publication Bias] --> m6[Subgroup Meta-Regression] --> m7[Validation vs R Stata]
    end

    subgraph PDA["PDA - Python Data Analysis (13 skills) - Copyright Ahsan Mandhar"]
        direction TB
        p1[Python Language Core] --> p2[NumPy Fundamentals] --> p3[NumPy Advanced] --> p4[Pandas Foundations] --> p5[Data Loading IO] --> p6[Data Cleaning] --> p7[Data Wrangling Reshape] --> p8[GroupBy Aggregation] --> p9[Time Series] --> p10[Plotting Visualization] --> p11[Modeling Libraries] --> p12[IPython Jupyter Workflow] --> p13[Analysis Examples]
    end

    subgraph MLK["MLK - Machine Learning (13 skills) - Copyright Ahsan Mandhar"]
        direction TB
        l1[Vectors Matrices Arrays] --> l2[Data Loading] --> l3[Data Wrangling] --> l4[Datetime Features] --> l5[Categorical Features] --> l6[Text Features] --> l7[Numerical Features] --> l8[Dimensionality Reduction] --> l9[Naive Bayes] --> l10[SVM] --> l11[Trees Forests] --> l12[Neural Networks] --> l13[Clustering]
    end

    subgraph FLOW["FLOW - Workflow (6 skills) - Copyright Ahsan Mandhar"]
        direction TB
        f1[Python Analysis Env] --> f2[Unit Hygiene] --> f3[Supervisor] --> f4[Analysis Pipeline Dunn] --> f5[Meta-Analysis Pipeline] --> f6[Corpus Extraction]
    end

## 🔮 Roadmap: Data Analysis Automation

- [x] Research toolkit: 61 method guides (SRM, MAA, PDA, MLK, FLOW)
- [x] TimesFM-1.0-200M zero-shot forecast pipeline
- [x] Mermaid skill-graph diagrams for all families
- [ ] Automated validity-condition checking per dataset
- [ ] Scheduled forecast re-runs (GitHub Actions)
- [ ] Auto-generated research reports from templates
- [ ] Integration with additional foundation models

---

**Built with** [TimesFM](https://github.com/google-research/timesfm) · [Mermaid](https://mermaid.js.org/) · [Python](https://www.python.org/)

### 🔒 Security — What Gets Pushed

Only essential, human-reviewed files are tracked in this repository. The following are **excluded** via `.gitignore`:

| Category | Excluded |
|----------|----------|
| Python cache | `__pycache__/`, `*.pyc`, `*.pyo` |
| Environments | `.venv/`, `env/`, `.env`, `*.log` |
| IDE | `.vscode/`, `.idea/`, `.oxlintrc.json` |
| OS junk | `.DS_Store`, `Thumbs.db` |
| Archives | `*.zip`, `*.tar.gz`, `*.gz` |
| AI/cache | `tmp/`, `.cache/`, `hf_timesfm_*/`, `.huggingface/` |
| AI-generated | `*_ai_generated*`, `*_generated.py` |
| Jupyter | `.ipynb_checkpoints/` |

**Tracked files only:** `README.md`, `.gitignore`, `LICENSE`, `requirements.txt`, `data/*.csv`, `analysis/*.py`, `skills/**/*.md`, `skills/mermaid/*.mmd`, `docs/*.mmd`.

---

### 📂 Repository Structure

```
research-automation-toolkit/
├── README.md                          # This file — UI with badges, narrative, embedded PNG
├── .gitignore                         # Security: excludes AI junk, cache, env, etc.
├── LICENSE                            # MIT
├── requirements.txt                   # Python dependencies
├── data/
│   ├── meta_analysis_publications_2015_2026.csv   # Historical data (2015–2026)
│   └── meta_analysis_forecast_2027_2031.png       # Forecast chart
├── analysis/
│   └── timesfm_forecast_v2.py         # TimesFM 1.0-200M zero-shot forecast script
├── skills/
│   ├── research-toolkit/
│   │   ├── SKILL.md                   # Routing guide for 61 method references
│   │   └── references/
│   │       ├── srm/   (22 files)
│   │       ├── maa/   (7 files)
│   │       ├── pda/   (13 files)
│   │       ├── mlk/   (13 files)
│   │       └── flow/  (6 files)
│   └── mermaid/
│       ├── srm-skill-graph.mmd        # Detailed Mermaid: SRM 22 skills
│       ├── maa-skill-graph.mmd        # Detailed Mermaid: MAA 7 skills
│       ├── pda-skill-graph.mmd        # Detailed Mermaid: PDA 13 skills
│       ├── mlk-skill-graph.mmd        # Detailed Mermaid: MLK 13 skills
│       ├── flow-skill-graph.mmd       # Detailed Mermaid: FLOW 6 skills
│       └── full-skill-graph.mmd       # Detailed Mermaid: All 61 skills overview
└── docs/
    └── architecture.mmd               # High-level automation architecture diagram
```

---

Read `skills/research-toolkit/SKILL.md` for the full routing guide.
    SRM --> MAA
    MAA --> PDA
    PDA --> MLK
    MLK --> FLOW

    class SRM,MAA,PDA,MLK,FLOW family
    linkStyle default stroke-width:2px,fill:none,stroke:#0288d1
```
    end

    class FLOW group
    linkStyle default stroke-width:2px,fill:none,stroke:#d81b60
```
    end

    class MLK group
    linkStyle default stroke-width:2px,fill:none,stroke:#fb8c00
```
    end

    class PDA group
    linkStyle default stroke-width:2px,fill:none,stroke:#43a047
```
    end

    class MAA group
    linkStyle default stroke-width:2px,fill:none,stroke:#8e24aa
```
    end

    class SRM group
    linkStyle default stroke-width:2px,fill:none,stroke:#0288d1
```

---

## 🎯 Skills Overview

### Skill Families (61 total references)

| Family | Prefix | Count | Focus Area |
|--------|--------|-------|------------|
| **SRM** — Statistical Research Methods | `srm-` | 22 | Study design, sampling, descriptive stats, CI, hypothesis tests (t, chi-square), correlation, regression, confounding & blinding, reporting |
| **MAA** — Meta-Analysis | `maa-` | 7 | Effect sizes, pooling models, heterogeneity, forest plots, publication bias, subgroup meta-regression, validation vs R/Stata |
| **PDA** — Python Data Analysis | `pda-` | 13 | Data loading/IO, cleaning, wrangling/reshape, groupby, time series, plotting, NumPy, pandas, Jupyter |
| **MLK** — Machine Learning | `mlk-` | 13 | Vectors/matrices/arrays, data loading/wrangling, datetime/categorical/text/numerical features, dim reduction, naive Bayes, SVM, trees/forests, neural networks, clustering |
| **FLOW** — Workflow | `flow-` | 6 | Python analysis env, unit hygiene, supervisor, analysis pipeline (Dunn), meta-analysis pipeline, corpus extraction |