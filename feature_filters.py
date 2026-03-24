
"""One-file feature filter comparer (7 filters) + subplot figure.

https://www.geeksforgeeks.org/machine-learning/feature-selection-techniques-in-machine-learning/

- Loads CSV(s), concatenates rows via Grade 3 import functinos, 
  leave scaling(normalization) to functions here to allow Chi2 MinMax scaling
- Runs 7 filter methods and prints top-k features per filter.
- Saves ONE comparison image with subplots (7 bar charts).

Filters:
1) Information Gain (Mutual Information)
2) Chi-square
3) Fisher's Score
4) Pearson |r|
5) Variance
6) Mean Absolute Difference (MAD)
7) Dispersion ratio (AM/GM)

Notes:
- Chi2 requires non-negative features: we min-max scale *only for chi2*.
- Pearson is computed vs label-encoded y .
"""

#import argparse
import numpy as np
import pandas as pd

from scipy.stats import pearsonr
from scipy.stats.mstats import gmean

from sklearn.feature_selection import mutual_info_classif, chi2
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import matplotlib
matplotlib.use("Agg")   # interaktiv: öppnar fönster , Agg inte interaktiv
import matplotlib.pyplot as plt


# ====== KÖR-SETTINGS   ======

LABEL_COL = "event"                       # din label-kolumn
TOPK = 15
OUT_FIG = "jämför_filter.png"
NORMALIZE_SCORES = True
# =====================================


#%%
# Tar fram unika events ur event kolumnen, normal och event och skapar en array med binaiserad events.  ["normal", "event", "normal"] → [0, 1, 0]
def _encode_y(y):
    y = np.asarray(y)
    uniq = pd.unique(y)
    m = {v:i for i,v in enumerate(uniq)} 
    return np.vectorize(m.get)(y) #https://numpy.org/doc/stable/reference/generated/numpy.vectorize.html

#%%
# Min max normaliserar mellan 0 och 1. 
def _normalize_01(scores):
    s = np.asarray(scores, dtype=float) # skapar en float array 
    s[~np.isfinite(s)] = 0.0            #  Ersätter NaN, +inf, -inf med 0.0, konstant feture för t ex pearson r
    lo, hi = float(np.min(s)), float(np.max(s))  # Hitta min, max väre
    if hi - lo < 1e-12:                 # ingen division med noll 
        return np.zeros_like(s)
    return (s - lo) / (hi - lo)         # returnera min-max normalisering

#%%
# returnerar k antal features med namn och score 
def _topk(names, scores, k):
    idx = np.argsort(scores)[::-1][:k]
    return [(names[i], float(scores[i])) for i in idx]



# =========================================================
#  7 filter functions Varje funktion returnerar värde med kolumn 
# =========================================================

#%%
# filter information gain  https://www.geeksforgeeks.org/machine-learning/information-gain-and-mutual-information-for-machine-learning/
# 
def f_information_gain(X, y):
    # Standardize helps MI behave more stably on continuous features
    Xs = StandardScaler().fit_transform(X) # 
    mi = mutual_info_classif(Xs, _encode_y(y), random_state=42, discrete_features='auto') # https://sklearner.com/scikit-learn-mutual_info_classif/
    return mi 
#%%
# https://www.geeksforgeeks.org/maths/chi-square-test/
# https://pythonguides.com/python-scipy-chi-square-test/
def f_chi2(X, y):
    # Chi2 requires non-negative
    Xn = MinMaxScaler().fit_transform(X)
    chi_vals, _ = chi2(Xn, _encode_y(y))
    return chi_vals

#%%
# https://www.geeksforgeeks.org/r-language/fishers-f-test-in-r-programming/
def f_fisher(X, y):
    y_arr = np.asarray(y)
    classes = pd.unique(y_arr)
    mu = X.mean(axis=0)
    num = np.zeros(X.shape[1])
    den = np.zeros(X.shape[1])
    for c in classes:
        Xc = X[y_arr == c]
        if len(Xc) == 0:
            continue
        mu_c = Xc.mean(axis=0)
        var_c = Xc.var(axis=0, ddof=1)
        n = Xc.shape[0]
        num += n * (mu_c - mu) ** 2
        den += n * (var_c + 1e-12)
    return num / (den + 1e-12)

#%%
# https://www.geeksforgeeks.org/maths/pearson-correlation-coefficient/
def f_pearson_abs_r(X, y):
    y_enc = _encode_y(y)
    out = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        x = X[:, j]
        if np.std(x) < 1e-12:
            out[j] = 0.0
        else:
            r, _ = pearsonr(x, y_enc)
            out[j] = abs(r)
    return out

#%%
# https://www.geeksforgeeks.org/machine-learning/variance-threshold/
def f_variance(X, y=None):
    return X.var(axis=0)

#%%
# https://www.geeksforgeeks.org/maths/mean-absolute-deviation/
def f_mad(X, y=None):
    mu = X.mean(axis=0)
    return np.mean(np.abs(X - mu), axis=0)

#%%
# https://www.geeksforgeeks.org/maths/measures-of-dispersion/
def f_dispersion_ratio(X, y=None, eps=1e-9):
    # Need positive for gmean; shift per feature if needed
    Xp = X.copy()
    mins = Xp.min(axis=0)
    shift = np.where(mins <= 0, -mins + eps, 0.0)
    Xp = Xp + shift
    am = Xp.mean(axis=0)
    gm = gmean(Xp, axis=0)
    return am / (gm + eps)


FILTERS = [
    ("InfoGain(MI)", f_information_gain),
    ("Chi2", f_chi2),
    ("Fisher", f_fisher),
    ("Pearson|r|", f_pearson_abs_r),
    ("Variance", f_variance),
    ("MAD", f_mad),
    ("Dispersion AM/GM", f_dispersion_ratio),
]

#%%
def eval_filters(df, Xdf,y):
    Xdf = Xdf.select_dtypes(include=[np.number])
    Xdf = Xdf.replace([np.inf, -np.inf], np.nan).fillna(Xdf.median(numeric_only=True))

    feat_names = list(Xdf.columns)
    X = Xdf.values.astype(float)

    print(f"Loaded rows={len(df)}  features={X.shape[1]}  label='{LABEL_COL}'")

    results = []
    for name, fn in FILTERS:
        scores = np.asarray(fn(X, y), dtype=float)
        if NORMALIZE_SCORES:
            scores = _normalize_01(scores)
        top = _topk(feat_names, scores, TOPK)

        top_str = ", ".join([f"{f}:{v:.3f}" for f, v in top[:min(8, len(top))]])
        print(f"{name:<14} -> {top_str}")

        results.append((name, top))

    # Subplots
    n = len(results); ncols = 2; nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 2.8*nrows), constrained_layout=True)
    axes = np.array(axes).reshape(-1)

    for i, (name, top) in enumerate(results):
        ax = axes[i]
        labels = [t[0] for t in top][::-1]
        vals = [t[1] for t in top][::-1]
        ax.barh(labels, vals)
        ax.set_title(name); ax.set_xlabel("score"); ax.grid(axis="x", alpha=0.2)

    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    plt.savefig(OUT_FIG, dpi=160)
    
    print(f"Saved subplot figure: {OUT_FIG}")
    plt.show()





