"""Wrapper features

Implements:
1) Forward Selection (SequentialFeatureSelector, direction='forward')
2) Backward Elimination (SequentialFeatureSelector, direction='backward')
3) Recursive Feature Elimination (RFE)

Results:
- Console output: best k + selected features per method
- One figure with 3 subplots: CV score vs number of selected features

How it works (short):
For each k (feature) in K_LIST, we cross-validate a Pipeline:
  StandardScaler -> Selector (SFS/RFE) -> LogisticRegression
and record mean CV score.

Pipeline (https://www.geeksforgeeks.org/data-science/whats-data-science-pipeline/)

"""

import warnings
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector, RFE
from sklearn.svm import SVC

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt






# ========================= EDIT THESE =========================
LABEL_COL = "event"                            # target column

SCORING = "f1_macro"  # multiclass-safe (was f1 for binary)
CV_SPLITS = 5
RANDOM_STATE = 42

MAX_K = 15               # max number av features att testa (wrappers kan vara slöa)
MAX_ITER = 2000
C_VALUE = 10  #  regularization strength C equal to 10.0, https://scikit-learn.org/stable/auto_examples/linear_model/plot_logistic_path.html#sphx-glr-auto-examples-linear-model-plot-logistic-path-py
K_STEP = 1
OUT_FIG = "wrapper_compare.png"
SHOW_PLOT = True
# ==============================================================

#%%
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
# https://realpython.com/logistic-regression-python/
def _make_estimator():
    return LogisticRegression(
        C = C_VALUE,
        max_iter=MAX_ITER,
        solver="lbfgs", # ‘lbfgs’, ‘liblinear’, ‘newton-cg’, ‘newton-cholesky’, ‘sag’, ‘saga’}, default=’lbfgs’
        class_weight="balanced"   # viktigt för obalanserad data
    )


#%%
# Cross-validation via StratifiedKFold
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
def _cv():
    return StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)


#%% 
# https://michaelallen1966.github.io/titanic/10_feature_selection_2_forward.html
def forward_selection_curve(Xdf, y, k_list):
    means, stds = [], []
    est = _make_estimator()
    for k in k_list:
        sfs = SequentialFeatureSelector(est, n_features_to_select=k, direction='forward', scoring=SCORING, cv=_cv(), n_jobs=None)
        pipe = Pipeline([('scaler', StandardScaler()), ('sfs', sfs), ('clf', est)])
        scores = cross_val_score(pipe, Xdf, y, scoring=SCORING, cv=_cv())
        means.append(scores.mean()); stds.append(scores.std())
    return np.array(means), np.array(stds)

#%%
# https://rstudio-pubs-static.s3.amazonaws.com/772806_6a9f85bfde0b410c8cacf5945b6fd03b.html
# https://mattkcole.com/2017/01/22/the-problem-with-backward-selection/
def backward_elimination_curve(Xdf, y, k_list):
    means, stds = [], []
    est = _make_estimator()
    for k in k_list:
        sfs = SequentialFeatureSelector(est, n_features_to_select=k, direction='backward', scoring=SCORING, cv=_cv(), n_jobs=None)
        pipe = Pipeline([('scaler', StandardScaler()), ('sfs', sfs), ('clf', est)])
        scores = cross_val_score(pipe, Xdf, y, scoring=SCORING, cv=_cv())
        means.append(scores.mean()); stds.append(scores.std())
    return np.array(means), np.array(stds)

#%%
# https://machinelearningmastery.com/rfe-feature-selection-in-python/
# https://stackoverflow.com/questions/65296195/sklearn-rfe-pipeline-and-cross-validation
def rfe_curve(Xdf, y, k_list):
    means, stds = [], []
    est = _make_estimator()
    for k in k_list:
        rfe = RFE(estimator=est, n_features_to_select=k, step=1)
        pipe = Pipeline([('scaler', StandardScaler()), ('rfe', rfe), ('clf', est)])
        scores = cross_val_score(pipe, Xdf, y, scoring=SCORING, cv=_cv())
        means.append(scores.mean()); stds.append(scores.std())
    return np.array(means), np.array(stds)

#%%
# hjälpfunktion för att inte behöva skriva samma för "forward" och "backward"
def _fit_and_get_features_sfs(Xdf, y, k, direction):
    est = _make_estimator()
    sfs = SequentialFeatureSelector(est, n_features_to_select=k, direction=direction, scoring=SCORING, cv=_cv())
    pipe = Pipeline([('scaler', StandardScaler()), ('sfs', sfs), ('clf', est)])
    pipe.fit(Xdf, y)
    mask = pipe.named_steps['sfs'].get_support()
    return list(Xdf.columns[mask])

#%%
# Pipeline Recursive Feature elemnination för att vara konsekvent. 
def _fit_and_get_features_rfe(Xdf, y, k):
    est = _make_estimator()
    rfe = RFE(estimator=est, n_features_to_select=k, step=1)
    pipe = Pipeline([('scaler', StandardScaler()), ('rfe', rfe), ('clf', est)])
    pipe.fit(Xdf, y)
    mask = pipe.named_steps['rfe'].get_support()
    return list(Xdf.columns[mask])

#%%
# hjälp funktion för att kunna jämföra plots from sklearn.svm import SVC


def Wrapper_Sfs(X2, y, best_params, scoring="f1", seed=42, k=1):
    """
    Wrapper: SFS (forward) på PCA-2D (X2). Väljer k features (default=1)
    och tränar en SVM(C,gamma) på valda features.
    Returnerar en Pipeline som kan predict() på 2D-grid.
    """
    X2 = np.asarray(X2, dtype=float)
    y = np.asarray(y).ravel().astype(int)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    svm = SVC(
        kernel="rbf",
        C=best_params["C"],
        gamma=best_params["gamma"],
        class_weight="balanced",
        random_state=seed
    )

    sfs = SequentialFeatureSelector(
        estimator=svm,
        n_features_to_select=min(k, X2.shape[1]),
        direction="forward",
        scoring=scoring,
        cv=cv
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("sfs", sfs),
        ("svm", svm),
    ])

    pipe.fit(X2, y)
    return pipe





def select_features_wrapper_sfs(Xdf, y, best_params, k=15, scoring="f1", seed=42):
    """
    Wrapper feature selection på ORIGINALFEATURES (Xdf).
    Returnerar listan med valda feature-namn.

    - Xdf: pandas DataFrame med originalfeatures
    - y: 1D array
    - best_params: dict med {"C": ..., "gamma": ...} från din SVM CV-tuning
    """
    Xdf = Xdf.copy()
    Xdf = Xdf.select_dtypes(include=[np.number])
    Xdf = Xdf.replace([np.inf, -np.inf], np.nan).fillna(Xdf.median(numeric_only=True)).fillna(0)

    y = np.asarray(y).ravel().astype(int)

    # Safe k
    k = int(max(1, min(k, Xdf.shape[1])))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    svm = SVC(
        kernel="rbf",
        C=best_params["C"],
        gamma=best_params["gamma"],
        class_weight="balanced",
        random_state=seed
    )

    sfs = SequentialFeatureSelector(
        estimator=svm,
        n_features_to_select=k,
        direction="forward",
        scoring=scoring,
        cv=cv
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("sfs", sfs),
    ])

    pipe.fit(Xdf, y)
    mask = pipe.named_steps["sfs"].get_support()
    selected = list(Xdf.columns[mask])
    return selected

#%%
# =========================================================
#  Utvärdera Wrappers
# =========================================================
def eval_wrappers(Xdf, y):
    #Xdf, y = _load_data()
        # numeric only + simple impute

    Xdf = Xdf.select_dtypes(include=[np.number])
    Xdf = Xdf.replace([np.inf, -np.inf], np.nan).fillna(Xdf.median(numeric_only=True))

    n_features = Xdf.shape[1]
    max_k = int(min(MAX_K, n_features))
    k_list = list(range(1, max_k + 1, int(K_STEP)))

    print(f"Loaded rows={len(Xdf)}  features={n_features}  label='{LABEL_COL}'")
    print(f"Wrapper search k=1..{max_k}  scoring='{SCORING}'  cv={CV_SPLITS}-fold")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f_mu, f_sd = forward_selection_curve(Xdf, y, k_list)
        b_mu, b_sd = backward_elimination_curve(Xdf, y, k_list)
        r_mu, r_sd = rfe_curve(Xdf, y, k_list)

    # pick best k (max mean score)
    best_f = k_list[int(np.argmax(f_mu))]
    best_b = k_list[int(np.argmax(b_mu))]
    best_r = k_list[int(np.argmax(r_mu))]

    feats_f = _fit_and_get_features_sfs(Xdf, y, best_f, 'forward')
    feats_b = _fit_and_get_features_sfs(Xdf, y, best_b, 'backward')
    feats_r = _fit_and_get_features_rfe(Xdf, y, best_r)

    print("\nBest selections:")
    print(f"Forward Selection  k={best_f:>2}  score={f_mu[k_list.index(best_f)]:.3f}  features={feats_f}")
    print(f"Backward Elim.     k={best_b:>2}  score={b_mu[k_list.index(best_b)]:.3f}  features={feats_b}")
    print(f"RFE               k={best_r:>2}  score={r_mu[k_list.index(best_r)]:.3f}  features={feats_r}")

    # ---- Plot (3 subplots) ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    def _plot(ax, title, mu, sd, best_k):
        ax.plot(k_list, mu, marker='o', linewidth=1.5)
        ax.fill_between(k_list, mu - sd, mu + sd, alpha=0.15)
        ax.axvline(best_k, linestyle='--', alpha=0.6)
        ax.set_title(title)
        ax.set_xlabel('Antal features (k)')
        ax.set_ylabel(f"CV {SCORING}")
        ax.grid(alpha=0.25)

    _plot(axes[0], 'Forward Selection (SFS)', f_mu, f_sd, best_f)
    _plot(axes[1], 'Backward Elimination (SFS)', b_mu, b_sd, best_b)
    _plot(axes[2], 'RFE', r_mu, r_sd, best_r)

    plt.savefig(OUT_FIG, dpi=160)
    print(f"\nSaved figure: {OUT_FIG}")

    if SHOW_PLOT:
        try:
            plt.show()
        except Exception:
            print("(Could not open interactive window — figure still saved.)")



