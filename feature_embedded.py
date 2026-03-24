"""Embedded feature selection (3 methods)

Methods:
1) L1 Regularization (Logistic Regression with L1)  -> non-zero coefficients kept
2) Random Forest                                 -> impurity-based feature importance
3) Gradient Boosting                             -> loss-reduction feature importance

Output:
- Console: CV score for each model + top features
- Figure: 3 subplots (L1 coefficients, RF importances, GB importances)

"""

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC
import matplotlib
matplotlib.use("Agg")   # interaktiv: öppnar fönster , Agg inte interaktiv
import matplotlib.pyplot as plt

# ========================= EDIT THESE =========================

SCORING = "accuracy"   # 'accuracy' or 'f1'
CV_SPLITS = 5
RANDOM_STATE = 42

TOPN = 12               # features to show in plots
OUT_FIG = "embedded_compare.png"
SHOW_PLOT = True
# ==============================================================



def sanitize_Xy(Xdf, y):
    # y: 1D int-array
    y = np.asarray(y).ravel().astype(int)

    # X: numeriskt, inga inf, inga NaN kvar
    Xdf = Xdf.copy()
    Xdf = Xdf.apply(pd.to_numeric, errors="coerce")
    Xdf = Xdf.replace([np.inf, -np.inf], np.nan)

    # median per kolumn, fallback 0 om helkolumn saknar data
    Xdf = Xdf.fillna(Xdf.median(numeric_only=True)).fillna(0)

    # sista säkerhetskollen
    if not np.isfinite(Xdf.values).all():
        bad = np.where(~np.isfinite(Xdf.values))
        raise ValueError(f"X contains NaN/inf after cleaning at indices (first) {bad[0][0]}, {bad[1][0]}")

    return Xdf, y


def _cv():
    return StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)


def _top_items(names, values, topn=TOPN):
    values = np.asarray(values, dtype=float)
    idx = np.argsort(values)[::-1][:topn]
    return [(names[i], float(values[i])) for i in idx]



def embedded_l1(Xdf, y):
    Xdf = Xdf.copy()
    Xdf = Xdf.apply(pd.to_numeric, errors="coerce")
    Xdf = Xdf.replace([np.inf, -np.inf], np.nan)
    Xdf = Xdf.fillna(Xdf.median(numeric_only=True)).fillna(0)

    y = np.asarray(y).ravel().astype(int)

    clf = LogisticRegression(
        penalty="l1", solver="saga", max_iter=5000,
        class_weight="balanced", random_state=RANDOM_STATE
    )
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    scores = cross_val_score(pipe, Xdf, y, scoring=SCORING, cv=_cv(), error_score="raise")
    #scores = cross_val_score(pipe, Xdf, y, scoring=SCORING, cv=_cv())

    pipe.fit(Xdf, y)

    coef = pipe.named_steps['clf'].coef_.ravel()
    abs_coef = np.abs(coef)
    selected = [name for name, c in zip(Xdf.columns, coef) if abs(c) > 1e-8]

    return {
        'name': 'L1 (LogReg/Lasso)',
        'cv_mean': float(scores.mean()),
        'cv_std': float(scores.std()),
        'importance': abs_coef,
        'signed_coef': coef,
        'selected': selected
    }


def embedded_rf(Xdf, y):

    rf = RandomForestClassifier(
        n_estimators=400,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample"
    )

    pipe_rf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", rf)
    ])

    scores = cross_val_score(pipe_rf, Xdf, y, scoring=SCORING, cv=_cv(), error_score="raise")
    pipe_rf.fit(Xdf, y)
    imp = pipe_rf.named_steps["rf"].feature_importances_

    return {
        'name': 'Random Forest',
        'cv_mean': float(scores.mean()),
        'cv_std': float(scores.std()),
        'importance': imp,
        'selected': [Xdf.columns[i] for i in np.argsort(imp)[::-1][:TOPN]]
    }


def embedded_gb(Xdf, y):

    gb = GradientBoostingClassifier(random_state=RANDOM_STATE)

    pipe_gb = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("gb", gb)
    ])

    scores = cross_val_score(pipe_gb, Xdf, y, scoring=SCORING, cv=_cv(), error_score="raise")
    pipe_gb.fit(Xdf, y)
    imp = pipe_gb.named_steps["gb"].feature_importances_

    return {
        'name': 'Gradient Boosting',
        'cv_mean': float(scores.mean()),
        'cv_std': float(scores.std()),
        'importance': imp,
        'selected': [Xdf.columns[i] for i in np.argsort(imp)[::-1][:TOPN]]
    }


def _plot_barh(ax, title, items):
    labels = [k for k, _ in items][::-1]
    vals = [v for _, v in items][::-1]
    ax.barh(labels, vals)
    ax.set_title(title)
    ax.grid(axis='x', alpha=0.25)


#%%
# hjälpfunktion för att kunna jämföra plots


def Embedded_f1(X2, y, best_params, seed=42):
    """
    Embedded: L1-LogReg som feature-selector (embedded), följt av SVM med best_params.
    Returnerar en Pipeline som kan användas direkt i plot_side_by_side_compare().
    """
    X2 = np.asarray(X2, dtype=float)
    y = np.asarray(y).ravel().astype(int)

    selector = SelectFromModel(
        LogisticRegression(
            penalty="l1",
            solver="liblinear",
            class_weight="balanced",
            random_state=seed,
            max_iter=3000
        ),
        max_features=min(2, X2.shape[1])
    )

    svm = SVC(kernel="rbf",
              C=best_params["C"],
              gamma=best_params["gamma"],
              class_weight="balanced",
              random_state=seed)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("sel", selector),
        ("svm", svm)
    ])
    pipe.fit(X2, y)

    # Säkerhet: om SelectFromModel väljer 0 features, fall back till båda
    if hasattr(pipe.named_steps["sel"], "get_support"):
        if not pipe.named_steps["sel"].get_support().any():
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("svm", svm)
            ])
            pipe.fit(X2, y)

    return pipe


def select_features_embedded_l1(Xdf, y, max_features=15, seed=42):
    """
    Embedded feature selection (L1 Logistic Regression) på ORIGINALFEATURES (Xdf).
    Returnerar listan med valda feature-namn.

    max_features: ungefärligt antal features att behålla (kan bli färre/ fler beroende på sparsitet).
    """
    Xdf = Xdf.copy()
    Xdf = Xdf.apply(pd.to_numeric, errors="coerce")
    Xdf = Xdf.replace([np.inf, -np.inf], np.nan)
    Xdf = Xdf.fillna(Xdf.median(numeric_only=True)).fillna(0)

    y = np.asarray(y).ravel().astype(int)

    base = LogisticRegression(
        penalty="l1",
        solver="saga",
        max_iter=5000,
        class_weight="balanced",
        random_state=seed
    )

    selector = SelectFromModel(base, max_features=min(int(max_features), Xdf.shape[1]))

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("sel", selector),
    ])

    pipe.fit(Xdf, y)
    mask = pipe.named_steps["sel"].get_support()
    selected = list(Xdf.columns[mask])

    # Fallback om den skulle välja 0
    if len(selected) == 0:
        selected = list(Xdf.columns[:min(int(max_features), Xdf.shape[1])])

    return selected
    
#%%
def eval_embedded_methods(Xdf, y):
    #Xdf, y = _load_data()

    Xdf, y = sanitize_Xy(Xdf, y)
    names = list(Xdf.columns)

    r1 = embedded_l1(Xdf, y)
    r2 = embedded_rf(Xdf, y)
    r3 = embedded_gb(Xdf, y)

    print("\nEmbedded methods (CV):")
    print(f"{r1['name']:<18}  {SCORING}={r1['cv_mean']:.3f}±{r1['cv_std']:.3f}  nonzero={len(r1['selected'])}  selected={r1['selected']}")
    print(f"{r2['name']:<18}  {SCORING}={r2['cv_mean']:.3f}±{r2['cv_std']:.3f}  top={r2['selected']}")
    print(f"{r3['name']:<18}  {SCORING}={r3['cv_mean']:.3f}±{r3['cv_std']:.3f}  top={r3['selected']}")

    # ---- Visualization ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)

    # L1: show signed coefficients (top by abs)
    abs_coef = r1['importance']
    top = _top_items(names, abs_coef, topn=TOPN)
    # keep sign for plotting
    signed = {n: r1['signed_coef'][names.index(n)] for n, _ in top}
    items_l1 = [(n, signed[n]) for n, _ in top]
    _plot_barh(axes[0], f"L1 Logistic (coef)\n{SCORING}={r1['cv_mean']:.3f}", items_l1)
    axes[0].axvline(0, color='k', linewidth=0.8)

    items_rf = _top_items(names, r2['importance'], topn=TOPN)
    _plot_barh(axes[1], f"Random Forest (importance)\n{SCORING}={r2['cv_mean']:.3f}", items_rf)

    items_gb = _top_items(names, r3['importance'], topn=TOPN)
    _plot_barh(axes[2], f"Gradient Boosting (importance)\n{SCORING}={r3['cv_mean']:.3f}", items_gb)

    plt.savefig(OUT_FIG, dpi=170)
    print(f"\nSaved figure: {OUT_FIG}")

    if SHOW_PLOT:
        try:
            plt.show()
        except Exception:
            print("(Could not open interactive window — figure still saved.)")



