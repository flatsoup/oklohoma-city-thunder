# NAME: Bogdan Khudoidodov | NUMBER: TODO_FILL_LAST4DIGITS
"""
project_code.py
OKC Analyst Intern Project -- NBA shot make-probability model.

Pipeline:
  1. Load training.csv.gz / testing.csv.gz
  2. Clean + parse raw columns (closestdefapproach array, booleans)
  3. Feature engineering:
       - shot geometry (distance/angle to rim center)
       - defender closeout dynamics (closing speed / acceleration from closestdefapproach)
       - contester aggregates
       - out-of-fold (K-Fold) target encoding for shooter/defender/team/zone, to avoid
         data leakage from using the raw outcome to encode a category that includes
         that same shot
       - basketball-motivated interaction terms
  4. Modeling: LightGBM (primary) bagged over a 5-fold Stratified K-Fold, blended with a
     spline Logistic Regression baseline when the blend improves OOF log-loss.
  5. Diagnostics: feature importance, SHAP-style contribution plots (via LightGBM's native
     pred_contrib, no extra heavy dependency), calibration plot, and a season-holdout
     generalization check (see note below on why this matters for this dataset).
  6. Writes final predictions into submission.csv (preserving the template's row order).

NOTE on season_id (important modeling decision -- see project_writeup for details):
  training.csv contains two seasons (fe055, 2676a) and testing.csv contains a THIRD,
  completely disjoint season (d7a2d). Because season_id in the test set never appears in
  training, it cannot be used as a predictive feature (it would be an always-unseen
  category at inference time), and a "group K-fold by season" is not meaningful with only
  two training groups. Instead we:
    (a) use plain Stratified K-Fold (on `outcome`) for model training / OOF target encoding
    (b) run a separate train-on-one-season / test-on-the-other diagnostic to estimate how
        much performance degrades under a season shift, which mimics what actually happens
        when scoring against testing.csv.
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RANDOM_STATE = 42
N_FOLDS = 5
RIM_X, RIM_Y = -41.75, 0.0

ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "training.csv.gz"
TEST_PATH = ROOT / "testing.csv.gz"
SUBMISSION_TEMPLATE_PATH = ROOT / "submission.csv"
FIGURES_DIR = ROOT / "project_writeup_assets"
FIGURES_DIR.mkdir(exist_ok=True)

TARGET_ENCODE_COLS = {
    # column -> smoothing strength (higher = trust the global mean more for low-count cats)
    "shooter_id": 25,
    "closestdef_id": 25,
    "off_team_id": 50,
    "def_team_id": 50,
    "zone": 10,
}

CATEGORICAL_FEATURES = ["shottype", "gamestate", "month", "zone"]


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    return train, test


# ---------------------------------------------------------------------------
# 2. Clean / parse
# ---------------------------------------------------------------------------
def parse_closestdefapproach(df):
    """closestdefapproach is a string '{d_1.0s,d_0.75s,d_0.5s,d_0.25s}'."""
    parsed = (
        df["closestdefapproach"]
        .str.strip("{}")
        .str.split(",", expand=True)
        .astype(float)
    )
    parsed.columns = ["app_100", "app_075", "app_050", "app_025"]
    return parsed


def basic_clean(df):
    df = df.copy()
    for c in ["contested", "three"]:
        if c in df.columns:
            df[c] = df[c].astype(int)
    if "outcome" in df.columns:
        df["outcome"] = df["outcome"].astype(int)
    approach = parse_closestdefapproach(df)
    df = pd.concat([df.reset_index(drop=True), approach.reset_index(drop=True)], axis=1)
    return df


# ---------------------------------------------------------------------------
# 3. Feature engineering
# ---------------------------------------------------------------------------
def add_geometry_features(df):
    df = df.copy()
    df["dx"] = df["locationx"] - RIM_X
    df["dy"] = df["locationy"] - RIM_Y
    df["abs_locationy"] = df["dy"].abs()
    # angle from straight-on (0 deg = top of key / straight out, 90 deg = along baseline)
    df["angle_deg"] = np.degrees(np.arctan2(df["abs_locationy"], df["dx"].clip(lower=0.1)))
    return df


def add_zone(df):
    df = df.copy()
    conditions = [
        df["distance"] <= 4,
        (df["three"] == 1) & (df["angle_deg"] >= 62),  # tight to the baseline -> corner 3
        (df["three"] == 1),
        df["distance"] <= 14,
    ]
    choices = ["restricted_area", "corner_three", "above_break_three", "paint"]
    df["zone"] = np.select(conditions, choices, default="midrange")
    return df


def add_defender_dynamics(df):
    df = df.copy()
    df["speed_100_075"] = (df["app_100"] - df["app_075"]) / 0.25
    df["speed_075_050"] = (df["app_075"] - df["app_050"]) / 0.25
    df["speed_050_025"] = (df["app_050"] - df["app_025"]) / 0.25
    df["speed_025_000"] = (df["app_025"] - df["closestdefdist"]) / 0.25
    df["avg_closing_speed"] = (df["app_100"] - df["closestdefdist"]) / 1.0
    df["closing_accel"] = df["speed_025_000"] - df["speed_100_075"]
    approach_cols = ["app_100", "app_075", "app_050", "app_025", "closestdefdist"]
    df["min_approach_dist"] = df[approach_cols].min(axis=1)
    df["recovery_gap"] = df["closestdefdist"] - df["min_approach_dist"]
    return df


def add_contester_features(df):
    df = df.copy()
    dist_cols = ["distcont1", "distcont2", "distcont3", "distcont4"]
    df["contest_min_dist"] = df[dist_cols].min(axis=1)
    df["contest_mean_dist"] = df[dist_cols].mean(axis=1)
    df["contest_max_dist"] = df[dist_cols].max(axis=1)
    return df


def add_interactions(df):
    df = df.copy()
    df["dist_x_defdist"] = df["distance"] * df["closestdefdist"]
    df["dribbles_x_shotclock"] = df["dribblesbefore"] * df["shotclock"]
    df["contestcount_x_distance"] = df["num_contesters"] * df["distance"]
    df["closingspeed_x_distance"] = df["avg_closing_speed"] * df["distance"]
    df["shooterspeed_x_dribbles"] = df["shooterspeed"] * df["dribblesbefore"]
    df["three_x_defdist"] = df["three"] * df["closestdefdist"]
    return df


def engineer_features(df):
    df = add_geometry_features(df)
    df = add_zone(df)
    df = add_defender_dynamics(df)
    df = add_contester_features(df)
    df = add_interactions(df)
    return df


# ---------------------------------------------------------------------------
# Out-of-fold target encoding
# ---------------------------------------------------------------------------
def oof_target_encode(train, test, col, folds, target="outcome", smoothing=20):
    """Leave-fold-out mean target encoding with additive smoothing.

    `folds` is the SAME list of (train_idx, val_idx) used later to train the models, so a
    row's encoded value is always derived only from rows outside its own validation fold --
    i.e. exactly the data the model sees when predicting that row. No leakage.
    """
    global_mean = train[target].mean()
    oof = np.full(len(train), np.nan)

    for tr_idx, val_idx in folds:
        stats = train.iloc[tr_idx].groupby(col)[target].agg(["mean", "count"])
        enc_map = (stats["mean"] * stats["count"] + global_mean * smoothing) / (
            stats["count"] + smoothing
        )
        oof[val_idx] = train.iloc[val_idx][col].map(enc_map).fillna(global_mean).values

    stats_full = train.groupby(col)[target].agg(["mean", "count"])
    enc_full = (stats_full["mean"] * stats_full["count"] + global_mean * smoothing) / (
        stats_full["count"] + smoothing
    )
    test_enc = test[col].map(enc_full).fillna(global_mean).values
    return oof, test_enc


def build_target_encodings(train, test, folds):
    train = train.copy()
    test = test.copy()
    for col, smoothing in TARGET_ENCODE_COLS.items():
        oof_col = f"{col}_te"
        oof, test_enc = oof_target_encode(train, test, col, folds, smoothing=smoothing)
        train[oof_col] = oof
        test[oof_col] = test_enc
    return train, test


# ---------------------------------------------------------------------------
# Feature list
# ---------------------------------------------------------------------------
def get_feature_columns():
    numeric = [
        "distance", "locationx", "locationy", "abs_locationy", "angle_deg",
        "dribblesbefore", "shotclock", "closestdefdist", "shooterspeed",
        "num_contesters", "distcont1", "distcont2", "distcont3", "distcont4",
        "contest_min_dist", "contest_mean_dist", "contest_max_dist",
        "app_100", "app_075", "app_050", "app_025",
        "speed_100_075", "speed_075_050", "speed_050_025", "speed_025_000",
        "avg_closing_speed", "closing_accel", "min_approach_dist", "recovery_gap",
        "dist_x_defdist", "dribbles_x_shotclock", "contestcount_x_distance",
        "closingspeed_x_distance", "shooterspeed_x_dribbles", "three_x_defdist",
        "contested", "three",
    ]
    target_encoded = [f"{c}_te" for c in TARGET_ENCODE_COLS]
    categorical = CATEGORICAL_FEATURES
    return numeric + target_encoded + categorical


def prep_model_frame(df, feature_cols):
    X = df[feature_cols].copy()
    for c in CATEGORICAL_FEATURES:
        X[c] = X[c].astype("category")
    return X


# ---------------------------------------------------------------------------
# 4. Modeling -- LightGBM bagged over folds
# ---------------------------------------------------------------------------
LGBM_PARAMS = dict(
    objective="binary",
    metric="binary_logloss",
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=150,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.5,
    n_estimators=3000,
    random_state=RANDOM_STATE,
    verbosity=-1,
)


def train_lightgbm_cv(train, feature_cols, folds):
    y = train["outcome"].values
    X = prep_model_frame(train, feature_cols)

    oof_pred = np.zeros(len(train))
    models = []
    fold_scores = []

    for i, (tr_idx, val_idx) in enumerate(folds):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        model.fit(
            X_tr, y_tr,
            eval_X=X_val, eval_y=y_val,
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        pred = model.predict_proba(X_val)[:, 1]
        oof_pred[val_idx] = pred
        score = log_loss(y_val, pred)
        fold_scores.append(score)
        models.append(model)
        print(f"  [LightGBM] fold {i+1}/{len(folds)}  logloss={score:.5f}  best_iter={model.best_iteration_}")

    cv_score = log_loss(y, oof_pred)
    print(f"  [LightGBM] OOF logloss={cv_score:.5f}  (fold mean {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f})")
    return models, oof_pred, cv_score


def predict_lightgbm_bagged(models, test, feature_cols):
    X_test = prep_model_frame(test, feature_cols)
    preds = np.zeros(len(test))
    for model in models:
        preds += model.predict_proba(X_test)[:, 1] / len(models)
    return preds


# ---------------------------------------------------------------------------
# 5a. Logistic regression baseline (interpretable comparison)
# ---------------------------------------------------------------------------
LR_SPLINE_FEATURES = ["distance", "closestdefdist", "shotclock", "angle_deg"]
LR_LINEAR_FEATURES = [
    "dribblesbefore", "shooterspeed", "num_contesters", "contest_min_dist",
    "avg_closing_speed", "closing_accel", "contested", "three",
    "shooter_id_te", "closestdef_id_te", "off_team_id_te", "def_team_id_te", "zone_te",
]
LR_CATEGORICAL_FEATURES = ["shottype", "gamestate", "month"]


def build_lr_pipeline():
    spline = Pipeline([
        ("impute_and_spline", SplineTransformer(n_knots=6, degree=3, include_bias=False)),
    ])
    pre = ColumnTransformer([
        ("spline", spline, LR_SPLINE_FEATURES),
        ("num", StandardScaler(), LR_LINEAR_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), LR_CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=2000, C=1.0)),
    ])


def train_lr_cv(train, test, folds):
    lr_cols = LR_SPLINE_FEATURES + LR_LINEAR_FEATURES + LR_CATEGORICAL_FEATURES
    train_lr = train[lr_cols].copy()
    test_lr = test[lr_cols].copy()
    # LR can't handle NaN -- impute contester/zone-related NaNs with a "no contester" sentinel
    for c in train_lr.columns:
        if train_lr[c].dtype.kind in "fc" and train_lr[c].isnull().any():
            fill = train_lr[c].median()
            train_lr[c] = train_lr[c].fillna(fill)
            test_lr[c] = test_lr[c].fillna(fill)

    y = train["outcome"].values
    oof_pred = np.zeros(len(train))
    models = []
    for i, (tr_idx, val_idx) in enumerate(folds):
        pipe = build_lr_pipeline()
        pipe.fit(train_lr.iloc[tr_idx], y[tr_idx])
        pred = pipe.predict_proba(train_lr.iloc[val_idx])[:, 1]
        oof_pred[val_idx] = pred
        models.append(pipe)
    cv_score = log_loss(y, oof_pred)
    print(f"  [LogReg+splines] OOF logloss={cv_score:.5f}")

    test_pred = np.zeros(len(test))
    for pipe in models:
        test_pred += pipe.predict_proba(test_lr)[:, 1] / len(models)
    return oof_pred, test_pred, cv_score


# ---------------------------------------------------------------------------
# 5b. Blend search
# ---------------------------------------------------------------------------
def find_best_blend(y, oof_a, oof_b):
    best_w, best_score = 1.0, log_loss(y, oof_a)
    for w in np.linspace(0, 1, 101):
        blend = w * oof_a + (1 - w) * oof_b
        score = log_loss(y, blend)
        if score < best_score:
            best_w, best_score = w, score
    return best_w, best_score


# ---------------------------------------------------------------------------
# Season-holdout diagnostic
# ---------------------------------------------------------------------------
def season_holdout_diagnostic(train, feature_cols):
    """Train on one season, validate on the other. Mimics the real train->test gap, since
    testing.csv is itself a third, disjoint season never seen in training."""
    seasons = train["season_id"].unique()
    assert len(seasons) == 2, "expected exactly two training seasons"
    results = {}
    for train_season, val_season in [(seasons[0], seasons[1]), (seasons[1], seasons[0])]:
        tr = train[train["season_id"] == train_season].reset_index(drop=True)
        va = train[train["season_id"] == val_season].reset_index(drop=True)

        # fresh target encoding fit only on `tr`, applied to `va` (no leakage across seasons)
        global_mean = tr["outcome"].mean()
        tr_enc, va_enc = tr.copy(), va.copy()
        for col, smoothing in TARGET_ENCODE_COLS.items():
            stats = tr.groupby(col)["outcome"].agg(["mean", "count"])
            enc_map = (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)
            tr_enc[f"{col}_te"] = tr[col].map(enc_map).fillna(global_mean)
            va_enc[f"{col}_te"] = va[col].map(enc_map).fillna(global_mean)

        X_tr = prep_model_frame(tr_enc, feature_cols)
        X_va = prep_model_frame(va_enc, feature_cols)
        params = dict(LGBM_PARAMS)
        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr, tr_enc["outcome"], eval_X=X_va, eval_y=va_enc["outcome"],
                   callbacks=[lgb.early_stopping(100, verbose=False)])
        pred = model.predict_proba(X_va)[:, 1]
        score = log_loss(va_enc["outcome"], pred)
        results[f"train={train_season}->val={val_season}"] = score
        print(f"  [season holdout] train on {train_season}, validate on {val_season}: logloss={score:.5f}")
    return results


# ---------------------------------------------------------------------------
# 6. Diagnostics / plots
# ---------------------------------------------------------------------------
def plot_feature_importance(models, feature_cols, path):
    importances = np.zeros(len(feature_cols))
    for model in models:
        importances += model.booster_.feature_importance(importance_type="gain")
    importances /= len(models)
    order = np.argsort(importances)[::-1][:20]
    names = [feature_cols[i] for i in order]
    vals = importances[order]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(range(len(names)), vals[::-1], color="#1f6fb2")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1], fontsize=9)
    ax.set_xlabel("LightGBM gain importance (avg across 5 folds)")
    ax.set_title("Top 20 feature importances")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_shap_summary(model, X_sample, feature_cols, path_bar, path_dependence):
    """Use LightGBM's native pred_contrib (equivalent to TreeSHAP) instead of the external
    shap package, to avoid pulling in a heavy numba/llvmlite dependency for one plot."""
    contrib = model.booster_.predict(X_sample, pred_contrib=True)
    contrib = contrib[:, :-1]  # drop the bias/expected-value column
    mean_abs = np.abs(contrib).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:20]
    names = [feature_cols[i] for i in order]
    vals = mean_abs[order]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(range(len(names)), vals[::-1], color="#d9534f")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1], fontsize=9)
    ax.set_xlabel("mean |SHAP value| (impact on predicted log-odds)")
    ax.set_title("Top 20 features by SHAP impact")
    fig.tight_layout()
    fig.savefig(path_bar, dpi=150)
    plt.close(fig)

    # dependence plot for the single most impactful feature
    top_idx = order[0]
    top_name = feature_cols[top_idx]
    fig, ax = plt.subplots(figsize=(7, 5))
    x_vals = X_sample[top_name].values if hasattr(X_sample[top_name], "values") else X_sample[top_name]
    sc = ax.scatter(x_vals, contrib[:, top_idx], s=4, alpha=0.3, c=X_sample["closestdefdist"],
                     cmap="coolwarm")
    ax.set_xlabel(top_name)
    ax.set_ylabel(f"SHAP value for {top_name}")
    ax.set_title(f"SHAP dependence: {top_name} (colored by closestdefdist)")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("closestdefdist")
    fig.tight_layout()
    fig.savefig(path_dependence, dpi=150)
    plt.close(fig)


def plot_calibration(y_true, y_pred, path, n_bins=15):
    bins = pd.qcut(y_pred, n_bins, duplicates="drop")
    df = pd.DataFrame({"y": y_true, "p": y_pred, "bin": bins})
    grouped = df.groupby("bin", observed=True).agg(mean_pred=("p", "mean"), mean_actual=("y", "mean"), n=("y", "size"))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    ax.plot(grouped["mean_pred"], grouped["mean_actual"], "o-", color="#1f6fb2", label="model")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed make rate")
    ax.set_title("Calibration curve (OOF predictions)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_zone_fgpct(train, path):
    grp = train.groupby("zone")["outcome"].agg(["mean", "count"]).sort_values("mean")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(grp.index, grp["mean"], color="#1f6fb2")
    for i, (m, n) in enumerate(zip(grp["mean"], grp["count"])):
        ax.text(m + 0.005, i, f"{m:.1%} (n={n:,})", va="center", fontsize=8)
    ax.set_xlabel("field goal %")
    ax.set_title("Observed FG% by court zone (training data)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. Submission
# ---------------------------------------------------------------------------
def write_submission(test, test_pred, path=SUBMISSION_TEMPLATE_PATH):
    template = pd.read_csv(path)
    pred_map = dict(zip(test["shot_id"], test_pred))
    missing = set(template["shot_id"]) - set(pred_map)
    assert not missing, f"{len(missing)} shot_ids in template have no prediction"
    template["make_prob"] = template["shot_id"].map(pred_map)
    assert template["make_prob"].notnull().all()
    assert list(template["shot_id"]) == list(pd.read_csv(path)["shot_id"]), "row order changed!"
    template.to_csv(path, index=False)
    print(f"  wrote {path} ({len(template)} rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=== 1. Loading data ===")
    train, test = load_data()
    train = basic_clean(train)
    test = basic_clean(test)

    print("=== 2. Feature engineering ===")
    train = engineer_features(train)
    test = engineer_features(test)
    feature_cols = get_feature_columns()
    print(f"  {len(feature_cols)} features")

    print("=== 3. Building CV folds + OOF target encoding ===")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    folds = list(skf.split(train, train["outcome"]))
    train, test = build_target_encodings(train, test, folds)

    print("=== 4. Naive baseline ===")
    naive_pred = np.full(len(train), train["outcome"].mean())
    naive_score = log_loss(train["outcome"], naive_pred)
    print(f"  naive (global mean) logloss={naive_score:.5f}")

    print("=== 5. LightGBM 5-fold CV ===")
    lgbm_models, lgbm_oof, lgbm_cv = train_lightgbm_cv(train, feature_cols, folds)

    print("=== 6. Logistic regression (spline) baseline ===")
    lr_oof, lr_test_pred, lr_cv = train_lr_cv(train, test, folds)

    print("=== 7. Blend search ===")
    y = train["outcome"].values
    best_w, blend_cv = find_best_blend(y, lgbm_oof, lr_oof)
    print(f"  best blend weight on LightGBM = {best_w:.2f}  blended OOF logloss={blend_cv:.5f}")
    use_blend = blend_cv < lgbm_cv - 1e-4
    print(f"  using {'BLEND' if use_blend else 'LightGBM only'} for final submission")

    print("=== 8. Season-holdout diagnostic ===")
    season_results = season_holdout_diagnostic(train, feature_cols)

    print("=== 9. Final test predictions ===")
    lgbm_test_pred = predict_lightgbm_bagged(lgbm_models, test, feature_cols)
    if use_blend:
        final_test_pred = best_w * lgbm_test_pred + (1 - best_w) * lr_test_pred
        final_oof = best_w * lgbm_oof + (1 - best_w) * lr_oof
    else:
        final_test_pred = lgbm_test_pred
        final_oof = lgbm_oof
    final_cv = log_loss(y, final_oof)
    print(f"  final OOF logloss={final_cv:.5f}")

    print("=== 10. Writing submission.csv ===")
    write_submission(test, final_test_pred)

    print("=== 11. Diagnostics / plots ===")
    plot_feature_importance(lgbm_models, feature_cols, FIGURES_DIR / "feature_importance.png")
    sample_idx = np.random.RandomState(RANDOM_STATE).choice(len(train), size=8000, replace=False)
    X_sample = prep_model_frame(train.iloc[sample_idx], feature_cols)
    plot_shap_summary(lgbm_models[0], X_sample, feature_cols,
                       FIGURES_DIR / "shap_bar.png", FIGURES_DIR / "shap_dependence.png")
    plot_calibration(y, lgbm_oof, FIGURES_DIR / "calibration.png")
    plot_zone_fgpct(train, FIGURES_DIR / "zone_fgpct.png")

    print("=== SUMMARY ===")
    summary = {
        "naive_baseline_logloss": naive_score,
        "lightgbm_oof_logloss": lgbm_cv,
        "logreg_spline_oof_logloss": lr_cv,
        "blend_weight_lightgbm": best_w,
        "blend_oof_logloss": blend_cv,
        "used_blend_for_submission": use_blend,
        "final_oof_logloss": final_cv,
        "season_holdout": season_results,
    }
    for k, v in summary.items():
        print(f"  {k}: {v}")

    import json
    with open(ROOT / "project_writeup_assets" / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


if __name__ == "__main__":
    main()
