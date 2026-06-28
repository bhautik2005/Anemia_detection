"""
=============================================================
 AnemiaFusionNet — Clinical Data Model (Module 2 of 3)
 Blood Cell Feature-based Anomaly Detection
=============================================================

YOUR DATASET COLUMNS:
─────────────────────
IDENTIFIERS:
  cell_id, dataset_source, microscope_model, staining_protocol

MORPHOLOGICAL FEATURES (numerical):
  cell_diameter_um, nucleus_area_pct, chromatin_density,
  cytoplasm_ratio, circularity, eccentricity,
  granularity_score, lobularity_score, membrane_smoothness,
  cell_area_px, perimeter_px

COLOR / STAIN FEATURES (numerical):
  mean_r, mean_g, mean_b, stain_intensity

BLOOD COUNT (clinical, numerical):
  wbc_count_per_ul, rbc_count_millions_per_ul,
  hemoglobin_g_dl, hematocrit_pct, platelet_count_per_ul,
  mcv_fl, mchc_g_dl

IMAGE METADATA (numerical):
  magnification_x, image_resolution_px,
  cytodiffusion_anomaly_score,
  cytodiffusion_classification_confidence,
  labeller_confidence_score

PATIENT INFO (categorical):
  patient_age_group, patient_sex

CELL TYPE INFO (categorical):
  cell_type, disease_category

TARGET:
  anomaly_label  →  0 = Normal, 1 = Anomalous
=============================================================
"""

# ── 0. Imports ────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import joblib
from io import StringIO

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score,
    precision_recall_curve, average_precision_score
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
import xgboost as xgb

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)


# ══════════════════════════════════════════════════════════
# SECTION 1 — Load & Understand Data
# ══════════════════════════════════════════════════════════

def load_data(csv_path: str) -> pd.DataFrame:
    """Load dataset and show a quick summary."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    print("=" * 55)
    print(" Clinical Data Model — Data Overview")
    print("=" * 55)
    print(f"Shape      : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Target dist:\n{df['anomaly_label'].value_counts().rename({0:'Normal', 1:'Anomaly'})}")
    print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

    return df


# ══════════════════════════════════════════════════════════
# SECTION 2 — Feature Engineering
# ══════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features from existing ones.
    These derived features often improve model accuracy
    because they capture biological relationships.
    """
    df = df.copy()

    # ── Ratio features ──
    # Nucleus-to-cell ratio: large nucleus = sign of abnormal cell
    df["nucleus_to_cell_ratio"] = df["nucleus_area_pct"] / (df["cell_area_px"] + 1e-6)

    # Color pallor index: anemic cells are paler (less red, more white)
    # Higher = more pale = more likely anemic
    df["pallor_index"] = (df["mean_r"] - df["mean_g"]) / (df["mean_r"] + df["mean_b"] + 1e-6)

    # Shape complexity: high perimeter relative to area = irregular shape
    df["shape_complexity"] = df["perimeter_px"] / (np.sqrt(df["cell_area_px"]) + 1e-6)

    # ── Blood count derived features ──
    # MCH (Mean Corpuscular Hemoglobin) approximation
    df["mch_approx"] = (df["hemoglobin_g_dl"] / df["rbc_count_millions_per_ul"].replace(0, np.nan)).fillna(0)

    # Anemia risk score (clinical rule-of-thumb)
    # Low Hgb + low MCV + low Hct = iron deficiency pattern
    df["anemia_risk_score"] = (
        (df["hemoglobin_g_dl"] < 12).astype(int) +
        (df["mcv_fl"] < 80).astype(int) +
        (df["hematocrit_pct"] < 36).astype(int)
    )  # Score: 0 (no risk) to 3 (high risk)

    # Stain color ratio (G/R ratio drops in anemic cells due to pallor)
    df["green_red_ratio"] = df["mean_g"] / (df["mean_r"] + 1e-6)

    # Morphology composite score
    df["morph_score"] = (
        df["circularity"] * 0.3 +
        df["membrane_smoothness"] * 0.3 +
        (1 - df["eccentricity"]) * 0.2 +
        (1 - df["granularity_score"] / 10) * 0.2
    )

    print(f"Feature engineering done → added 7 new features")
    return df


# ══════════════════════════════════════════════════════════
# SECTION 3 — Preprocessing Pipeline
# ══════════════════════════════════════════════════════════

# Column groups
NUMERICAL_FEATURES = [
    # Morphological
    "cell_diameter_um", "nucleus_area_pct", "chromatin_density",
    "cytoplasm_ratio", "circularity", "eccentricity",
    "granularity_score", "lobularity_score", "membrane_smoothness",
    "cell_area_px", "perimeter_px",
    # Color
    "mean_r", "mean_g", "mean_b", "stain_intensity",
    # Blood counts
    "wbc_count_per_ul", "rbc_count_millions_per_ul",
    "hemoglobin_g_dl", "hematocrit_pct", "platelet_count_per_ul",
    "mcv_fl", "mchc_g_dl",
    # Image / confidence
    "cytodiffusion_anomaly_score",
    "cytodiffusion_classification_confidence",
    "labeller_confidence_score",
    # Engineered
    "nucleus_to_cell_ratio", "pallor_index", "shape_complexity",
    "mch_approx", "anemia_risk_score", "green_red_ratio", "morph_score",
]

CATEGORICAL_FEATURES = [
    "patient_age_group",   # Adult / Elderly / Child
    "patient_sex",         # M / F
    "staining_protocol",   # Wright / Giemsa
    "microscope_model",    # Zeiss_Axio / Leica_DM2000 / Olympus_BX51
    "dataset_source",      # CytoData / PBC_Dataset
]

# Columns to DROP (identifiers, leakage risks)
DROP_COLUMNS = [
    "cell_id",
    "cell_type",           # too directly linked to label (data leakage risk)
    "disease_category",    # same — this IS the label in another form
    "magnification_x",     # always 100, no information
    "image_resolution_px", # mostly constant
]

TARGET = "anomaly_label"


def preprocess(df: pd.DataFrame):
    """
    Full preprocessing pipeline:
      1. Drop irrelevant columns
      2. Encode categoricals
      3. Impute missing values
      4. Scale numericals
    Returns X, y, scaler, encoders
    """
    df = df.copy()

    # Drop columns
    df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns], inplace=True)

    # Separate target
    y = df[TARGET].values
    df.drop(columns=[TARGET], inplace=True)

    # ── Encode categorical columns ──
    encoders = {}
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str).fillna("unknown"))
            encoders[col] = le

    # ── Keep only known features ──
    feature_cols = [c for c in NUMERICAL_FEATURES + CATEGORICAL_FEATURES if c in df.columns]
    X = df[feature_cols].copy()

    # ── Impute missing with median ──
    for col in X.columns:
        if X[col].isnull().any():
            X[col].fillna(X[col].median(), inplace=True)

    # ── Scale ──
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns
    )

    print(f"Preprocessing done → X shape: {X_scaled.shape}, features: {list(X_scaled.columns[:5])}...")
    return X_scaled, y, scaler, encoders, feature_cols


# ══════════════════════════════════════════════════════════
# SECTION 4 — Model Definitions
# ══════════════════════════════════════════════════════════

def get_models():
    """
    Returns 4 different ML models to compare.
    We then pick the best one (or ensemble them).

    Why these models?
    ─────────────────
    XGBoost        → best accuracy on tabular data, handles missing values
    Random Forest  → robust, interpretable, less overfitting
    Gradient Boost → like XGBoost but scikit-learn native
    Logistic Reg   → simple baseline, very interpretable
    """
    models = {
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=1,  # set to neg/pos ratio if imbalanced
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced",  # handles imbalance
            random_state=42,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=3,
            random_state=42,
        ),
        "LogisticRegression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        ),
    }
    return models


# ══════════════════════════════════════════════════════════
# SECTION 5 — Model Comparison with Cross-Validation
# ══════════════════════════════════════════════════════════

def compare_models(X, y, cv_folds=5):
    """
    Runs 5-fold cross-validation on all models.
    This gives a fair comparison without using the test set.
    """
    print("\n" + "=" * 55)
    print(" Model Comparison (5-Fold Cross-Validation)")
    print("=" * 55)

    models  = get_models()
    cv      = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    results = {}

    for name, model in models.items():
        f1_scores  = cross_val_score(model, X, y, cv=cv, scoring="f1",       n_jobs=-1)
        auc_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc",  n_jobs=-1)
        acc_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)

        results[name] = {
            "f1_mean":  f1_scores.mean(),
            "f1_std":   f1_scores.std(),
            "auc_mean": auc_scores.mean(),
            "auc_std":  auc_scores.std(),
            "acc_mean": acc_scores.mean(),
        }

        print(f"{name:22s} | F1: {f1_scores.mean():.4f} ±{f1_scores.std():.3f} "
              f"| AUC: {auc_scores.mean():.4f} | Acc: {acc_scores.mean():.4f}")

    # Pick best model by F1
    best_name = max(results, key=lambda k: results[k]["f1_mean"])
    print(f"\n✓ Best model: {best_name} (F1={results[best_name]['f1_mean']:.4f})")
    return results, best_name


# ══════════════════════════════════════════════════════════
# SECTION 6 — Train Final Model
# ══════════════════════════════════════════════════════════

def train_final_model(X_train, X_test, y_train, y_test, best_name: str):
    """
    Train the best model on train set and evaluate on test set.
    Also builds an ensemble (voting) for robustness.
    """
    print("\n" + "=" * 55)
    print(f" Training Final Model: {best_name}")
    print("=" * 55)

    models = get_models()
    model  = models[best_name]

    # XGBoost: use early stopping with eval set
    if best_name == "XGBoost":
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )
    else:
        model.fit(X_train, y_train)

    # Predictions
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    acc = (y_pred == y_test).mean()
    f1  = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    ap  = average_precision_score(y_test, y_proba)

    print(f"Test Accuracy  : {acc:.4f}")
    print(f"Test F1 Score  : {f1:.4f}")
    print(f"Test AUC-ROC   : {auc:.4f}")
    print(f"Avg Precision  : {ap:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Anomaly"]))

    return model, y_pred, y_proba


# ══════════════════════════════════════════════════════════
# SECTION 7 — Feature Importance
# ══════════════════════════════════════════════════════════

def plot_feature_importance(model, feature_names: list, model_name: str, top_n=20):
    """Shows which features matter most for the prediction."""

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        # For logistic regression use absolute coefficients
        importances = np.abs(model.coef_[0])

    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_scores   = [importances[i] for i in indices]

    plt.figure(figsize=(10, 6))
    colors = ["#2a78d6" if s > np.mean(top_scores) else "#B5D4F4" for s in top_scores]
    bars = plt.barh(range(len(top_features)), top_scores[::-1], color=colors[::-1])
    plt.yticks(range(len(top_features)), top_features[::-1], fontsize=9)
    plt.xlabel("Feature Importance Score")
    plt.title(f"Top {top_n} Important Features — {model_name}")
    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Feature importance plot saved → outputs/feature_importance.png")

    print(f"\nTop 10 most important features:")
    for i, (feat, score) in enumerate(zip(top_features[:10], top_scores[:10]), 1):
        print(f"  {i:2d}. {feat:<40s} {score:.4f}")


# ══════════════════════════════════════════════════════════
# SECTION 8 — Plots
# ══════════════════════════════════════════════════════════

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Anomaly"],
                yticklabels=["Normal", "Anomaly"])
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("outputs/clinical_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Confusion matrix saved → outputs/clinical_confusion_matrix.png")


def plot_roc_pr_curves(y_true, y_proba):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    axes[0].plot(fpr, tpr, color="#2a78d6", label=f"AUC = {auc:.3f}")
    axes[0].plot([0,1],[0,1],"k--", alpha=0.4)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend()

    # Precision-Recall
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    axes[1].plot(rec, prec, color="#1baf7a", label=f"AP = {ap:.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("outputs/clinical_roc_pr.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("ROC & PR curves saved → outputs/clinical_roc_pr.png")


def plot_correlation_heatmap(X: pd.DataFrame):
    """Shows how features relate to each other — helps spot redundant features."""
    plt.figure(figsize=(14, 10))
    corr = X[NUMERICAL_FEATURES[:15]].corr()  # top 15 for readability
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0,
                annot=False, linewidths=0.3, vmin=-1, vmax=1)
    plt.title("Feature Correlation Matrix (top 15 numerical features)")
    plt.tight_layout()
    plt.savefig("outputs/correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Correlation heatmap saved → outputs/correlation_heatmap.png")


# ══════════════════════════════════════════════════════════
# SECTION 9 — Feature Vector Extractor (for Fusion)
# ══════════════════════════════════════════════════════════

def extract_clinical_features(df_new: pd.DataFrame, scaler, encoders, feature_cols) -> np.ndarray:
    """
    Extracts 64-dim feature vector for fusion with eye image model.
    Used in AnemiaFusionNet Transformer Fusion module.

    Returns:
        np.ndarray of shape (N, 64)
    """
    df_new = engineer_features(df_new)

    for col, le in encoders.items():
        if col in df_new.columns:
            df_new[col] = df_new[col].astype(str).map(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )

    X = df_new[[c for c in feature_cols if c in df_new.columns]].fillna(0)
    X_scaled = scaler.transform(X)

    # Project to 64-dim via PCA-like truncation
    # (In full pipeline, replace with nn.Linear layer output)
    return X_scaled[:, :64] if X_scaled.shape[1] >= 64 else X_scaled


# ══════════════════════════════════════════════════════════
# SECTION 10 — Single Sample Prediction
# ══════════════════════════════════════════════════════════

def predict_single_cell(cell_data: dict, model, scaler, encoders, feature_cols) -> dict:
    """
    Predict anomaly for a single blood cell record.

    Usage:
        cell = {
            "cell_diameter_um": 15.18,
            "hemoglobin_g_dl": 11.7,
            "patient_sex": "F",
            "patient_age_group": "Elderly",
            ... (other features)
        }
        result = predict_single_cell(cell, model, scaler, encoders, feature_cols)
        print(result["prediction"])  # → "Anomaly" or "Normal"
    """
    df_single = pd.DataFrame([cell_data])
    df_single = engineer_features(df_single)

    for col, le in encoders.items():
        if col in df_single.columns:
            val = str(df_single[col].iloc[0])
            df_single[col] = le.transform([val])[0] if val in le.classes_ else 0

    X = df_single[[c for c in feature_cols if c in df_single.columns]].fillna(0)
    X_scaled = scaler.transform(X)

    pred  = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]

    return {
        "prediction":      "Anomaly" if pred == 1 else "Normal",
        "confidence":      round(float(proba[pred]), 4),
        "prob_anomaly":    round(float(proba[1]), 4),
        "prob_normal":     round(float(proba[0]), 4),
        "feature_vector":  X_scaled,  # for fusion
    }


# ══════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════

def main():
    # ── 1. Load ──
    CSV_PATH ="blood_cell_anomaly_detection.csv"   # ← change to your CSV filename
    df = load_data(CSV_PATH)

    # ── 2. Feature Engineering ──
    df = engineer_features(df)

    # ── 3. Preprocess ──
    X, y, scaler, encoders, feature_cols = preprocess(df)

    # ── 4. Correlation heatmap ──
    plot_correlation_heatmap(X)

    # ── 5. Train/Val/Test split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"\nTrain: {len(X_train)}  |  Test: {len(X_test)}")

    # ── 6. Compare models ──
    results, best_name = compare_models(X_train, y_train, cv_folds=5)

    # ── 7. Train best model on full train set ──
    model, y_pred, y_proba = train_final_model(
        X_train, X_test, y_train, y_test, best_name
    )

    # ── 8. Plots ──
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_pr_curves(y_test, y_proba)
    plot_feature_importance(model, feature_cols, best_name, top_n=20)

    # ── 9. Save model ──
    joblib.dump({
        "model":       model,
        "scaler":      scaler,
        "encoders":    encoders,
        "feature_cols": feature_cols,
        "best_name":   best_name,
    }, "outputs/clinical_model.pkl")
    print("\nModel saved → outputs/clinical_model.pkl")

    # ── 10. Example single prediction ──
    example_cell = {
        "cell_diameter_um":   15.18,
        "nucleus_area_pct":   58.8,
        "chromatin_density":  0.542,
        "cytoplasm_ratio":    0.301,
        "circularity":        0.563,
        "eccentricity":       0.529,
        "granularity_score":  4.11,
        "lobularity_score":   6.6,
        "membrane_smoothness": 0.8,
        "cell_area_px":       445,
        "perimeter_px":       90,
        "mean_r":             215,
        "mean_g":             141,
        "mean_b":             160,
        "stain_intensity":    0.555,
        "patient_age_group":  "Elderly",
        "patient_sex":        "F",
        "wbc_count_per_ul":   6352,
        "rbc_count_millions_per_ul": 4.44,
        "hemoglobin_g_dl":    11.7,
        "hematocrit_pct":     43.4,
        "platelet_count_per_ul": 257383,
        "mcv_fl":             85.5,
        "mchc_g_dl":          31.4,
        "cytodiffusion_anomaly_score": 0.7649,
        "cytodiffusion_classification_confidence": 0.5726,
        "labeller_confidence_score": 0.567,
        "staining_protocol":  "Giemsa",
        "microscope_model":   "Zeiss_Axio",
        "dataset_source":     "CytoData",
    }

    result = predict_single_cell(example_cell, model, scaler, encoders, feature_cols)
    print(f"\nExample prediction:")
    print(f"  Prediction  : {result['prediction']}")
    print(f"  Confidence  : {result['confidence']:.4f}")
    print(f"  Prob Anomaly: {result['prob_anomaly']:.4f}")
    print(f"  Prob Normal : {result['prob_normal']:.4f}")
    print("\nDone! All outputs saved to outputs/")


if __name__ == "__main__":
    main()