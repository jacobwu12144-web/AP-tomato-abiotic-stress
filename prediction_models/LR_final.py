import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    roc_curve,
)

# USER PARAMETERS

REAL_INPUT_CSV = 

OUTPUT_CSV = "logistic_regression_predictions_fixed.csv"
MODEL_PATH = "logistic_model_fixed.pkl"
SCALER_PATH = "scaler_fixed.pkl"
POLY_PATH = "poly_transform_fixed.pkl"
THRESHOLD_PATH = "threshold_fixed.pkl"

TEST_SIZE = 0.25
RANDOM_STATE = 42
N_SPLITS = 5
LAMBDA_GRID = np.logspace(-3, 2, 20)

FEATURE_COLS = ["Amplitude", "tau_rise", "tau_fall", "Undershoot", "tau_rec"]
LABEL_COL = "Label"


df = pd.read_csv(REAL_INPUT_CSV)
df.columns = df.columns.str.strip()

# Fix naming mismatch if needed
if "tau_recovery" in df.columns and "tau_rec" not in df.columns:
    df = df.rename(columns={"tau_recovery": "tau_rec"})

# Create binary labels from stress column (6th column)
# 'a_control' -> 0, everything else -> 1
label_source_col = df.columns[5]
df[label_source_col] = df[label_source_col].fillna("").astype(str)
df[LABEL_COL] = (~df[label_source_col].str.lower().str.startswith("a")).astype(int)

# Convert features to numeric
df[FEATURE_COLS] = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
df = df.dropna(subset=FEATURE_COLS + [LABEL_COL])

# Remove non-positive time constants (required for log transform)
valid_mask = (df["tau_rise"] > 0) & (df["tau_fall"] > 0) & (df["tau_rec"] > 0)
df = df[valid_mask].copy()

print(f"Total real recordings after cleaning: {len(df)}")
print(f"Class distribution:\n{df[LABEL_COL].value_counts().to_string()}")

# ===========================================================
# FEATURE ENGINEERING

# Log transforms
df["log_tau_rise"] = np.log(df["tau_rise"])
df["log_tau_fall"] = np.log(df["tau_fall"])
df["log_tau_rec"] = np.log(df["tau_rec"])

# Biophysical interaction terms
df["rise_fall_ratio"] = df["tau_rise"] / df["tau_fall"]
df["A_tau_rec"] = df["Amplitude"] * df["tau_rec"]
df["U_tau_fall"] = df["Undershoot"] * df["tau_fall"]

ENGINEERED_COLS = [
    "Amplitude", "tau_rise", "tau_fall", "Undershoot", "tau_rec",
    "log_tau_rise", "log_tau_fall", "log_tau_rec",
    "rise_fall_ratio", "A_tau_rec", "U_tau_fall",
]

df = df.dropna(subset=ENGINEERED_COLS)

X = df[ENGINEERED_COLS].values
y = df[LABEL_COL].values.astype(int)

print(f"\nFinal dataset shape: {X.shape}")
print(f"Any NaNs remaining: {np.isnan(X).any()}")

# TRAIN / TEST SPLIT  — SPLIT ONCE

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

print(f"\nTrain set: {X_train.shape[0]} sessions")
print(f"Test set:  {X_test.shape[0]} sessions")
print(f"Train class distribution: 0={np.sum(y_train==0)}, 1={np.sum(y_train==1)}")
print(f"Test class distribution:  0={np.sum(y_test==0)}, 1={np.sum(y_test==1)}")

# Polynomial expansion
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

print(f"Feature space after polynomial expansion: {X_train_poly.shape[1]} dimensions")

# CROSS-VALIDATED LAMBDA SELECTION (on training set only)

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

best_lambda = None
best_mean_auc = -np.inf

print("\nCross-validating regularization strength...\n")

for lam in LAMBDA_GRID:
    aucs = []

    for train_idx, val_idx in skf.split(X_train_poly, y_train):
        X_tr, X_val = X_train_poly[train_idx], X_train_poly[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        sc = StandardScaler()
        X_tr_scaled = sc.fit_transform(X_tr)
        X_val_scaled = sc.transform(X_val)

        mdl = LogisticRegression(
            penalty="l2", C=1.0 / lam, solver="lbfgs", max_iter=3000
        )
        mdl.fit(X_tr_scaled, y_tr)

        y_val_prob = mdl.predict_proba(X_val_scaled)[:, 1]
        aucs.append(roc_auc_score(y_val, y_val_prob))

    mean_auc = np.mean(aucs)
    print(f"  Lambda={lam:.5f}  CV AUC={mean_auc:.4f} ± {np.std(aucs):.4f}")

    if mean_auc > best_mean_auc:
        best_mean_auc = mean_auc
        best_lambda = lam

print(f"\nSelected lambda: {best_lambda:.5f}  (CV AUC={best_mean_auc:.4f})")

# FINAL MODEL TRAINING (on full training set)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_poly)
X_test_scaled = scaler.transform(X_test_poly)

final_model = LogisticRegression(
    penalty="l2", C=1.0 / best_lambda, solver="lbfgs", max_iter=3000
)
final_model.fit(X_train_scaled, y_train)

# SAVE MODEL + TRANSFORMS

joblib.dump(final_model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
joblib.dump(poly, POLY_PATH)

# TEST SET EVALUATION — single locked evaluation, no iteration

# Using default 0.50 threshold as stated in the paper
THRESHOLD = 0.50

y_test_prob = final_model.predict_proba(X_test_scaled)[:, 1]
y_test_pred = (y_test_prob >= THRESHOLD).astype(int)

acc = accuracy_score(y_test, y_test_pred)
prec = precision_score(y_test, y_test_pred)
rec = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)
auc = roc_auc_score(y_test, y_test_prob)

# Confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
fpr = fp / (fp + tn)

print("\n" + "=" * 60)
print("TEST SET RESULTS (real data only, single locked evaluation)")
print("=" * 60)
print(f"Test set size: {len(y_test)} sessions")
print(f"Threshold: {THRESHOLD}")
print(f"")
print(f"Accuracy:    {acc:.4f}")
print(f"Precision:   {prec:.4f}")
print(f"Recall:      {rec:.4f}")
print(f"F1-score:    {f1:.4f}")
print(f"ROC-AUC:     {auc:.4f}")
print(f"")
print(f"Sensitivity: {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"FPR:         {fpr:.4f}")
print(f"")
print(f"Confusion matrix:")
print(f"  TN={tn}  FP={fp}")
print(f"  FN={fn}  TP={tp}")
print(f"")
print(classification_report(y_test, y_test_pred, target_names=["Control", "Stressed"]))

# AUC 95% CI (bootstrap approximation)

rng = np.random.RandomState(RANDOM_STATE)
n_bootstrap = 10000
boot_aucs = []

for _ in range(n_bootstrap):
    idx = rng.choice(len(y_test), size=len(y_test), replace=True)
    if len(np.unique(y_test[idx])) < 2:
        continue
    boot_aucs.append(roc_auc_score(y_test[idx], y_test_prob[idx]))

boot_aucs = np.array(boot_aucs)
ci_lower = np.percentile(boot_aucs, 2.5)
ci_upper = np.percentile(boot_aucs, 97.5)

print(f"ROC-AUC 95% CI (bootstrap, {n_bootstrap} resamples): {ci_lower:.4f} – {ci_upper:.4f}")

# ROC CURVE PLOT

fpr_curve, tpr_curve, _ = roc_curve(y_test, y_test_prob)

plt.figure(figsize=(6, 6))
plt.plot(fpr_curve, tpr_curve, color="darkorange", lw=2, label=f"LR (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Logistic Regression")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve_fixed.png", dpi=300)
plt.show()

# FEATURE CONTRIBUTIONS

coef_poly = final_model.coef_.flatten()
poly_feature_names = poly.get_feature_names_out(ENGINEERED_COLS)

feature_contrib_df = pd.DataFrame({
    "feature": poly_feature_names,
    "coefficient": coef_poly,
})
feature_contrib_df["abs_coeff"] = np.abs(feature_contrib_df["coefficient"])
feature_contrib_df = feature_contrib_df.sort_values("abs_coeff", ascending=False)

print("\nTop 20 feature contributions:")
print(feature_contrib_df.head(20).to_string(index=False))

# SAVE PREDICTIONS

output_df = pd.DataFrame({
    "true_label": y_test,
    "predicted_probability": y_test_prob,
    "predicted_class": y_test_pred,
})
output_df.to_csv(OUTPUT_CSV, index=False)

print(f"\nPredictions saved to: {OUTPUT_CSV}")
print(f"Total rows in output (= test set size): {len(output_df)}")