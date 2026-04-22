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
    classification_report,
    roc_curve
)

# USER PARAMETERS

INPUT_CSV = 
OUTPUT_CSV = "logistic_regression_predictions.csv"

MODEL_PATH = "logistic_model.pkl"
SCALER_PATH = "scaler.pkl"
POLY_PATH = "poly_transform.pkl"
THRESHOLD_PATH = "threshold.pkl"

TEST_SIZE = 0.25
RANDOM_STATE = 42
N_SPLITS = 5
LAMBDA_GRID = np.logspace(-3, 2, 20)


# LOAD DATA

df = pd.read_csv(INPUT_CSV)
df.columns = df.columns.str.strip()

FEATURE_COLS = ["Amplitude", "tau_rise", "tau_fall", "Undershoot", "tau_rec"]
LABEL_COL = "Label"

df[FEATURE_COLS] = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce")
df = df.dropna()

# Remove non-positive time constants BEFORE log
valid_mask = (df["tau_rise"] > 0) & (df["tau_fall"] > 0) & (df["tau_rec"] > 0)
df = df[valid_mask].copy()

# Log transforms
df["log_tau_rise"] = np.log(df["tau_rise"])
df["log_tau_fall"] = np.log(df["tau_fall"])
df["log_tau_rec"]  = np.log(df["tau_rec"])

# Biophysical interaction terms
df["rise_fall_ratio"] = df["tau_rise"] / df["tau_fall"]
df["A_tau_rec"] = df["Amplitude"] * df["tau_rec"]
df["U_tau_fall"] = df["Undershoot"] * df["tau_fall"]

ENGINEERED_COLS = [
    "Amplitude", "tau_rise", "tau_fall", "Undershoot", "tau_rec",
    "log_tau_rise", "log_tau_fall", "log_tau_rec",
    "rise_fall_ratio", "A_tau_rec", "U_tau_fall"
]

df = df.dropna()
X = df[ENGINEERED_COLS].values
y = df[LABEL_COL].values.astype(int)

print("\nFinal dataset shape after cleaning:", X.shape)
print("Any NaNs remaining?", np.isnan(X).any())

# TRAIN / TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly  = poly.transform(X_test)


# CROSS-VALIDATED LAMBDA SELECTION

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

best_lambda = None
best_mean_auc = -np.inf

print("\nCross-validating regularization strength...\n")

for lam in LAMBDA_GRID:
    aucs, f1s, accs = [], [], []

    for train_idx, val_idx in skf.split(X_train_poly, y_train):
        X_tr, X_val = X_train_poly[train_idx], X_train_poly[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val)

        model = LogisticRegression(
            penalty="l2",
            C=1.0/lam,
            solver="lbfgs",
            max_iter=3000
        )
        model.fit(X_tr_scaled, y_tr)

        y_val_prob = model.predict_proba(X_val_scaled)[:, 1]
        y_val_pred = (y_val_prob >= 0.5).astype(int)

        aucs.append(roc_auc_score(y_val, y_val_prob))
        f1s.append(f1_score(y_val, y_val_pred))
        accs.append(accuracy_score(y_val, y_val_pred))

    # Compute mean ± std for CV folds
    mean_auc = np.mean(aucs)
    std_auc  = np.std(aucs)
    mean_f1  = np.mean(f1s)
    std_f1   = np.std(f1s)
    mean_acc = np.mean(accs)
    std_acc  = np.std(accs)

    print(f"Lambda={lam:.5f} | AUC={mean_auc:.4f}±{std_auc:.4f} | "
          f"F1={mean_f1:.4f}±{std_f1:.4f} | ACC={mean_acc:.4f}±{std_acc:.4f}")

    if mean_auc > best_mean_auc:
        best_mean_auc = mean_auc
        best_lambda = lam

print(f"\nSelected lambda: {best_lambda:.5f} | CV AUC={best_mean_auc:.4f}")


# FINAL MODEL TRAINING


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_poly)
X_test_scaled  = scaler.transform(X_test_poly)

final_model = LogisticRegression(
    penalty="l2",
    C=1.0/best_lambda,
    solver="lbfgs",
    max_iter=3000
)
final_model.fit(X_train_scaled, y_train)


# SAVE MODEL + TRANSFORMS

joblib.dump(final_model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
joblib.dump(poly, POLY_PATH)

print(f"\nModel saved to: {MODEL_PATH}")
print(f"Scaler saved to: {SCALER_PATH}")
print(f"Polynomial transformer saved to: {POLY_PATH}")

# Optimal threshold using training set
train_probs = final_model.predict_proba(X_train_scaled)[:, 1]
thresholds = np.linspace(0.01, 0.99, 200)
f1_scores = [f1_score(y_train, (train_probs >= t).astype(int)) for t in thresholds]
best_threshold = thresholds[np.argmax(f1_scores)]
print(f"\nOptimal threshold (train F1): {best_threshold:.3f}")

joblib.dump(best_threshold, THRESHOLD_PATH)
print(f"Threshold saved to: {THRESHOLD_PATH}")



# FEATURE CONTRIBUTIONS


coef_poly = final_model.coef_.flatten()
poly_feature_names = poly.get_feature_names_out(ENGINEERED_COLS)

feature_contrib_df = pd.DataFrame({
    "feature": poly_feature_names,
    "coefficient": coef_poly
})
feature_contrib_df["abs_coeff"] = np.abs(feature_contrib_df["coefficient"])
feature_contrib_df = feature_contrib_df.sort_values("abs_coeff", ascending=False)
print("\nTop feature contributions:")
print(feature_contrib_df.head(20))


# TEST EVALUATION WITH STD


# Use StratifiedKFold on TEST set to compute mean ± std
test_skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
test_aucs, test_f1s, test_accs = [], [], []

for test_idx_train, test_idx_val in test_skf.split(X_test_scaled, y_test):
    X_tst_tr, X_tst_val = X_test_scaled[test_idx_train], X_test_scaled[test_idx_val]
    y_tst_tr, y_tst_val = y_test[test_idx_train], y_test[test_idx_val]

    y_val_prob = final_model.predict_proba(X_tst_val)[:, 1]
    y_val_pred = (y_val_prob >= best_threshold).astype(int)

    test_aucs.append(roc_auc_score(y_tst_val, y_val_prob))
    test_f1s.append(f1_score(y_tst_val, y_val_pred))
    test_accs.append(accuracy_score(y_tst_val, y_val_pred))

mean_auc  = np.mean(test_aucs)
std_auc   = np.std(test_aucs)
mean_f1   = np.mean(test_f1s)
std_f1    = np.std(test_f1s)
mean_acc  = np.mean(test_accs)
std_acc   = np.std(test_accs)

print("\n=== Test Metrics (mean ± std across folds) ===")
print(f"ROC-AUC: {mean_auc:.4f} ± {std_auc:.4f}")
print(f"F1 Score: {mean_f1:.4f} ± {std_f1:.4f}")
print(f"Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, (final_model.predict_proba(X_test_scaled)[:,1]>=best_threshold).astype(int)))

# Save predictions
output_df = pd.DataFrame({
    "true_label": y_test,
    "predicted_probability": final_model.predict_proba(X_test_scaled)[:,1],
    "predicted_class": (final_model.predict_proba(X_test_scaled)[:,1] >= best_threshold).astype(int)
})
output_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nPredictions saved to: {OUTPUT_CSV}")
