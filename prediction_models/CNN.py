import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve


# USER PARAMETERS


ROOT_FOLDER =   # UPDATE THIS
TEST_SIZE = 0.25
RANDOM_STATE = 42
EPOCHS = 30
LR = 1e-3
BATCH_SIZE = 4
MAX_SEQ_LEN = 200000
THRESHOLD = 0.50

# CNN MODEL (unchanged from original)

class WaveformCNN(nn.Module):
    def __init__(self, input_channels=1, dropout=0.3):
        super(WaveformCNN, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.global_pool(x).squeeze(-1)
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits


# DATASET

class DenoisedWaveformDataset(Dataset):
    def __init__(self, root_folder, max_seq_len=200000):
        self.waveforms = []
        self.sample_ids = []
        self.labels = []
        self.max_seq_len = max_seq_len

        for folder_name in sorted(os.listdir(root_folder)):
            folder_path = os.path.join(root_folder, folder_name)
            if not os.path.isdir(folder_path):
                continue

            label = 0 if folder_name == "a_control" else 1
            csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
            print(f"Found {len(csv_files)} CSV files in folder '{folder_name}'")

            for csv_file in csv_files:
                df = pd.read_csv(csv_file, header=None)
                waveform = df.iloc[:, 1].to_numpy(dtype=np.float32)

                # Normalize
                waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-8)

                # Pad/truncate
                if len(waveform) > max_seq_len:
                    waveform = waveform[:max_seq_len]
                elif len(waveform) < max_seq_len:
                    waveform = np.pad(waveform, (0, max_seq_len - len(waveform)))

                self.waveforms.append(np.expand_dims(waveform, axis=0))
                self.sample_ids.append(f"{folder_name}_{os.path.basename(csv_file)}")
                self.labels.append(label)

        print(f"Total samples loaded: {len(self.waveforms)}")

    def __len__(self):
        return len(self.waveforms)

    def __getitem__(self, idx):
        waveform = torch.tensor(self.waveforms[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        sample_id = self.sample_ids[idx]
        return waveform, label, sample_id



# METRICS

def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    sensitivity = recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "fpr": fpr,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# TRAINING


def train_model(model, train_loader, val_loader, device, epochs=30, lr=1e-3):
    save_dir = "model_outputs"
    os.makedirs(save_dir, exist_ok=True)

    # Compute pos_weight from training set
    train_labels_list = []
    for _, labels, _ in train_loader.dataset:
        train_labels_list.append(labels.item() if isinstance(labels, torch.Tensor) else labels)

    # Alternative: iterate through indices
    if len(train_labels_list) == 0:
        ds = train_loader.dataset
        if hasattr(ds, 'dataset'):  # it's a Subset
            train_labels_list = [ds.dataset.labels[i] for i in ds.indices]
        else:
            train_labels_list = ds.labels

    num_pos = sum(train_labels_list)
    num_neg = len(train_labels_list) - num_pos
    pos_weight = torch.tensor([num_neg / max(num_pos, 1)], dtype=torch.float32).to(device)
    print(f"\nTraining: {num_neg} control, {num_pos} stressed, pos_weight={pos_weight.item():.3f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_f1 = 0

    for epoch in range(epochs):
        # TRAIN
        model.train()
        epoch_loss = 0

        for waveforms, labels, _ in train_loader:
            waveforms = waveforms.to(device)
            labels = labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            logits = model(waveforms)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)

        # VALIDATION
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for waveforms, labels, _ in val_loader:
                waveforms = waveforms.to(device)
                logits = model(waveforms)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                preds = (probs >= THRESHOLD).astype(int)
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        m = compute_metrics(all_labels, all_preds)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | "
              f"Val Acc: {m['accuracy']:.4f}, F1: {m['f1']:.4f}")

        # Save best model by validation F1
        if m["f1"] > best_val_f1:
            best_val_f1 = m["f1"]
            path = os.path.join(save_dir, "best_model.pth")
            torch.save(model.state_dict(), path)

    # Save final model
    final_path = os.path.join(save_dir, "final_model.pth")
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete. Best val F1: {best_val_f1:.4f}")

    return os.path.join(save_dir, "best_model.pth")


# TEST EVALUATION — single locked evaluation on held-out data

def evaluate_test_set(model, test_loader, device):
    model.eval()

    all_probs, all_labels, all_ids = [], [], []

    with torch.no_grad():
        for waveforms, labels, ids in test_loader:
            waveforms = waveforms.to(device)
            logits = model(waveforms)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
            all_ids.extend(ids)

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = (y_prob >= THRESHOLD).astype(int)

    m = compute_metrics(y_true, y_pred)

    # ROC-AUC
    auc = roc_auc_score(y_true, y_prob)

    # Bootstrap 95% CI for AUC
    rng = np.random.RandomState(RANDOM_STATE)
    n_bootstrap = 10000
    boot_aucs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boot_aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    boot_aucs = np.array(boot_aucs)
    ci_lower = np.percentile(boot_aucs, 2.5)
    ci_upper = np.percentile(boot_aucs, 97.5)

    # Print results
    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    print(f"Test set size: {len(y_true)} sessions")
    print(f"Threshold: {THRESHOLD}")
    print(f"")
    print(f"Accuracy:    {m['accuracy']:.4f}")
    print(f"Precision:   {m['precision']:.4f}")
    print(f"Recall:      {m['recall']:.4f}")
    print(f"F1-score:    {m['f1']:.4f}")
    print(f"ROC-AUC:     {auc:.4f}")
    print(f"")
    print(f"Sensitivity: {m['sensitivity']:.4f}")
    print(f"Specificity: {m['specificity']:.4f}")
    print(f"FPR:         {m['fpr']:.4f}")
    print(f"")
    print(f"Confusion matrix:")
    print(f"  TN={m['tn']}  FP={m['fp']}")
    print(f"  FN={m['fn']}  TP={m['tp']}")
    print(f"")
    print(f"ROC-AUC 95% CI (bootstrap, {n_bootstrap} resamples): {ci_lower:.4f} – {ci_upper:.4f}")

    # ROC curve plot
    fpr_curve, tpr_curve, _ = roc_curve(y_true, y_prob)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 6))
    plt.plot(fpr_curve, tpr_curve, color="steelblue", lw=2, label=f"CNN (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — 1D-CNN")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("roc_curve_cnn_fixed.png", dpi=300)
    plt.show()

    # Save predictions
    output_df = pd.DataFrame({
        "sample_id": all_ids,
        "true_label": y_true.astype(int),
        "predicted_probability": y_prob,
        "predicted_class": y_pred,
    })
    output_df.to_csv("cnn_test_predictions_fixed.csv", index=False)
    print(f"\nPredictions saved to: cnn_test_predictions_fixed.csv")
    print(f"Total rows in output (= test set size): {len(output_df)}")

    return auc, m


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # Set seeds for reproducibility
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    # Load all data
    dataset = DenoisedWaveformDataset(ROOT_FOLDER, max_seq_len=MAX_SEQ_LEN)

    all_labels = np.array(dataset.labels)
    all_indices = np.arange(len(dataset))

    print(f"\nTotal samples: {len(dataset)}")
    print(f"Class distribution: 0={np.sum(all_labels==0)}, 1={np.sum(all_labels==1)}")

    # ---- STRATIFIED 75/25 SPLIT with fixed seed ----
    train_indices, test_indices = train_test_split(
        all_indices,
        test_size=TEST_SIZE,
        stratify=all_labels,
        random_state=RANDOM_STATE,
    )

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    print(f"\nTrain set: {len(train_indices)} sessions")
    print(f"Test set:  {len(test_indices)} sessions")
    print(f"Train class dist: 0={np.sum(all_labels[train_indices]==0)}, 1={np.sum(all_labels[train_indices]==1)}")
    print(f"Test class dist:  0={np.sum(all_labels[test_indices]==0)}, 1={np.sum(all_labels[test_indices]==1)}")

    # Weighted sampler for training (handles class imbalance)
    train_labels = all_labels[train_indices]
    class_counts = np.array([np.sum(train_labels == t) for t in [0, 1]])
    weight = 1.0 / class_counts
    samples_weight = np.array([weight[t] for t in train_labels])
    sampler = WeightedRandomSampler(
        torch.from_numpy(samples_weight).double(), len(samples_weight)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = WaveformCNN().to(device)

    best_model_path = train_model(
        model, train_loader, test_loader, device, epochs=EPOCHS, lr=LR
    )

    # ---- LOAD BEST MODEL AND EVALUATE ON TEST SET ----
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print(f"\nLoaded best model from: {best_model_path}")

    evaluate_test_set(model, test_loader, device)