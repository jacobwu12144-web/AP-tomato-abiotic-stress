import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, random_split
import torch.nn as nn
import torch.nn.functional as F

# CNN Model

class WaveformCNN(nn.Module):
    def __init__(self, input_channels=1, embedding_dim=128, dropout=0.3):
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


# Dataset

class DenoisedWaveformDataset(Dataset):
    def __init__(self, root_folder, max_seq_len=200000):
        self.waveforms = []
        self.sample_ids = []
        self.labels = []
        self.max_seq_len = max_seq_len

        for folder_name in os.listdir(root_folder):
            folder_path = os.path.join(root_folder, folder_name)
            if not os.path.isdir(folder_path):
                continue

            label = 0 if folder_name == "a_control" else 1
            csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
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



# Metrics

def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = np.mean(y_true == y_pred)
    tp = np.sum((y_true==1)&(y_pred==1))
    tn = np.sum((y_true==0)&(y_pred==0))
    fp = np.sum((y_true==0)&(y_pred==1))
    fn = np.sum((y_true==1)&(y_pred==0))

    precision = tp/(tp+fp) if (tp+fp)>0 else 0
    recall = tp/(tp+fn) if (tp+fn)>0 else 0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0

    return accuracy, precision, recall, f1



# Training

def train_model(model, train_loader, val_loader, device, epochs=10, lr=1e-3):

    save_dir = "model_outputs"
    os.makedirs(save_dir, exist_ok=True)

    num_pos = sum([label.item() for _, label, _ in train_loader.dataset])
    num_neg = len(train_loader.dataset) - num_pos
    pos_weight = torch.tensor([num_neg / max(num_pos, 1)], dtype=torch.float32).to(device)

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
                preds = (probs >= 0.5).astype(int)

                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        acc, prec, rec, f1 = compute_metrics(all_labels, all_preds)

        print(f"\nEpoch {epoch+1}")
        print(f"Loss: {avg_loss:.4f}")
        print(f"Val → Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")

        # SAVE BEST MODEL
        if f1 > best_val_f1:
            best_val_f1 = f1
            path = os.path.join(save_dir, "best_model.pth")
            torch.save(model.state_dict(), path)
            print(f"Saved best model → {os.path.abspath(path)}")

    # SAVE FINAL MODEL
    final_path = os.path.join(save_dir, "final_model.pth")
    torch.save(model.state_dict(), final_path)
    print(f"\n Saved FINAL model → {os.path.abspath(final_path)}")



# Prediction

def predict_and_save(model, dataset, device):
    model.eval()
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    all_ids, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for waveforms, _, ids in loader:
            waveforms = waveforms.to(device)

            logits = model(waveforms)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds = (probs >= 0.5).astype(int)

            all_ids.extend(ids)
            all_probs.extend(probs)
            all_preds.extend(preds)

    df = pd.DataFrame({
        "sample_id": all_ids,
        "prediction": all_preds,
        "probability": all_probs
    })

    df.to_csv("predictions.csv", index=False)
    print("Predictions saved to predictions.csv")



# MAIN

if __name__ == "__main__":

    root_folder = 
    dataset = DenoisedWaveformDataset(root_folder)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_labels = [dataset.labels[i] for i in train_dataset.indices]
    class_counts = np.array([sum(np.array(train_labels)==t) for t in [0,1]])
    weight = 1. / class_counts
    samples_weight = np.array([weight[t] for t in train_labels])

    sampler = WeightedRandomSampler(torch.from_numpy(samples_weight).double(), len(samples_weight))

    train_loader = DataLoader(train_dataset, batch_size=4, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = WaveformCNN().to(device)

    # TRAIN
    train_model(model, train_loader, val_loader, device)

    # PREDICT
    predict_and_save(model, dataset, device)
