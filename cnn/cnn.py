import os
import glob
import time
import numpy as np
import pandas as pd
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score,f1_score,confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau


#Für Evaluation
LABEL = os.getenv("TRAFFIC_LABEL", "unknown")

#dataset
class SSHDataset(Dataset):
    def __init__(self, csv_files: List[str], labels: List[int], max_len: int = 200):
        self.csv_files = csv_files
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.csv_files)

    def __getitem__(self, idx):
        csv_file = self.csv_files[idx]
        label = self.labels[idx]
        features = preprocess_csv(csv_file, self.max_len)
        if features is None:
            return None
        return torch.tensor(features), torch.tensor(label, dtype=torch.float32)

def collate_fn(batch):
    # batch = Liste von (features, label) oder None
    batch = [b for b in batch if b is not None]  
    if len(batch) == 0:
        return torch.empty(0), torch.empty(0)  # leeren Batch zurückgeben
    return torch.utils.data._utils.collate.default_collate(batch)

#datenvorbereitung
def preprocess_csv(csv_file, max_len=200):
    try:
        df = pd.read_csv(csv_file, on_bad_lines = "skip")
    except Exception as e:
        file_name = os.path.basename(csv_file)
        print(f"Fehler beim Einlesen von: {file_name} mit  {e}", flush = True)
        return None
    
    if df.empty:
        return None

    #Zeitstempel
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
        df["delta_t"] = df["timestamp"].diff().dt.total_seconds().fillna(0)
    else:
        df["delta_t"] = 0

    # Flags in binäre Features 
    df["psh"] = df["flags"].apply(lambda f: int("P" in str(f)) if pd.notna(f) else 0)
    df["ackf"] = df["flags"].apply(lambda f: int("A" in str(f)) if pd.notna(f) else 0)

    # Feature-Matrix
    feature_columns = [
        "payload_len",
        "delta_t",
        "src_port",
        "dst_port",
        "psh",
        "ackf",
        "window"
    ]
    # Fehlende Spalten auffüllen
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    features = df[feature_columns].fillna(0).values

    # Padding/Kürzen
    seq_len = len(features)
    if seq_len < max_len:
        pad_len = max_len - seq_len
        pad = np.zeros((pad_len, features.shape[1]))
        features = np.vstack([features, pad])
        padding_ratio = pad_len / max_len
    else:
        features = features[:max_len]
        padding_ratio = 0.0

    # padding_ratio als zusätzliche Spalte an jedes Zeilenfeature anhängen
    #padding_feature = np.full((max_len, 1), padding_ratio, dtype=np.float32)
    #features = np.hstack([features, padding_feature])  

    return features.astype(np.float32)

#cnn
class SSHCNN(nn.Module):
    def __init__(self, n_features=7, max_len=200):  #n_features = 8,wenn padding_ratio genutzt wird
        super(SSHCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=7, out_channels=64, kernel_size=5)# bei padd_ration n=8
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Eingabe: (batch, seq_len, n_features) → (batch, n_features, seq_len)
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.global_pool(x).squeeze(-1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x.squeeze(-1)

#training
def train_model(train_loader, val_loader, epochs=1, lr=1e-3, device="cpu"):
    
    model = SSHCNN().to(device)
    
    # Verhältnis: wie stark Klasse 1 unterrepräsentiert ist
    pos_weight = torch.tensor([len(ssh_files) / len(tunnel_files)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight = pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    metrics_file = "/data/train_metrics.csv"
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    if not os.path.exists(metrics_file):
        pd.DataFrame(columns=["epoch",
            "train_loss","train_acc","train_precision","train_recall","train_f1",
            "val_loss","val_acc","val_precision","val_recall","val_f1"]).to_csv(metrics_file, index=False)

    cm_file = "/data/cm.csv"
    os.makedirs(os.path.dirname(cm_file), exist_ok=True)
    if not os.path.exists(cm_file):  
        pd.DataFrame(columns=["epoch","phase","cm"]).to_csv(cm_file, index=False)  


    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, total = 0, 0, 0
        train_preds,train_labels = [],[]

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X.size(0)
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            train_correct += (preds == y).sum().item()
            total += y.size(0)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(y.cpu().numpy())

        train_acc = np.mean(np.array(train_preds) == np.array(train_labels))
        train_prec = precision_score(train_labels, train_preds, zero_division=0)
        train_rec = recall_score(train_labels, train_preds, zero_division=0)
        train_f1 = f1_score(train_labels, train_preds, zero_division=0)
        avg_train_loss = train_loss / total

        #train cm
        train_cm = confusion_matrix(train_labels, train_preds)
        pd.DataFrame([{
            "epoch": epoch+1,
            "phase": "train",
            "cm": train_cm.tolist()  # in Liste umwandeln für CSV
        }]).to_csv(cm_file, mode='a', header=False, index=False)

        # Validation
        if val_loader is not None:
            model.eval()
            val_correct, val_total,val_loss = 0,0, 0
            val_preds =[]
            val_labels = []
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(device), y.to(device)
                    outputs = model(X)
                    loss = criterion(outputs,y)
                    val_loss += loss.item() * X.size(0)
                    val_total += y.size(0)
                    preds = (torch.sigmoid(outputs) >= 0.5).float()
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(y.cpu().numpy())
            
            val_acc = np.mean(np.array(val_preds) == np.array(val_labels))
            val_prec = precision_score(val_labels, val_preds, zero_division=0)
            val_rec = recall_score(val_labels, val_preds, zero_division=0)
            val_f1 = f1_score(val_labels, val_preds, zero_division=0)
            avg_val_loss = val_loss / val_total

             # Validation cm
            val_cm = confusion_matrix(val_labels, val_preds)
            pd.DataFrame([{
                "epoch": epoch+1,
                "phase": "val",
                "cm": val_cm.tolist()  
            }]).to_csv(cm_file, mode='a', header=False, index=False)

            scheduler.step(avg_val_loss)
        else:
            val_acc = val_prec = val_rec = val_f1 = avg_val_loss = None
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f}, Prec: {train_prec:.4f}, Rec: {train_rec:.4f}, F1: {train_f1:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.4f}, Prec: {val_prec:.4f}, Rec: {val_rec:.4f}, F1: {val_f1:.4f}", flush=True)


        # metrik - CSV aktualisieren
        pd.DataFrame([{
            "epoch": epoch+1,
            "train_loss": avg_train_loss,
            "train_acc": train_acc,
            "train_precision": train_prec,
            "train_recall": train_rec,
            "train_f1": train_f1,
            "val_loss": avg_val_loss,
            "val_acc": val_acc,
            "val_precision": val_prec,
            "val_recall": val_rec,
            "val_f1": val_f1
        }]).to_csv(metrics_file, mode='a', header=False, index=False)


    torch.save(model.state_dict(), "ssh_cnn.pt")
    print("Modell gespeichert als ssh_cnn.pt")
    return model


#fortlaufend

def predict_csv(model, csv_file, device="cpu"):
    features = preprocess_csv(csv_file, max_len=200)
    X = torch.tensor(features).unsqueeze(0).to(device)  # (1, seq_len, n_features)
    with torch.no_grad():
        output = model(X)
        pred = (output >= 0.5).float().item()
    return int(pred)  # 0 = ssh, 1 = tunnel


#training+ Live
if __name__ == "__main__":
    # Trainingsdaten laden
    print("Start: Data Processing")
    log_dir = "/sniffer/logs"
    os.makedirs(log_dir, exist_ok=True)
    all_files = glob.glob(os.path.join(log_dir, "*.csv"))
    
    ssh_files = [f for f in all_files if os.path.basename(f).startswith("0_")]
    tunnel_files = [f for f in all_files if os.path.basename(f).startswith("1_")]

    print(f"Anzahl SSH Files : {len(ssh_files)}",flush = True)
    print(f"Anzahl Tunnel Files :{len(tunnel_files)}  ", flush=True)

    train_files = ssh_files + tunnel_files  
    train_labels = [0] * len(ssh_files) + [1] * len(tunnel_files)

    #wenn die csv-datei leer ist,wird diese beim training nicht berücksichtigt
    valid_files = []
    valid_labels = []
    for f, label in zip(train_files, train_labels):
        if preprocess_csv(f) is not None:
            valid_files.append(f)
            valid_labels.append(label)

    train_files = valid_files  
    train_labels = valid_labels

    if len(train_files) == 0:
        print("Keine CSV-Dateien gefunden! Training abgebrochen.",flush = True)
        exit(1)

    dataset = SSHDataset(train_files, train_labels, max_len=200)

    # Train/Val Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, collate_fn=collate_fn)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Start: Training on {device}")

    # training
    model = train_model(train_loader, val_loader, epochs=4, lr=1e-3, device=device)

    #live-logs
    results_csv = "/data/predictions.csv"
    os.makedirs("/data", exist_ok=True)
   
    seen_files = set()
    skipped_files = 0
    
    all_logs = glob.glob(os.path.join(log_dir, "*.csv"))
    initial_test_files = [f for f in all_logs if not os.path.basename(f).startswith(("0_","1_"))]

    print(f"Vorhandene Testdateien ohne Präfix: {len(initial_test_files)}", flush=True)

    for f in initial_test_files:
        features = preprocess_csv(f, max_len=200)
        if features is None:
            skipped_files +=1
            print(f"{f} konnte nicht gelesen werden", flush=True)
            #seen_files.add(f)
            continue

        X = torch.tensor(features).unsqueeze(0).to(device)
        with torch.no_grad():
            output = torch.sigmoid(model(X))
            pred = (output >= 0.5).float().item()
        
        prediction = "ssh-tunnelled" if pred == 1 else "ssh-not tunnelled"
        base = os.path.basename(f)
        true_label = LABEL
        
        #print(f"[Initial] {base} → predicted: {prediction}, true: {true_label}", flush=True)
        
        pd.DataFrame([{
            "file": base,
            "prediction": prediction,
            "true_label": true_label,
            "timestamp": pd.Timestamp.now()
        }]).to_csv(results_csv, mode='a', header=not os.path.exists(results_csv), index=False)
        
        seen_files.add(f)
    #print(f"Anzahl übersprungener Dateien: {skipped_files}", flush=True)

        #print(" Warte auf neue Verbindungen in logs/ ...",flush = True)
    while True:
        all_logs = glob.glob(os.path.join(log_dir, "*.csv"))
        #print("DEBUG: Alle CSVs im Log:", all_logs, flush = True)

        test_files = [f for f in all_logs if not os.path.basename(f).startswith(("0_","1_")) and f not in seen_files]
        print(f"Start: LiveLoop: {len(test_files)} neue Dateien:", flush = True)

        if test_files:
            # Features aller neuen Dateien vorbereiten
            #print(f" {len(test_files)} neue Datei(en) gefunden...", flush=True)
            feature_list = []
            file_order = []
            for f in test_files:
                features = preprocess_csv(f, max_len=200)
                if features is None:
                    print(f"Datei {f} wird übersprungen", flush= True)
                    seen_files.add(f)
                    continue
                # Debug-Ausgabe der Feature-Matrix
                #print(f"DEBUG: Features von {f} Shape:", features.shape, "erste 3 Zeilen:\n", features[:3, :], flush=True)
                feature_list.append(features)
                file_order.append(f)
                seen_files.add(f)


            if feature_list:
                X_batch = torch.tensor(np.stack(feature_list), dtype=torch.float32).to(device)  # (batch, seq_len, n_features)
            
                # Predictions in einem Batch
                with torch.no_grad():
                    outputs = model(X_batch)
                    preds = (outputs >= 0.5).float().cpu().numpy()
            
                # Ergebnisse ausgeben
                for f, pred in zip(file_order, preds):
                    # Modell-Prediction
                    prediction = "ssh-tunnelled" if pred == 1 else "ssh-not tunnelled"

                    # True Label aus Dateiname
                    base = os.path.basename(f)
                    true_label = LABEL
                    
                    #print(f"DEBUG: {base} → predicted: {prediction}, true: {true_label}", flush=True)

                    # Prediction + Label speichern
                    pd.DataFrame([{
                        "file": base,
                        "prediction": prediction,
                        "true_label": true_label,
                        "timestamp": pd.Timestamp.now()
                    }]).to_csv(results_csv, mode='a', header=not os.path.exists(results_csv), index=False)

                    # korrekt oder falsch
                    if true_label != "unknown":
                        print(f"[+] {base} → predicted: {prediction}, true: {true_label}, correct: {prediction == true_label}", flush = True)

        time.sleep(5)  # alle 5 Sek
