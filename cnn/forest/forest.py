import os
import glob
import time
import numpy as np
import pandas as pd
import joblib
from typing import List, Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score, roc_auc_score, log_loss
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict


#Für Evaluation
LABEL = os.getenv("TRAFFIC_LABEL", "unknown")

#Preprocessor von CNN übernommen
def preprocess_csv(csv_file, max_len=200):
    try:
        # Make an explicit copy so we never work on a view
        df = pd.read_csv(csv_file, on_bad_lines="skip").copy()
    except Exception as e:
        file_name = os.path.basename(csv_file)
        print(f"Fehler beim Einlesen von: {file_name} mit  {e}", flush=True)
        return None

    if df.empty:
        return None

    #timestamps
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
        df.loc[:, "timestamp"] = ts
        df.loc[:, "delta_t"] = ts.diff().dt.total_seconds().fillna(0.0)
    else:
        df.loc[:, "delta_t"] = 0.0

    #flags (vectorized, no .apply)
    if "flags" in df.columns:
        fstr = df["flags"].astype(str)
        df.loc[:, "psh"]  = fstr.str.contains("P", na=False).astype("int8")
        df.loc[:, "ackf"] = fstr.str.contains("A", na=False).astype("int8")
    else:
        df.loc[:, "psh"] = 0
        df.loc[:, "ackf"] = 0

    #Feature columns auswählen
    feature_columns = ["payload_len", "delta_t", "src_port", "dst_port", "psh", "ackf", "window"]
    for col in feature_columns:
        if col not in df.columns:
            df.loc[:, col] = 0

    features = df.loc[:, feature_columns].fillna(0).to_numpy(dtype=np.float32)

    #pad / crop zur max_len
    seq_len = len(features)
    if seq_len < max_len:
        pad = np.zeros((max_len - seq_len, features.shape[1]), dtype=np.float32)
        features = np.vstack([features, pad])
    else:
        features = features[:max_len]

    return features


#Helpers für Random Forest pipeline
def file_to_vector(csv_path: str, max_len: int = 200) -> np.ndarray | None:
    #Preprocess und flatten zu 1D feature vector für sklearn
    feats = preprocess_csv(csv_path, max_len=max_len)
    if feats is None:
        return None
    return feats.reshape(-1)  # (max_len * 7,)

def build_dataset(file_paths: List[str], labels: List[int], max_len: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    X_list, y_list = [], []
    for fp, y in zip(file_paths, labels):
        v = file_to_vector(fp, max_len=max_len)
        if v is not None:
            X_list.append(v)
            y_list.append(y)
    if len(X_list) == 0:
        return np.empty((0, max_len * 7)), np.empty((0,))
    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list).astype(int)
    return X, y

def log_confusion_matrix(cm_file: str, epoch: int, phase: str, cm: np.ndarray):
    pd.DataFrame([{
        "epoch": epoch,
        "phase": phase,
        "cm": cm.tolist()
    }]).to_csv(cm_file, mode='a', header=not os.path.exists(cm_file), index=False)


#Training
def train_random_forest(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_estimators: int = 400,
        max_depth: int | None = None,
        random_state: int = 42
):
    #class_weight='balanced' hilft bei imbalanced data, !alt: ohne cross validation!
    # rf = RandomForestClassifier(
    #     n_estimators=n_estimators,
    #     max_depth=max_depth,
    #     n_jobs=-1,
    #     random_state=random_state,
    #     class_weight="balanced",
    #     min_samples_split=2,
    #     min_samples_leaf=1,
    #     max_features="sqrt",
    #     oob_score=True
    # )
    # rf.fit(X_train, y_train)

    rf = RandomForestClassifier(
        n_estimators=400, max_depth=None, max_features='sqrt',
        class_weight='balanced', n_jobs=-1, random_state=42, oob_score=False
    )

    #5-fold stratified cross validation (shuffled für Robustheit)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    #Out-of-fold Wahrscheinlichkeiten
    oof_proba = cross_val_predict(
        rf, X, y, cv=cv, method='predict_proba', n_jobs=-1
    )[:, 1]

    #Auswählen von threshold
    ths = np.linspace(0.1, 0.9, 81)
    f1s = [f1_score(y, (oof_proba >= t).astype(int)) for t in ths]
    best_t = ths[int(np.argmax(f1s))]

    #OOF metrics beim threshold
    y_oof = (oof_proba >= best_t).astype(int)
    metrics = {
        "ACC": accuracy_score(y, y_oof),
        "PREC": precision_score(y, y_oof, zero_division=0),
        "REC": recall_score(y, y_oof, zero_division=0),
        "F1": f1_score(y, y_oof, zero_division=0),
        "AUC": roc_auc_score(y, oof_proba)
    }
    print("OOF metrics:", metrics, "best_threshold:", round(best_t, 3))


    rf.fit(X, y)

    #Evaluieren
    def eval_split(X, y):
        if len(X) == 0:
            return {k: None for k in ["loss","acc","prec","rec","f1","auc"]}
        proba = rf.predict_proba(X)[:, 1]
        preds = (proba >= 0.5).astype(int)  # or use your tuned threshold

        #real loss pro split
        loss = None
        if len(np.unique(y)) == 2:
            loss = log_loss(y, np.c_[1 - proba, proba], labels=[0, 1])

        return {
            "loss": loss,
            "acc": accuracy_score(y, preds),
            "prec": precision_score(y, preds, zero_division=0),
            "rec": recall_score(y, preds, zero_division=0),
            "f1": f1_score(y, preds, zero_division=0),
            "auc": roc_auc_score(y, proba) if len(np.unique(y)) == 2 else None,
            "preds": preds
    }

    train_metrics = eval_split(X_train, y_train)
    val_metrics = eval_split(X_val, y_val)

    return rf, train_metrics, val_metrics


#Hauptscript
if __name__ == "__main__":
    print("Start: Data Processing (RF)")
    log_dir = "/sniffer/logs"
    os.makedirs(log_dir, exist_ok=True)
    all_files = glob.glob(os.path.join(log_dir, "*.csv"))

    ssh_files = [f for f in all_files if os.path.basename(f).startswith("0_")]
    tunnel_files = [f for f in all_files if os.path.basename(f).startswith("1_")]

    print(f"Anzahl SSH Files : {len(ssh_files)}", flush=True)
    print(f"Anzahl Tunnel Files : {len(tunnel_files)}", flush=True)

    train_files = ssh_files + tunnel_files
    train_labels = [0] * len(ssh_files) + [1] * len(tunnel_files)

    #Rausfiltern von unlesbaren csv-Dateien
    valid_files, valid_labels = [], []
    for f, label in zip(train_files, train_labels):
        if preprocess_csv(f) is not None:
            valid_files.append(f)
            valid_labels.append(label)

    train_files = valid_files
    train_labels = valid_labels

    if len(train_files) == 0:
        print("Keine CSV-Dateien gefunden! Training abgebrochen.", flush=True)
        exit(1)

    #Erstellen von Datensatz für sklearn
    X, y = build_dataset(train_files, train_labels, max_len=200)

    #Train/Val split, ist unnötig
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    #Gleiche Pfade wie beim CNN fpr metric logs
    metrics_file = "/data/train_metrics.csv"
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    if not os.path.exists(metrics_file):
        pd.DataFrame(columns=[
            "epoch",
            "train_loss","train_acc","train_precision","train_recall","train_f1",
            "val_loss","val_acc","val_precision","val_recall","val_f1","val_auc"
        ]).to_csv(metrics_file, index=False)

    cm_file = "/data/cm.csv"
    os.makedirs(os.path.dirname(cm_file), exist_ok=True)
    if not os.path.exists(cm_file):
        pd.DataFrame(columns=["epoch","phase","cm"]).to_csv(cm_file, index=False)


    rf_model, tr_m, va_m = train_random_forest(X_train, y_train, X_val, y_val)

    #Log confusion matrices
    log_confusion_matrix(cm_file, epoch=1, phase="train", cm=confusion_matrix(y_train, tr_m["preds"]))
    log_confusion_matrix(cm_file, epoch=1, phase="val",   cm=confusion_matrix(y_val,   va_m["preds"]))

    #Print Metriken
    print(
        f"Epoch 1/1 | "
        f"Train Loss: {tr_m['loss']:.4f}, Acc: {tr_m['acc']:.4f}, Prec: {tr_m['prec']:.4f}, Rec: {tr_m['rec']:.4f}, F1: {tr_m['f1']:.4f} | "
        f"Val Loss: {va_m['loss']:.4f}, Acc: {va_m['acc']:.4f}, Prec: {va_m['prec']:.4f}, Rec: {va_m['rec']:.4f}, F1: {va_m['f1']:.4f}, AUC: {va_m['auc'] if va_m['auc'] is not None else float('nan'):.4f}",
        flush=True
    )

    pd.DataFrame([{
        "epoch": 1,
        "train_loss": tr_m["loss"],
        "train_acc": tr_m["acc"],
        "train_precision": tr_m["prec"],
        "train_recall": tr_m["rec"],
        "train_f1": tr_m["f1"],
        "val_loss": va_m["loss"],
        "val_acc": va_m["acc"],
        "val_precision": va_m["prec"],
        "val_recall": va_m["rec"],
        "val_f1": va_m["f1"],
        "val_auc": va_m["auc"]
    }]).to_csv(metrics_file, mode='a', header=False, index=False)

    #Modell speichern
    joblib.dump(rf_model, "ssh_rf.joblib")
    print("Modell gespeichert als ssh_rf.joblib")


    #Batch prediction auf Dateien ohne Label
    results_csv = "/data/predictions.csv"
    os.makedirs("/data", exist_ok=True)

    seen_files = set()
    skipped_files = 0

    all_logs = glob.glob(os.path.join(log_dir, "*.csv"))
    initial_test_files = [f for f in all_logs if not os.path.basename(f).startswith(("0_", "1_"))]

    print(f"Vorhandene Testdateien ohne Präfix: {len(initial_test_files)}", flush=True)

    for f in initial_test_files:
        vec = file_to_vector(f, max_len=200)
        if vec is None:
            skipped_files += 1
            print(f"{f} konnte nicht gelesen werden", flush=True)
            continue

        proba = rf_model.predict_proba([vec])[0, 1]
        pred_label = int(proba >= 0.5)
        prediction = "ssh-tunnelled" if pred_label == 1 else "ssh-not tunnelled"

        base = os.path.basename(f)
        true_label = LABEL

        pd.DataFrame([{
            "file": base,
            "prediction": prediction,
            "proba_tunnel": float(proba),
            "true_label": true_label,
            "timestamp": pd.Timestamp.now()
        }]).to_csv(results_csv, mode='a', header=not os.path.exists(results_csv), index=False)

        seen_files.add(f)


    #Live loop
    while True:
        all_logs = glob.glob(os.path.join(log_dir, "*.csv"))
        test_files = [f for f in all_logs if not os.path.basename(f).startswith(("0_", "1_")) and f not in seen_files]
        print(f"Start: LiveLoop: {len(test_files)} neue Dateien:", flush=True)

        for f in test_files:
            vec = file_to_vector(f, max_len=200)
            if vec is None:
                print(f"Datei {f} wird übersprungen", flush=True)
                seen_files.add(f)
                continue

            proba = rf_model.predict_proba([vec])[0, 1]
            pred_label = int(proba >= 0.5)
            prediction = "ssh-tunnelled" if pred_label == 1 else "ssh-not tunnelled"

            base = os.path.basename(f)
            true_label = os.getenv("TRAFFIC_LABEL", "unknown")


            pd.DataFrame([{
                "file": base,
                "prediction": prediction,
                "proba_tunnel": float(proba),
                "true_label": true_label,
                "timestamp": pd.Timestamp.now()
            }]).to_csv(results_csv, mode='a', header=not os.path.exists(results_csv), index=False)

            seen_files.add(f)

        time.sleep(5)
