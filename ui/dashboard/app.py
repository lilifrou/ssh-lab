from flask import Flask, render_template, redirect, url_for, request, make_response, Blueprint
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, base64
import pandas as pd
import os
import seaborn as sns
import matplotlib.dates as mdates

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True #für automates reloaden, wenn sich code veröndert hat


#Dateipfade
RF_CSV_FILE     = "/data/predictions.csv"
RF_METRICS_FILE = "/data/train_metrics.csv"
RF_CM_FILE      = "/data/cm.csv"

CNN_CSV_FILE     = "/cnn/predictions.csv"
CNN_METRICS_FILE = "/cnn/train_metrics.csv"
CNN_CM_FILE      = "/cnn/cm.csv"


#Shared helpers mit Modus
def render_dashboard(metrics_path, cm_path, title_prefix="", mode="rf"):
    # Prüfen, ob METRICS_FILE existiert
    if not os.path.exists(metrics_path):
        return "<h1>Metrics file not found</h1>"

    metrics = pd.read_csv(metrics_path)
    #Letzte Epoche für Summary-Metriken
    last_epoch = metrics.iloc[-1]

    train_metrics = {
        'Loss': last_epoch.get('train_loss'),
        'Accuracy': last_epoch.get('train_acc'),
        'Precision': last_epoch.get('train_precision'),
        'Recall': last_epoch.get('train_recall'),
        'F1': last_epoch.get('train_f1')
    }
    val_metrics = {
        'Loss': last_epoch.get('val_loss'),
        'Accuracy': last_epoch.get('val_acc'),
        'Precision': last_epoch.get('val_precision'),
        'Recall': last_epoch.get('val_recall'),
        'F1': last_epoch.get('val_f1')
    }

    #Accuracy
    fig, ax = plt.subplots(figsize=(8,4))
    if 'train_acc' in metrics.columns:
        ax.plot(metrics['epoch'], metrics['train_acc'], label='Train Accuracy')
    if 'val_acc' in metrics.columns:
        ax.plot(metrics['epoch'], metrics['val_acc'], label='Validation Accuracy')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
    ax.set_title(f'{title_prefix} Train vs Validation Accuracy'.strip())
    ax.legend()
    buf = io.BytesIO(); plt.savefig(buf, format='png', bbox_inches='tight'); buf.seek(0)
    acc_image = base64.b64encode(buf.read()).decode(); buf.close(); plt.close(fig)

    #Loss
    fig, ax = plt.subplots(figsize=(8,4))
    if 'train_loss' in metrics.columns:
        ax.plot(metrics['epoch'], metrics['train_loss'], label='Train Loss')
    if 'val_loss' in metrics.columns:
        ax.plot(metrics['epoch'], metrics['val_loss'], label='Validation Loss')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title(f'{title_prefix} Train vs Validation Loss'.strip())
    ax.legend()
    buf = io.BytesIO(); plt.savefig(buf, format='png', bbox_inches='tight'); buf.seek(0)
    loss_image = base64.b64encode(buf.read()).decode(); buf.close(); plt.close(fig)

    #Confusion matrix
    cm_image = None
    if os.path.exists(cm_path):
        cm_data = pd.read_csv(cm_path)
        if not cm_data.empty:
            chosen = cm_data[cm_data["phase"] == "val"].iloc[-1] if "phase" in cm_data and not cm_data[cm_data["phase"] == "val"].empty else cm_data.iloc[-1]
            cm = eval(chosen["cm"]) if isinstance(chosen["cm"], str) else chosen["cm"]
            fig, ax = plt.subplots(figsize=(4,4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
            ax.set_xlabel("Predicted"); ax.set_ylabel("True")
            title_epoch = chosen['epoch'] if 'epoch' in chosen else ''
            ax.set_title(f"{title_prefix} Confusion Matrix (Epoch {title_epoch})".strip())
            buf = io.BytesIO(); plt.savefig(buf, format="png", bbox_inches="tight"); buf.seek(0)
            cm_image = base64.b64encode(buf.read()).decode(); buf.close(); plt.close(fig)

    return render_template(
        'index.html',
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        acc_image=acc_image,
        loss_image=loss_image,
        cm_image=cm_image,
        mode=mode,                    # hier wird Modus übertragen
    )

def render_live_table(csv_path, page_title="Live Predictions", mode="rf"):
    if not os.path.exists(csv_path):
        return "<h1>No predictions yet</h1>"
    df = pd.read_csv(csv_path)
    cols = [c for c in ['file','prediction','proba_tunnel','timestamp','true_label'] if c in df.columns]
    if not cols:
        cols = df.columns.tolist()
    live_table = df[cols]
    return render_template(
        "live.html",
        tables=[live_table.to_html(
            classes="table table-striped table-hover align-middle text-center shadow-sm rounded",
            index=False,
            border=0
        )],
        titles=[page_title],
        mode=mode,                       # <— pass mode
    )

# def render_traffic_chart(csv_path, title="Packets per Minute", mode="rf"):
#     if not os.path.exists(csv_path):
#         return "<h1>No predictions yet</h1>"
#     data = pd.read_csv(csv_path, parse_dates=['timestamp'])
#     data['timestamp'] = pd.to_datetime(data['timestamp'])
#     traffic = (
#         data.set_index('timestamp')
#         .resample('15T')
#         .size()
#     )
#     fig, ax = plt.subplots(figsize=(12,5))
#     ax.plot(traffic.index, traffic.values, marker='None', linestyle='-')
#     ax.set_title(title); ax.set_xlabel("Time"); ax.set_ylabel("Packet Count"); ax.grid(True)
#     ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m %H:%M'))
#     fig.autofmt_xdate(rotation=45)
#     buf = io.BytesIO(); plt.savefig(buf, format='png', bbox_inches='tight'); buf.seek(0)
#     chart_base64 = base64.b64encode(buf.read()).decode('utf-8'); buf.close(); plt.close()
#     return render_template('traffic_chart.html', chart=chart_base64, mode=mode)  # <— pass mode
def render_traffic_chart(csv_path, title="Packets per 15 minutes", mode="rf", freq="15T"):
    import os, io, base64
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for servers
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from flask import render_template

    # Early exits for missing/empty file
    if not os.path.exists(csv_path) or os.stat(csv_path).st_size == 0:
        return "<h1>No predictions yet</h1>"

    # Your sample row:
    # 192.168.100.100_2222_172.21.0.2_40410_2025-09-19T16-30-56.380554.csv,
    # ssh-not tunnelled,0.0,ssh-not-tunneled,2025-09-19 20:39:43.181401
    col_names = ["capture_file", "raw_label", "score", "label", "timestamp"]

    try:
        df = pd.read_csv(
            csv_path,
            header=None,                 # file has no header
            names=col_names,             # assign names
            engine="python",
            on_bad_lines="skip",         # skip malformed lines
            parse_dates=["timestamp"],   # parse last column as datetime
        )
    except Exception:
        # Fallback: try reading without fixed schema, then coerce last column to datetime
        df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")
        ts_col = df.columns[-1]
        df["timestamp"] = pd.to_datetime(df[ts_col], errors="coerce")

    # Keep only rows with valid timestamps
    df = df.dropna(subset=["timestamp"])
    if df.empty:
        return "<h1>No predictions yet</h1>"

    # Resample counts per window (default 15 minutes)
    traffic = (
        df.sort_values("timestamp")
        .set_index("timestamp")
        .resample(freq)
        .size()
    )

    if traffic.empty:
        return "<h1>No predictions yet</h1>"

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(traffic.index, traffic.values, marker=None, linestyle='-')
    ax.set_title(title or f"Packets per {freq}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Packet Count")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m %H:%M'))
    fig.autofmt_xdate(rotation=45)

    # Encode as base64 for template
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    chart_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    return render_template('traffic_chart.html', chart=chart_base64, mode=mode)


#Blueprints (stable prefixes)
rf_bp = Blueprint('rf', __name__, url_prefix='/rf')
cnn_bp = Blueprint('cnn', __name__, url_prefix='/cnn')

#RF routes
@rf_bp.route('/')
def rf_index():
    return render_dashboard(RF_METRICS_FILE, RF_CM_FILE, title_prefix="RF", mode="rf")

@rf_bp.route('/live')
def rf_live():
    return render_live_table(RF_CSV_FILE, page_title="RF Live Predictions", mode="rf")

@rf_bp.route('/traffic_chart')
def rf_traffic_chart():
    return render_traffic_chart(RF_CSV_FILE, title="RF Packets per Minute", mode="rf")

#CNN routes
@cnn_bp.route('/')
def cnn_index():
    return render_dashboard(CNN_METRICS_FILE, CNN_CM_FILE, title_prefix="CNN", mode="cnn")

@cnn_bp.route('/live')
def cnn_live():
    return render_live_table(CNN_CSV_FILE, page_title="CNN Live Predictions", mode="cnn")

@cnn_bp.route('/traffic_chart')
def cnn_traffic_chart():
    return render_traffic_chart(CNN_CSV_FILE, title="CNN Packets per Minute", mode="cnn")

app.register_blueprint(rf_bp)
app.register_blueprint(cnn_bp)


#Modusauswahl

#Redirect zum letzten ausgewählten Modus des Nutzers
@app.route('/')
def landing():
    mode = request.cookies.get('mode', 'rf')
    if mode not in ('rf', 'cnn'):
        mode = 'rf'
    return redirect(url_for(f"{mode}.{'rf_index' if mode=='rf' else 'cnn_index'}"))

#Hier Modus setzen und zur root von Modus wechseln
@app.route('/set_mode/<mode>')
def set_mode(mode):

    if mode not in ('rf', 'cnn'):
        mode = 'rf'
    resp = make_response(redirect(url_for(f"{mode}.{'rf_index' if mode=='rf' else 'cnn_index'}")))
    resp.set_cookie('mode', mode, max_age=30*24*3600, samesite='Lax')
    return resp

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
