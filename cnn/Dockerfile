FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime


WORKDIR /app

# Abhängigkeiten kopieren
COPY requirements.txt ./

# Abhängigkeiten installieren
RUN pip install --no-cache-dir -r requirements.txt

# gesamten Code kopieren
COPY . .

# Logs-Ordner für Live-Prediction
#RUN mkdir -p /sniffer/logs /data

# CNN-Skript 
CMD ["python", "cnn.py"]
