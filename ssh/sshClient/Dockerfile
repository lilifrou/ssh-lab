FROM python:3.14.0b1-slim

# Kopieren des private keys
COPY id_rsa /root/.ssh/id_rsa
RUN chmod 600 /root/.ssh/id_rsa

# Tools installieren
RUN apt-get update && apt-get install -y openssh-client iproute2 curl && \
    pip install --no-cache-dir scapy && \
    apt-get install -y curl && \
    apt-get clean


# Work Directory setzen
WORKDIR /app



# Starte Container mit Script
CMD ["sh", "-c", "python $TRAFFIC_LABEL.py"]
