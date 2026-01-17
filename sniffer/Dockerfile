FROM alpine:latest

# Install dependencies
RUN apk update && apk add --no-cache \
    tshark \
    libpcap \
    tcpdump \
    iproute2 \
    python3 \
    py3-pip \
    scapy

# Copy sniffer script
COPY sniffer.py /sniffer/sniffer.py

WORKDIR /sniffer

# Run sniffer script
CMD ["python3", "sniffer.py"]


