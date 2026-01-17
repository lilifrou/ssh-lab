from scapy.all import sniff, TCP, IP
from datetime import datetime
import csv
import os

# Logs-Ordner
log_dir = "logs/"
os.makedirs(log_dir, exist_ok=True)

MAX_CONNECTIONS = 5000
finished_connections = 0

# Verbindungsbuffer
connections = {}  # key: (src, sport, dst, dport) -> list of packets
start_times = {}  # key -> startzeit als String

def get_connection_key(ip, tcp):
    return (ip.src, tcp.sport, ip.dst, tcp.dport)

def save_connection_to_file(key, packets):
    src, sport, dst, dport = key
    start_time = start_times.get(key, datetime.now().isoformat()).replace(":", "-")
    filename = f"{src}_{sport}_{dst}_{dport}_{start_time}.csv"
    filepath = os.path.join(log_dir, filename)

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "timestamp", "src_port", "dst_port",
            "flags", "seq", "ack", "window", "payload_len"
        ])
        writer.writeheader()
        writer.writerows(packets)

def log_packet(pkt):
    global finished_connections   # <--- wichtig!
    if pkt.haslayer(IP) and pkt.haslayer(TCP):
        now = datetime.now().isoformat()
        ip = pkt[IP]
        tcp = pkt[TCP]
        key = get_connection_key(ip, tcp)

        packet_data = {
            "timestamp": now,
            "src_port": tcp.sport,
            "dst_port": tcp.dport,
            "flags": str(tcp.flags),
            "seq": tcp.seq,
            "ack": tcp.ack,
            "window": tcp.window,
            "payload_len": len(tcp.payload)
        }

        if key not in connections:
            connections[key] = []
            start_times[key] = now

        connections[key].append(packet_data)
        print(f"[Tracking] {key} Flags={tcp.flags}")


    # Verbindung beenden bei FIN oder RST
        if "F" in str(tcp.flags) or "R" in str(tcp.flags):
            save_connection_to_file(key, connections[key])
            del connections[key]
            del start_times[key]
            finished_connections += 1   # <--- Zähler hoch
            if finished_connections >= MAX_CONNECTIONS:
                print("Max erreicht")
                os._exit(0)

# Auf eth0 lauschen
sniff(filter="tcp", iface="eth0", prn=log_packet, store=0)
