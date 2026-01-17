Das Repository für die Gruppe sshustlers.

# Für server:

Falls komische Fehler auftreten, config löschen und neu erstellen lassen

# Für tunneling:

Falls tunneling nicht funktioniert, überprüfen, ob Eintrag  AllowTcpForwarding auf yes ist, in sshd_config vom ssh-server.

# Für die Vpn:

2 docker networks:

network_1: 192.168.100.0/24

network_2: 172.21.0.0/16

#Für ssh-server:
Das ist wichtig für die VPN:
ip route add 172.21.0.0/16 via 192.168.100.2 (!ist nicht automatisch, Skript macht server kaputt!)
-> ALSO: Falls der Server nicht antwortet, trotz funktionierender VPN Verbindun, dann manuell diesen Befehl ausführen.

#Für ssh-client-vpn:
Das ist wichtig für die VPN:
ip route add 192.168.100.0/24 via 172.21.0.3  (ist automatisch)

Um ssh-client-vpn mit server zu bauen:
- docker compose -f docker-compose-vpn.yml build --no-cache

Um zu starten:
- docker compose -f docker-compose-vpn.yml up -d

Um sniffer-container mit vpn zu benutzen:

network_mode: "container:ssh-client-vpn"

Sonst:

network_mode: "container:ssh-client"

Um ein neues profil ssh-user.ovpn zu erstellen:
docker exec -u root openvpn-as /bin/sh /profiles/create.sh

Um das neue Profil im client zu benutzen:
docker exec -it ssh-client-vpn openvpn /vpn/profiles/ssh-user-autologin.ovpn

**Automatisiert mit diesem Skript:**

/openvpn/profile.sh

Um einen besseren Datensatz zu generieren, der unabhängiger von VPN Konfigurationen ist, diese beiden Befehle in profile.sh abwechselnd auskommentieren

# Für die erste Hälfte laufen lassen
docker exec -it ssh-client-vpn \
  openvpn --config /vpn/profiles/ssh-user-autologin.ovpn \
          --daemon \
          --proto udp \
          --tun-mtu 1400 \
          --mssfix 1360 \
          --txqueuelen 100 \
          --data-ciphers AES-128-GCM \
          --ncp-disable \
          --pull-filter ignore "compress" \
          --pull-filter ignore "comp-lzo"

# Für eine Hälfte laufen lassen
docker exec -it ssh-client-vpn \
openvpn --config /vpn/profiles/ssh-user-autologin.ovpn \
--daemon \
--proto tcp \
--tun-mtu 1500 \
--mssfix 0 \
--txqueuelen 1000 \
--data-ciphers AES-256-GCM \
--ncp-disable \
--pull-filter ignore "compress" \
--pull-filter ignore "comp-lzo"


# docker external:

volumes:
- profiles
- sniffer_captures

networks:
- network_1
- network_2

#Anleitung zum  Starten von client, server mit VPN-Verbindung

1. openvpn container starten
2. in ssh/ : docker compose -f docker-compose-vpn.yml up -d
3. in openvpn/ : sh profile.sh
Für Sniffer: Der namespace bei network_mode in der docker-compose.yml muss geändert werden von "ssh-client zu ssh-client-vpn"

Um alles automatisch zu starten: Ausführen von /sshustlers/start-vpn.sh

#Anleitung zum Starten von client, ohne VPN-Verbindung

1. ssh/ : docker compose -f docker-compose.yml up -d
Für Sniffer: Der namespace bei network_mode in der docker-compose.yml muss geändert werden von "ssh-client-vpn zu ssh-client"

Um alles automatisch zu starten: Ausführen von /sshustlers/start.sh

----------

Um alles zu stoppen, unabhängig ob mit VPN oder ohne: /sshustlers/stop.sh

----------

Für die Präsentation wurde ein extra Datensatz gefertigt, aufgrund von technischen Problemen mit dem Server

Da sind die 4 Datensätze nur 200 Sessions groß, statt 5000.

----------

#Für die Steuerung der Skripte ab jetzt:

in .env:

Plain-Skript:
TRAFFIC_LABEL: ssh-not-tunneled

Tunneling-Skript: 
TRAFFIC_LABEL: ssh-tunneled 

#Auf dem Server
Unsere finale Umgebung is in /home/bp/final/sshustlers/
