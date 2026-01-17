#!/bin/bash

#Kopieren von create.sh in den openvpn container, hat in Dockerfile nicht funktioniert
docker cp create.sh openvpn-as:/profiles/create.sh

#Ausführen
docker exec -u root openvpn-as /bin/sh /profiles/create.sh

#Für den client, start.sh wurde entfernt wegen Datensatzproblemen
docker exec -it ssh-client-vpn ip route add 192.168.100.0/24 via 172.21.0.3

#Da bei jedem Neustart dieser Command ausgeführt werden muss, ist das hier ein Quickfix, weil Script immer laufen muss
docker exec -it ssh-server ip route add 172.21.0.0/16 via 192.168.100.2

#Initiiert die Vpn-Verbindung mit der neuen Profildatei im Hintergrund
#docker exec -it ssh-client-vpn openvpn --config /vpn/profiles/ssh-user-autologin.ovpn --daemon

# Für 100 laufen lassen
#docker exec -it ssh-client-vpn \
#  openvpn --config /vpn/profiles/ssh-user-autologin.ovpn \
#          --daemon \
#          --proto udp \
#          --tun-mtu 1400 \
#          --mssfix 1360 \
#          --txqueuelen 100 \
#          --data-ciphers AES-128-GCM \
#          --ncp-disable \
#          --pull-filter ignore "compress" \
#          --pull-filter ignore "comp-lzo"

# Für 100 laufen lassen
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

# Insgesamt 4 * laufen lassen, für 400 Einträge, im tatsächlichen Projekt 2500 * 4 = 10.000