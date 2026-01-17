#!/bin/bash

#OpenVpn
docker compose -f openvpn/docker-compose.yml build
docker compose -f openvpn/docker-compose.yml up -d

#SSH server und client
docker compose -f ssh/docker-compose-vpn.yml build
docker compose -f ssh/docker-compose-vpn.yml up -d

sh openvpn/profile.sh

#sniffer
docker compose -f sniffer/docker-compose.yml build
docker compose -f sniffer/docker-compose.yml up -d


#CNN
docker compose -f cnn/docker-compose.yml build
docker compose -f cnn/docker-compose.yml up -d

#Random Forest
docker compose -f cnn/docker-compose-forest.yml build
docker compose -f cnn/docker-compose-forest.yml up -d

#Ui
docker compose -f ui/docker-compose.yml build
docker compose -f ui/docker-compose.yml up -d


echo "All containers newly built and started! VPN connection initiated!"
