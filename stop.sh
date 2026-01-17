#!/bin/bash

#SSH server und client
docker compose -f ssh/docker-compose.yml down
docker compose -f ssh/docker-compose-vpn.yml down

#sniffer
docker compose -f sniffer/docker-compose.yml down

#CNN
docker compose -f cnn/docker-compose.yml down

#Random Forest
docker compose -f cnn/docker-compose-forest.yml down

#Vpn
docker compose -f openvpn/docker-compose.yml down

#Ui
docker compose -f ui/docker-compose.yml down

echo "All containers stopped!"
