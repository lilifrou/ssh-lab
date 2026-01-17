#!/bin/bash

#SSH server und client
docker compose -f ssh/docker-compose.yml build
docker compose -f ssh/docker-compose.yml up -d

#sniffer
docker compose -f sniffer/docker-compose.yml build
docker compose -f sniffer/docker-compose.yml up -d

#CNN container
docker compose -f cnn/docker-compose.yml build
docker compose -f cnn/docker-compose.yml up -d

#Random Forest
docker compose -f cnn/docker-compose-forest.yml build
docker compose -f cnn/docker-compose-forest.yml up -d

#Ui
docker compose -f ui/docker-compose.yml build
docker compose -f ui/docker-compose.yml up -d

echo "All containers newly built and started!"
