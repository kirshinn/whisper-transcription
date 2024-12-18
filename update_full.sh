#!/bin/bash
git pull origin main
docker compose down
docker system prune -y
docker compose up --build -d
