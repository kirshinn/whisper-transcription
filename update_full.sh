#!/bin/bash
git reset --hard HEAD
git pull origin main
docker compose down
docker system prune -a -f
docker compose up --build -d
