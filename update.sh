#!/bin/bash
git reset --hard HEAD
git pull origin main
docker compose restart
