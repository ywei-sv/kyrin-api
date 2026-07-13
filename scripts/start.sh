#!/bin/bash
# Start kyrin-api with .env variables loaded
cd "$(dirname "$0")/.."
set -o allexport
source .env
set +o allexport
exec .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 5271
