#!/bin/bash
set -e

echo "Building database from API..."
uv run python /app/scripts/build_db.py

echo "Starting Streamlit..."
exec uv run streamlit run /app/han_tyumi/main.py --server.port=8080 --server.address=0.0.0.0
