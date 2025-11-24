#!/bin/bash

set -e

echo "🎬 Starting YouTube Shorts Generator (Docker)"

# Start backend
echo "🚀 Starting backend server..."
cd /app
export PYTHONPATH=/app
exec python3 backend/main.py
