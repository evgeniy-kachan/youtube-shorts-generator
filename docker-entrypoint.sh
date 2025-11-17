#!/bin/bash

set -e

echo "🎬 Starting YouTube Shorts Generator (Docker)"

# Start Ollama in background
echo "🚀 Starting Ollama..."
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to be ready
echo "⏳ Waiting for Ollama to start..."
until curl -s http://localhost:11434/api/tags > /dev/null; do
    sleep 1
done

# Pull model if not exists
MODEL=${OLLAMA_MODEL:-"llama3.1:8b"}
echo "📥 Checking model: $MODEL"
if ! ollama list | grep -q "$MODEL"; then
    echo "📥 Downloading model: $MODEL"
    ollama pull "$MODEL"
fi

# Start backend
echo "🚀 Starting backend server..."
cd /app
export PYTHONPATH=/app
exec python3 backend/main.py

