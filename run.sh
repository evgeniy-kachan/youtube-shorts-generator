#!/bin/bash

# YouTube Shorts Generator - Startup Script

set -e

echo "🎬 Starting YouTube Shorts Generator..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run install.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "⚠️  Ollama is not running. Starting Ollama..."
    ollama serve > /dev/null 2>&1 &
    sleep 5
fi

# Check if model is available
MODEL=${OLLAMA_MODEL:-"llama3.1:8b"}
if ! ollama list | grep -q "$MODEL"; then
    echo "📥 Downloading model: $MODEL"
    ollama pull "$MODEL"
fi

# Create necessary directories
mkdir -p temp output

# Start backend
echo "🚀 Starting backend server..."
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python backend/main.py

