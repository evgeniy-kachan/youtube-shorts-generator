#!/bin/bash

# YouTube Shorts Generator - Installation Script

set -e

echo "🎬 Installing YouTube Shorts Generator..."

# Check Python version
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.10"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Python 3.10+ is required. Current version: $PYTHON_VERSION"
    exit 1
fi

echo "✅ Python version: $PYTHON_VERSION"

# Check CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  WARNING: nvidia-smi not found. GPU acceleration may not work."
else
    echo "✅ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name --format=csv,noheader
fi

# Check FFmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "❌ FFmpeg is not installed. Please install it:"
    echo "   sudo apt install ffmpeg"
    exit 1
fi

echo "✅ FFmpeg installed"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
echo "📦 Installing Python dependencies..."

# Install NumPy first
pip install "numpy>=1.23.0,<2.0.0"

# Install PyTorch with CUDA
echo "🔥 Installing PyTorch with CUDA support..."
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cat > .env << 'EOF'
HOST=0.0.0.0
PORT=8000

DEEPSEEK_API_KEY=replace_with_your_key
DEEPSEEK_MODEL=deepseek-reasoner
DEEPSEEK_BASE_URL=https://api.deepseek.com

WHISPER_MODEL=large-v3
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=float16

MAX_VIDEO_DURATION=7200
TEMP_DIR=./temp
OUTPUT_DIR=./output

CUDA_VISIBLE_DEVICES=0
TTS_ENABLE_MARKUP=true
TTS_MARKUP_MAX_TOKENS=200
EOF
    echo "✅ .env file created"
else
    echo "✅ .env file already exists"
fi

echo "ℹ️  Make sure to update DEEPSEEK_API_KEY in .env with your personal key."

# Download Whisper model (will be downloaded on first use)
echo "📥 Whisper model will be downloaded automatically on first use"

# Install frontend dependencies
if [ -d "frontend" ]; then
    echo "📦 Installing frontend dependencies..."
    cd frontend
    if command -v npm &> /dev/null; then
        npm install
        echo "✅ Frontend dependencies installed"
    else
        echo "⚠️  npm not found. Skipping frontend installation."
        echo "   To install frontend, install Node.js and run: cd frontend && npm install"
    fi
    cd ..
fi

# Create directories
mkdir -p temp output

echo ""
echo "✅ Installation complete!"
echo ""
echo "🚀 To start the server, run:"
echo "   ./run.sh"
echo ""
echo "📖 Or manually:"
echo "   source venv/bin/activate"
echo "   python backend/main.py"
echo ""

