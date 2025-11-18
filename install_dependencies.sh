#!/bin/bash
# Install Python dependencies with correct order to avoid conflicts

set -e

echo "📦 Installing dependencies for YouTube Shorts Generator..."

# Check if in venv
if [ -z "$VIRTUAL_ENV" ]; then
    echo "❌ Please activate virtual environment first:"
    echo "   source venv/bin/activate"
    exit 1
fi

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install NumPy first (base dependency)
echo "📦 Installing NumPy..."
pip install "numpy>=1.23.0,<2.0.0"

# Install PyTorch with CUDA support
echo "🔥 Installing PyTorch with CUDA 11.8..."
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install from requirements.txt (without torch/numpy)
echo "📦 Installing other dependencies..."
pip install -r requirements.txt

# Verify installation
echo ""
echo "🔍 Verifying installation..."
python << 'EOF'
import sys
print(f"Python: {sys.version}")

try:
    import numpy as np
    print(f"✅ NumPy: {np.__version__}")
except:
    print("❌ NumPy failed")

try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"✅ CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"❌ PyTorch: {e}")

try:
    from faster_whisper import WhisperModel
    print("✅ Faster Whisper: OK")
except Exception as e:
    print(f"❌ Faster Whisper: {e}")

try:
    import transformers
    print(f"✅ Transformers: {transformers.__version__}")
except:
    print("❌ Transformers failed")

try:
    import ollama
    print("✅ Ollama client: OK")
except:
    print("❌ Ollama failed")

print("\n🎉 Installation verification complete!")
EOF

echo ""
echo "✅ All dependencies installed successfully!"

