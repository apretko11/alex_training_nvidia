#!/usr/bin/env bash
set -e

ENV_NAME="alex_training"
PYTHON_VERSION="3.10"
REQ_FILE="alex_training_env.txt"
MINICONDA_DIR="$HOME/miniconda3"

echo "=== Checking for conda ==="

if ! command -v conda &> /dev/null && [ ! -d "$MINICONDA_DIR" ]; then
    echo "Conda not found. Installing Miniconda..."

    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $MINICONDA_DIR
    rm miniconda.sh

    echo "Initializing conda..."
    $MINICONDA_DIR/bin/conda init
    source ~/.bashrc
else
    echo "Conda already installed."
fi

# Ensure conda is usable in non-interactive script
source $MINICONDA_DIR/etc/profile.d/conda.sh

echo "=== Accepting Conda ToS (if needed) ==="
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

echo "=== Removing existing env if present ==="
if conda env list | grep -q "^$ENV_NAME"; then
    conda remove -n $ENV_NAME --all -y
fi

echo "=== Creating conda environment: $ENV_NAME ==="
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

conda activate $ENV_NAME

echo "=== Installing PyTorch (CUDA 12.1 wheel) ==="
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121

echo "=== Checking torchrun ==="
if ! command -v torchrun &> /dev/null; then
    echo "torchrun not found; creating wrapper in conda env..."
    cat > "$CONDA_PREFIX/bin/torchrun" <<'EOF'
#!/usr/bin/env bash
exec python -m torch.distributed.run "$@"
EOF
    chmod +x "$CONDA_PREFIX/bin/torchrun"
fi

echo "=== Installing remaining requirements from $REQ_FILE ==="
pip install -r $REQ_FILE

echo "=== Installing local LLaMA-Factory editable ==="
pip install -e llamafac/LLaMA-Factory

echo "=== Verifying CUDA ==="
python - <<EOF2
import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("CUDA version (torch):", torch.version.cuda)
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
EOF2

echo "=== Environment setup complete ==="
