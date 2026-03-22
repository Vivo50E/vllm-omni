#!/bin/bash
# Environment setup script for benchmark

set -e

echo "============================================================"
echo " Setting up environment for Qwen3-Omni benchmark"
echo "============================================================"

# Check if we're in the right directory
if [ ! -f "../../setup.py" ] && [ ! -f "../../pyproject.toml" ]; then
    echo "ERROR: Please run this script from benchmarks/qwen3-omni/"
    exit 1
fi

cd ../..

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: ${PYTHON_VERSION}"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install vllm-omni
echo "Installing vllm-omni from current branch..."
pip install -e .

# Install benchmark dependencies
echo "Installing benchmark dependencies..."
pip install aiohttp numpy tqdm matplotlib

# Verify installation
echo ""
echo "Verifying installation..."
which vllm-omni
vllm-omni --version || echo "vllm-omni installed (no --version flag)"

echo ""
echo "============================================================"
echo " Environment setup complete!"
echo "============================================================"
echo ""
echo "To run benchmarks, use:"
echo "  source venv/bin/activate"
echo "  cd benchmarks/qwen3-omni"
echo "  bash benchmark_three_modes.sh"
echo ""
