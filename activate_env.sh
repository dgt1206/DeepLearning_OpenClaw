#!/bin/bash
# Conda 环境快速激活脚本

export PATH="$HOME/miniconda3/bin:$PATH"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate DL_OpenClaw

echo "✅ Conda 环境已激活: DL_OpenClaw"
echo "🐍 Python: $(python --version)"
echo "📦 PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "💻 Device: CPU"
echo ""
echo "使用方法:"
echo "  source /DeepLearning_OpenClaw/activate_env.sh"
