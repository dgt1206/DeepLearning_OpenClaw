# 🐍 Conda Environment Setup

## Environment Info
- **Name**: `DL_OpenClaw`
- **Python**: 3.10.19
- **PyTorch**: 2.10.0+cpu (CPU-only version)
- **Location**: `~/miniconda3/envs/DL_OpenClaw`

## Installed Packages
- PyTorch 2.10.0 (CPU)
- TorchVision 0.25.0
- TorchAudio 2.10.0
- NumPy 2.2.6
- Pandas 2.3.3
- Matplotlib 3.10.8
- Seaborn 0.13.2
- Scikit-learn 1.7.2
- TensorBoard 2.20.0
- WandB 0.25.0
- TQDM, PyYAML, Pillow

## Quick Activation

### Method 1: Use the activation script (Recommended)
```bash
source /DeepLearning_OpenClaw/activate_env.sh
```

### Method 2: Manual activation
```bash
export PATH="$HOME/miniconda3/bin:$PATH"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate DL_OpenClaw
```

## Verification
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

## Common Commands

### List environments
```bash
conda env list
```

### Deactivate environment
```bash
conda deactivate
```

### Install additional packages
```bash
conda activate DL_OpenClaw
pip install <package_name>
```

### Export environment
```bash
conda activate DL_OpenClaw
conda env export > environment.yml
```

### Remove environment (if needed)
```bash
conda env remove -n DL_OpenClaw
```

## Notes
- ⚠️ This is a **CPU-only** environment (no CUDA)
- For GPU support, you need to:
  1. Install NVIDIA drivers
  2. Install CUDA toolkit
  3. Reinstall PyTorch with CUDA:
     ```bash
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
     ```

## Sub-Agent Usage
When spawning sub-agents, ensure they use this environment:
```python
sessions_spawn(
    runtime="subagent",
    mode="session",
    label="data-analyst",
    task="分析数据集",
    cwd="/DeepLearning_OpenClaw",
    # Add environment activation in the task
)
```

---
**Environment created**: 2025-03-09  
**Last updated**: 2025-03-09
