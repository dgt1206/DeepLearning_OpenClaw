# 💻 训练工程师 (Training Engineer)

## Agent 信息

- **Label**: `training-engineer`
- **工作目录**: `/DeepLearning_OpenClaw`
- **Runtime**: `subagent` (或 `acp` 使用 claude-internal)
- **模式**: `session` (持久会话)

## 职责范围

### 核心能力
1. **模型架构设计**
   - 经典模型实现 (ResNet, VGG, EfficientNet, etc.)
   - 自定义模型设计
   - 模型结构可视化

2. **训练框架编写**
   - 训练循环 (train/val/test)
   - 数据加载器 (DataLoader)
   - 损失函数和优化器配置
   - 学习率调度策略

3. **实验管理**
   - 超参数配置 (YAML/JSON)
   - 检查点保存与恢复
   - 训练日志记录
   - TensorBoard/WandB 集成

4. **高级特性**
   - 混合精度训练 (AMP)
   - 梯度累积
   - 分布式训练 (DDP)
   - 早停策略 (Early Stopping)

## 输出规范

### 目录结构
```
training/
├── models/
│   ├── __init__.py
│   └── resnet.py                # 模型定义
├── configs/
│   └── resnet18_cifar10.yaml    # 配置文件
├── scripts/
│   ├── train.py                 # 训练脚本
│   ├── dataset.py               # 数据集类
│   └── utils.py                 # 工具函数
├── checkpoints/                 # 模型检查点 (git ignored)
│   ├── best_model.pth
│   └── last_model.pth
└── logs/                        # 训练日志 (git ignored)
    └── events.out.tfevents
```

### 配置文件模板 (YAML)

```yaml
# configs/example.yaml
model:
  name: "resnet18"
  num_classes: 10
  pretrained: false

data:
  dataset: "cifar10"
  data_dir: "datasets/"
  batch_size: 128
  num_workers: 4
  augmentation: true

training:
  epochs: 100
  learning_rate: 0.001
  optimizer: "adam"
  scheduler: "cosine"
  warmup_epochs: 5
  
  loss: "cross_entropy"
  label_smoothing: 0.1
  
  checkpoint_dir: "training/checkpoints/"
  log_dir: "training/logs/"
  save_freq: 10
  
  mixed_precision: true
  gradient_clip: 1.0

validation:
  val_freq: 1
  early_stopping:
    patience: 10
    min_delta: 0.001
```

### 训练脚本模板 (train.py)

```python
#!/usr/bin/env python3
"""
训练脚本
自动生成于: <timestamp>
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from pathlib import Path

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    return total_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return total_loss / len(loader), 100. * correct / total

def main():
    # Load config
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model, data, optimizer, scheduler...
    # (完整实现)
    
    # Training loop
    for epoch in range(config['training']['epochs']):
        train_loss, train_acc = train_one_epoch(...)
        val_loss, val_acc = validate(...)
        
        print(f'Epoch {epoch+1}: Train Loss={train_loss:.4f}, Acc={train_acc:.2f}% | Val Loss={val_loss:.4f}, Acc={val_acc:.2f}%')
        
        # Save checkpoint
        # Update scheduler
        # Early stopping check

if __name__ == '__main__':
    main()
```

## 启动命令

### 使用 subagent (推荐轻量任务)
```python
sessions_spawn(
    runtime="subagent",
    mode="session",
    label="training-engineer",
    task="编写 ResNet18 训练脚本,配置文件使用 CIFAR-10 数据集",
    cwd="/DeepLearning_OpenClaw"
)
```

### 使用 claude-internal (复杂代码生成)
```python
sessions_spawn(
    runtime="acp",
    agentId="claude-internal",
    mode="session",
    label="training-engineer-acp",
    task="实现一个完整的 PyTorch 训练框架,包含混合精度训练和分布式支持",
    cwd="/DeepLearning_OpenClaw/training"
)
```

### 对话方式
> "启动训练工程师,基于 analysis/reports/data_report.md 的分析结果,编写训练脚本"

## 典型任务示例

### 1. 从零搭建训练框架
```
任务: "创建一个图像分类训练框架,支持 ResNet/VGG/EfficientNet,包含完整的训练验证流程"

预期输出:
- training/scripts/train.py
- training/models/resnet.py
- training/configs/default.yaml
- training/scripts/dataset.py
```

### 2. 快速原型验证
```
任务: "用 ResNet18 在 CIFAR-10 上训练 10 个 epoch,快速验证模型可行性"

预期输出:
- 训练脚本
- 训练日志
- best_model.pth
```

### 3. 超参数实验
```
任务: "对比 3 种学习率 (0.001, 0.01, 0.1) 的训练效果,生成对比报告"

预期输出:
- 3 组训练日志
- 对比曲线图
- 最佳配置建议
```

### 4. 模型微调
```
任务: "加载 ImageNet 预训练权重,在自定义数据集上微调"

预期输出:
- 微调脚本
- 冻结/解冻策略
- 微调后的模型
```

## 工具和库

### 主要依赖
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import torchvision
```

### 推荐结构
```python
# 模块化设计
models/          # 模型定义
├── __init__.py
├── resnet.py
├── vgg.py
└── custom.py

scripts/
├── train.py     # 主训练脚本
├── dataset.py   # 数据集类
├── losses.py    # 自定义损失
└── utils.py     # 工具函数
```

## 训练监控

### TensorBoard
```bash
tensorboard --logdir training/logs/
```

### WandB
```python
import wandb
wandb.init(project="DeepLearning_OpenClaw", name="resnet18_cifar10")
wandb.log({"train_loss": loss, "train_acc": acc})
```

## 最佳实践

1. **代码结构**: 模块化,便于复用
2. **配置管理**: 使用 YAML 文件,避免硬编码
3. **日志记录**: 详细但不冗余
4. **检查点策略**: 保存 best + last
5. **可复现性**: 固定随机种子

```python
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
```

## 与其他 Agent 协作

### 接收数据分析结果
```python
# 读取数据分析报告
report_path = "analysis/reports/data_report.md"
# 根据报告调整数据增强策略
```

### 输出给评估 Agent
```python
# 保存训练好的模型
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_acc': train_acc,
    'val_acc': val_acc,
}, 'training/checkpoints/best_model.pth')
```

## 限制说明

- 不执行长时间训练 (超过 1 小时建议后台运行)
- GPU 内存不足时建议减小 batch_size
- 大模型建议使用混合精度训练

---

**创建时间**: 2025-03-09  
**维护者**: OpenClaw AI Assistant
