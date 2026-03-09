# 🧠 DeepLearning_OpenClaw

基于 OpenClaw 的深度学习多 Agent 协作平台

## 📖 项目简介

这是一个由 **OpenClaw** 管理的深度学习工作流自动化项目，通过 3 个专业化 Sub-Agent 协同工作：

- 🔍 **数据分析师 (Data Analyst)** - 数据探索与质量分析
- 💻 **训练工程师 (Training Engineer)** - 模型训练代码编写与执行
- 📊 **模型评估师 (Model Evaluator)** - 性能评估与优化建议

## 🏗️ 项目结构

```
DeepLearning_OpenClaw/
├── datasets/              # 数据集目录 (git ignored)
├── analysis/              # 数据分析输出
│   ├── reports/          # 分析报告
│   └── visualizations/   # 可视化图表
├── training/              # 训练相关
│   ├── models/           # 模型定义
│   ├── configs/          # 配置文件
│   ├── scripts/          # 训练脚本
│   ├── checkpoints/      # 模型检查点 (git ignored)
│   └── logs/             # 训练日志 (git ignored)
├── evaluation/            # 评估结果
│   ├── reports/          # 评估报告
│   └── metrics/          # 性能指标
└── agents/                # Sub-Agent 配置文档
    ├── data_analyst.md
    ├── training_engineer.md
    └── model_evaluator.md
```

## 🤖 Sub-Agent 说明

### 1️⃣ 数据分析师 (Data Analyst)

**职责**: 数据集加载、统计分析、可视化、质量报告

**输出位置**: `analysis/`

**使用方法**:
```bash
# 通过 OpenClaw 调用
sessions_spawn(
    runtime="subagent",
    mode="session",
    label="data-analyst",
    task="分析 datasets/cifar10 数据集",
    cwd="/DeepLearning_OpenClaw"
)
```

**典型任务**:
- 数据集探索（shape, dtype, 缺失值）
- 统计特征分析（分布、相关性）
- 数据可视化（分布图、热力图）
- 数据质量报告生成

---

### 2️⃣ 训练工程师 (Training Engineer)

**职责**: 模型架构设计、训练代码编写、实验管理

**输出位置**: `training/`

**使用方法**:
```bash
sessions_spawn(
    runtime="subagent",
    mode="session",
    label="training-engineer",
    task="基于 ResNet18 训练图像分类模型",
    cwd="/DeepLearning_OpenClaw"
)
```

**典型任务**:
- 编写训练脚本 (train.py)
- 定义模型架构 (models/)
- 配置超参数 (configs/)
- 训练日志记录
- 模型检查点保存

---

### 3️⃣ 模型评估师 (Model Evaluator)

**职责**: 模型性能评估、错误分析、优化建议

**输出位置**: `evaluation/`

**使用方法**:
```bash
sessions_spawn(
    runtime="subagent",
    mode="session",
    label="model-evaluator",
    task="评估 training/checkpoints/best_model.pth 性能",
    cwd="/DeepLearning_OpenClaw"
)
```

**典型任务**:
- 计算评估指标 (Acc/F1/AUC/mAP)
- 混淆矩阵分析
- 错误案例分析
- 模型复杂度分析
- 优化建议生成

---

## 🔄 典型工作流

### 示例：训练一个图像分类模型

```bash
# 1. 数据分析
→ Agent: Data Analyst
→ 输入: datasets/my_dataset
→ 输出: analysis/reports/data_report.md

# 2. 模型训练
→ Agent: Training Engineer
→ 输入: 数据分析报告
→ 输出: training/scripts/train.py + 训练好的模型

# 3. 性能评估
→ Agent: Model Evaluator
→ 输入: 训练好的模型
→ 输出: evaluation/reports/evaluation_report.md

# 4. 迭代优化
→ 根据评估建议调整超参数，重新训练
```

---

## 🛠️ 环境配置

### 依赖安装

```bash
pip install -r requirements.txt
```

### 主要依赖
- PyTorch / TensorFlow
- NumPy, Pandas, Matplotlib
- scikit-learn
- tqdm, wandb (可选)

---

## 📊 监控与管理

### 查看运行中的 Agent

```bash
# 通过 OpenClaw 查看
subagents list
```

### 与 Agent 交互

```bash
# 发送消息
sessions_send(label="data-analyst", message="检查数据平衡性")

# 查看历史
sessions_history(label="training-engineer")
```

---

## 📝 开发规范

### Git Commit 规范

- `feat:` 新功能
- `fix:` Bug 修复
- `docs:` 文档更新
- `refactor:` 代码重构
- `perf:` 性能优化
- `test:` 测试相关
- `chore:` 构建/工具链

### 代码风格

- Python: PEP 8
- 使用 black 格式化
- 使用 type hints

---

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License

---

## 🔗 相关链接

- [OpenClaw 文档](https://docs.openclaw.ai)
- [PyTorch 文档](https://pytorch.org/docs/)
- [GitHub 仓库](https://github.com/dgt1206/DeepLearning_OpenClaw)

---

**由 OpenClaw AI 助手自动生成和管理** 🦞
