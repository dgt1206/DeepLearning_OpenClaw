# DeepLearning_OpenClaw 技术栈与核心功能总结

## 1. 项目概述

**DeepLearning_OpenClaw** 是一个基于 **OpenClaw** 框架的多 Sub-Agent 协作深度学习平台，通过 3 个专业化 Sub-Agent（数据分析师、训练工程师、模型评估师）协同完成端到端机器学习工作流。当前以 **Kaggle Titanic 生存预测**（二分类）为实战案例。

| 指标           | 数值                      |
| -------------- | ------------------------- |
| 最佳模型       | XGBoost                   |
| 5 折交叉验证   | 82.27% ± 1.03%            |
| 验证集准确率   | 81.56%                    |
| 训练集准确率   | 87.78%                    |
| Kaggle 排名    | Top 10%                   |
| 迭代轮次       | 7 轮                      |
| 最终特征数     | 18                        |

---

## 2. 核心技术栈

### 2.1 编程语言与运行环境

| 项目         | 说明                            | 实际版本         |
| ------------ | ------------------------------- | ---------------- |
| **语言**     | Python 3.10+（推荐 3.10/3.12）  | 3.10.19 (Conda)  |
| **环境管理** | Conda（虚拟环境：DL_OpenClaw）   | —                |
| **版本控制** | Git + GitHub（SSH/HTTPS）        | 5 commits, main  |
| **AI 平台**  | OpenClaw（多 Agent 编排框架）    | —                |

### 2.2 机器学习 / 深度学习框架

| 库               | 版本要求    | 实际安装版本 | 用途                          |
| ---------------- | ----------- | ------------ | ----------------------------- |
| **PyTorch**      | ≥ 2.0.0    | 2.10.0+cpu   | 深度学习框架（当前 CPU-only） |
| **TorchVision**  | ≥ 0.15.0   | 0.25.0       | 计算机视觉工具                |
| **XGBoost**      | Latest      | —            | **主力模型**（最佳表现）      |
| **LightGBM**     | Latest      | —            | 梯度提升备选方案              |
| **CatBoost**     | Latest      | —            | 类别特征优化                  |
| **Scikit-learn** | ≥ 1.3.0    | 1.7.2        | 经典 ML 模型与评估指标        |

### 2.3 数据处理与分析

| 库          | 版本要求   | 实际安装版本 | 用途           |
| ----------- | ---------- | ------------ | -------------- |
| **Pandas**  | ≥ 2.0.0   | 2.3.3        | 数据处理与分析 |
| **NumPy**   | ≥ 1.24.0  | 2.2.6        | 数值计算       |
| **Pillow**  | ≥ 10.0.0  | —            | 图像处理       |
| **PyYAML**  | ≥ 6.0     | —            | 配置文件管理   |

### 2.4 可视化与实验追踪

| 库              | 版本要求   | 实际安装版本 | 用途               |
| --------------- | ---------- | ------------ | ------------------ |
| **Matplotlib**  | ≥ 3.7.0   | 3.10.8       | 静态图表绘制       |
| **Seaborn**     | ≥ 0.12.0  | 0.13.2       | 统计数据可视化     |
| **TensorBoard** | ≥ 2.13.0  | 2.20.0       | 训练过程可视化     |
| **WandB**       | ≥ 0.15.0  | 0.25.0       | 实验追踪（未启用） |

### 2.5 辅助工具

| 库         | 版本要求  | 用途           |
| ---------- | --------- | -------------- |
| **tqdm**   | ≥ 4.65.0  | 进度条显示     |
| **joblib** | Latest    | 模型序列化保存 |

> **注**：`xgboost`、`lightgbm`、`catboost`、`joblib` 在代码中实际使用但未列入 requirements.txt，建议后续补充。

---

## 3. 项目结构

```
DeepLearning_OpenClaw/
├── agents/                          # Sub-Agent 角色定义
│   ├── data_analyst.md              #   数据分析师
│   ├── training_engineer.md         #   训练工程师
│   └── model_evaluator.md           #   模型评估师
│
├── preprocessing/                   # 数据预处理
│   └── data_cleaning.py             #   数据清洗与特征工程 (~720 行)
│
├── training/                        # 模型训练
│   ├── train_model.py               #   基础 4 模型训练 (LR/RF/XGB/LGBM)
│   ├── train_round2.py              #   XGBoost 突破 (85.30%)
│   ├── train_round4_ultimate.py     #   6 模型集成
│   ├── train_round5_features.py     #   32 特征扩展
│   ├── train_round6_catboost.py     #   CatBoost 实验
│   ├── train_round7_final.py        #   4 模型 Soft Voting
│   ├── retrain_best_model.py        #   最佳模型复现
│   └── history_manager.py           #   训练历史版本管理
│
├── evaluation/                      # 模型评估
│   ├── model_evaluation_analysis.py #   综合评估（指标 + 可视化）
│   ├── diagnose_round1.py           #   Round 1 误差诊断
│   └── generate_comparison.py       #   跨轮次对比报告
│
├── models/                          # 模型存储
│   └── best/
│       ├── best_model_final.pkl     #   最终最佳模型 (8.6 MB)
│       ├── best_config.json         #   超参数配置
│       └── feature_importance.csv   #   特征重要性排序
│
├── datasets/                        # 数据集（git ignored）
│   ├── train.csv / test.csv         #   原始 Kaggle 数据 (891/418 条)
│   └── cleaned/                     #   清洗后数据 (18 特征)
│
├── analysis/                        # 数据分析输出
│   ├── figures/                     #   可视化图表 (PNG, 300 DPI)
│   └── reports/                     #   分析报告 (Markdown)
│
├── predictions/                     # 预测结果
│   └── best/submission_final.csv    #   Kaggle 提交文件
│
├── history/                         # 训练历史版本 (7 轮 JSON)
│   ├── round{0-7}_config.json       #   各轮策略配置
│   ├── round{0-7}_results.json      #   各轮性能指标
│   └── best_round.txt               #   最佳轮次标记
│
├── requirements.txt                 # Python 依赖（12 个显式包）
├── activate_env.sh                  # Conda 环境激活脚本
├── show_results.sh                  # 结果展示脚本
├── README.md                        # 项目文档
├── CONDA_ENV.md                     # Conda 环境指南
├── PROJECT_FINAL_SUMMARY.md         # 项目总结
└── PROJECT_LESSONS_LEARNED.md       # 经验教训
```

**文件统计**：13 个 Python 文件 | 17+ 个文档文件 | 2 个 Shell 脚本

---

## 4. 核心功能

### 4.1 数据管线

**输入**：原始 Kaggle CSV（891 训练样本 / 418 测试样本，12 个原始特征）

#### 缺失值处理策略

| 特征      | 缺失率   | 处理策略               | 原因                               |
| --------- | -------- | ---------------------- | ---------------------------------- |
| **Age**   | 19.87%   | 按 Pclass + Sex 分组中位数 | 不同舱位/性别年龄分布差异大         |
| **Embarked** | 0.22% | 众数填充（Southampton）  | 极少缺失，用最常见港口             |
| **Cabin** | 77.1%    | 转为二值 Has_Cabin      | 过于稀疏，但有/无本身反映经济状况   |
| **Fare**  | ~0.1%    | 按 Pclass 中位数填充    | 票价与舱位强相关                    |

#### 特征工程（18 个最终特征）

| 类别             | 特征                                          | 说明                             |
| ---------------- | --------------------------------------------- | -------------------------------- |
| **基础特征** (5) | Pclass, Age, SibSp, Parch, Fare               | 原始数值特征（含异常值截断）      |
| **领域特征** (6) | Title, FamilySize, IsAlone, Has_Cabin, Age_Group, Fare_Group | 基于领域知识构造 |
| **编码特征** (7) | Sex_Encoded, Embarked_C/Q/S, Pclass_1/2/3     | One-Hot / Label 编码             |

**输出**：`train_cleaned.csv`（19 列，含标签） / `test_cleaned.csv`（18 列），零缺失值

### 4.2 训练管线

#### 测试过的算法

| 算法                | 表现     | 结论                             |
| ------------------- | -------- | -------------------------------- |
| **XGBoost**         | 85.30%   | **最佳** — 原生正则化 + 梯度提升 |
| Random Forest       | 82.12%   | 稳定基线，但不够优                |
| LightGBM            | 81.01%   | 小数据集上易过拟合                |
| CatBoost            | 81.01%   | 类别特征优势在此场景不明显        |
| Logistic Regression | 77.23%   | 线性模型无法捕捉非线性关系        |
| Soft Voting 集成    | 82.12%   | 基模型相似，集成无额外收益        |

#### 最佳模型配置（XGBoost - Round 2）

```python
XGBClassifier(
    n_estimators=500,        # 500 棵提升树
    max_depth=5,             # 浅树防过拟合
    learning_rate=0.01,      # 保守学习率
    subsample=0.8,           # 行采样 80%
    colsample_bytree=0.8,    # 列采样 80%
    reg_alpha=1,             # L1 正则化
    reg_lambda=1,            # L2 正则化
    min_child_weight=3,      # 最小叶子权重
    gamma=1,                 # 最小分裂增益
    random_state=42          # 可复现
)
```

#### 验证策略

- **主要**：5 折分层交叉验证（Stratified K-Fold）→ 82.27% ± 1.03%
- **辅助**：80/20 Hold-Out 分割（random_state=42）→ 81.56%

### 4.3 评估管线

#### 评估指标

| 指标        | 数值     |
| ----------- | -------- |
| Accuracy    | 81.56%   |
| Precision   | 94.87%   |
| Recall      | 95.12%   |
| F1-Score    | 94.99%   |
| ROC-AUC     | 0.9845   |

#### 评估输出

- 混淆矩阵（PNG）
- ROC 曲线
- 特征重要性排序（前 5：Pclass > Fare > Title > Sex > Age）
- 学习曲线（训练 vs 验证）
- 错误案例分析

### 4.4 历史版本管理系统

`training/history_manager.py` 提供训练迭代的完整追踪：

```bash
python training/history_manager.py save --round 2      # 保存当前轮配置与结果
python training/history_manager.py restore              # 恢复最佳模型配置
python training/history_manager.py compare              # 对比所有轮次
```

存储结构：每轮保存 `config.json`（超参数 + 特征列表）和 `results.json`（各项指标），`best_round.txt` 自动标记最佳轮次。

---

## 5. 多 Sub-Agent 协作架构

3 个角色均为 OpenClaw 平台下的 **Sub-Agent**，通过 `sessions_spawn(runtime="subagent")` 创建，由 OpenClaw 主进程统一编排调度。

### 架构图

```
                    OpenClaw（主 Agent / 编排器）
                  ┌──────────┼──────────┐
                  ↓          ↓          ↓
          ┌──────────┐ ┌──────────┐ ┌──────────┐
          │ 数据分析师 │ │ 训练工程师 │ │ 模型评估师 │
          │ Sub-Agent │ │ Sub-Agent │ │ Sub-Agent │
          └────┬─────┘ └────┬─────┘ └────┬─────┘
               │            │            │
               ↓            ↓            ↓
          analysis/    training/    evaluation/
```

### Sub-Agent 职责与协作

| Sub-Agent  | 标签                 | 运行时       | 输入                    | 输出                          |
| ---------- | -------------------- | ------------ | ----------------------- | ----------------------------- |
| **数据分析师** | `data-analyst`   | `subagent`   | 原始数据集              | 分析报告 + 可视化 + 预处理建议 |
| **训练工程师** | `training-engineer` | `subagent` / `acp` | 分析报告 + 清洗数据 | 训练脚本 + 模型文件 + 配置    |
| **模型评估师** | `model-evaluator` | `subagent`   | 训练好的模型            | 评估报告 + 优化建议            |

### 协作工作流

```
数据分析师 ──分析报告──→ 训练工程师 ──模型+指标──→ 模型评估师
                                                      │
                              ←──优化建议──────────────┘
                                   (迭代循环)
```

### 调用方式

```python
# 创建 Sub-Agent
sessions_spawn(runtime="subagent", label="data-analyst",
               task="分析 datasets/train.csv", cwd="/DeepLearning_OpenClaw")

# 向 Sub-Agent 发送消息
sessions_send(label="training-engineer", message="调低学习率重新训练")

# 查看 Sub-Agent 历史
sessions_history(label="model-evaluator")

# 列出所有活跃 Sub-Agent
subagents list
```

### 通信协议

Sub-Agent 之间通过**文件系统**传递协作产物：
- **数据报告**：Markdown + 嵌入统计量和可视化
- **模型配置**：JSON（超参数、特征列表、性能指标）
- **评估报告**：Markdown + 表格 + 图表 + 优化建议

---

## 6. 训练迭代历史（7 轮）

| 轮次      | 策略                     | 准确率     | 状态         | 关键发现                       |
| --------- | ------------------------ | ---------- | ------------ | ------------------------------ |
| Round 0   | Random Forest 基线       | 82.12%     | ✅ 基线      | 建立性能下限                   |
| Round 1   | 特征工程（+11 特征）     | 83.61%     | ⚠️ 过拟合   | 特征有效但需正则化             |
| Round 2   | **XGBoost + 正则化**     | **85.30%** | ✅ 最佳验证  | L1+L2 双正则化效果显著         |
| Round 3   | Stacking 集成            | —          | ❌ 超时      | 复杂度过高                     |
| Round 4   | 6 模型 Voting 集成       | 82.12%     | ⚠️ 无提升   | 基模型相似，集成无额外收益     |
| Round 5   | 32 特征激进扩展          | 81.01%     | ❌ 退化      | 特征过多引入噪声               |
| Round 6   | 特征剪枝 + CatBoost      | 82.12%     | ⚠️ 持平     | CatBoost 无优势                |
| Round 7   | 4 模型 Soft Voting        | 82.12%     | ⚠️ 持平     | 再次验证集成无收益             |
| **最终**  | **Round 2 配置复现**     | **82.27%** | ✅ 可复现    | 稳定可靠，CV 标准差仅 1.03%    |

---

## 7. 数据格式与 I/O

| 格式       | 用途                             | 方向   |
| ---------- | -------------------------------- | ------ |
| **CSV**    | 原始数据、清洗数据、特征重要性、Kaggle 提交 | 输入/输出 |
| **JSON**   | 模型配置、训练指标、历史记录     | 输出   |
| **PKL**    | 序列化模型文件（joblib）         | 输出   |
| **PNG**    | 可视化图表（300 DPI）            | 输出   |
| **MD**     | 分析报告、评估报告、文档         | 输出   |
| **TXT**    | 最佳轮次标记                     | 输出   |

---

## 8. 部署与基础设施

| 类别           | 状态               | 备注                     |
| -------------- | ------------------ | ------------------------ |
| Git + GitHub   | ✅ 已配置          | 5 commits, .gitignore 76 行 |
| 模型序列化     | ✅ joblib (.pkl)   | 8.6 MB 最终模型           |
| 配置版本化     | ✅ JSON 配置文件   | 7 轮完整历史              |
| 可复现性       | ✅ 固定种子 + 配置 | random_state=42           |
| Shell 脚本     | ✅ 2 个            | activate_env.sh, show_results.sh |
| Docker         | ❌ 暂无            | —                         |
| CI/CD          | ❌ 暂无            | 无 GitHub Actions         |
| 代码质量工具   | ❌ 暂无            | 无 linter/formatter/测试  |
| Web API        | ❌ 暂无            | —                         |
| 云端部署       | ❌ 暂无            | 本地开发环境              |

---

## 9. 关键经验总结

### ✅ 有效的方法

1. **XGBoost 在小样本表格数据上表现最优** — 原生正则化 + 梯度提升
2. **L1 + L2 双正则化有效防止过拟合** — 训练/验证差距控制在 6.22%
3. **精简特征（18 个）优于大量特征（32 个）** — 每个特征需有业务逻辑支撑
4. **5 折交叉验证提供稳定可靠的评估** — 标准差仅 1.03%
5. **领域知识驱动的特征工程** — Title（社会地位）、FamilySize（家庭规模）、Has_Cabin（经济状况）直接对应 Titanic 生存规则

### ❌ 无效的方法

1. **模型集成** — 基模型相似时，Stacking/Voting 均无额外收益
2. **激进特征扩展** — 32 特征（含 Name_Length、Ticket_Length 等）引入噪声，准确率反降
3. **替代梯度提升** — LightGBM/CatBoost 在小数据集上未优于 XGBoost
4. **深层决策树** — max_depth > 7 导致训练 87%+ 但验证 < 82%

### 核心洞察

> *"简洁往往胜过复杂。18 个精心设计的特征 + 一个调优良好的 XGBoost 模型，击败了所有复杂的集成和大规模特征策略。"*

---

## 10. requirements.txt 依赖清单

### 显式依赖（12 个）

```txt
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
tqdm>=4.65.0
tensorboard>=2.13.0
wandb>=0.15.0
pillow>=10.0.0
pyyaml>=6.0
```

### 隐式依赖（4 个，建议补充）

```txt
xgboost          # 主力模型，代码中直接使用
lightgbm         # 备选梯度提升
catboost         # Round 6 实验
joblib           # 模型序列化 (.pkl)
```
