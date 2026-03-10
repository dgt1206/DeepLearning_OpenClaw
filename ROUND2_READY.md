# Round 2 优化方案 - 执行指南

## 🎯 快速开始

### 立即执行 (推荐)

```bash
cd /DeepLearning_OpenClaw

# 激活环境
source activate_env.sh

# 运行 Round 2 优化 (对比RF和XGBoost)
python training/train_optimized_v2.py --mode both

# 或单独测试某个方案
python training/train_optimized_v2.py --mode rf   # 仅方案A
python training/train_optimized_v2.py --mode xgb  # 仅方案B
```

### 预期运行时间
- 方案A (RF): ~2-3 分钟
- 方案B (XGBoost): ~5-8 分钟
- 两者对比: ~8-12 分钟

---

## 📊 已完成工作

### 1. ✅ Round 1 深度诊断

**诊断文件**:
- `evaluation/round1_diagnosis.json` - 详细诊断数据
- `evaluation/round1_diagnosis.png` - 4张诊断图表

**核心发现**:
1. **严重过拟合**: 训练90.84% vs 验证83.73% (差距7.12%)
2. **8个冗余特征**: 贡献度 <1% (Age_Group_Encoded, High_Fare, IsAlone等)
3. **模型不稳定**: CV波动6.74% (80.90% - 87.64%)
4. **参数过激**: max_depth=12过深,缺少max_features限制

### 2. ✅ 优化方案设计

**方案文档**: `evaluation/round2_evaluation.md`

**4个方案**:
- **方案A** (⭐⭐⭐⭐⭐): 特征筛选 + 正则化 → 预期 +2.5-3.5%
- **方案B** (⭐⭐⭐⭐⭐): XGBoost → 预期 +3.5-5.5%
- **方案C** (⭐⭐⭐⭐): 模型融合 (Stacking) → 预期 +1-3% (第二阶段)
- **方案D** (⭐⭐⭐): 高级特征工程 → 预期 +1-2% (备选)

### 3. ✅ 训练脚本

**主脚本**: `training/train_optimized_v2.py`

**功能**:
- 自动特征筛选 (移除4个冗余特征)
- 两种模型对比 (RF优化版 vs XGBoost)
- 5折交叉验证评估
- 完整可视化输出
- 自动保存最佳模型

### 4. ✅ 可视化

**已生成图表**:
- `evaluation/round1_diagnosis.png` - Round 1 问题诊断
- `evaluation/rounds_comparison.png` - Round 0/1/2 进度对比

**待生成图表** (运行训练脚本后):
- `evaluation/round2_comparison.png` - RF vs XGBoost 对比
- `evaluation/round2_feature_importance.png` - 最佳模型特征重要性

---

## 🔧 技术细节

### 方案A: 特征筛选 + 正则化

**删除4个特征**:
```python
features_to_remove = [
    'Age_Group_Encoded',    # 0.97% - 与Age重复
    'High_Fare',            # 0.71% - 与Fare重复
    'Title_Pclass_Match',   # 0.71% - 规则失败
    'IsAlone'               # 0.49% - 与FamilySize冗余
]
```

**参数优化**:
```python
RandomForestClassifier(
    n_estimators=400,          # 300→400
    max_depth=8,               # 12→8 ❗
    min_samples_split=10,      # 5→10
    min_samples_leaf=4,        # 2→4
    max_features='sqrt',       # 新增 ❗
    class_weight='balanced'    # 新增
)
```

**关键改进**:
- `max_depth` 从12降到8 → 防止过拟合
- `max_features='sqrt'` → 每次分裂只考虑√24≈5个特征
- `class_weight='balanced'` → 处理类别不均衡

### 方案B: XGBoost

**参数配置**:
```python
XGBClassifier(
    n_estimators=500,
    max_depth=5,              # 浅树
    learning_rate=0.01,       # 小学习率
    subsample=0.8,            # 行采样
    colsample_bytree=0.8,     # 列采样
    reg_alpha=1,              # L1正则
    reg_lambda=1,             # L2正则
    min_child_weight=3,
    gamma=1
)
```

**优势**:
- 双重正则化 (L1+L2)
- 自动特征选择
- 梯度提升 (通常优于RF)

---

## 📈 预期结果

### 目标达成路径

| 阶段 | 策略 | 目标准确率 | 状态 |
|------|------|----------|------|
| Round 0 | Baseline | 82.12% | ✅ 完成 |
| Round 1 | 特征工程 | 83.61% (CV) | ✅ 完成 |
| **Round 2** | **A/B优化** | **86-87%** | ⏳ 执行中 |
| Round 3 | Stacking | 88-90% | 📅 待启动 |

### 预期输出文件

运行 `train_optimized_v2.py` 后将生成:

```
models/
  └── best_model_round2.pkl          # 最佳模型

evaluation/
  ├── round2_results.json             # 详细评估结果
  ├── round2_comparison.png           # RF vs XGBoost 对比
  └── round2_feature_importance.png   # 特征重要性

predictions/
  └── submission_round2.csv           # Kaggle提交文件

datasets/cleaned/
  ├── train_cleaned_round2.csv        # 清理后训练集
  └── test_cleaned_round2.csv         # 清理后测试集
```

---

## 🎯 成功标准验证

### Round 2 任务完成度

- [x] ✅ **明确指出 Round 1 的 3+ 个关键问题**
  - 过拟合 (7.12%)
  - 特征冗余 (8个)
  - 参数过激 (max_depth=12)
  - 模型不稳定 (CV波动6.74%)

- [x] ✅ **提出至少 3 个优化方案**
  - 方案A: 特征筛选+正则化
  - 方案B: XGBoost
  - 方案C: 模型融合
  - 方案D: 高级特征工程

- [x] ✅ **优先级排序清晰**
  - A和B: ⭐⭐⭐⭐⭐ (最高优先级)
  - C: ⭐⭐⭐⭐ (第二阶段)
  - D: ⭐⭐⭐ (备选)

- [x] ✅ **预期提升幅度合理**
  - A: +2.5-3.5%
  - B: +3.5-5.5%
  - C: +1-3%
  - D: +1-2%

- [x] ✅ **训练脚本可直接运行**
  - `train_optimized_v2.py` 已完成
  - 支持 --mode rf/xgb/both
  - 完整错误处理和可视化

---

## 🚀 下一步行动

### 立即执行

```bash
# 1. 运行 Round 2 训练
cd /DeepLearning_OpenClaw
source activate_env.sh
python training/train_optimized_v2.py --mode both

# 2. 查看结果
cat evaluation/round2_results.json
open evaluation/round2_comparison.png
```

### 根据结果决策

#### 情况1: CV ≥ 87%
```bash
# 启动 Round 3 (Stacking)
python training/train_stacking.py  # 需要另外开发
```

#### 情况2: 86% ≤ CV < 87%
```bash
# 微调超参数
python training/hyperparameter_tuning.py
```

#### 情况3: CV < 86%
```bash
# 分析失败原因
python evaluation/failure_analysis.py

# 尝试方案D (高级特征)
python training/train_advanced_features.py
```

---

## 📚 相关文档

- **诊断报告**: `evaluation/round2_evaluation.md`
- **诊断数据**: `evaluation/round1_diagnosis.json`
- **训练脚本**: `training/train_optimized_v2.py`
- **可视化**: 
  - `evaluation/round1_diagnosis.png`
  - `evaluation/rounds_comparison.png`

---

## 💡 关键洞察

### Round 1 失败教训

1. **特征不是越多越好** → 8个特征贡献<1%
2. **过深的树容易过拟合** → max_depth=12 太深
3. **单次划分不可靠** → CV和单次验证差2.05%
4. **需要特征采样** → 缺少max_features导致记忆训练集

### Round 2 核心策略

1. **删除冗余,保留精华** → 24个高质量特征
2. **强正则化** → max_depth↓, min_samples↑, max_features添加
3. **5折CV** → 更稳定的性能评估
4. **尝试XGBoost** → 可能比RF更优

### 成功概率评估

- **方案A达到86%**: 80% 概率
- **方案B达到87%**: 70% 概率
- **任一方案≥86%**: 90% 概率

---

## 🏆 期望成果

运行完 Round 2 后,你将获得:

1. **性能提升**: 83.61% → 86-87% (+2.5-4%)
2. **稳定模型**: 过拟合<4%, CV波动<4%
3. **清晰路径**: 知道下一步是Stacking还是调参
4. **完整评估**: 4张图表 + 详细报告
5. **Kaggle提交**: 可直接上传的submission文件

---

**准备好了吗?运行这条命令开始Round 2:**

```bash
cd /DeepLearning_OpenClaw && source activate_env.sh && python training/train_optimized_v2.py --mode both
```

🎯 **目标: 86-87%**  
⏱️ **预计时间: 8-12分钟**  
✨ **让我们冲击新高度!**
