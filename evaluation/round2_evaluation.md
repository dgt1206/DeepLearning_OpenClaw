# Round 2 模型评估与优化方案

## 📊 当前状态总览

| 指标 | Round 0 (基线) | Round 1 (特征工程) | 目标 | 差距 |
|------|---------------|-------------------|------|------|
| **验证集准确率** | 82.12% | 81.56% | 90% | **❌ -0.56%** |
| **CV均值** | - | 83.61% ± 2.54% | 90% | -6.39% |
| **过拟合差距** | - | **7.12%** | <3% | 严重超标 |
| **模型稳定性** | - | CV波动 6.74% | <3% | ⚠️ 不稳定 |

**核心问题**: Round 1 性能反而下降,暴露出特征冗余和严重过拟合。

---

## 🔍 Round 1 失败原因深度分析

### 1. 严重过拟合 (7.12% 差距) ⚠️⚠️⚠️

**诊断数据**:
- 训练集准确率: **90.84%** ± 0.77%
- 验证集准确率: **83.73%** ± 3.11%
- **差距**: 7.12% (远超 3% 健康阈值)

**根本原因**:
```python
RandomForestClassifier(
    n_estimators=300,  # ✅ 合理
    max_depth=12,      # ❌ 过深!树过度拟合训练数据
    min_samples_split=5,   # ⚠️ 不够严格
    min_samples_leaf=2,    # ⚠️ 允许过小的叶子
    # 缺少 max_features 限制 ❌
)
```

**学习曲线分析**:
- 训练样本达到600+后,训练集准确率持续>90%
- 验证集准确率在80-84%震荡,未继续提升
- 典型的高方差(high variance)问题

**改进方向**:
1. 降低 `max_depth` (12 → 8-10)
2. 提高 `min_samples_split` (5 → 10)
3. 提高 `min_samples_leaf` (2 → 4)
4. **添加 `max_features='sqrt'`** 进行特征采样

---

### 2. 特征冗余 (8个低贡献特征) ⚠️⚠️

**低贡献特征列表** (<1% importance):

| 特征名 | 重要性 | 贡献度 | 建议 |
|--------|--------|--------|------|
| `Age_Group_Encoded` | 0.0097 | 0.97% | **删除** |
| `Parch` | 0.0078 | 0.78% | 保留(原始特征) |
| `High_Fare` | 0.0071 | 0.71% | **删除** |
| `Title_Pclass_Match` | 0.0071 | 0.71% | **删除** |
| `Embarked_C` | 0.0070 | 0.70% | 保留(One-Hot) |
| `Pclass_2` | 0.0068 | 0.68% | 保留(One-Hot) |
| `IsAlone` | 0.0049 | 0.49% | **删除** |
| `Embarked_Q` | 0.0047 | 0.47% | 保留(One-Hot) |

**候选删除特征** (4个):
1. `Age_Group_Encoded` - 与 `Age` 和 `Age_Category` 重复
2. `High_Fare` - 与 `Fare` 和 `Fare_Quartile` 重复
3. `Title_Pclass_Match` - 手工规则特征,效果不佳
4. `IsAlone` - 与 `FamilySize` 完全冗余

**预期收益**: 移除噪音特征,降低模型复杂度,减少过拟合

---

### 3. 超参数过激 ⚠️

**Round 1 参数问题**:
- `max_depth=12` → 对于891条数据过深
- 缺少 `max_features` → 每次分裂考虑所有28个特征,容易记忆训练集

**Random Forest 最佳实践**:
```python
# Titanic 数据集 (n=891, features≈30) 推荐参数
{
    'n_estimators': 300-500,        # 足够多的树
    'max_depth': 8-10,              # 浅树防过拟合
    'min_samples_split': 10-15,     # 更保守的分裂
    'min_samples_leaf': 4-5,        # 叶子节点最小样本
    'max_features': 'sqrt',         # 特征采样 (≈√28 = 5)
    'class_weight': 'balanced'      # 处理类别不均衡
}
```

---

### 4. 模型不稳定 (CV波动 6.74%) ⚠️

**5折交叉验证结果**:
```
Fold 1: 82.68%
Fold 2: 81.46%
Fold 3: 87.64%  ← 异常高
Fold 4: 80.90%  ← 异常低
Fold 5: 85.39%
Mean: 83.61% ± 2.54%
Range: 6.74%
```

**原因分析**:
- Fold间差距过大 → 模型对数据划分敏感
- 说明模型泛化能力弱,记忆了训练集特定模式
- 需要更强的正则化

---

## ✅ Round 1 成功部分 (保留)

### 1. 高质量特征 (Top 10)

| 排名 | 特征 | 重要性 | 解读 |
|------|------|--------|------|
| 1 | **Sex_Pclass** | 12.48% | ⭐⭐⭐ 性别与舱位交互 |
| 2 | **Title_Encoded** | 11.82% | ⭐⭐⭐ 社会地位 |
| 3 | **Age_Fare** | 9.13% | ⭐⭐⭐ 年龄与票价交互 |
| 4 | **Sex_Encoded** | 8.26% | ⭐⭐⭐ 性别 (女性优先) |
| 5 | **Fare** | 7.84% | ⭐⭐ 财富指标 |
| 6 | **Age** | 7.23% | ⭐⭐ 年龄 (儿童优先) |
| 7 | **Woman_or_Child** | 5.74% | ⭐⭐ 救生规则 |
| 8 | **Title_Pclass** | 4.89% | ⭐ 称谓舱位交互 |
| 9 | **FamilySize_Pclass** | 3.43% | ⭐ 家庭舱位交互 |
| 10 | **Pclass** | 3.36% | ⭐ 舱位等级 |

**结论**: 交互特征贡献了 **前3名** (12.48% + 9.13% + 4.89% = 26.5%)

### 2. CV均值高于单次验证

- CV均值: **83.61%**
- 单次验证: 81.56%
- 差距: +2.05%

**说明**: 模型有潜力,只是单次 train_test_split 运气不好或过拟合严重

---

## 🎯 Round 2 优化方案 (按优先级排序)

### 方案 A: 特征筛选 + 正则化增强 ⭐⭐⭐⭐⭐

**优先级**: 最高  
**预期提升**: 83.61% → **86-87%** (+2.5-3.5%)  
**实施难度**: 低 (修改代码 <50行)

#### 核心策略

##### 1. 移除4个冗余特征

```python
# 删除列表
features_to_remove = [
    'Age_Group_Encoded',    # 与 Age/Age_Category 重复
    'High_Fare',            # 与 Fare/Fare_Quartile 重复
    'Title_Pclass_Match',   # 手工规则失败
    'IsAlone'               # 与 FamilySize 完全冗余
]

X = X.drop(columns=features_to_remove)
# 特征数: 28 → 24
```

##### 2. 强化正则化参数

```python
RandomForestClassifier(
    n_estimators=400,           # 300→400 (更多树平滑预测)
    max_depth=8,                # 12→8 ❗关键改动
    min_samples_split=10,       # 5→10
    min_samples_leaf=4,         # 2→4
    max_features='sqrt',        # 新增 (≈√24=5 features)
    class_weight='balanced',    # 处理类别不均衡
    random_state=42,
    n_jobs=-1
)
```

##### 3. 使用 StratifiedKFold 替代单次划分

```python
# 不再使用 train_test_split
# 改用5折CV的平均预测
from sklearn.model_selection import StratifiedKFold, cross_val_predict

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred_cv = cross_val_predict(model, X, y, cv=skf)
cv_accuracy = accuracy_score(y, y_pred_cv)
```

#### 预期效果

| 指标 | Round 1 | 方案A预期 | 改善 |
|------|---------|----------|------|
| CV均值 | 83.61% | **86-87%** | +2.5-3.5% |
| 过拟合差距 | 7.12% | **<4%** | ✅ |
| CV波动 | 6.74% | **<4%** | ✅ |
| 特征数 | 28 | 24 | -4 |

---

### 方案 B: 切换到 XGBoost ⭐⭐⭐⭐⭐

**优先级**: 最高 (可与方案A并行)  
**预期提升**: 83.61% → **87-89%** (+3.5-5.5%)  
**实施难度**: 中 (需安装 xgboost)

#### 为什么选择 XGBoost?

1. **更好的正则化**: L1 (alpha) + L2 (lambda) 双重正则
2. **自动特征选择**: 内置特征重要性筛选
3. **抗过拟合**: 早停 (early stopping) 机制
4. **Titanic 竞赛验证**: Top 10% 解决方案多用 XGBoost/LightGBM

#### 推荐参数

```python
import xgboost as xgb

model = xgb.XGBClassifier(
    n_estimators=500,          # 迭代次数
    max_depth=5,               # 浅树 (3-6)
    learning_rate=0.01,        # 小学习率 (0.01-0.05)
    subsample=0.8,             # 行采样 80%
    colsample_bytree=0.8,      # 列采样 80%
    reg_alpha=1,               # L1正则
    reg_lambda=1,              # L2正则
    min_child_weight=3,        # 叶子节点最小权重
    gamma=1,                   # 分裂最小损失降低
    scale_pos_weight=1,        # 类别权重
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42,
    early_stopping_rounds=50   # 早停
)

# 训练时监控验证集
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=50
)
```

#### 特点对比

| 特性 | Random Forest | XGBoost |
|------|--------------|---------|
| 正则化 | 弱 (树结构) | 强 (L1+L2+Gamma) |
| 早停机制 | ❌ | ✅ |
| 类别不均衡 | class_weight | scale_pos_weight |
| 训练速度 | 快 | 中等 |
| 可解释性 | 高 | 中 |
| Titanic 表现 | 85% | 87-89% |

#### 预期效果

| 指标 | Round 1 (RF) | 方案B (XGB) | 改善 |
|------|-------------|------------|------|
| CV均值 | 83.61% | **87-89%** | +3.5-5.5% |
| 过拟合差距 | 7.12% | **<3%** | ✅✅ |
| 训练时间 | 快 | 中等 | - |

---

### 方案 C: 模型融合 (Stacking Ensemble) ⭐⭐⭐⭐

**优先级**: 第二阶段 (当单模型≥86%后)  
**预期提升**: 87% → **88-90%** (+1-3%)  
**实施难度**: 高

#### 融合架构

```
Level 0 (Base Models):
┌─────────────────┐
│ Random Forest   │ → 预测概率1
│ (方案A优化版)   │
└─────────────────┘
┌─────────────────┐
│ XGBoost         │ → 预测概率2
│ (方案B)         │
└─────────────────┘
┌─────────────────┐
│ LightGBM        │ → 预测概率3
│ (新增)          │
└─────────────────┘
         ↓
Level 1 (Meta Model):
┌─────────────────┐
│ Logistic        │ → 最终预测
│ Regression      │
└─────────────────┘
```

#### 实现代码

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb

# Base models
rf_optimized = RandomForestClassifier(**best_rf_params)
xgb_model = xgb.XGBClassifier(**best_xgb_params)
lgb_model = lgb.LGBMClassifier(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.01,
    num_leaves=31,
    random_state=42
)

# Stacking
stacking_model = StackingClassifier(
    estimators=[
        ('rf', rf_optimized),
        ('xgb', xgb_model),
        ('lgb', lgb_model)
    ],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5,
    passthrough=False  # 只用预测概率,不用原始特征
)

stacking_model.fit(X_train, y_train)
```

#### 预期效果

- **多样性**: RF(树集成) + XGB(Boosting) + LGB(Boosting)
- **稳定性**: 5折CV的元模型训练
- **性能**: 顶级选手必用技术,Titanic Top 5% 标配

---

### 方案 D: 高级特征工程 ⭐⭐⭐

**优先级**: 第二阶段 (如方案A/B不够)  
**预期提升**: +1-2%  
**实施难度**: 中

#### 新特征设计

##### 1. Deck (甲板层)

```python
def extract_deck(cabin):
    if pd.isna(cabin):
        return 'Unknown'
    return cabin[0]  # A/B/C/D/E/F/G

train_df['Deck'] = train_df['Cabin'].apply(extract_deck)

# Deck 生存率 (历史数据):
# A: 46.7%  (高层,逃生较易)
# B: 74.5%  (贵族区,优先救援)
# C: 59.3%
# D: 75.0%
# E: 75.0%
# F: 61.5%
# G: 50.0%
# Unknown: 29.9%  (无舱位信息,多为3等舱)
```

##### 2. Name_Length (姓名长度)

```python
train_df['Name_Length'] = train_df['Name'].str.len()

# 假设: 贵族姓名更长 (带头衔、中间名)
# 示例:
# "Braund, Mr. Owen Harris" (24字符)
# "Cumings, Mrs. John Bradley (Florence Briggs Thayer)" (54字符) ← 贵族
```

##### 3. Ticket_Frequency (票号重复次数)

```python
ticket_counts = train_df['Ticket'].value_counts()
train_df['Ticket_Frequency'] = train_df['Ticket'].map(ticket_counts)

# 票号重复 → 团体/家庭购票
# Frequency高 → 可能整体获救或整体遇难
```

##### 4. Age × Sex 交互 (细粒度)

```python
# 替代粗粒度的 Woman_or_Child
train_df['Age_Sex_Category'] = pd.cut(
    train_df['Age'],
    bins=[0, 5, 12, 18, 35, 60, 100],
    labels=['Infant', 'Child', 'Teen', 'Young', 'Adult', 'Senior']
).astype(str) + '_' + train_df['Sex']

# 生成: Child_male, Child_female, Adult_male, Adult_female...
# One-Hot编码后12个特征
```

#### 预期贡献

| 特征 | 预期重要性 | 原因 |
|------|----------|------|
| Deck | 3-5% | 反映舱位位置,与逃生难度相关 |
| Name_Length | 1-2% | 间接反映社会地位 |
| Ticket_Frequency | 2-3% | 家庭/团体效应 |
| Age_Sex_Category | 2-4% | 比简单交互更精细 |

---

## 📋 Round 2 执行计划

### 阶段1: 快速迭代 (目标: 86-87%)

**时间**: 1-2小时  
**策略**: 方案A + 方案B 并行测试

```bash
# 1. 方案A: 特征筛选 + 正则化
python training/train_optimized_v2.py --mode feature_pruning

# 2. 方案B: XGBoost
python training/train_optimized_v2.py --mode xgboost

# 3. 对比结果,选择最佳
```

**预期输出**:
- `models/best_model_round2_rf.pkl` (方案A)
- `models/best_model_round2_xgb.pkl` (方案B)
- `evaluation/round2_comparison.csv`

### 阶段2: 集成冲刺 (目标: 88-90%)

**时间**: 2-3小时  
**策略**: 方案C (Stacking) 或 方案D (高级特征)

**条件**: 当阶段1达到 86% 后启动

```bash
# 如果阶段1 < 86%
→ 启动方案D (高级特征工程)

# 如果阶段1 ≥ 86%
→ 启动方案C (模型融合)
```

---

## 🎯 成功标准

### 必达指标 (Round 2)

- [x] ✅ 明确指出 Round 1 的 3+ 个关键问题
- [x] ✅ 提出至少 4 个优化方案
- [x] ✅ 优先级排序清晰 (⭐数量)
- [x] ✅ 预期提升幅度合理 (2-5%)
- [ ] ⏳ 训练脚本可直接运行

### 性能目标

| 阶段 | 目标准确率 | 策略 |
|------|----------|------|
| Round 2.1 | **86-87%** | 方案A/B |
| Round 2.2 | **88-89%** | 方案C/D |
| Round 3 | **90%+** | 超参数调优 + 数据增强 |

---

## 📊 方案对比矩阵

| 方案 | 优先级 | 预期提升 | 实施难度 | 推荐使用场景 |
|------|--------|---------|---------|------------|
| **A: 特征筛选+正则化** | ⭐⭐⭐⭐⭐ | +2.5-3.5% | 低 | 快速验证,稳定改进 |
| **B: XGBoost** | ⭐⭐⭐⭐⭐ | +3.5-5.5% | 中 | 追求性能,有GPU更佳 |
| **C: 模型融合** | ⭐⭐⭐⭐ | +1-3% | 高 | 最后冲刺,已达86% |
| **D: 高级特征** | ⭐⭐⭐ | +1-2% | 中 | 方案B不够时补充 |

### 推荐组合

#### 保守策略 (稳求86%)
```
A(特征筛选) → A结果 ≥ 86% ? 完成 : D(高级特征)
```

#### 激进策略 (冲刺88%)
```
B(XGBoost) → B结果 ≥ 87% ? C(Stacking) : A+D
```

#### 平衡策略 (推荐⭐)
```
A + B 并行 → 选最佳 (≥86%) → C(Stacking) → 目标88-90%
```

---

## 📝 下一步行动

1. ✅ **查看诊断结果**
   ```bash
   cat evaluation/round1_diagnosis.json
   open evaluation/round1_diagnosis.png  # 查看可视化
   ```

2. ⏳ **运行 Round 2 训练脚本**
   ```bash
   python training/train_optimized_v2.py
   ```

3. ⏳ **评估 Round 2 结果**
   ```bash
   python evaluation/evaluate_round2.py
   ```

4. ⏳ **决策下一轮策略**
   - 如果 ≥ 86% → 启动 Stacking
   - 如果 < 86% → 分析失败,调整参数

---

## 🔗 相关文件

- **诊断数据**: `evaluation/round1_diagnosis.json`
- **诊断可视化**: `evaluation/round1_diagnosis.png`
- **Round 1 模型**: `models/best_model_round1.pkl`
- **Round 1 数据**: `datasets/cleaned/train_enhanced_v1.csv`
- **Round 2 脚本**: `training/train_optimized_v2.py` (即将生成)

---

**报告生成时间**: 2026-03-09  
**分析师**: AI Model Evaluator (Round 2)  
**状态**: ✅ 诊断完成,方案就绪
