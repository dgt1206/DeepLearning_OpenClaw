# 🎯 Round 2 优化训练报告

**训练时间**: 2026-03-09 20:19 GMT+8  
**任务**: 特征筛选 + XGBoost 双重正则化

---

## 📊 历史对比

| 轮次 | 模型 | 准确率 | 状态 |
|------|------|--------|------|
| **Round 0** | Random Forest (基线) | 82.12% | ✅ 基准 |
| **Round 1** | Random Forest (特征工程) | 83.61% | ⚠️ 过拟合 (训练98.43%) |
| **Round 2 - 方案A** | Random Forest (优化版) | 82.83% | ❌ 退步 -0.78% |
| **Round 2 - 方案B** | **XGBoost** | **85.30%** | ✅ **最佳** |

---

## 🎉 最佳结果

### XGBoost (方案B)
- **5折CV准确率**: 85.30% ± 1.99%
- **提升幅度**: +1.69% (相比 Round 1)
- **距离目标**: 4.70% (目标90%)
- **过拟合控制**: ✅ 良好 (CV稳定)

### 各折得分
| 折数 | 准确率 |
|------|--------|
| Fold 1 | 84.92% |
| Fold 2 | **88.76%** ⭐ |
| Fold 3 | 82.58% |
| Fold 4 | 85.39% |
| Fold 5 | 84.83% |

**稳定性**: 最大-最小差 6.18% (可接受)

---

## 🔧 技术改进

### 1. 特征筛选
**移除 4 个冗余特征** (贡献 <1%):
- `Age_Group_Encoded`
- `High_Fare`
- `Title_Pclass_Match`
- `IsAlone`

**结果**: 30 → 26 特征 (减少 13%)

### 2. Random Forest 优化 (方案A)
```python
# 参数调整
n_estimators:      300  →  400
max_depth:         12   →  8      # 防过拟合 ⭐
min_samples_split: 5    →  10
min_samples_leaf:  2    →  4
max_features:      None →  'sqrt' # 特征采样 ⭐
class_weight:      None →  'balanced'
```

**结果**: 82.83% (未达预期，反而退步)

### 3. XGBoost 引入 (方案B) ⭐
```python
xgb_params = {
    'n_estimators': 500,
    'max_depth': 5,              # 浅树
    'learning_rate': 0.01,       # 小学习率
    'subsample': 0.8,            # 行采样
    'colsample_bytree': 0.8,     # 列采样
    'reg_alpha': 1,              # L1正则
    'reg_lambda': 1,             # L2正则
    'min_child_weight': 3,
    'gamma': 1
}
```

**结果**: **85.30%** ✅ (提升 1.69%)

### 4. 评估策略升级
- **替代**: 单次 train_test_split
- **使用**: 5折交叉验证
- **优势**: 更稳定、减少随机性影响

---

## 📈 性能指标

### XGBoost 详细报告

#### 分类报告
```
              precision    recall  f1-score   support

Not Survived     0.85       0.92      0.89       549
    Survived     0.85       0.75      0.80       342

    accuracy                         0.85       891
```

#### 混淆矩阵
```
                预测
              Not Survived  Survived
真实 Not Survived   504        45
     Survived        86       256
```

**关键洞察**:
- ✅ 高召回率 (92%) for Not Survived
- ⚠️ Survived 召回率偏低 (75%) - 改进空间

---

## 🔍 失败分析 (方案A)

### Random Forest 优化版为何退步？
1. **过度正则化**: max_depth 8 可能过于保守
2. **特征采样**: max_features='sqrt' 损失了部分关键特征组合
3. **class_weight='balanced'**: 可能破坏了原有的最优平衡

### 教训
- Random Forest 在此数据集已达上限 (~83%)
- 需要更强的泛化能力 → XGBoost 的正则化优势

---

## 🎯 状态评估

### 当前位置
- **Round 2 最佳**: 85.30%
- **距离目标**: 4.70% (目标90%)
- **阶段进度**: 第2轮/共3轮

### 决策
⏭️ **建议启动 Round 3 (模型融合)**

**理由**:
1. ✅ 准确率 ≥ 85% (达到继续条件)
2. ✅ XGBoost 显示强泛化能力
3. ✅ 过拟合已控制 (CV稳定)
4. 📊 单模型提升趋缓，需要融合策略

---

## 📂 输出文件

1. ✅ `models/best_model_round2.pkl` - XGBoost 模型
2. ✅ `evaluation/round2_results.json` - 详细指标
3. ✅ `evaluation/round2_comparison.png` - RF vs XGBoost 对比
4. ✅ `evaluation/round2_feature_importance.png` - 特征重要性
5. ✅ `predictions/submission_round2.csv` - 测试集预测

---

## 🚀 下一步计划 (Round 3)

### 模型融合策略
1. **Stacking**:
   - Base: XGBoost + Random Forest + Logistic Regression
   - Meta: XGBoost (轻量版)

2. **Voting**:
   - Soft voting 多模型集成
   - 权重优化

3. **Blending**:
   - 训练集二次划分
   - 避免信息泄漏

### 预期提升
- **保守**: 86-87%
- **乐观**: 88-89%
- **冲刺目标**: 90%+

---

## 📌 总结

### ✅ 成功点
1. 成功引入 XGBoost，提升 1.69%
2. 5折CV评估更稳定
3. 特征筛选减少冗余
4. 过拟合控制良好

### ⚠️ 待改进
1. Survived 类别召回率 (75%)
2. 特征工程天花板 (可能已达上限)
3. 单模型提升空间有限

### 🎯 关键洞察
**模型选择比超参数调整更重要**: XGBoost 的原生正则化能力 > Random Forest 的后期优化

---

**训练完成时间**: 2026-03-09 20:19  
**状态**: ⏭️ 准备启动 Round 3  
**报告生成**: training/training_report_round2.md
