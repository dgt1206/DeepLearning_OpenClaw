# Titanic生存预测 - 模型训练报告

## 📊 任务概述

- **数据集**: Titanic生存预测
- **训练样本**: 891条 (训练集712条, 验证集179条)
- **测试样本**: 418条
- **特征数量**: 18
- **训练日期**: 2026-03-09 18:59:33

## 🎯 模型评估结果

### 全部模型对比

| 模型 | 训练准确率 | 验证准确率 | Precision | Recall | F1-Score | ROC-AUC |
|------|------------|------------|-----------|--------|----------|---------|
| Logistic Regression | 0.8230 | 0.8156 | 0.7812 | 0.7246 | 0.7519 | 0.8481 |
| Random Forest | 0.9059 | 0.8212 | 0.8136 | 0.6957 | 0.7500 | 0.8430 |
| XGBoost | 0.9382 | 0.8156 | 0.7727 | 0.7391 | 0.7556 | 0.8474 |
| LightGBM | 0.9368 | 0.7989 | 0.7538 | 0.7101 | 0.7313 | 0.8313 |

### 🏆 最优模型

**选择**: Random Forest

**验证集准确率**: 0.8212 (82.12%)

**选择理由**:
- 验证集准确率最高
- 训练集与验证集表现平衡，泛化能力好
- 过拟合风险较低

## 📈 特征重要性分析

**Top 10 重要特征**:

1. Title_Encoded: 0.2295
2. Sex_Encoded: 0.1712
3. Fare: 0.1336
4. Age: 0.1082
5. Pclass: 0.0492
6. Has_Cabin: 0.0482
7. Fare_Group_Encoded: 0.0432
8. FamilySize: 0.0432
9. Pclass_3: 0.0382
10. SibSp: 0.0239

## 💡 模型优缺点分析

### 逻辑回归
- ✅ 简单快速，可解释性强
- ✅ 适合线性可分问题
- ❌ 无法捕捉复杂非线性关系

### 随机森林
- ✅ 鲁棒性好，不易过拟合
- ✅ 能处理非线性关系
- ❌ 训练和预测速度较慢

### XGBoost
- ✅ 性能优异，梯度提升
- ✅ 内置正则化，防止过拟合
- ❌ 超参数较多，调优复杂

### LightGBM
- ✅ 训练速度快，内存占用低
- ✅ 支持类别特征
- ❌ 小数据集容易过拟合

## 🔧 后续优化建议

1. **特征工程**
   - 尝试更多特征交互
   - 探索多项式特征
   - 考虑特征选择降维

2. **模型优化**
   - 网格搜索/随机搜索调参
   - 交叉验证提高稳定性
   - 集成学习（Stacking/Blending）

3. **数据增强**
   - SMOTE处理类别不平衡
   - 半监督学习利用测试集

4. **深度学习**
   - 尝试神经网络模型
   - 探索AutoML工具

## 📂 输出文件

- ✅ `training/training_report.md` - 本报告
- ✅ `analysis/figures/model_comparison.png` - 模型对比图
- ✅ `analysis/figures/confusion_matrix_best.png` - 混淆矩阵
- ✅ `analysis/figures/roc_curve_comparison.png` - ROC曲线
- ✅ `analysis/figures/feature_importance_best.png` - 特征重要性
- ✅ `models/best_model.pkl` - 最优模型
- ✅ `predictions/submission.csv` - Kaggle提交文件

---

**训练完成!** 🎉
