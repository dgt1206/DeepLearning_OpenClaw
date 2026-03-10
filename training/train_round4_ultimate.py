#!/usr/bin/env python3
"""
Round 4: 终极优化 - 冲刺最大准确率
目标: 87-89%
策略: Stacking + Voting + 超参数优化 + 内存清理
"""

import pandas as pd
import numpy as np
import joblib
import gc
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("Round 4: 终极优化训练")
print("=" * 80)

# ============================================================================
# 1. 内存清理函数
# ============================================================================
def clean_memory():
    """强制清理内存和CPU缓存"""
    gc.collect()
    import ctypes
    libc = ctypes.CDLL("libc.so.6")
    libc.malloc_trim(0)
    print("✅ 内存已清理")

# ============================================================================
# 2. 数据加载
# ============================================================================
print("\n[1/6] 加载数据...")
train_df = pd.read_csv('datasets/cleaned/train_cleaned.csv')
X = train_df.drop(['PassengerId', 'Survived'], axis=1)
y = train_df['Survived']

print(f"数据集大小: {len(train_df)}, 特征数: {X.shape[1]}")

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clean_memory()

# ============================================================================
# 3. Base Models (4个)
# ============================================================================
print("\n[2/6] 训练 Base Models...")

# Model 1: XGBoost (Round 2 最优配置)
print("训练 XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1,
    reg_lambda=1,
    min_child_weight=3,
    gamma=1,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)
xgb_score = xgb_model.score(X_val, y_val)
print(f"XGBoost 准确率: {xgb_score*100:.2f}%")
clean_memory()

# Model 2: LightGBM
print("训练 LightGBM...")
lgb_model = lgb.LGBMClassifier(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.01,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1,
    reg_lambda=1,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
lgb_model.fit(X_train, y_train)
lgb_score = lgb_model.score(X_val, y_val)
print(f"LightGBM 准确率: {lgb_score*100:.2f}%")
clean_memory()

# Model 3: Random Forest (保留基线)
print("训练 Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_score = rf_model.score(X_val, y_val)
print(f"Random Forest 准确率: {rf_score*100:.2f}%")
clean_memory()

# Model 4: Logistic Regression (线性基线)
print("训练 Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
lr_model.fit(X_train, y_train)
lr_score = lr_model.score(X_val, y_val)
print(f"Logistic Regression 准确率: {lr_score*100:.2f}%")
clean_memory()

# ============================================================================
# 4. 集成策略 1: Voting (软投票)
# ============================================================================
print("\n[3/6] 训练 Voting Classifier...")
voting = VotingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('lgb', lgb_model),
        ('rf', rf_model),
        ('lr', lr_model)
    ],
    voting='soft',
    weights=[3, 2, 1, 1],  # XGBoost 权重最高
    n_jobs=-1
)
voting.fit(X_train, y_train)
voting_score = voting.score(X_val, y_val)
print(f"Voting 准确率: {voting_score*100:.2f}%")
clean_memory()

# ============================================================================
# 5. 集成策略 2: Stacking
# ============================================================================
print("\n[4/6] 训练 Stacking Classifier...")
stacking = StackingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('lgb', lgb_model),
        ('rf', rf_model)
    ],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5,
    n_jobs=-1
)
stacking.fit(X_train, y_train)
stacking_score = stacking.score(X_val, y_val)
print(f"Stacking 准确率: {stacking_score*100:.2f}%")
clean_memory()

# ============================================================================
# 6. 选择最佳模型
# ============================================================================
print("\n[5/6] 选择最佳模型...")
models = {
    'XGBoost': (xgb_model, xgb_score),
    'LightGBM': (lgb_model, lgb_score),
    'Random Forest': (rf_model, rf_score),
    'Logistic Regression': (lr_model, lr_score),
    'Voting': (voting, voting_score),
    'Stacking': (stacking, stacking_score)
}

print("\n模型对比:")
for name, (model, score) in models.items():
    print(f"  {name:20s}: {score*100:.2f}%")

best_name = max(models, key=lambda k: models[k][1])
best_model, best_score = models[best_name]
print(f"\n✅ 最佳模型: {best_name} ({best_score*100:.2f}%)")

# ============================================================================
# 7. 保存结果
# ============================================================================
print("\n[6/6] 保存结果...")
joblib.dump(best_model, 'models/best_model_round4.pkl')
print("✅ 模型已保存: models/best_model_round4.pkl")

# 生成测试集预测
test_df = pd.read_csv('datasets/cleaned/test_cleaned.csv')
X_test = test_df.drop(['PassengerId', 'Survived'], axis=1, errors='ignore')
test_pred = best_model.predict(X_test)

submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': test_pred.astype(int)
})
os.makedirs('predictions', exist_ok=True)
submission.to_csv('predictions/submission_round4.csv', index=False)
print("✅ 预测已保存: predictions/submission_round4.csv")

# 生成报告
report = f"""# Round 4 训练报告

## 结果汇总

| 模型 | 准确率 |
|------|--------|
| XGBoost | {xgb_score*100:.2f}% |
| LightGBM | {lgb_score*100:.2f}% |
| Random Forest | {rf_score*100:.2f}% |
| Logistic Regression | {lr_score*100:.2f}% |
| **Voting** | **{voting_score*100:.2f}%** |
| **Stacking** | **{stacking_score*100:.2f}%** |

## 最佳模型
- **名称**: {best_name}
- **准确率**: {best_score*100:.2f}%
- **相比 Round 2 提升**: {(best_score - 0.8530)*100:+.2f}%

## 状态
- {'✅ 达到目标' if best_score >= 0.87 else '⏭️ 需要继续优化'}
"""

with open('training/training_report_round4.md', 'w') as f:
    f.write(report)
print("✅ 报告已保存: training/training_report_round4.md")

# 最终清理
clean_memory()

print("\n" + "=" * 80)
print(f"Round 4 完成！最佳准确率: {best_score*100:.2f}%")
print("=" * 80)
