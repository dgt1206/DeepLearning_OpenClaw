#!/usr/bin/env python3
"""
Round 7: 终极融合 - 最后冲刺
策略: 
1. 加载 Round 2 最优模型 (85.30%)
2. 训练多个模型并融合
3. 超参数优化
4. 加权集成
目标: 87-88%
"""

import pandas as pd
import numpy as np
import joblib
import gc
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("Round 7: 终极融合 - 最后冲刺")
print("=" * 80)

def clean_memory():
    gc.collect()
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except:
        pass

# ============================================================================
# 1. 加载数据（使用 Round 2 的最优特征集）
# ============================================================================
print("\n[1/6] 加载数据...")
train_df = pd.read_csv('datasets/cleaned/train_cleaned.csv')
X = train_df.drop(['PassengerId', 'Survived'], axis=1)
y = train_df['Survived']

print(f"特征数: {X.shape[1]}")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================================================
# 2. 训练多个优化模型
# ============================================================================
print("\n[2/6] 训练基础模型...")

# Model 1: XGBoost (优化版)
print("  训练 XGBoost...")
xgb1 = xgb.XGBClassifier(
    n_estimators=600,
    max_depth=6,
    learning_rate=0.008,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=1.5,
    reg_lambda=1.5,
    min_child_weight=4,
    gamma=1.2,
    random_state=42,
    n_jobs=-1
)
xgb1.fit(X_train, y_train)
xgb1_score = xgb1.score(X_val, y_val)
print(f"    XGBoost-1: {xgb1_score*100:.2f}%")
clean_memory()

# Model 2: XGBoost (不同参数)
print("  训练 XGBoost-2...")
xgb2 = xgb.XGBClassifier(
    n_estimators=800,
    max_depth=4,
    learning_rate=0.012,
    subsample=0.75,
    colsample_bytree=0.75,
    reg_alpha=2,
    reg_lambda=2,
    min_child_weight=5,
    gamma=2,
    random_state=43,
    n_jobs=-1
)
xgb2.fit(X_train, y_train)
xgb2_score = xgb2.score(X_val, y_val)
print(f"    XGBoost-2: {xgb2_score*100:.2f}%")
clean_memory()

# Model 3: LightGBM
print("  训练 LightGBM...")
lgbm = lgb.LGBMClassifier(
    n_estimators=700,
    max_depth=6,
    learning_rate=0.01,
    num_leaves=40,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1.5,
    reg_lambda=1.5,
    min_child_weight=5,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
lgbm.fit(X_train, y_train)
lgbm_score = lgbm.score(X_val, y_val)
print(f"    LightGBM: {lgbm_score*100:.2f}%")
clean_memory()

# Model 4: CatBoost
print("  训练 CatBoost...")
catb = CatBoostClassifier(
    iterations=1200,
    depth=7,
    learning_rate=0.008,
    l2_leaf_reg=5,
    random_seed=42,
    verbose=False
)
catb.fit(X_train, y_train, verbose=False)
catb_score = catb.score(X_val, y_val)
print(f"    CatBoost: {catb_score*100:.2f}%")
clean_memory()

# ============================================================================
# 3. 策略1: Soft Voting (加权投票)
# ============================================================================
print("\n[3/6] Soft Voting 集成...")

# 根据验证集性能计算权重
weights = []
scores_map = {
    'xgb1': xgb1_score,
    'xgb2': xgb2_score,
    'lgbm': lgbm_score,
    'catb': catb_score
}

# 归一化权重（性能越好权重越高）
total_score = sum(scores_map.values())
for name, score in scores_map.items():
    weight = score / total_score * 10
    weights.append(weight)
    print(f"  {name}: 权重 = {weight:.2f}")

voting = VotingClassifier(
    estimators=[
        ('xgb1', xgb1),
        ('xgb2', xgb2),
        ('lgbm', lgbm),
        ('catb', catb)
    ],
    voting='soft',
    weights=weights,
    n_jobs=-1
)
voting.fit(X_train, y_train)
voting_score = voting.score(X_val, y_val)
print(f"  Voting 准确率: {voting_score*100:.2f}%")
clean_memory()

# ============================================================================
# 4. 策略2: 简单平均
# ============================================================================
print("\n[4/6] 简单平均集成...")

y_pred_proba = (
    xgb1.predict_proba(X_val)[:, 1] +
    xgb2.predict_proba(X_val)[:, 1] +
    lgbm.predict_proba(X_val)[:, 1] +
    catb.predict_proba(X_val)[:, 1]
) / 4

y_pred_avg = (y_pred_proba > 0.5).astype(int)
avg_score = (y_pred_avg == y_val).sum() / len(y_val)
print(f"  平均集成准确率: {avg_score*100:.2f}%")

# ============================================================================
# 5. 选择最佳策略
# ============================================================================
print("\n[5/6] 选择最佳策略...")

strategies = {
    'XGBoost-1': (xgb1, xgb1_score),
    'XGBoost-2': (xgb2, xgb2_score),
    'LightGBM': (lgbm, lgbm_score),
    'CatBoost': (catb, catb_score),
    'Soft Voting': (voting, voting_score),
    'Simple Average': ('avg', avg_score)
}

print("\n所有策略对比:")
for name, (model, score) in strategies.items():
    print(f"  {name:20s}: {score*100:.2f}%")

# 找出最佳策略
best_name = max(strategies, key=lambda k: strategies[k][1])
best_model, best_score = strategies[best_name]

print(f"\n✅ 最佳策略: {best_name} ({best_score*100:.2f}%)")

# ============================================================================
# 6. 保存结果
# ============================================================================
print("\n[6/6] 保存结果...")

if best_name != 'Simple Average':
    joblib.dump(best_model, 'models/best_model_round7.pkl')
    print("✅ 模型已保存")

# 预测测试集
test_df = pd.read_csv('datasets/cleaned/test_cleaned.csv')
X_test = test_df.drop(['PassengerId', 'Survived'], axis=1, errors='ignore')

if best_name == 'Simple Average':
    test_pred_proba = (
        xgb1.predict_proba(X_test)[:, 1] +
        xgb2.predict_proba(X_test)[:, 1] +
        lgbm.predict_proba(X_test)[:, 1] +
        catb.predict_proba(X_test)[:, 1]
    ) / 4
    test_pred = (test_pred_proba > 0.5).astype(int)
else:
    test_pred = best_model.predict(X_test)

submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'].astype(int),
    'Survived': test_pred.astype(int)
})

os.makedirs('predictions', exist_ok=True)
submission.to_csv('predictions/submission_round7_final.csv', index=False)
print("✅ 预测已保存")

# 生成最终报告
report = f"""# Round 7 最终报告

## 训练的模型

| 模型 | 验证集准确率 |
|------|-------------|
| XGBoost-1 (优化) | {xgb1_score*100:.2f}% |
| XGBoost-2 (变体) | {xgb2_score*100:.2f}% |
| LightGBM | {lgbm_score*100:.2f}% |
| CatBoost | {catb_score*100:.2f}% |
| **Soft Voting** | **{voting_score*100:.2f}%** |
| **Simple Average** | **{avg_score*100:.2f}%** |

## 最佳策略
- **名称**: {best_name}
- **准确率**: {best_score*100:.2f}%
- **相比 Round 2 (85.30%)**: {(best_score - 0.8530)*100:+.2f}%

## 项目总结

### 完整训练历程
- Round 0: 82.12% (基线)
- Round 1: 83.61% (特征工程)
- Round 2: 85.30% (XGBoost) ✅ 历史最佳
- Round 3-4: 评估失败
- Round 5: 81.01% (特征过多)
- Round 6: 82.12% (精简特征)
- Round 7: {best_score*100:.2f}% (融合策略)

### 最终状态
{'✅ 成功' if best_score >= 0.85 else '⚠️ 未达预期'}

## 结论
{'项目达到了 85%+ 的优秀水平！' if best_score >= 0.85 else '虽未达到 90% 的极限目标，但通过多轮迭代获得了宝贵经验。'}
"""

with open('training/training_report_round7_final.md', 'w') as f:
    f.write(report)

clean_memory()

print("\n" + "=" * 80)
print(f"🎉 Round 7 完成！最终成绩: {best_score*100:.2f}%")
print("项目训练完毕！")
print("=" * 80)
