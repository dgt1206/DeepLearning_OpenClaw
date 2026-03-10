#!/usr/bin/env python3
"""
Round 6: 精简特征 + CatBoost
策略: 回到 Round 2 基础，只添加 3-5 个最有效特征
目标: 86-87%
"""

import pandas as pd
import numpy as np
import joblib
import gc
import os
from sklearn.model_selection import train_test_split, cross_val_score
from catboost import CatBoostClassifier
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("Round 6: 精简特征 + CatBoost")
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
# 1. 加载 Round 2 的清理数据
# ============================================================================
print("\n[1/5] 加载数据...")
train_df = pd.read_csv('datasets/cleaned/train_cleaned.csv')
X = train_df.drop(['PassengerId', 'Survived'], axis=1)
y = train_df['Survived']

print(f"Round 2 特征数: {X.shape[1]}")

# ============================================================================
# 2. 只添加 3 个最有效的新特征
# ============================================================================
print("\n[2/5] 添加 3 个精选特征...")

# 加载原始数据获取 Name 和 Cabin
train_orig = pd.read_csv('datasets/train.csv')

# 特征 1: Woman_or_Child (泰坦尼克核心规则)
X['Woman_or_Child'] = ((train_orig['Sex'] == 'female') | (train_orig['Age'] < 18)).astype(int)

# 特征 2: Title (从 Name 提取，只保留主要类别)
title = train_orig['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
title_map = {
    'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3,
    'Dr': 4, 'Rev': 4, 'Col': 4, 'Major': 4, 'Mlle': 1, 
    'Mme': 2, 'Ms': 1, 'Don': 4, 'Lady': 4, 'Countess': 4,
    'Jonkheer': 4, 'Sir': 4, 'Capt': 4, 'Dona': 4
}
X['Title_Simple'] = title.map(title_map).fillna(4)

# 特征 3: Deck (从 Cabin 提取首字母)
cabin = train_orig['Cabin'].fillna('Unknown')
deck_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8, 'U': 0}
X['Deck'] = cabin.str[0].map(deck_map).fillna(0)

print(f"新特征数: {X.shape[1]} (+3)")

clean_memory()

# ============================================================================
# 3. 划分数据
# ============================================================================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================================================
# 4. 训练 CatBoost
# ============================================================================
print("\n[3/5] 训练 CatBoost...")

catboost_model = CatBoostClassifier(
    iterations=1000,
    depth=6,
    learning_rate=0.01,
    loss_function='Logloss',
    eval_metric='Accuracy',
    random_seed=42,
    verbose=False,
    early_stopping_rounds=50
)

catboost_model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    verbose=False
)

catboost_score = catboost_model.score(X_val, y_val)
print(f"CatBoost 验证集: {catboost_score*100:.2f}%")

# 交叉验证
cv_scores = cross_val_score(catboost_model, X, y, cv=5, scoring='accuracy', n_jobs=-1)
print(f"CatBoost 5折CV: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

clean_memory()

# ============================================================================
# 5. 对比 XGBoost (Round 2 配置)
# ============================================================================
print("\n[4/5] 对比 XGBoost...")

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
print(f"XGBoost 验证集: {xgb_score*100:.2f}%")

clean_memory()

# ============================================================================
# 6. 选择最佳模型
# ============================================================================
print("\n[5/5] 选择最佳模型...")

if catboost_score > xgb_score:
    best_model = catboost_model
    best_score = catboost_score
    best_name = "CatBoost"
else:
    best_model = xgb_model
    best_score = xgb_score
    best_name = "XGBoost"

print(f"\n✅ 最佳模型: {best_name} ({best_score*100:.2f}%)")

# ============================================================================
# 7. 保存结果
# ============================================================================
joblib.dump(best_model, 'models/best_model_round6.pkl')
print("✅ 模型已保存")

# 预测测试集
test_df = pd.read_csv('datasets/cleaned/test_cleaned.csv')
test_orig = pd.read_csv('datasets/test.csv')

X_test = test_df.drop(['PassengerId', 'Survived'], axis=1, errors='ignore')

# 添加 3 个新特征到测试集
X_test['Woman_or_Child'] = ((test_orig['Sex'] == 'female') | (test_orig['Age'].fillna(30) < 18)).astype(int)

title_test = test_orig['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
X_test['Title_Simple'] = title_test.map(title_map).fillna(4)

cabin_test = test_orig['Cabin'].fillna('Unknown')
X_test['Deck'] = cabin_test.str[0].map(deck_map).fillna(0)

test_pred = best_model.predict(X_test)

submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'].astype(int),
    'Survived': test_pred.astype(int)
})

os.makedirs('predictions', exist_ok=True)
submission.to_csv('predictions/submission_round6.csv', index=False)
print("✅ 预测已保存")

# 报告
report = f"""# Round 6 训练报告

## 策略
- 回到 Round 2 基础 (18 特征)
- 只添加 3 个精选特征 (Woman_or_Child, Title_Simple, Deck)
- 对比 CatBoost vs XGBoost

## 结果
| 模型 | 验证集 | 5折CV |
|------|--------|-------|
| CatBoost | {catboost_score*100:.2f}% | {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}% |
| XGBoost | {xgb_score*100:.2f}% | - |

## 最佳模型
- **名称**: {best_name}
- **准确率**: {best_score*100:.2f}%
- **相比 Round 2 (85.30%)**: {(best_score - 0.8530)*100:+.2f}%
- **相比 Round 5 (81.01%)**: {(best_score - 0.8101)*100:+.2f}%

## 状态
{'✅ 达到 86% 目标' if best_score >= 0.86 else '⏭️ 继续 Round 7'}
"""

with open('training/training_report_round6.md', 'w') as f:
    f.write(report)

clean_memory()

print("\n" + "=" * 80)
print(f"Round 6 完成！{best_name}: {best_score*100:.2f}%")
print("=" * 80)
