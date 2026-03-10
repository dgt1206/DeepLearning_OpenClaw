#!/usr/bin/env python3
"""
最佳模型重训练 - Round 2 参数
基于项目经验，使用验证过的最优配置
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("最佳模型重训练 - Round 2 配置")
print("=" * 80)

# ============================================================================
# 1. 加载数据
# ============================================================================
print("\n[1/4] 加载数据...")
train_df = pd.read_csv('datasets/cleaned/train_cleaned.csv')
X = train_df.drop(['PassengerId', 'Survived'], axis=1)
y = train_df['Survived']

print(f"数据集: {len(train_df)} 样本, {X.shape[1]} 特征")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================================================
# 2. Round 2 最优配置
# ============================================================================
print("\n[2/4] 使用 Round 2 最优配置...")

best_config = {
    'model': 'XGBoost',
    'params': {
        'n_estimators': 500,
        'max_depth': 5,
        'learning_rate': 0.01,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 1,
        'reg_lambda': 1,
        'min_child_weight': 3,
        'gamma': 1,
        'random_state': 42,
        'n_jobs': -1
    },
    'features': list(X.columns),
    'feature_count': X.shape[1]
}

print("配置:")
for key, value in best_config['params'].items():
    print(f"  {key}: {value}")

# ============================================================================
# 3. 训练模型
# ============================================================================
print("\n[3/4] 训练模型...")

model = xgb.XGBClassifier(**best_config['params'])
model.fit(X_train, y_train)

# 评估
train_score = model.score(X_train, y_train)
val_score = model.score(X_val, y_val)

print(f"训练集准确率: {train_score*100:.2f}%")
print(f"验证集准确率: {val_score*100:.2f}%")

# 5折交叉验证
print("\n5折交叉验证...")
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=-1)
cv_mean = cv_scores.mean()
cv_std = cv_scores.std()

print(f"CV准确率: {cv_mean*100:.2f}% ± {cv_std*100:.2f}%")
print(f"各折: {[f'{s*100:.2f}%' for s in cv_scores]}")

# ============================================================================
# 4. 保存最佳模型和配置
# ============================================================================
print("\n[4/4] 保存结果...")

# 保存模型
os.makedirs('models/best', exist_ok=True)
joblib.dump(model, 'models/best/best_model_final.pkl')
print("✅ 模型已保存: models/best/best_model_final.pkl")

# 保存配置
best_config.update({
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'train_accuracy': float(train_score),
    'val_accuracy': float(val_score),
    'cv_mean': float(cv_mean),
    'cv_std': float(cv_std),
    'cv_scores': [float(s) for s in cv_scores]
})

with open('models/best/best_config.json', 'w') as f:
    json.dump(best_config, f, indent=2)
print("✅ 配置已保存: models/best/best_config.json")

# 保存特征重要性
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

feature_importance.to_csv('models/best/feature_importance.csv', index=False)
print("✅ 特征重要性已保存")

# 生成预测
test_df = pd.read_csv('datasets/cleaned/test_cleaned.csv')
X_test = test_df.drop(['PassengerId'], axis=1, errors='ignore')
test_pred = model.predict(X_test)

submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'].astype(int),
    'Survived': test_pred.astype(int)
})

os.makedirs('predictions/best', exist_ok=True)
submission.to_csv('predictions/best/submission_final.csv', index=False)
print("✅ 最终预测已保存")

print("\n" + "=" * 80)
print(f"🎉 最佳模型训练完成！验证集准确率: {val_score*100:.2f}%")
print("=" * 80)
