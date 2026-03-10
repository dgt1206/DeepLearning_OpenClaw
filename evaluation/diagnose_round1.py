#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Round 1 深度诊断脚本
分析过拟合、特征冗余、交叉验证稳定性
"""

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve, cross_val_score
import json
import os

os.makedirs('evaluation', exist_ok=True)

print('=== 🔍 Round 1 深度诊断分析 ===\n')

# 1. 加载 Round 1 模型和增强数据
print('📂 加载 Round 1 模型和增强数据...')
model_r1 = joblib.load('models/best_model_round1.pkl')

# 尝试加载增强数据
if os.path.exists('datasets/cleaned/train_enhanced_v1.csv'):
    train_df = pd.read_csv('datasets/cleaned/train_enhanced_v1.csv')
    print('✓ 使用增强特征数据集')
else:
    # 如果没有,需要重新生成特征
    print('⚠️ 增强特征数据集不存在,重新生成...')
    exec(open('training/train_optimized_v1.py').read().split('def train_optimized_model')[0])
    train_df = pd.read_csv('datasets/cleaned/train_cleaned.csv')
    train_df = create_advanced_features(train_df.copy())

X = train_df.drop(['PassengerId', 'Survived'], axis=1, errors='ignore')
y = train_df['Survived']

print(f'✓ 数据形状: {X.shape}')
print(f'✓ 特征数量: {X.shape[1]}')
print(f'✓ 模型期望特征数: {model_r1.n_features_in_}\n')

if X.shape[1] != model_r1.n_features_in_:
    print(f'❌ 特征数不匹配! 数据:{X.shape[1]} vs 模型:{model_r1.n_features_in_}')
    print('使用基础数据 + 特征工程函数...\n')

# 2. 特征重要性深度分析
print('📊 特征重要性分析:')
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model_r1.feature_importances_
}).sort_values('importance', ascending=False)

print('\n前10个重要特征:')
for idx, row in feature_importance.head(10).iterrows():
    print(f'  {row["feature"]:25s} {row["importance"]:.4f} ({row["importance"]*100:.2f}%)')

print('\n⚠️ 低贡献特征 (<1%):')
low_importance = feature_importance[feature_importance['importance'] < 0.01]
if len(low_importance) > 0:
    for idx, row in low_importance.iterrows():
        print(f'  {row["feature"]:25s} {row["importance"]:.4f} ({row["importance"]*100:.2f}%)')
    print(f'\n共 {len(low_importance)} 个低贡献特征')
else:
    print('  无低贡献特征')

# 3. 过拟合诊断 - 学习曲线
print('\n\n🧪 过拟合诊断 - 计算学习曲线...')
train_sizes, train_scores, val_scores = learning_curve(
    model_r1, X, y, 
    cv=5, 
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy',
    n_jobs=-1
)

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)

print(f'\n训练集准确率: {train_mean[-1]:.4f} ± {train_std[-1]:.4f}')
print(f'验证集准确率: {val_mean[-1]:.4f} ± {val_std[-1]:.4f}')
overfit_gap = train_mean[-1] - val_mean[-1]
print(f'过拟合差距: {overfit_gap*100:.2f}%')

if overfit_gap > 0.05:
    print('⚠️ 严重过拟合!')
elif overfit_gap > 0.03:
    print('⚠️ 中度过拟合')
else:
    print('✅ 过拟合程度可接受')

# 4. 交叉验证稳定性分析
print('\n\n📈 5折交叉验证详细分析...')
cv_scores = cross_val_score(model_r1, X, y, cv=5, scoring='accuracy')
cv_scores_list = [f'{s:.4f}' for s in cv_scores]
print(f'各折得分: {cv_scores_list}')
print(f'平均: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')
print(f'最大-最小差: {(cv_scores.max() - cv_scores.min())*100:.2f}%')

if cv_scores.std() > 0.02:
    print('⚠️ 模型不稳定,不同折之间差异较大')
else:
    print('✅ 模型稳定')

# 5. 保存诊断数据
diagnosis_data = {
    'round': 1,
    'model_features': int(model_r1.n_features_in_),
    'data_features': int(X.shape[1]),
    'feature_importance': feature_importance.to_dict('records'),
    'low_importance_features': low_importance['feature'].tolist(),
    'low_importance_count': len(low_importance),
    'overfit_gap': float(overfit_gap),
    'cv_mean': float(cv_scores.mean()),
    'cv_std': float(cv_scores.std()),
    'cv_scores': cv_scores.tolist(),
    'train_mean': float(train_mean[-1]),
    'val_mean': float(val_mean[-1]),
    'learning_curve': {
        'train_sizes': train_sizes.tolist(),
        'train_mean': train_mean.tolist(),
        'val_mean': val_mean.tolist()
    }
}

with open('evaluation/round1_diagnosis.json', 'w') as f:
    json.dump(diagnosis_data, f, indent=2)

print('\n✅ 诊断数据已保存到 evaluation/round1_diagnosis.json')

# 6. 生成可视化
print('\n📊 生成诊断可视化...')
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# 图1: Top 20 特征重要性
ax1 = fig.add_subplot(gs[0, :])
top20 = feature_importance.head(20)
colors = ['red' if imp < 0.01 else 'steelblue' for imp in top20['importance']]
ax1.barh(range(len(top20)), top20['importance'], color=colors)
ax1.set_yticks(range(len(top20)))
ax1.set_yticklabels(top20['feature'])
ax1.set_xlabel('Importance', fontsize=12)
ax1.set_title('Top 20 Feature Importance (Round 1) - Red: <1%', fontsize=14, fontweight='bold')
ax1.invert_yaxis()
ax1.grid(True, alpha=0.3, axis='x')

# 图2: 学习曲线
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(train_sizes, train_mean, 'o-', label='Training', linewidth=2, markersize=8, color='green')
ax2.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color='green')
ax2.plot(train_sizes, val_mean, 'o-', label='Validation', linewidth=2, markersize=8, color='orange')
ax2.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2, color='orange')
ax2.set_xlabel('Training Size', fontsize=12)
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.set_title(f'Learning Curve - Overfit Gap: {overfit_gap*100:.2f}%', fontsize=12, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

# 图3: CV折分布
ax3 = fig.add_subplot(gs[1, 1])
ax3.bar(range(1, 6), cv_scores, color='steelblue', edgecolor='black')
ax3.axhline(cv_scores.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {cv_scores.mean():.4f}')
ax3.set_xlabel('Fold', fontsize=12)
ax3.set_ylabel('Accuracy', fontsize=12)
ax3.set_title('5-Fold Cross Validation Scores', fontsize=12, fontweight='bold')
ax3.set_xticks(range(1, 6))
ax3.set_ylim([0.75, 0.90])
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3, axis='y')

# 图4: 低贡献特征详情
ax4 = fig.add_subplot(gs[2, :])
if len(low_importance) > 0:
    ax4.barh(range(len(low_importance)), low_importance['importance'], color='red', edgecolor='darkred')
    ax4.set_yticks(range(len(low_importance)))
    ax4.set_yticklabels(low_importance['feature'])
    ax4.set_xlabel('Importance', fontsize=12)
    ax4.set_title(f'Low Contribution Features (<1%) - {len(low_importance)} features to remove', 
                 fontsize=12, fontweight='bold', color='red')
    ax4.invert_yaxis()
    ax4.grid(True, alpha=0.3, axis='x')
else:
    ax4.text(0.5, 0.5, 'No low importance features found', 
            ha='center', va='center', fontsize=14, fontweight='bold')
    ax4.set_title('Low Contribution Features', fontsize=12, fontweight='bold')
    ax4.axis('off')

plt.savefig('evaluation/round1_diagnosis.png', dpi=150, bbox_inches='tight')
print('✓ 保存到 evaluation/round1_diagnosis.png')

print('\n' + '='*70)
print('🎯 Round 1 诊断完成!')
print('='*70)
print(f'\n核心发现:')
print(f'  1. 过拟合差距: {overfit_gap*100:.2f}% (训练{train_mean[-1]:.4f} vs 验证{val_mean[-1]:.4f})')
print(f'  2. 低贡献特征: {len(low_importance)} 个 (<1% importance)')
print(f'  3. CV稳定性: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')
print(f'  4. CV vs 单次验证差距: {(cv_scores.mean() - val_mean[-1])*100:.2f}%')
