#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型深度评估分析 - Round 1
分析Random Forest模型的性能瓶颈,提出优化方案
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, roc_auc_score
)
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style('whitegrid')

RANDOM_STATE = 42

def create_directories():
    """创建输出目录"""
    os.makedirs('evaluation', exist_ok=True)
    os.makedirs('analysis/figures', exist_ok=True)
    print("✅ 目录结构准备完成\n")

def load_model_and_data():
    """加载模型和数据"""
    print("📂 加载模型和数据...")
    
    # 加载模型
    model = joblib.load('models/best_model.pkl')
    print(f"✅ 模型类型: {type(model).__name__}")
    
    # 加载数据
    train_df = pd.read_csv('datasets/cleaned/train_cleaned.csv')
    X = train_df.drop(['PassengerId', 'Survived'], axis=1)
    y = train_df['Survived']
    
    # 使用相同的划分方式
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"✅ 训练集: {X_train.shape}")
    print(f"✅ 验证集: {X_val.shape}")
    print(f"✅ 特征数量: {X.shape[1]}\n")
    
    return model, X_train, X_val, y_train, y_val, X.columns.tolist()

def analyze_errors(model, X_val, y_val, feature_names):
    """错误分析"""
    print("=" * 60)
    print("🔍 第1部分: 错误分析")
    print("=" * 60)
    
    # 预测
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # 混淆矩阵
    cm = confusion_matrix(y_val, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print("\n📊 混淆矩阵:")
    print(f"真阴性 (TN): {tn} | 假阳性 (FP): {fp}")
    print(f"假阴性 (FN): {fn} | 真阳性 (TP): {tp}")
    
    # 错误统计
    total_errors = fp + fn
    error_rate = total_errors / len(y_val)
    
    print(f"\n❌ 总错误数: {total_errors}/{len(y_val)} ({error_rate:.2%})")
    print(f"   - 假阳性 (预测存活但实际未存活): {fp} ({fp/total_errors:.1%})")
    print(f"   - 假阴性 (预测未存活但实际存活): {fn} ({fn/total_errors:.1%})")
    
    # 错误样本分析
    errors_mask = (y_pred != y_val)
    error_df = X_val[errors_mask].copy()
    error_df['true_label'] = y_val[errors_mask].values
    error_df['pred_label'] = y_pred[errors_mask]
    error_df['pred_proba'] = y_pred_proba[errors_mask]
    error_df['error_type'] = error_df.apply(
        lambda x: 'False Positive' if x['pred_label'] == 1 else 'False Negative', 
        axis=1
    )
    
    print("\n📈 分类报告:")
    print(classification_report(y_val, y_pred, target_names=['Not Survived', 'Survived']))
    
    # 不确定样本分析
    uncertain_mask = (y_pred_proba > 0.4) & (y_pred_proba < 0.6)
    n_uncertain = uncertain_mask.sum()
    print(f"\n🤔 预测不确定样本 (概率0.4-0.6): {n_uncertain} ({n_uncertain/len(y_val):.1%})")
    
    # 错误样本特征分析
    print("\n📋 错误样本特征统计:")
    for error_type in ['False Positive', 'False Negative']:
        subset = error_df[error_df['error_type'] == error_type]
        print(f"\n{error_type} ({len(subset)} 样本):")
        
        # 显示前几个最重要特征的统计
        important_features = ['Sex_Encoded', 'Pclass', 'Age', 'Fare', 'Title_Encoded']
        for feat in important_features:
            if feat in subset.columns:
                mean_val = subset[feat].mean()
                print(f"  {feat}: 均值={mean_val:.2f}")
    
    return error_df, y_pred, y_pred_proba

def analyze_prediction_confidence(y_val, y_pred_proba, y_pred):
    """预测置信度分析"""
    print("\n" + "=" * 60)
    print("🎯 第2部分: 预测置信度分析")
    print("=" * 60)
    
    # 按置信度分组
    confidence_bins = [0, 0.3, 0.4, 0.6, 0.7, 1.0]
    confidence_labels = ['Very Low (0-0.3)', 'Low (0.3-0.4)', 
                        'Uncertain (0.4-0.6)', 'High (0.6-0.7)', 
                        'Very High (0.7-1.0)']
    
    confidence_groups = pd.cut(y_pred_proba, bins=confidence_bins, 
                              labels=confidence_labels, include_lowest=True)
    
    print("\n📊 各置信度区间的样本分布:")
    for label in confidence_labels:
        mask = (confidence_groups == label)
        n_samples = mask.sum()
        if n_samples > 0:
            accuracy = (y_val[mask] == y_pred[mask]).mean()
            print(f"{label}: {n_samples} 样本, 准确率 {accuracy:.1%}")
    
    return confidence_groups

def analyze_learning_curve(model, X_train, y_train):
    """学习曲线分析 - 诊断欠拟合/过拟合"""
    print("\n" + "=" * 60)
    print("📈 第3部分: 学习曲线分析")
    print("=" * 60)
    
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train, 
        cv=5, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy',
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    print(f"\n📊 学习曲线统计:")
    print(f"最大训练样本时:")
    print(f"  训练集准确率: {train_mean[-1]:.4f} ± {train_std[-1]:.4f}")
    print(f"  验证集准确率: {val_mean[-1]:.4f} ± {val_std[-1]:.4f}")
    print(f"  差距: {train_mean[-1] - val_mean[-1]:.4f}")
    
    # 诊断
    gap = train_mean[-1] - val_mean[-1]
    if gap > 0.05:
        print("\n⚠️ 诊断: 存在过拟合倾向 (训练集显著高于验证集)")
    elif val_mean[-1] < 0.85:
        print("\n⚠️ 诊断: 存在欠拟合倾向 (训练集和验证集都不够高)")
    else:
        print("\n✅ 诊断: 模型拟合较好")
    
    return train_sizes, train_mean, train_std, val_mean, val_std

def analyze_feature_importance(model, X_val, y_val, feature_names):
    """特征重要性与错误率分析"""
    print("\n" + "=" * 60)
    print("🔬 第4部分: 特征重要性分析")
    print("=" * 60)
    
    # 基础特征重要性
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n📊 Top 10 特征重要性:")
    print(feature_importance.head(10).to_string(index=False))
    
    # Permutation Importance (更准确)
    print("\n🔄 计算 Permutation Importance...")
    perm_importance = permutation_importance(
        model, X_val, y_val, 
        n_repeats=10, 
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    perm_feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': perm_importance.importances_mean,
        'importance_std': perm_importance.importances_std
    }).sort_values('importance_mean', ascending=False)
    
    print("\n📊 Top 10 Permutation Importance:")
    print(perm_feature_importance.head(10).to_string(index=False))
    
    # 比较两种重要性
    print("\n🔍 重要性对比分析:")
    top_5_basic = set(feature_importance.head(5)['feature'])
    top_5_perm = set(perm_feature_importance.head(5)['feature'])
    common = top_5_basic & top_5_perm
    print(f"两种方法共同的Top5特征: {common}")
    
    return feature_importance, perm_feature_importance

def analyze_data_distribution(X_val, y_val):
    """数据分布分析"""
    print("\n" + "=" * 60)
    print("📊 第5部分: 数据分布分析")
    print("=" * 60)
    
    # 类别不平衡
    class_dist = y_val.value_counts()
    print(f"\n类别分布:")
    print(f"  未存活 (0): {class_dist[0]} ({class_dist[0]/len(y_val):.1%})")
    print(f"  存活 (1): {class_dist[1]} ({class_dist[1]/len(y_val):.1%})")
    
    imbalance_ratio = class_dist[0] / class_dist[1]
    print(f"  不平衡比例: {imbalance_ratio:.2f}:1")
    
    if imbalance_ratio > 1.5:
        print("⚠️ 存在类别不平衡问题")
    else:
        print("✅ 类别分布较为均衡")
    
    return class_dist

def generate_visualizations(error_df, y_val, y_pred_proba, confidence_groups,
                          train_sizes, train_mean, train_std, val_mean, val_std,
                          feature_importance, perm_feature_importance):
    """生成所有可视化"""
    print("\n" + "=" * 60)
    print("🎨 生成可视化图表")
    print("=" * 60)
    
    # 图1: 错误分析
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Error Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # 1.1 错误类型分布
    ax1 = axes[0, 0]
    error_counts = error_df['error_type'].value_counts()
    colors = ['#ff6b6b', '#4ecdc4']
    ax1.bar(error_counts.index, error_counts.values, color=colors, alpha=0.7)
    ax1.set_title('Error Type Distribution', fontweight='bold')
    ax1.set_ylabel('Count')
    ax1.set_xlabel('Error Type')
    for i, v in enumerate(error_counts.values):
        ax1.text(i, v + 1, str(v), ha='center', va='bottom', fontweight='bold')
    
    # 1.2 错误样本的预测概率分布
    ax2 = axes[0, 1]
    ax2.hist(error_df[error_df['error_type'] == 'False Positive']['pred_proba'], 
             bins=20, alpha=0.6, label='False Positive', color='#ff6b6b')
    ax2.hist(error_df[error_df['error_type'] == 'False Negative']['pred_proba'], 
             bins=20, alpha=0.6, label='False Negative', color='#4ecdc4')
    ax2.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Decision Boundary')
    ax2.set_title('Prediction Probability of Error Samples', fontweight='bold')
    ax2.set_xlabel('Prediction Probability')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    
    # 1.3 关键特征的错误分布 (Sex)
    ax3 = axes[1, 0]
    if 'Sex_Encoded' in error_df.columns:
        error_by_sex = error_df.groupby(['Sex_Encoded', 'error_type']).size().unstack(fill_value=0)
        error_by_sex.plot(kind='bar', ax=ax3, color=colors, alpha=0.7)
        ax3.set_title('Errors by Sex', fontweight='bold')
        ax3.set_xlabel('Sex (0=Female, 1=Male)')
        ax3.set_ylabel('Error Count')
        ax3.legend(title='Error Type')
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=0)
    
    # 1.4 关键特征的错误分布 (Pclass)
    ax4 = axes[1, 1]
    if 'Pclass' in error_df.columns:
        error_by_pclass = error_df.groupby(['Pclass', 'error_type']).size().unstack(fill_value=0)
        error_by_pclass.plot(kind='bar', ax=ax4, color=colors, alpha=0.7)
        ax4.set_title('Errors by Passenger Class', fontweight='bold')
        ax4.set_xlabel('Pclass')
        ax4.set_ylabel('Error Count')
        ax4.legend(title='Error Type')
        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=0)
    
    plt.tight_layout()
    plt.savefig('analysis/figures/error_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ 保存: analysis/figures/error_analysis.png")
    plt.close()
    
    # 图2: 预测置信度分布
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Prediction Confidence Analysis', fontsize=16, fontweight='bold')
    
    # 2.1 概率分布直方图
    ax1 = axes[0]
    ax1.hist(y_pred_proba[y_val == 0], bins=30, alpha=0.5, label='Not Survived (True)', color='#ff6b6b')
    ax1.hist(y_pred_proba[y_val == 1], bins=30, alpha=0.5, label='Survived (True)', color='#4ecdc4')
    ax1.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
    ax1.axvspan(0.4, 0.6, alpha=0.2, color='yellow', label='Uncertain Zone')
    ax1.set_xlabel('Prediction Probability', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Probability Distribution by True Label', fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2.2 置信度区间的准确率
    ax2 = axes[1]
    confidence_accuracy = []
    confidence_counts = []
    labels = ['Very Low\n(0-0.3)', 'Low\n(0.3-0.4)', 'Uncertain\n(0.4-0.6)', 
              'High\n(0.6-0.7)', 'Very High\n(0.7-1.0)']
    
    for label in confidence_groups.categories:
        mask = (confidence_groups == label)
        if mask.sum() > 0:
            acc = (y_val[mask] == (y_pred_proba[mask] > 0.5)).mean()
            confidence_accuracy.append(acc)
            confidence_counts.append(mask.sum())
        else:
            confidence_accuracy.append(0)
            confidence_counts.append(0)
    
    bars = ax2.bar(range(len(confidence_accuracy)), confidence_accuracy, 
                   color=['#d32f2f', '#ff6f00', '#fbc02d', '#689f38', '#388e3c'], alpha=0.7)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Accuracy by Confidence Level', fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.axhline(0.8212, color='red', linestyle='--', linewidth=2, label='Overall Accuracy')
    ax2.grid(alpha=0.3, axis='y')
    ax2.legend()
    
    # 添加样本数标签
    for i, (bar, count) in enumerate(zip(bars, confidence_counts)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.1%}\n(n={count})', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('analysis/figures/prediction_confidence.png', dpi=300, bbox_inches='tight')
    print("✅ 保存: analysis/figures/prediction_confidence.png")
    plt.close()
    
    # 图3: 学习曲线
    plt.figure(figsize=(12, 8))
    plt.plot(train_sizes, train_mean, 'o-', color='#2196F3', linewidth=2, 
             markersize=8, label='Training Score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                     alpha=0.2, color='#2196F3')
    
    plt.plot(train_sizes, val_mean, 'o-', color='#FF5722', linewidth=2,
             markersize=8, label='Cross-Validation Score')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                     alpha=0.2, color='#FF5722')
    
    plt.axhline(0.90, color='green', linestyle='--', linewidth=2, label='Target (90%)')
    plt.axhline(train_mean[-1], color='blue', linestyle=':', alpha=0.5)
    plt.axhline(val_mean[-1], color='red', linestyle=':', alpha=0.5)
    
    plt.xlabel('Training Set Size', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy Score', fontsize=14, fontweight='bold')
    plt.title('Learning Curve - Random Forest', fontsize=16, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(alpha=0.3)
    plt.ylim(0.7, 1.0)
    
    # 添加诊断文本
    gap = train_mean[-1] - val_mean[-1]
    diagnosis = f"Train: {train_mean[-1]:.3f} | Val: {val_mean[-1]:.3f} | Gap: {gap:.3f}"
    plt.text(0.02, 0.98, diagnosis, transform=plt.gca().transAxes,
             fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', 
             facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('analysis/figures/learning_curve.png', dpi=300, bbox_inches='tight')
    print("✅ 保存: analysis/figures/learning_curve.png")
    plt.close()
    
    # 图4: 特征重要性对比
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Feature Importance Comparison', fontsize=16, fontweight='bold')
    
    # 4.1 基础特征重要性
    ax1 = axes[0]
    top_features = feature_importance.head(15)
    y_pos = np.arange(len(top_features))
    ax1.barh(y_pos, top_features['importance'], color='steelblue', alpha=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(top_features['feature'])
    ax1.invert_yaxis()
    ax1.set_xlabel('Importance (Gini)', fontsize=12)
    ax1.set_title('Tree-based Feature Importance', fontweight='bold')
    ax1.grid(alpha=0.3, axis='x')
    
    # 4.2 Permutation重要性
    ax2 = axes[1]
    top_perm_features = perm_feature_importance.head(15)
    y_pos = np.arange(len(top_perm_features))
    ax2.barh(y_pos, top_perm_features['importance_mean'], 
             xerr=top_perm_features['importance_std'], 
             color='coral', alpha=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(top_perm_features['feature'])
    ax2.invert_yaxis()
    ax2.set_xlabel('Importance (Permutation)', fontsize=12)
    ax2.set_title('Permutation Feature Importance', fontweight='bold')
    ax2.grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('analysis/figures/feature_importance_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ 保存: analysis/figures/feature_importance_comparison.png")
    plt.close()

def generate_optimization_report(error_df, y_val, y_pred_proba, 
                                train_mean, val_mean, feature_importance):
    """生成优化建议报告"""
    print("\n" + "=" * 60)
    print("📝 生成优化建议报告")
    print("=" * 60)
    
    # 计算关键指标
    total_errors = len(error_df)
    error_rate = total_errors / len(y_val)
    uncertain_count = ((y_pred_proba > 0.4) & (y_pred_proba < 0.6)).sum()
    gap = train_mean[-1] - val_mean[-1]
    
    fp_count = len(error_df[error_df['error_type'] == 'False Positive'])
    fn_count = len(error_df[error_df['error_type'] == 'False Negative'])
    
    report = f"""# 模型深度评估报告 - Round 1

**评估日期**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}  
**模型**: Random Forest  
**当前验证集准确率**: 82.12%  
**目标准确率**: 90.00%  
**性能差距**: 7.88%

---

## 📊 1. 错误分析总结

### 1.1 错误统计

| 指标 | 数值 | 占比 |
|------|------|------|
| **总错误数** | {total_errors} | {error_rate:.2%} |
| 假阳性 (FP) | {fp_count} | {fp_count/total_errors:.1%} |
| 假阴性 (FN) | {fn_count} | {fn_count/total_errors:.1%} |
| 不确定样本 (0.4-0.6) | {uncertain_count} | {uncertain_count/len(y_val):.1%} |

### 1.2 错误模式识别

#### 🔴 假阳性 (False Positive) - 预测存活但实际未存活

**典型特征**:
"""
    
    # 分析假阳性样本的特征
    fp_samples = error_df[error_df['error_type'] == 'False Positive']
    if len(fp_samples) > 0:
        report += "\n"
        for feat in ['Sex_Encoded', 'Pclass', 'Age', 'Fare']:
            if feat in fp_samples.columns:
                mean_val = fp_samples[feat].mean()
                report += f"- {feat}: 均值 {mean_val:.2f}\n"
        
        report += f"\n**分析**: {fp_count} 个假阳性样本中,模型倾向于高估某些乘客的存活概率。"
    
    report += """

#### 🔵 假阴性 (False Negative) - 预测未存活但实际存活

**典型特征**:
"""
    
    # 分析假阴性样本的特征
    fn_samples = error_df[error_df['error_type'] == 'False Negative']
    if len(fn_samples) > 0:
        report += "\n"
        for feat in ['Sex_Encoded', 'Pclass', 'Age', 'Fare']:
            if feat in fn_samples.columns:
                mean_val = fn_samples[feat].mean()
                report += f"- {feat}: 均值 {mean_val:.2f}\n"
        
        report += f"\n**分析**: {fn_count} 个假阴性样本中,模型倾向于低估某些乘客的存活概率。"
    
    report += """

### 1.3 关键发现

1. **预测不确定性高**: {uncertain_rate:.1%} 的样本预测概率在 0.4-0.6 之间,说明模型在边界案例上犹豫不决
2. **特征表达不足**: 当前特征可能无法充分区分某些相似样本
3. **决策边界模糊**: 需要更清晰的特征组合来提升决策置信度

---

## 🔍 2. 性能瓶颈诊断

### 2.1 模型拟合状态

| 指标 | 训练集 | 验证集 | 差距 |
|------|--------|--------|------|
| **准确率** | {train_acc:.4f} | {val_acc:.4f} | {gap:.4f} |

**诊断结果**: {diagnosis}

### 2.2 瓶颈分析

#### ⚠️ 问题1: 模型容量不足
- 当前随机森林可能无法捕捉所有复杂模式
- 验证集准确率离目标还有 7.88% 的差距
- 需要更强的模型或更好的特征

#### ⚠️ 问题2: 特征工程不充分
- 现有特征主要是单一特征,缺乏交互项
- Top特征 (Title, Sex, Fare) 虽然重要,但单独使用可能遗漏组合信息
- 年龄、票价等连续特征的分箱可能不够细致

#### ⚠️ 问题3: 超参数未充分优化
- 当前使用的是固定参数 (n_estimators=200, max_depth=10)
- 未进行系统的网格搜索或贝叶斯优化
- 可能存在更优的参数组合

#### ⚠️ 问题4: 单模型局限性
- 依赖单一算法 (Random Forest)
- 未尝试集成多个模型的优势
- 缺少模型多样性

---

## 🚀 3. 优化方案排序

基于错误分析和瓶颈诊断,按**预期提升效果**排序的优化方案:

### 优先级1: 高级特征工程 ⭐⭐⭐⭐⭐
**预期提升**: 2-3%

**实施方案**:

#### 3.1.1 特征交互项
```python
# 关键交互特征
X['Sex_Pclass'] = X['Sex_Encoded'] * X['Pclass']
X['Sex_Age'] = X['Sex_Encoded'] * X['Age']
X['Fare_Pclass'] = X['Fare'] * X['Pclass']
X['Title_Pclass'] = X['Title_Encoded'] * X['Pclass']
X['FamilySize_Pclass'] = X['FamilySize'] * X['Pclass']

# 家庭组合
X['IsAlone'] = (X['FamilySize'] == 1).astype(int)
X['FamilyType'] = pd.cut(X['FamilySize'], bins=[0, 1, 4, 20], 
                         labels=['Alone', 'Small', 'Large'])
```

#### 3.1.2 更细粒度的分箱
```python
# 年龄细分
X['Age_Group'] = pd.cut(X['Age'], 
                       bins=[0, 5, 12, 18, 35, 60, 100],
                       labels=['Infant', 'Child', 'Teen', 'Young', 'Adult', 'Senior'])

# 票价细分 (基于四分位数)
X['Fare_Quartile'] = pd.qcut(X['Fare'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

# 船舱等级细分
X['Cabin_Deck'] = X['Cabin'].str[0]  # 提取船舱甲板字母
```

#### 3.1.3 领域知识特征
```python
# 女性和儿童优先原则
X['Woman_or_Child'] = ((X['Sex_Encoded'] == 0) | (X['Age'] < 18)).astype(int)

# 高价票持有者
X['High_Fare'] = (X['Fare'] > X['Fare'].quantile(0.75)).astype(int)

# 标题与舱位的一致性
X['Title_Pclass_Match'] = (
    ((X['Title_Encoded'] == 3) & (X['Pclass'] == 1)) |  # Mr + 1st class
    ((X['Title_Encoded'] == 2) & (X['Pclass'] <= 2))     # Mrs/Miss + upper class
).astype(int)
```

**预期效果**: 通过捕捉特征间的非线性关系,预计可提升 2-3%

---

### 优先级2: 集成学习 (Stacking) ⭐⭐⭐⭐
**预期提升**: 1.5-2.5%

**实施方案**:

#### 3.2.1 异质模型堆叠
```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# 基模型 (多样化)
base_models = [
    ('rf', RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42)),
    ('xgb', XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05)),
    ('gb', GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.05))
]

# 元模型
meta_model = LogisticRegression(max_iter=1000)

# Stacking
stacking_clf = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5
)
```

#### 3.2.2 Soft Voting
```python
from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=300, max_depth=12)),
        ('xgb', XGBClassifier(n_estimators=300, max_depth=6)),
        ('lgb', LGBMClassifier(n_estimators=300, max_depth=6))
    ],
    voting='soft',  # 软投票使用概率
    weights=[2, 1, 1]  # 权重可调优
)
```

**预期效果**: 组合多个模型的优势,降低单模型偏差

---

### 优先级3: 超参数精细调优 ⭐⭐⭐⭐
**预期提升**: 1-2%

**实施方案**:

#### 3.3.1 贝叶斯优化 (推荐)
```python
from skopt import BayesSearchCV
from skopt.space import Real, Integer

# 搜索空间
param_space = {{
    'n_estimators': Integer(200, 500),
    'max_depth': Integer(8, 20),
    'min_samples_split': Integer(2, 20),
    'min_samples_leaf': Integer(1, 10),
    'max_features': Real(0.5, 1.0),
    'max_samples': Real(0.7, 1.0)
}}

bayes_search = BayesSearchCV(
    RandomForestClassifier(random_state=42),
    param_space,
    n_iter=50,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)
```

#### 3.3.2 关键参数建议
- `n_estimators`: 200 → **300-500** (更多树,更稳定)
- `max_depth`: 10 → **12-15** (允许更深的树)
- `min_samples_split`: 5 → **2-10** (调整节点分裂条件)
- `max_features`: sqrt → **0.6-0.8** (控制特征采样)

**预期效果**: 找到更优参数组合,提升 1-2%

---

### 优先级4: 高级算法尝试 ⭐⭐⭐
**预期提升**: 0.5-1.5%

**实施方案**:

#### 3.4.1 CatBoost (适合类别特征)
```python
from catboost import CatBoostClassifier

cat_features = ['Pclass', 'Sex_Encoded', 'Embarked_Encoded', 'Title_Encoded']

catboost_clf = CatBoostClassifier(
    iterations=500,
    depth=8,
    learning_rate=0.05,
    cat_features=cat_features,
    verbose=False,
    random_state=42
)
```

#### 3.4.2 LightGBM (高效梯度提升)
```python
import lightgbm as lgb

lgb_clf = lgb.LGBMClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    num_leaves=50,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
```

**预期效果**: 算法本身的性能优势可能带来小幅提升

---

### 优先级5: 数据层面优化 ⭐⭐⭐
**预期提升**: 0.5-1%

**实施方案**:

#### 3.5.1 交叉验证替代单次划分
```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
print(f"CV Mean: {{cv_scores.mean():.4f}} ± {{cv_scores.std():.4f}}")
```

#### 3.5.2 SMOTE处理类别不平衡 (谨慎使用)
```python
from imblearn.over_sampling import SMOTE

# 仅在训练集使用
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
```

**注意**: Titanic数据集类别不平衡不严重 (约 1.5:1),SMOTE可能作用有限

#### 3.5.3 调整训练集比例
```python
# test_size: 0.2 → 0.15
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
```

**预期效果**: 更多训练数据,更稳健的评估

---

## 📋 4. 具体实施建议

### 4.1 第一阶段 (预计提升 3-4%)

**任务清单**:
1. ✅ 实现所有特征工程增强
2. ✅ 使用贝叶斯优化调参
3. ✅ 使用5折交叉验证

**时间估计**: 2-3小时

**代码模板**: 见附录 A

### 4.2 第二阶段 (预计再提升 2-3%)

**任务清单**:
1. ✅ 实现 Stacking 集成
2. ✅ 尝试 CatBoost/LightGBM
3. ✅ 融合多模型预测

**时间估计**: 2-3小时

**代码模板**: 见附录 B

### 4.3 评估标准

- 验证集准确率 > 85%: 第一阶段成功
- 验证集准确率 > 88%: 第二阶段成功
- 验证集准确率 ≥ 90%: 达成目标 🎯

---

## 🎯 5. 最高优先级推荐

**立即执行**:
1. **特征交互项** (Sex × Pclass, Age × Fare, Title × Pclass)
2. **Woman_or_Child 特征** (利用"女性和儿童优先"规则)
3. **超参数优化** (n_estimators=300, max_depth=12-15)

**理由**:
- 特征工程成本低,收益高
- 符合领域知识 (泰坦尼克救生规则)
- 超参数调优简单直接

**预期组合效果**: 82.12% → **85-86%** (提升 3-4%)

---

## 📚 附录

### 附录 A: 第一阶段完整代码框架

```python
# 1. 特征工程
def create_advanced_features(df):
    # 交互特征
    df['Sex_Pclass'] = df['Sex_Encoded'] * df['Pclass']
    df['Sex_Age'] = df['Sex_Encoded'] * df['Age']
    df['Fare_Pclass'] = df['Fare'] * df['Pclass']
    
    # 领域知识特征
    df['Woman_or_Child'] = ((df['Sex_Encoded'] == 0) | (df['Age'] < 18)).astype(int)
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    return df

# 2. 超参数优化
from skopt import BayesSearchCV
param_space = {{
    'n_estimators': Integer(200, 500),
    'max_depth': Integer(10, 18),
    'min_samples_split': Integer(2, 15)
}}

bayes_opt = BayesSearchCV(
    RandomForestClassifier(random_state=42),
    param_space, n_iter=30, cv=5, n_jobs=-1
)
bayes_opt.fit(X_train, y_train)

# 3. 交叉验证评估
cv_scores = cross_val_score(bayes_opt.best_estimator_, X, y, cv=5)
print(f"CV准确率: {{cv_scores.mean():.4f}} ± {{cv_scores.std():.4f}}")
```

### 附录 B: 第二阶段Stacking示例

```python
from sklearn.ensemble import StackingClassifier

base_models = [
    ('rf', RandomForestClassifier(**best_rf_params)),
    ('xgb', XGBClassifier(n_estimators=300, max_depth=6)),
    ('lgb', LGBMClassifier(n_estimators=300, max_depth=6))
]

stacking = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5
)

stacking.fit(X_train, y_train)
stacking_score = stacking.score(X_val, y_val)
print(f"Stacking准确率: {{stacking_score:.4f}}")
```

---

## 📊 6. 风险与注意事项

### ⚠️ 风险
1. **过拟合风险**: 特征过多可能导致过拟合,需密切监控验证集表现
2. **计算成本**: Stacking和贝叶斯优化计算量较大
3. **特征泄露**: 确保测试集不参与任何训练过程

### ✅ 缓解措施
- 使用交叉验证评估
- 监控训练集与验证集差距
- 特征选择去除冗余特征
- 正则化参数调优

---

**报告生成完成!** 🎉

**下一步**: 将优先级1方案传递给训练工程师,开始第一轮优化迭代!
""".format(
        uncertain_rate=uncertain_count/len(y_val)*100,
        train_acc=train_mean[-1],
        val_acc=val_mean[-1],
        gap=gap,
        diagnosis="存在轻微过拟合" if gap > 0.05 else "拟合良好"
    )
    
    # 保存报告
    with open('evaluation/round1_evaluation.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("✅ 保存: evaluation/round1_evaluation.md")
    print(f"   报告长度: {len(report)} 字符")

def main():
    """主流程"""
    print("=" * 60)
    print("🔬 模型深度评估分析 - Round 1")
    print("=" * 60)
    print("目标: 从 82.12% 提升至 90%")
    print("=" * 60 + "\n")
    
    # 创建目录
    create_directories()
    
    # 加载数据和模型
    model, X_train, X_val, y_train, y_val, feature_names = load_model_and_data()
    
    # 第1部分: 错误分析
    error_df, y_pred, y_pred_proba = analyze_errors(model, X_val, y_val, feature_names)
    
    # 第2部分: 预测置信度分析
    confidence_groups = analyze_prediction_confidence(y_val, y_pred_proba, y_pred)
    
    # 第3部分: 学习曲线分析
    train_sizes, train_mean, train_std, val_mean, val_std = analyze_learning_curve(
        model, X_train, y_train
    )
    
    # 第4部分: 特征重要性分析
    feature_importance, perm_feature_importance = analyze_feature_importance(
        model, X_val, y_val, feature_names
    )
    
    # 第5部分: 数据分布分析
    class_dist = analyze_data_distribution(X_val, y_val)
    
    # 生成可视化
    generate_visualizations(
        error_df, y_val, y_pred_proba, confidence_groups,
        train_sizes, train_mean, train_std, val_mean, val_std,
        feature_importance, perm_feature_importance
    )
    
    # 生成最终报告
    generate_optimization_report(
        error_df, y_val, y_pred_proba,
        train_mean, val_mean, feature_importance
    )
    
    print("\n" + "=" * 60)
    print("✅ 评估分析完成!")
    print("=" * 60)
    print("\n📂 输出文件:")
    print("  1. evaluation/round1_evaluation.md - 详细评估报告")
    print("  2. analysis/figures/error_analysis.png")
    print("  3. analysis/figures/prediction_confidence.png")
    print("  4. analysis/figures/learning_curve.png")
    print("  5. analysis/figures/feature_importance_comparison.png")
    print("\n🎯 下一步: 执行优先级1的特征工程方案!")

if __name__ == '__main__':
    main()
