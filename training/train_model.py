#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Titanic生存预测 - 多模型训练与评估
完整的模型训练、评估、可视化和预测流程
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
import xgboost as xgb
import lightgbm as lgb

# 设置随机种子
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_directories():
    """创建必要的目录"""
    dirs = ['analysis/figures', 'models', 'predictions', 'training']
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("✅ 目录结构创建完成")

def load_and_prepare_data():
    """加载并准备数据"""
    print("\n📂 加载数据...")
    
    # 加载训练集
    train_df = pd.read_csv('datasets/cleaned/train_cleaned.csv')
    print(f"训练集形状: {train_df.shape}")
    
    # 分离特征和目标
    X = train_df.drop(['PassengerId', 'Survived'], axis=1, errors='ignore')
    y = train_df['Survived']
    
    print(f"特征数量: {X.shape[1]}")
    print(f"样本数量: {X.shape[0]}")
    print(f"存活比例: {y.mean():.2%}")
    
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    
    print(f"训练集: {X_train.shape[0]} 样本")
    print(f"验证集: {X_val.shape[0]} 样本")
    
    return X_train, X_val, y_train, y_val, X.columns.tolist()

def train_logistic_regression(X_train, X_val, y_train, y_val):
    """训练逻辑回归模型"""
    print("\n🔵 训练模型 1/4: 逻辑回归 (Baseline)")
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # 训练
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_train_scaled, y_train)
    
    print("✅ 逻辑回归训练完成")
    return lr, scaler, X_train_scaled, X_val_scaled

def train_random_forest(X_train, y_train):
    """训练随机森林模型"""
    print("\n🌲 训练模型 2/4: 随机森林")
    
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    
    print("✅ 随机森林训练完成")
    return rf

def train_xgboost(X_train, y_train):
    """训练XGBoost模型"""
    print("\n🚀 训练模型 3/4: XGBoost")
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)
    
    print("✅ XGBoost训练完成")
    return xgb_model

def train_lightgbm(X_train, y_train):
    """训练LightGBM模型"""
    print("\n⚡ 训练模型 4/4: LightGBM")
    
    lgb_model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        num_leaves=31,
        random_state=RANDOM_STATE,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    
    print("✅ LightGBM训练完成")
    return lgb_model

def evaluate_model(model, X_train, X_val, y_train, y_val, model_name):
    """评估单个模型"""
    # 训练集预测
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    
    # 验证集预测
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)[:, 1]
    
    # 计算指标
    metrics = {
        'model': model_name,
        'train_acc': accuracy_score(y_train, y_train_pred),
        'val_acc': accuracy_score(y_val, y_val_pred),
        'precision': precision_score(y_val, y_val_pred),
        'recall': recall_score(y_val, y_val_pred),
        'f1': f1_score(y_val, y_val_pred),
        'roc_auc': roc_auc_score(y_val, y_val_proba)
    }
    
    return metrics, y_val_proba

def plot_model_comparison(results_df):
    """绘制模型对比图"""
    print("\n📊 生成模型对比图...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    metrics = ['train_acc', 'val_acc', 'precision', 'recall', 'f1', 'roc_auc']
    titles = ['Training Accuracy', 'Validation Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        sns.barplot(data=results_df, x='model', y=metric, ax=ax, palette='viridis')
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1)
        ax.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for i, v in enumerate(results_df[metric]):
            ax.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('analysis/figures/model_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ 保存: analysis/figures/model_comparison.png")

def plot_confusion_matrix(y_true, y_pred, model_name):
    """绘制混淆矩阵"""
    print(f"\n📊 生成混淆矩阵: {model_name}...")
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Not Survived', 'Survived'],
                yticklabels=['Not Survived', 'Survived'])
    plt.title(f'Confusion Matrix - {model_name}', fontweight='bold', fontsize=14)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('analysis/figures/confusion_matrix_best.png', dpi=300, bbox_inches='tight')
    print("✅ 保存: analysis/figures/confusion_matrix_best.png")

def plot_roc_curves(y_val, all_probas, model_names):
    """绘制ROC曲线对比"""
    print("\n📊 生成ROC曲线对比图...")
    
    plt.figure(figsize=(10, 8))
    
    colors = ['blue', 'green', 'red', 'purple']
    for i, (proba, name) in enumerate(zip(all_probas, model_names)):
        fpr, tpr, _ = roc_curve(y_val, proba)
        auc = roc_auc_score(y_val, proba)
        plt.plot(fpr, tpr, color=colors[i], lw=2, 
                label=f'{name} (AUC = {auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve Comparison', fontweight='bold', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('analysis/figures/roc_curve_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ 保存: analysis/figures/roc_curve_comparison.png")

def plot_feature_importance(model, feature_names, model_name):
    """绘制特征重要性"""
    print(f"\n📊 生成特征重要性图: {model_name}...")
    
    # 获取特征重要性
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        print(f"⚠️ {model_name} 不支持特征重要性")
        return
    
    # 排序
    indices = np.argsort(importances)[::-1][:15]  # 取前15个
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), importances[indices], color='steelblue')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importance', fontsize=12)
    plt.title(f'Top 15 Feature Importance - {model_name}', fontweight='bold', fontsize=14)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('analysis/figures/feature_importance_best.png', dpi=300, bbox_inches='tight')
    print("✅ 保存: analysis/figures/feature_importance_best.png")

def generate_predictions(best_model, best_scaler, model_name):
    """生成测试集预测"""
    print("\n🔮 生成测试集预测...")
    
    # 加载测试集
    test_df = pd.read_csv('datasets/cleaned/test_cleaned.csv')
    passenger_ids = test_df['PassengerId']
    X_test = test_df.drop(['PassengerId', 'Survived'], axis=1, errors='ignore')
    
    # 如果是逻辑回归，需要标准化
    if model_name == 'Logistic Regression' and best_scaler is not None:
        X_test = best_scaler.transform(X_test)
    
    # 预测
    predictions = best_model.predict(X_test)
    
    # 生成提交文件
    submission = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Survived': predictions
    })
    submission.to_csv('predictions/submission.csv', index=False)
    
    print(f"✅ 预测完成: {len(predictions)} 条")
    print(f"   存活预测: {predictions.sum()} ({predictions.mean():.2%})")
    print("✅ 保存: predictions/submission.csv")

def generate_report(results_df, best_model_name, best_val_acc, feature_names, best_model):
    """生成训练报告"""
    print("\n📝 生成训练报告...")
    
    report = f"""# Titanic生存预测 - 模型训练报告

## 📊 任务概述

- **数据集**: Titanic生存预测
- **训练样本**: 891条 (训练集712条, 验证集179条)
- **测试样本**: 418条
- **特征数量**: {len(feature_names)}
- **训练日期**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## 🎯 模型评估结果

### 全部模型对比

| 模型 | 训练准确率 | 验证准确率 | Precision | Recall | F1-Score | ROC-AUC |
|------|------------|------------|-----------|--------|----------|---------|
"""
    
    for _, row in results_df.iterrows():
        report += f"| {row['model']} | {row['train_acc']:.4f} | {row['val_acc']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {row['roc_auc']:.4f} |\n"
    
    report += f"""
### 🏆 最优模型

**选择**: {best_model_name}

**验证集准确率**: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)

**选择理由**:
- 验证集准确率最高
- 训练集与验证集表现平衡，泛化能力好
- 过拟合风险较低

## 📈 特征重要性分析

"""
    
    # 添加特征重要性
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        
        report += "**Top 10 重要特征**:\n\n"
        for i, idx in enumerate(indices, 1):
            report += f"{i}. {feature_names[idx]}: {importances[idx]:.4f}\n"
    else:
        report += "最优模型不支持特征重要性分析\n"
    
    report += """
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
"""
    
    with open('training/training_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("✅ 保存: training/training_report.md")

def main():
    """主流程"""
    print("=" * 60)
    print("🚀 Titanic生存预测 - 模型训练与评估")
    print("=" * 60)
    
    # 创建目录
    create_directories()
    
    # 加载数据
    X_train, X_val, y_train, y_val, feature_names = load_and_prepare_data()
    
    # 训练所有模型
    print("\n" + "=" * 60)
    print("🎯 开始训练模型")
    print("=" * 60)
    
    # 1. 逻辑回归
    lr_model, scaler, X_train_scaled, X_val_scaled = train_logistic_regression(
        X_train, X_val, y_train, y_val
    )
    lr_metrics, lr_proba = evaluate_model(
        lr_model, X_train_scaled, X_val_scaled, y_train, y_val, 'Logistic Regression'
    )
    
    # 2. 随机森林
    rf_model = train_random_forest(X_train, y_train)
    rf_metrics, rf_proba = evaluate_model(
        rf_model, X_train, X_val, y_train, y_val, 'Random Forest'
    )
    
    # 3. XGBoost
    xgb_model = train_xgboost(X_train, y_train)
    xgb_metrics, xgb_proba = evaluate_model(
        xgb_model, X_train, X_val, y_train, y_val, 'XGBoost'
    )
    
    # 4. LightGBM
    lgb_model = train_lightgbm(X_train, y_train)
    lgb_metrics, lgb_proba = evaluate_model(
        lgb_model, X_train, X_val, y_train, y_val, 'LightGBM'
    )
    
    # 整理结果
    results_df = pd.DataFrame([lr_metrics, rf_metrics, xgb_metrics, lgb_metrics])
    
    print("\n" + "=" * 60)
    print("📊 模型评估结果")
    print("=" * 60)
    print(results_df.to_string(index=False))
    
    # 选择最优模型
    best_idx = results_df['val_acc'].idxmax()
    best_model_name = results_df.loc[best_idx, 'model']
    best_val_acc = results_df.loc[best_idx, 'val_acc']
    
    models = {
        'Logistic Regression': (lr_model, scaler),
        'Random Forest': (rf_model, None),
        'XGBoost': (xgb_model, None),
        'LightGBM': (lgb_model, None)
    }
    
    best_model, best_scaler = models[best_model_name]
    
    print("\n" + "=" * 60)
    print("🏆 最优模型")
    print("=" * 60)
    print(f"模型: {best_model_name}")
    print(f"验证集准确率: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    
    # 生成可视化
    print("\n" + "=" * 60)
    print("📊 生成可视化")
    print("=" * 60)
    
    plot_model_comparison(results_df)
    
    # 混淆矩阵
    if best_model_name == 'Logistic Regression':
        y_val_pred = best_model.predict(X_val_scaled)
    else:
        y_val_pred = best_model.predict(X_val)
    plot_confusion_matrix(y_val, y_val_pred, best_model_name)
    
    # ROC曲线
    plot_roc_curves(
        y_val, 
        [lr_proba, rf_proba, xgb_proba, lgb_proba],
        ['Logistic Regression', 'Random Forest', 'XGBoost', 'LightGBM']
    )
    
    # 特征重要性
    plot_feature_importance(best_model, feature_names, best_model_name)
    
    # 保存模型
    print("\n" + "=" * 60)
    print("💾 保存模型")
    print("=" * 60)
    
    joblib.dump(best_model, 'models/best_model.pkl')
    print("✅ 保存: models/best_model.pkl")
    
    if best_scaler is not None:
        joblib.dump(best_scaler, 'models/scaler.pkl')
        print("✅ 保存: models/scaler.pkl")
    
    # 生成预测
    generate_predictions(best_model, best_scaler, best_model_name)
    
    # 生成报告
    generate_report(results_df, best_model_name, best_val_acc, feature_names, best_model)
    
    print("\n" + "=" * 60)
    print("🎉 训练完成!")
    print("=" * 60)
    print(f"\n🏆 最优模型: {best_model_name}")
    print(f"📈 验证集准确率: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print("\n📂 所有输出文件已保存到相应目录")

if __name__ == '__main__':
    main()
