#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Round 2 优化训练脚本
实现方案A (特征筛选+正则化) 和 方案B (XGBoost)
目标: 83.61% → 86-87%+

使用方法:
    python train_optimized_v2.py --mode rf      # 方案A: Random Forest优化
    python train_optimized_v2.py --mode xgb     # 方案B: XGBoost
    python train_optimized_v2.py --mode both    # 两者对比 (推荐)
"""

import argparse
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42

def load_and_clean_data():
    """加载并清理数据"""
    print("📂 加载数据...")
    
    # 加载 Round 1 的增强数据
    train_df = pd.read_csv('datasets/cleaned/train_enhanced_v1.csv')
    test_df = pd.read_csv('datasets/cleaned/test_enhanced_v1.csv')
    
    print(f"✓ 训练集: {train_df.shape}")
    print(f"✓ 测试集: {test_df.shape}\n")
    
    return train_df, test_df

def feature_pruning(train_df, test_df):
    """方案A: 特征筛选"""
    print("✂️ 方案A: 特征筛选...")
    
    # 基于 Round 1 诊断,移除4个低贡献特征
    features_to_remove = [
        'Age_Group_Encoded',    # 0.97% - 与Age/Age_Category重复
        'High_Fare',            # 0.71% - 与Fare/Fare_Quartile重复  
        'Title_Pclass_Match',   # 0.71% - 手工规则失败
        'IsAlone'               # 0.49% - 与FamilySize完全冗余
    ]
    
    print(f"删除特征: {features_to_remove}")
    
    # 检查特征是否存在
    existing_to_remove = [f for f in features_to_remove if f in train_df.columns]
    
    train_cleaned = train_df.drop(columns=existing_to_remove, errors='ignore')
    test_cleaned = test_df.drop(columns=existing_to_remove, errors='ignore')
    
    print(f"✓ 特征数: {train_df.shape[1]} → {train_cleaned.shape[1]} (移除 {len(existing_to_remove)} 个)\n")
    
    return train_cleaned, test_cleaned

def train_random_forest_v2(X_train, y_train):
    """方案A: 优化的Random Forest"""
    print("🌲 训练优化的Random Forest (方案A)...")
    print("参数调整:")
    print("  n_estimators:      300  →  400")
    print("  max_depth:         12   →  8   ❗关键")
    print("  min_samples_split: 5    →  10")
    print("  min_samples_leaf:  2    →  4")
    print("  max_features:      None →  'sqrt'  ❗新增")
    print("  class_weight:      None →  'balanced'\n")
    
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=8,                    # 防过拟合
        min_samples_split=10,
        min_samples_leaf=4,
        max_features='sqrt',            # 特征采样
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0
    )
    
    model.fit(X_train, y_train)
    print("✅ Random Forest 训练完成\n")
    
    return model

def train_xgboost(X_train, y_train, X_val=None, y_val=None):
    """方案B: XGBoost模型"""
    try:
        import xgboost as xgb
    except ImportError:
        print("❌ XGBoost 未安装,请运行: pip install xgboost")
        return None
    
    print("🚀 训练 XGBoost (方案B)...")
    print("参数:")
    print("  n_estimators:      500")
    print("  max_depth:         5    (浅树)")
    print("  learning_rate:     0.01 (小学习率)")
    print("  subsample:         0.8  (行采样)")
    print("  colsample_bytree:  0.8  (列采样)")
    print("  reg_alpha:         1    (L1正则)")
    print("  reg_lambda:        1    (L2正则)")
    print("  min_child_weight:  3")
    print("  gamma:             1\n")
    
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1,
        reg_lambda=1,
        min_child_weight=3,
        gamma=1,
        scale_pos_weight=1,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0
    )
    
    # 如果有验证集,使用早停
    if X_val is not None and y_val is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
    else:
        model.fit(X_train, y_train)
    
    print("✅ XGBoost 训练完成\n")
    
    return model

def evaluate_model(model, X, y, model_name="Model"):
    """全面评估模型"""
    print(f"=" * 70)
    print(f"📊 {model_name} 评估")
    print("=" * 70)
    
    # 1. 5折交叉验证
    print("\n🔄 5折交叉验证...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy', n_jobs=-1)
    
    print(f"各折得分: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"平均准确率: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"最大-最小差: {(cv_scores.max() - cv_scores.min())*100:.2f}%")
    
    # 2. CV预测 (无泄漏的验证集预测)
    y_pred_cv = cross_val_predict(model, X, y, cv=skf)
    cv_accuracy = accuracy_score(y, y_pred_cv)
    
    print(f"\nCV预测准确率: {cv_accuracy:.4f} ({cv_accuracy*100:.2f}%)")
    
    # 3. 分类报告
    print(f"\n详细分类报告:")
    print(classification_report(y, y_pred_cv, 
                               target_names=['Not Survived', 'Survived'],
                               digits=4))
    
    # 4. 混淆矩阵
    cm = confusion_matrix(y, y_pred_cv)
    print("混淆矩阵:")
    print(f"  预测 Not Survived | Survived")
    print(f"真实 Not Survived: {cm[0][0]:4d}         {cm[0][1]:4d}")
    print(f"真实 Survived:     {cm[1][0]:4d}         {cm[1][1]:4d}")
    
    # 5. 与Round 1对比
    round1_cv = 0.8361
    improvement = cv_accuracy - round1_cv
    
    print(f"\n📈 性能对比:")
    print(f"  Round 1 CV:  {round1_cv:.4f} (83.61%)")
    print(f"  Round 2 CV:  {cv_accuracy:.4f} ({cv_accuracy*100:.2f}%)")
    print(f"  提升幅度:    {improvement:+.4f} ({improvement*100:+.2f}%)")
    
    if cv_accuracy >= 0.87:
        print("  🎉🎉 已达到目标! (≥87%)")
    elif cv_accuracy >= 0.86:
        print("  ✅✅ 接近目标! (≥86%)")
    elif cv_accuracy >= 0.85:
        print("  ✅ 有进步,继续优化 (≥85%)")
    else:
        print("  ⚠️ 未达预期,需调整策略")
    
    return {
        'cv_mean': cv_accuracy,
        'cv_std': cv_scores.std(),
        'cv_scores': cv_scores.tolist(),
        'improvement': improvement,
        'confusion_matrix': cm.tolist()
    }

def plot_comparison(results_dict, save_path='evaluation/round2_comparison.png'):
    """可视化对比不同方案"""
    print("\n📊 生成对比可视化...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 图1: CV准确率对比
    ax1 = axes[0, 0]
    models = list(results_dict.keys())
    cv_means = [results_dict[m]['cv_mean'] for m in models]
    cv_stds = [results_dict[m]['cv_std'] for m in models]
    
    x = np.arange(len(models))
    bars = ax1.bar(x, cv_means, yerr=cv_stds, capsize=5, alpha=0.8, edgecolor='black')
    
    # 颜色编码
    colors = ['steelblue' if cv >= 0.86 else 'orange' for cv in cv_means]
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax1.axhline(0.8361, color='red', linestyle='--', linewidth=2, label='Round 1 (83.61%)')
    ax1.axhline(0.86, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Target (86%)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Round 2 Model Comparison - CV Mean', fontsize=14, fontweight='bold')
    ax1.set_ylim([0.80, 0.90])
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 在柱子上显示数值
    for i, (mean, std) in enumerate(zip(cv_means, cv_stds)):
        ax1.text(i, mean + std + 0.005, f'{mean:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 图2: 提升幅度对比
    ax2 = axes[0, 1]
    improvements = [results_dict[m]['improvement'] * 100 for m in models]
    bars2 = ax2.bar(x, improvements, alpha=0.8, edgecolor='black')
    
    for bar, imp in zip(bars2, improvements):
        bar.set_color('green' if imp > 0 else 'red')
    
    ax2.axhline(0, color='black', linewidth=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=12)
    ax2.set_ylabel('Improvement (%)', fontsize=12)
    ax2.set_title('Improvement vs Round 1', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for i, imp in enumerate(improvements):
        ax2.text(i, imp + 0.2, f'{imp:+.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # 图3: CV各折详情 (取第一个模型)
    ax3 = axes[1, 0]
    first_model = models[0]
    cv_scores = results_dict[first_model]['cv_scores']
    ax3.bar(range(1, 6), cv_scores, color='steelblue', edgecolor='black', alpha=0.8)
    ax3.axhline(np.mean(cv_scores), color='red', linestyle='--', linewidth=2, 
               label=f"Mean: {np.mean(cv_scores):.4f}")
    ax3.set_xlabel('Fold', fontsize=12)
    ax3.set_ylabel('Accuracy', fontsize=12)
    ax3.set_title(f'{first_model} - 5-Fold CV Scores', fontsize=12, fontweight='bold')
    ax3.set_xticks(range(1, 6))
    ax3.set_ylim([0.75, 0.90])
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 图4: Round 0/1/2 进展曲线
    ax4 = axes[1, 1]
    rounds = ['Round 0\n(Baseline)', 'Round 1\n(Feature Eng.)', f'Round 2\n({models[0]})']
    round_scores = [0.8212, 0.8361, cv_means[0]]
    
    ax4.plot(rounds, round_scores, 'o-', linewidth=3, markersize=12, color='purple')
    ax4.axhline(0.90, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Goal (90%)')
    ax4.set_ylabel('Accuracy', fontsize=12)
    ax4.set_title('Progress Across Rounds', fontsize=14, fontweight='bold')
    ax4.set_ylim([0.80, 0.92])
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3, axis='y')
    
    for i, score in enumerate(round_scores):
        ax4.text(i, score + 0.005, f'{score:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ 保存到 {save_path}\n")

def main():
    parser = argparse.ArgumentParser(description='Round 2 优化训练')
    parser.add_argument('--mode', type=str, default='both', 
                       choices=['rf', 'xgb', 'both'],
                       help='训练模式: rf(方案A), xgb(方案B), both(对比)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("🚀 Round 2 优化训练")
    print("=" * 70)
    print(f"模式: {args.mode.upper()}")
    print(f"目标: 83.61% → 86-87%+\n")
    
    # 1. 加载数据
    train_df, test_df = load_and_clean_data()
    
    # 2. 特征筛选
    train_cleaned, test_cleaned = feature_pruning(train_df, test_df)
    
    # 3. 准备训练数据
    X = train_cleaned.drop(['PassengerId', 'Survived'], axis=1, errors='ignore')
    y = train_cleaned['Survived']
    
    print(f"📊 最终数据:")
    print(f"  样本数: {X.shape[0]}")
    print(f"  特征数: {X.shape[1]}")
    print(f"  特征列表: {list(X.columns)}\n")
    
    results = {}
    models = {}
    
    # 4. 训练模型
    if args.mode in ['rf', 'both']:
        print("\n" + "=" * 70)
        print("方案A: Random Forest 优化")
        print("=" * 70 + "\n")
        
        model_rf = train_random_forest_v2(X, y)
        results['RF_Optimized'] = evaluate_model(model_rf, X, y, "Random Forest (方案A)")
        models['RF_Optimized'] = model_rf
    
    if args.mode in ['xgb', 'both']:
        print("\n" + "=" * 70)
        print("方案B: XGBoost")
        print("=" * 70 + "\n")
        
        model_xgb = train_xgboost(X, y)
        if model_xgb:
            results['XGBoost'] = evaluate_model(model_xgb, X, y, "XGBoost (方案B)")
            models['XGBoost'] = model_xgb
    
    # 5. 对比可视化
    if len(results) > 0:
        plot_comparison(results)
    
    # 6. 选择最佳模型并保存
    print("\n" + "=" * 70)
    print("💾 保存最佳模型")
    print("=" * 70)
    
    best_model_name = max(results.keys(), key=lambda k: results[k]['cv_mean'])
    best_model = models[best_model_name]
    best_score = results[best_model_name]['cv_mean']
    
    print(f"\n最佳模型: {best_model_name}")
    print(f"CV准确率: {best_score:.4f} ({best_score*100:.2f}%)")
    
    # 保存模型
    joblib.dump(best_model, 'models/best_model_round2.pkl')
    print("✓ 保存: models/best_model_round2.pkl")
    
    # 保存清理后的数据
    train_cleaned.to_csv('datasets/cleaned/train_cleaned_round2.csv', index=False)
    test_cleaned.to_csv('datasets/cleaned/test_cleaned_round2.csv', index=False)
    print("✓ 保存清理后的数据")
    
    # 保存评估结果
    results['best_model'] = best_model_name
    results['best_score'] = best_score
    
    with open('evaluation/round2_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("✓ 保存: evaluation/round2_results.json")
    
    # 7. 生成测试集预测
    print("\n🔮 生成测试集预测...")
    X_test = test_cleaned.drop(['PassengerId', 'Survived'], axis=1, errors='ignore')
    test_predictions = best_model.predict(X_test)
    
    submission = pd.DataFrame({
        'PassengerId': test_cleaned['PassengerId'],
        'Survived': test_predictions
    })
    submission.to_csv('predictions/submission_round2.csv', index=False)
    print("✓ 保存: predictions/submission_round2.csv")
    
    # 8. 生成特征重要性图
    if hasattr(best_model, 'feature_importances_'):
        print("\n📊 生成特征重要性图...")
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(12, 8))
        top20 = feature_importance.head(20)
        plt.barh(range(len(top20)), top20['importance'], color='steelblue', edgecolor='black')
        plt.yticks(range(len(top20)), top20['feature'])
        plt.xlabel('Importance', fontsize=12)
        plt.title(f'{best_model_name} - Top 20 Feature Importance', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig('evaluation/round2_feature_importance.png', dpi=150, bbox_inches='tight')
        print("✓ 保存: evaluation/round2_feature_importance.png")
    
    # 9. 总结
    print("\n" + "=" * 70)
    print("🎉 Round 2 训练完成!")
    print("=" * 70)
    
    print(f"\n核心改进:")
    print(f"  ✅ 移除 4 个冗余特征")
    print(f"  ✅ 强化正则化 (max_depth 12→8, 新增 max_features)")
    print(f"  ✅ 使用 5折CV 替代单次划分")
    print(f"  ✅ 最佳模型: {best_model_name}")
    print(f"  ✅ CV准确率: {best_score:.4f} ({best_score*100:.2f}%)")
    print(f"  ✅ 提升幅度: {results[best_model_name]['improvement']:+.4f} ({results[best_model_name]['improvement']*100:+.2f}%)")
    
    print(f"\n📂 输出文件:")
    print(f"  1. models/best_model_round2.pkl")
    print(f"  2. evaluation/round2_results.json")
    print(f"  3. evaluation/round2_comparison.png")
    print(f"  4. evaluation/round2_feature_importance.png")
    print(f"  5. predictions/submission_round2.csv")
    
    if best_score >= 0.87:
        print(f"\n🎯 下一步: 启动 Round 3 (Stacking) 冲击 90%!")
    elif best_score >= 0.86:
        print(f"\n🎯 下一步: 微调参数或添加高级特征")
    else:
        print(f"\n🎯 下一步: 分析失败原因,调整策略")

if __name__ == '__main__':
    main()
