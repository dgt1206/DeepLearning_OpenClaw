#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
优先级1优化方案 - 快速实施版
基于round1评估结果,实现最高ROI的优化方案

预期提升: 82.12% → 85-86% (3-4%)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

RANDOM_STATE = 42

def create_advanced_features(df):
    """
    创建高级特征
    基于Round1评估的核心建议
    """
    print("🔧 创建高级特征...")
    
    # ==================== 优先级最高 ====================
    # 1. Woman_or_Child - 泰坦尼克"女性和儿童优先"规则
    df['Woman_or_Child'] = ((df['Sex_Encoded'] == 0) | (df['Age'] < 18)).astype(int)
    print("  ✅ Woman_or_Child (女性和儿童优先)")
    
    # ==================== 交互特征 ====================
    # 2. Sex × Pclass - 性别与舱位的组合效应
    df['Sex_Pclass'] = df['Sex_Encoded'] * df['Pclass']
    print("  ✅ Sex_Pclass (性别×舱位)")
    
    # 3. Age × Fare - 年龄与票价的关系
    df['Age_Fare'] = df['Age'] * df['Fare']
    print("  ✅ Age_Fare (年龄×票价)")
    
    # 4. Title × Pclass - 称谓与舱位的一致性
    df['Title_Pclass'] = df['Title_Encoded'] * df['Pclass']
    print("  ✅ Title_Pclass (称谓×舱位)")
    
    # 5. FamilySize × Pclass
    df['FamilySize_Pclass'] = df['FamilySize'] * df['Pclass']
    print("  ✅ FamilySize_Pclass (家庭规模×舱位)")
    
    # ==================== 领域知识特征 ====================
    # 6. IsAlone - 独自旅行
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    print("  ✅ IsAlone (独自旅行)")
    
    # 7. High_Fare - 高价票持有者 (前25%)
    fare_q75 = df['Fare'].quantile(0.75)
    df['High_Fare'] = (df['Fare'] > fare_q75).astype(int)
    print("  ✅ High_Fare (高价票)")
    
    # 8. Title_Pclass_Match - 标题与舱位一致性
    # 假设 Title_Encoded: 0=Master, 1=Miss, 2=Mrs, 3=Mr, 4=Other
    df['Title_Pclass_Match'] = (
        ((df['Title_Encoded'] == 3) & (df['Pclass'] == 1)) |  # Mr + 1st class
        ((df['Title_Encoded'].isin([1, 2])) & (df['Pclass'] <= 2))  # Miss/Mrs + upper class
    ).astype(int)
    print("  ✅ Title_Pclass_Match (称谓舱位匹配)")
    
    # ==================== 更细粒度的分组 ====================
    # 9. 家庭类型
    df['FamilyType'] = pd.cut(
        df['FamilySize'], 
        bins=[0, 1, 4, 20], 
        labels=[0, 1, 2]  # 0=Alone, 1=Small, 2=Large
    ).astype(int)
    print("  ✅ FamilyType (家庭类型)")
    
    # 10. 年龄段细分
    df['Age_Category'] = pd.cut(
        df['Age'],
        bins=[0, 5, 12, 18, 35, 60, 100],
        labels=[0, 1, 2, 3, 4, 5]  # Infant, Child, Teen, Young, Adult, Senior
    ).astype(int)
    print("  ✅ Age_Category (年龄细分)")
    
    # 11. 票价四分位
    df['Fare_Quartile'] = pd.qcut(
        df['Fare'], 
        q=4, 
        labels=[0, 1, 2, 3],
        duplicates='drop'
    ).astype(int)
    print("  ✅ Fare_Quartile (票价分组)")
    
    print(f"✅ 新增 11 个高级特征,当前特征数: {df.shape[1]}\n")
    return df

def train_optimized_model():
    """训练优化后的模型"""
    print("=" * 60)
    print("🚀 优先级1优化方案 - 训练开始")
    print("=" * 60)
    print("目标: 82.12% → 85-86%\n")
    
    # 1. 加载原始清理后数据
    print("📂 加载数据...")
    train_df = pd.read_csv('datasets/cleaned/train_cleaned.csv')
    test_df = pd.read_csv('datasets/cleaned/test_cleaned.csv')
    print(f"训练集形状: {train_df.shape}")
    print(f"测试集形状: {test_df.shape}\n")
    
    # 2. 创建高级特征
    train_enhanced = create_advanced_features(train_df.copy())
    test_enhanced = create_advanced_features(test_df.copy())
    
    # 3. 准备训练数据
    X = train_enhanced.drop(['PassengerId', 'Survived'], axis=1, errors='ignore')
    y = train_enhanced['Survived']
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"📊 数据划分:")
    print(f"  训练集: {X_train.shape}")
    print(f"  验证集: {X_val.shape}")
    print(f"  特征数: {X.shape[1]} (原始18 + 新增11)\n")
    
    # 4. 训练优化后的Random Forest
    print("🌲 训练优化后的Random Forest...")
    print("参数调整:")
    print("  n_estimators: 200 → 300")
    print("  max_depth:    10  → 12")
    print("  其他参数保持不变\n")
    
    model = RandomForestClassifier(
        n_estimators=300,      # ↑ 从200
        max_depth=12,          # ↑ 从10
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0
    )
    
    model.fit(X_train, y_train)
    print("✅ 训练完成\n")
    
    # 5. 评估
    print("=" * 60)
    print("📊 模型评估")
    print("=" * 60)
    
    # 训练集
    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    
    # 验证集
    y_val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    
    print(f"\n基础指标:")
    print(f"  训练集准确率: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"  验证集准确率: {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"  过拟合程度:   {train_acc - val_acc:.4f}")
    
    # 对比原始模型
    original_val_acc = 0.8212
    improvement = val_acc - original_val_acc
    print(f"\n📈 性能提升:")
    print(f"  原始模型: {original_val_acc:.4f} (82.12%)")
    print(f"  优化模型: {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"  提升幅度: {improvement:+.4f} ({improvement*100:+.2f}%)")
    
    if val_acc >= 0.85:
        print("  🎉 已达到第一阶段目标 (≥85%)!")
    elif val_acc >= 0.84:
        print("  ✅ 接近第一阶段目标,继续优化!")
    else:
        print("  ⚠️ 未达预期,需要进一步调整")
    
    print(f"\n详细分类报告:")
    print(classification_report(y_val, y_val_pred, 
                               target_names=['Not Survived', 'Survived']))
    
    # 6. 交叉验证
    print("=" * 60)
    print("🔄 5折交叉验证")
    print("=" * 60)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy', n_jobs=-1)
    
    print(f"\nCV准确率: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"各折结果: {[f'{s:.4f}' for s in cv_scores]}")
    
    if cv_scores.mean() >= 0.85:
        print("✅ CV均值达标!")
    
    # 7. 特征重要性分析
    print("\n" + "=" * 60)
    print("📊 新特征重要性分析")
    print("=" * 60)
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 15 特征:")
    print(feature_importance.head(15).to_string(index=False))
    
    # 检查新特征的贡献
    new_features = [
        'Woman_or_Child', 'Sex_Pclass', 'Age_Fare', 'Title_Pclass',
        'FamilySize_Pclass', 'IsAlone', 'High_Fare', 'Title_Pclass_Match',
        'FamilyType', 'Age_Category', 'Fare_Quartile'
    ]
    
    new_feat_importance = feature_importance[feature_importance['feature'].isin(new_features)]
    print(f"\n新增特征重要性:")
    print(new_feat_importance.to_string(index=False))
    
    new_feat_total_importance = new_feat_importance['importance'].sum()
    print(f"\n新特征总贡献: {new_feat_total_importance:.4f} ({new_feat_total_importance*100:.1f}%)")
    
    # 8. 保存模型
    print("\n" + "=" * 60)
    print("💾 保存优化后的模型")
    print("=" * 60)
    
    joblib.dump(model, 'models/optimized_model_v1.pkl')
    print("✅ 保存: models/optimized_model_v1.pkl")
    
    # 保存增强后的数据
    train_enhanced.to_csv('datasets/cleaned/train_enhanced_v1.csv', index=False)
    test_enhanced.to_csv('datasets/cleaned/test_enhanced_v1.csv', index=False)
    print("✅ 保存增强数据集")
    
    # 9. 生成测试集预测
    print("\n🔮 生成测试集预测...")
    X_test = test_enhanced.drop(['PassengerId', 'Survived'], axis=1, errors='ignore')
    test_predictions = model.predict(X_test)
    
    submission = pd.DataFrame({
        'PassengerId': test_enhanced['PassengerId'],
        'Survived': test_predictions
    })
    submission.to_csv('predictions/submission_v1_optimized.csv', index=False)
    print("✅ 保存: predictions/submission_v1_optimized.csv")
    
    # 10. 总结
    print("\n" + "=" * 60)
    print("🎉 优化完成!")
    print("=" * 60)
    
    print(f"\n核心改进:")
    print(f"  ✅ 新增 11 个高级特征")
    print(f"  ✅ 参数优化 (n_estimators=300, max_depth=12)")
    print(f"  ✅ 验证集准确率: {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"  ✅ 性能提升: {improvement:+.4f} ({improvement*100:+.2f}%)")
    
    print(f"\n📂 输出文件:")
    print(f"  1. models/optimized_model_v1.pkl")
    print(f"  2. datasets/cleaned/train_enhanced_v1.csv")
    print(f"  3. datasets/cleaned/test_enhanced_v1.csv")
    print(f"  4. predictions/submission_v1_optimized.csv")
    
    if val_acc >= 0.85:
        print(f"\n🎯 下一步: 启动第二阶段 (集成学习) 冲击88%!")
    else:
        print(f"\n🎯 下一步: 微调参数或尝试更多特征交互")
    
    return model, val_acc, improvement

if __name__ == '__main__':
    model, val_acc, improvement = train_optimized_model()
