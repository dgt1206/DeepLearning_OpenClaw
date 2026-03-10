#!/usr/bin/env python3
"""
Round 5: 特征工程深挖
目标: 87%
策略: 领域知识 + 特征交互 + 文本特征
"""

import pandas as pd
import numpy as np
import joblib
import gc
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("Round 5: 特征工程深挖")
print("=" * 80)

def clean_memory():
    gc.collect()
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except:
        pass
    print("✅ 内存已清理")

# ============================================================================
# 1. 加载原始数据
# ============================================================================
print("\n[1/5] 加载原始数据...")
train_df = pd.read_csv('datasets/train.csv')
test_df = pd.read_csv('datasets/test.csv')

print(f"训练集: {len(train_df)}, 测试集: {len(test_df)}")

# 合并以便统一特征工程
test_df['Survived'] = -1
combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)

# ============================================================================
# 2. 高级特征工程
# ============================================================================
print("\n[2/5] 特征工程...")

# 基础清理
combined['Age'].fillna(combined.groupby(['Pclass', 'Sex'])['Age'].transform('median'), inplace=True)
combined['Fare'].fillna(combined.groupby('Pclass')['Fare'].transform('median'), inplace=True)
combined['Embarked'].fillna('S', inplace=True)

# ===== 1. 领域知识特征 =====
print("  - 领域知识特征...")

# 女性和儿童优先规则（泰坦尼克号核心规则）
combined['Is_Woman'] = (combined['Sex'] == 'female').astype(int)
combined['Is_Child'] = (combined['Age'] < 18).astype(int)
combined['Woman_or_Child'] = ((combined['Sex'] == 'female') | (combined['Age'] < 18)).astype(int)

# 富有成年男性（最危险群体）
combined['Rich_Adult_Male'] = (
    (combined['Sex'] == 'male') & 
    (combined['Age'] >= 18) & 
    (combined['Fare'] > combined['Fare'].quantile(0.75))
).astype(int)

# 家庭规模特征
combined['FamilySize'] = combined['SibSp'] + combined['Parch'] + 1
combined['IsAlone'] = (combined['FamilySize'] == 1).astype(int)
combined['SmallFamily'] = (combined['FamilySize'] <= 3).astype(int)
combined['LargeFamily'] = (combined['FamilySize'] >= 5).astype(int)

# ===== 2. 文本特征深挖 =====
print("  - 文本特征...")

# Title 提取（更细粒度）
combined['Title'] = combined['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
# 合并稀有 Title
title_mapping = {
    'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
    'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
    'Mlle': 'Miss', 'Mme': 'Mrs', 'Ms': 'Miss',
    'Don': 'Rare', 'Dona': 'Rare', 'Lady': 'Rare', 'Countess': 'Rare',
    'Jonkheer': 'Rare', 'Sir': 'Rare', 'Capt': 'Rare'
}
combined['Title'] = combined['Title'].map(title_mapping).fillna('Rare')

# Name 长度（社会地位指标）
combined['Name_Length'] = combined['Name'].str.len()

# Cabin 特征
combined['Has_Cabin'] = combined['Cabin'].notna().astype(int)
combined['Deck'] = combined['Cabin'].str[0].fillna('Unknown')
combined['Cabin_Multiple'] = combined['Cabin'].apply(lambda x: 0 if pd.isna(x) else len(str(x).split(' ')))

# Ticket 特征
combined['Ticket_Prefix'] = combined['Ticket'].apply(lambda x: str(x).split()[0] if len(str(x).split()) > 1 else 'None')
combined['Ticket_Frequency'] = combined.groupby('Ticket')['Ticket'].transform('count')
combined['Ticket_Length'] = combined['Ticket'].str.len()

# ===== 3. 特征交互 =====
print("  - 特征交互...")

# 核心交互
combined['Sex_Pclass'] = combined['Sex'].map({'male': 0, 'female': 1}) * 10 + combined['Pclass']
combined['Age_Pclass'] = combined['Age'] * combined['Pclass']
combined['Age_Fare'] = combined['Age'] * combined['Fare']
combined['Fare_Per_Person'] = combined['Fare'] / combined['FamilySize']

# Title × Pclass
le_title = LabelEncoder()
combined['Title_Encoded'] = le_title.fit_transform(combined['Title'])
combined['Title_Pclass'] = combined['Title_Encoded'] * 10 + combined['Pclass']

# Age × FamilySize
combined['Age_FamilySize'] = combined['Age'] * combined['FamilySize']

# 三路交互
combined['Sex_Pclass_Age'] = (
    combined['Sex'].map({'male': 0, 'female': 1}) * 100 + 
    combined['Pclass'] * 10 + 
    combined['Age'] // 10
)

# ===== 4. 分箱特征 =====
print("  - 分箱特征...")

# Age 分组
combined['Age_Group'] = pd.cut(combined['Age'], bins=[0, 12, 18, 35, 60, 100], 
                               labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])

# Fare 分组
combined['Fare_Group'] = pd.qcut(combined['Fare'].rank(method='first'), q=5, 
                                 labels=['Very_Low', 'Low', 'Medium', 'High', 'Very_High'])

# ===== 5. 编码 =====
print("  - 编码...")

# 类别特征编码
le_sex = LabelEncoder()
le_embarked = LabelEncoder()
le_deck = LabelEncoder()
le_ticket_prefix = LabelEncoder()
le_age_group = LabelEncoder()
le_fare_group = LabelEncoder()

combined['Sex_Encoded'] = le_sex.fit_transform(combined['Sex'])
combined['Embarked_Encoded'] = le_embarked.fit_transform(combined['Embarked'])
combined['Deck_Encoded'] = le_deck.fit_transform(combined['Deck'])
combined['Ticket_Prefix_Encoded'] = le_ticket_prefix.fit_transform(combined['Ticket_Prefix'])
combined['Age_Group_Encoded'] = le_age_group.fit_transform(combined['Age_Group'])
combined['Fare_Group_Encoded'] = le_fare_group.fit_transform(combined['Fare_Group'])

# 选择特征
feature_cols = [
    'Pclass', 'Sex_Encoded', 'Age', 'SibSp', 'Parch', 'Fare',
    'Embarked_Encoded', 'Has_Cabin', 'Deck_Encoded',
    'Is_Woman', 'Is_Child', 'Woman_or_Child', 'Rich_Adult_Male',
    'FamilySize', 'IsAlone', 'SmallFamily', 'LargeFamily',
    'Title_Encoded', 'Name_Length', 'Cabin_Multiple',
    'Ticket_Prefix_Encoded', 'Ticket_Frequency', 'Ticket_Length',
    'Sex_Pclass', 'Age_Pclass', 'Age_Fare', 'Fare_Per_Person',
    'Title_Pclass', 'Age_FamilySize', 'Sex_Pclass_Age',
    'Age_Group_Encoded', 'Fare_Group_Encoded'
]

print(f"✅ 总特征数: {len(feature_cols)}")

# 分离训练集和测试集
train = combined[combined['Survived'] != -1].copy()
test = combined[combined['Survived'] == -1].copy()

X = train[feature_cols]
y = train['Survived']
X_test = test[feature_cols]

clean_memory()

# ============================================================================
# 3. 训练模型
# ============================================================================
print("\n[3/5] 训练 XGBoost...")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,  # 增加深度以捕捉复杂交互
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

model.fit(X_train, y_train)

# 验证集评估
val_score = model.score(X_val, y_val)
print(f"验证集准确率: {val_score*100:.2f}%")

# 5折交叉验证
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=-1)
print(f"5折CV准确率: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

clean_memory()

# ============================================================================
# 4. 保存结果
# ============================================================================
print("\n[4/5] 保存结果...")

joblib.dump(model, 'models/best_model_round5.pkl')
print("✅ 模型已保存")

# 预测
test_pred = model.predict(X_test)
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'].astype(int),
    'Survived': test_pred.astype(int)
})

os.makedirs('predictions', exist_ok=True)
submission.to_csv('predictions/submission_round5.csv', index=False)
print("✅ 预测已保存")

# 报告
report = f"""# Round 5 训练报告

## 特征工程
- 领域知识特征: 8 个
- 文本特征: 9 个
- 特征交互: 9 个
- 分箱特征: 2 个
- **总特征数**: {len(feature_cols)}

## 结果
- **验证集准确率**: {val_score*100:.2f}%
- **5折CV准确率**: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%
- **相比 Round 2 (85.30%)**: {(val_score - 0.8530)*100:+.2f}%

## 状态
{'✅ 达到 87% 目标' if val_score >= 0.87 else '⏭️ 继续 Round 6'}
"""

with open('training/training_report_round5.md', 'w') as f:
    f.write(report)
print("✅ 报告已保存")

clean_memory()

print("\n" + "=" * 80)
print(f"Round 5 完成！准确率: {val_score*100:.2f}%")
print("=" * 80)
