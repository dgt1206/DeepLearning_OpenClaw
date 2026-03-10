# Round 6 训练报告

## 策略
- 回到 Round 2 基础 (18 特征)
- 只添加 3 个精选特征 (Woman_or_Child, Title_Simple, Deck)
- 对比 CatBoost vs XGBoost

## 结果
| 模型 | 验证集 | 5折CV |
|------|--------|-------|
| CatBoost | 82.12% | 83.16% ± 2.52% |
| XGBoost | 82.12% | - |

## 最佳模型
- **名称**: XGBoost
- **准确率**: 82.12%
- **相比 Round 2 (85.30%)**: -3.18%
- **相比 Round 5 (81.01%)**: +1.11%

## 状态
⏭️ 继续 Round 7
