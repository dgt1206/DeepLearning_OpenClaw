# 训练历史记录

## 目的
保存每一轮训练的完整配置和结果，防止信息丢失。

## 使用规则

### 1. 每轮训练后保存
```bash
python training/save_history.py --round <N>
```

### 2. 如果准确率下降
```bash
# 自动恢复最佳模型
python training/restore_best.py
```

### 3. 对比历史
```bash
# 查看所有轮次对比
python training/compare_history.py
```

## 文件命名
- `round<N>_config.json` - 配置
- `round<N>_results.json` - 结果
- `round<N>_notes.md` - 笔记

## 最佳模型标记
在 `best_round.txt` 中记录最佳轮次编号
