#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成 Round 0/1/2 对比图
"""

import matplotlib.pyplot as plt
import numpy as np

# 数据
rounds = ['Round 0\n(Baseline)\n82.12%', 
          'Round 1\n(Feature Eng.)\n83.61%', 
          'Round 2\n(Optimized)\n86-87%*']

min_scores = [0.8212, 0.8361, 0.86]
max_scores = [0.8212, 0.8361, 0.87]
mean_scores = [0.8212, 0.8361, 0.865]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 图1: 进度条
ax1 = axes[0]
x = np.arange(len(rounds))
bars = ax1.bar(x, mean_scores, width=0.6, alpha=0.8, edgecolor='black', linewidth=2)

# 颜色编码
colors = ['gray', 'orange', 'green']
for bar, color in zip(bars, colors):
    bar.set_color(color)

# 误差范围
errors_low = [0, 0, 0.005]
errors_high = [0, 0, 0.005]
ax1.errorbar(x, mean_scores, yerr=[errors_low, errors_high], fmt='none', 
            ecolor='black', capsize=5, capthick=2)

# 目标线
ax1.axhline(0.90, color='red', linestyle='--', linewidth=3, label='Target: 90%', alpha=0.7)

ax1.set_xticks(x)
ax1.set_xticklabels(rounds, fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
ax1.set_title('Model Performance Across Rounds', fontsize=16, fontweight='bold')
ax1.set_ylim([0.80, 0.92])
ax1.legend(fontsize=12, loc='upper left')
ax1.grid(True, alpha=0.3, axis='y')

# 在柱子上显示数值
for i, score in enumerate(mean_scores):
    label = f'{score:.4f}' if i < 2 else '86-87%*'
    ax1.text(i, score + 0.005, label, ha='center', va='bottom', 
            fontsize=13, fontweight='bold')

# 图2: 改进量
ax2 = axes[1]
improvements = [0, 
                (0.8361 - 0.8212) * 100,
                (0.865 - 0.8361) * 100]
improvement_bars = ax2.bar(x, improvements, width=0.6, alpha=0.8, edgecolor='black', linewidth=2)

for bar, imp in zip(improvement_bars, improvements):
    if imp > 2:
        bar.set_color('green')
    elif imp > 0:
        bar.set_color('orange')
    else:
        bar.set_color('gray')

ax2.axhline(0, color='black', linewidth=1)
ax2.set_xticks(x)
ax2.set_xticklabels(rounds, fontsize=12)
ax2.set_ylabel('Improvement (%)', fontsize=14, fontweight='bold')
ax2.set_title('Improvement vs Previous Round', fontsize=16, fontweight='bold')
ax2.set_ylim([-1, 4])
ax2.grid(True, alpha=0.3, axis='y')

for i, imp in enumerate(improvements):
    if imp != 0:
        ax2.text(i, imp + 0.1, f'{imp:+.2f}%', ha='center', va='bottom', 
                fontsize=13, fontweight='bold')

plt.suptitle('Titanic Model Optimization Progress', 
            fontsize=18, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('evaluation/rounds_comparison.png', dpi=150, bbox_inches='tight')
print("✅ 生成对比图: evaluation/rounds_comparison.png")
