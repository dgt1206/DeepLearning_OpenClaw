#!/usr/bin/env python3
"""
训练历史管理工具
功能:
1. 保存每轮训练配置和结果
2. 自动恢复最佳模型
3. 对比历史轮次
"""

import json
import os
import joblib
import shutil
from datetime import datetime
import pandas as pd
import argparse

class TrainingHistory:
    def __init__(self, project_root='/DeepLearning_OpenClaw'):
        self.project_root = project_root
        self.history_dir = os.path.join(project_root, 'history')
        self.best_model_dir = os.path.join(project_root, 'models/best')
        
        os.makedirs(self.history_dir, exist_ok=True)
        os.makedirs(self.best_model_dir, exist_ok=True)
    
    def save_round(self, round_num, config, results, notes=''):
        """保存一轮训练的所有信息"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # 保存配置
        config_data = {
            'round': round_num,
            'timestamp': timestamp,
            'config': config,
            'notes': notes
        }
        
        config_file = os.path.join(self.history_dir, f'round{round_num}_config.json')
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        # 保存结果
        results_data = {
            'round': round_num,
            'timestamp': timestamp,
            'results': results
        }
        
        results_file = os.path.join(self.history_dir, f'round{round_num}_results.json')
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"✅ Round {round_num} 历史已保存")
        
        # 检查是否是最佳
        self._check_and_update_best(round_num, results)
    
    def _check_and_update_best(self, round_num, results):
        """检查并更新最佳模型"""
        current_acc = results.get('val_accuracy', 0)
        
        # 读取历史最佳
        best_file = os.path.join(self.history_dir, 'best_round.txt')
        
        if os.path.exists(best_file):
            with open(best_file, 'r') as f:
                content = f.read().strip().split('\n')
                best_round = int(content[0])
                best_acc = float(content[1])
            
            if current_acc > best_acc:
                print(f"🎉 新的最佳成绩! {best_acc*100:.2f}% → {current_acc*100:.2f}%")
                self._save_as_best(round_num, current_acc)
            else:
                print(f"⚠️  准确率下降: {current_acc*100:.2f}% < {best_acc*100:.2f}% (Round {best_round})")
                print(f"💡 建议: 使用 `python training/restore_best.py` 恢复最佳模型")
        else:
            # 第一次
            self._save_as_best(round_num, current_acc)
    
    def _save_as_best(self, round_num, accuracy):
        """保存为最佳模型"""
        best_file = os.path.join(self.history_dir, 'best_round.txt')
        with open(best_file, 'w') as f:
            f.write(f"{round_num}\n{accuracy}")
        
        # 复制模型文件 (如果存在)
        model_file = os.path.join(self.project_root, f'models/best_model_round{round_num}.pkl')
        if os.path.exists(model_file):
            dest = os.path.join(self.best_model_dir, 'best_model_final.pkl')
            shutil.copy(model_file, dest)
            print(f"✅ 模型已复制到 models/best/")
    
    def restore_best(self):
        """恢复最佳模型"""
        best_file = os.path.join(self.history_dir, 'best_round.txt')
        
        if not os.path.exists(best_file):
            print("❌ 未找到最佳模型记录")
            return
        
        with open(best_file, 'r') as f:
            content = f.read().strip().split('\n')
            best_round = int(content[0])
            best_acc = float(content[1])
        
        print(f"恢复 Round {best_round} 的最佳模型 (准确率: {best_acc*100:.2f}%)")
        
        # 加载配置
        config_file = os.path.join(self.history_dir, f'round{best_round}_config.json')
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        print("最佳配置:")
        print(json.dumps(config_data['config'], indent=2))
    
    def compare_all(self):
        """对比所有轮次"""
        history_files = [f for f in os.listdir(self.history_dir) if f.endswith('_results.json')]
        
        if not history_files:
            print("❌ 未找到历史记录")
            return
        
        records = []
        for f in sorted(history_files):
            with open(os.path.join(self.history_dir, f), 'r') as file:
                data = json.load(file)
                records.append({
                    'Round': data['round'],
                    'Val Accuracy': f"{data['results'].get('val_accuracy', 0)*100:.2f}%",
                    'CV Mean': f"{data['results'].get('cv_mean', 0)*100:.2f}%",
                    'Timestamp': data['timestamp']
                })
        
        df = pd.DataFrame(records)
        print("\n历史对比:")
        print(df.to_string(index=False))
        
        # 标记最佳
        best_file = os.path.join(self.history_dir, 'best_round.txt')
        if os.path.exists(best_file):
            with open(best_file, 'r') as f:
                best_round = int(f.read().strip().split('\n')[0])
            print(f"\n✅ 最佳: Round {best_round}")

def main():
    parser = argparse.ArgumentParser(description='训练历史管理')
    parser.add_argument('action', choices=['save', 'restore', 'compare'], 
                        help='操作: save/restore/compare')
    parser.add_argument('--round', type=int, help='轮次编号 (save时需要)')
    
    args = parser.parse_args()
    
    history = TrainingHistory()
    
    if args.action == 'save':
        if args.round is None:
            print("❌ 需要指定 --round <N>")
            return
        
        # 示例: 从 best_config.json 加载
        config_file = '/DeepLearning_OpenClaw/models/best/best_config.json'
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                data = json.load(f)
            
            history.save_round(
                round_num=args.round,
                config=data['params'],
                results={
                    'train_accuracy': data['train_accuracy'],
                    'val_accuracy': data['val_accuracy'],
                    'cv_mean': data['cv_mean'],
                    'cv_std': data['cv_std']
                },
                notes=f"Round {args.round} - 详见配置文件"
            )
        else:
            print("❌ 未找到配置文件")
    
    elif args.action == 'restore':
        history.restore_best()
    
    elif args.action == 'compare':
        history.compare_all()

if __name__ == '__main__':
    main()
