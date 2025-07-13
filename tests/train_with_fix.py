#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
带有BatchNorm修复的train.py版本
在运行训练之前先应用运行时修复
"""

import sys
import os
from pathlib import Path

# 获取项目根目录
script_dir = Path(__file__).parent
project_root = script_dir.parent
src_path = project_root / 'src'

# 确保使用本地源码
sys.path.insert(0, str(src_path))

def apply_batchnorm_fix():
    """应用BatchNorm修复"""
    print("=" * 50)
    print("应用BatchNorm修复...")
    print("=" * 50)
    
    try:
        import torch.nn as nn
        
        # 首先尝试导入
        from graspdataprocessing.machine_learning_module.ANN import ANNClassifier
        
        # 保存原始方法
        original_build_model = ANNClassifier._build_model
        
        def patched_build_model(self):
            """修复的_build_model方法"""
            model = nn.Sequential(
                nn.Linear(self.input_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),  # 使用LayerNorm替代BatchNorm1d
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.LayerNorm(self.hidden_size // 2),  # 使用LayerNorm替代BatchNorm1d
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.hidden_size // 2, self.output_size)
            ).to(self.device)
            
            self._initialize_weights(model)
            return model
        
        # 应用修复
        ANNClassifier._build_model = patched_build_model
        print("✅ BatchNorm修复已应用")
        
        # 快速验证
        test_classifier = ANNClassifier(input_size=245, output_size=2)
        test_model = test_classifier._build_model()
        has_batchnorm = any(isinstance(m, nn.BatchNorm1d) for m in test_model.modules())
        
        if not has_batchnorm:
            print("✅ 修复验证成功，没有BatchNorm1d")
            return True
        else:
            print("❌ 修复验证失败，仍有BatchNorm1d")
            return False
            
    except Exception as e:
        print(f"❌ BatchNorm修复失败: {e}")
        return False

def main():
    """主函数"""
    print("带有BatchNorm修复的训练脚本")
    print(f"项目根目录: {project_root}")
    print(f"源码路径: {src_path}")
    
    # 应用修复
    if not apply_batchnorm_fix():
        print("BatchNorm修复失败，无法继续")
        sys.exit(1)
    
    # 现在导入并运行原始的train脚本
    print("\n" + "=" * 50)
    print("开始运行训练脚本...")
    print("=" * 50)
    
    try:
        # 导入原始train脚本的main函数
        train_script_path = script_dir / 'ml_csf_choosing' / 'train.py'
        
        if not train_script_path.exists():
            print(f"❌ 训练脚本不存在: {train_script_path}")
            sys.exit(1)
        
        # 添加train脚本目录到路径
        sys.path.insert(0, str(train_script_path.parent))
        
        # 导入train模块
        import importlib.util
        spec = importlib.util.spec_from_file_location("train", train_script_path)
        train_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(train_module)
        
        # 解析命令行参数并运行
        import argparse
        parser = argparse.ArgumentParser(description='机器学习训练程序（带BatchNorm修复）')
        parser.add_argument('--config', type=str, default='config.toml', help='配置文件路径')
        args = parser.parse_args()
        
        # 加载配置并运行
        import graspdataprocessing as gdp
        cfg = gdp.load_config(args.config)
        train_module.main(cfg)
        
        print("\n✅ 训练完成!")
        
    except FileNotFoundError:
        print(f"❌ 配置文件 {args.config} 不存在")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()