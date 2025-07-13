#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
强制修复BatchNorm的训练脚本
在导入任何模块之前就进行修复
"""

import sys
import os
from pathlib import Path

# 获取项目根目录
script_dir = Path(__file__).parent
project_root = script_dir.parent
src_path = project_root / 'src'

print("强制BatchNorm修复训练脚本")
print(f"项目根目录: {project_root}")
print(f"源码路径: {src_path}")

# 确保使用本地源码
if str(src_path) in sys.path:
    sys.path.remove(str(src_path))
sys.path.insert(0, str(src_path))

print(f"✅ Python路径已设置: {src_path}")

# 第一步：在导入任何graspdataprocessing之前进行模块级别的修复
print("=" * 60)
print("第一步：预防性模块修复")
print("=" * 60)

# 导入torch和nn
import torch
import torch.nn as nn

# 保存原始的BatchNorm1d
original_BatchNorm1d = nn.BatchNorm1d

def LayerNormReplacement(num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None):
    """BatchNorm1d的LayerNorm替代函数"""
    print(f"🔧 BatchNorm1d被替换为LayerNorm (num_features={num_features})")
    return nn.LayerNorm(num_features, eps=eps, elementwise_affine=affine, device=device, dtype=dtype)

# 全局替换BatchNorm1d
nn.BatchNorm1d = LayerNormReplacement
print("✅ 全局BatchNorm1d已被LayerNorm替换")

# 第二步：导入并测试
print("\n" + "=" * 60)
print("第二步：导入模块并测试")
print("=" * 60)

try:
    import graspdataprocessing as gdp
    from graspdataprocessing.machine_learning_module.ANN import ANNClassifier
    
    print("✅ graspdataprocessing模块导入成功")
    
    # 测试模型创建
    test_classifier = ANNClassifier(input_size=245, output_size=2)
    test_model = test_classifier._build_model()
    
    # 检查模型结构
    print("\n模型结构:")
    for i, layer in enumerate(test_model):
        print(f"  {i}: {layer}")
    
    # 验证没有原始BatchNorm1d
    has_original_batchnorm = any(isinstance(m, original_BatchNorm1d) for m in test_model.modules())
    has_layernorm = any(isinstance(m, nn.LayerNorm) for m in test_model.modules())
    
    print(f"\n验证结果:")
    print(f"  原始BatchNorm1d: {has_original_batchnorm}")
    print(f"  LayerNorm: {has_layernorm}")
    
    if has_original_batchnorm:
        print("❌ 仍然包含原始BatchNorm1d!")
        sys.exit(1)
    
    # 测试训练模式下的batch_size=1
    print(f"\n测试训练模式batch_size=1:")
    test_input = torch.randn(1, 245)
    test_model.train()  # 设为训练模式
    
    try:
        output = test_model(test_input)
        print(f"✅ 成功: {test_input.shape} -> {output.shape}")
    except Exception as e:
        print(f"❌ 失败: {e}")
        if "Expected more than 1 value per channel" in str(e):
            print("这是BatchNorm错误，修复失败!")
            sys.exit(1)
    
    print("✅ 所有测试通过，开始训练...")
    
except Exception as e:
    print(f"❌ 模块导入或测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 第三步：运行训练
print("\n" + "=" * 60)
print("第三步：运行训练")
print("=" * 60)

try:
    # 导入原始train脚本
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
    
    # 确保train模块也使用我们修复的nn
    train_module.torch = torch
    train_module.nn = nn
    
    spec.loader.exec_module(train_module)
    
    # 解析命令行参数并运行
    import argparse
    parser = argparse.ArgumentParser(description='强制修复BatchNorm的机器学习训练程序')
    parser.add_argument('--config', type=str, default='config.toml', help='配置文件路径')
    args = parser.parse_args()
    
    # 加载配置并运行
    cfg = gdp.load_config(args.config)
    
    print(f"开始运行训练，配置: {args.config}")
    train_module.main(cfg)
    
    print("\n🎉 训练完成!")
    
except FileNotFoundError:
    print(f"❌ 配置文件 {args.config} 不存在")
    sys.exit(1)
except Exception as e:
    print(f"❌ 训练失败: {e}")
    import traceback
    traceback.print_exc()
    
    # 检查是否是BatchNorm错误
    if "Expected more than 1 value per channel" in str(e):
        print("\n💡 这仍然是BatchNorm错误！")
        print("可能的原因:")
        print("1. 有其他地方创建了BatchNorm1d")
        print("2. 模块缓存问题")
        print("3. 多线程/多进程问题")
        
        # 尝试找到出错位置
        tb = traceback.format_exc()
        if "batchnorm.py" in tb:
            print("4. PyTorch内部仍在使用原始BatchNorm")
    
    sys.exit(1)