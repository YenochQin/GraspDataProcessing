#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
BatchNorm运行时修复脚本
用于在运行时替换BatchNorm1d为LayerNorm，解决batch_size=1的问题
"""

import sys
import os
from pathlib import Path

def apply_runtime_fix():
    """应用运行时修复"""
    print("=" * 60)
    print("BatchNorm运行时修复")
    print("=" * 60)
    
    try:
        import torch.nn as nn
        from graspdataprocessing.machine_learning_module.ANN import ANNClassifier
        
        # 保存原始方法
        original_build_model = ANNClassifier._build_model
        
        def patched_build_model(self):
            """修复的_build_model方法，强制使用LayerNorm"""
            print(f"应用修复: 使用LayerNorm替代BatchNorm1d")
            
            model = nn.Sequential(
                nn.Linear(self.input_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),  # 强制使用LayerNorm
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.LayerNorm(self.hidden_size // 2),  # 强制使用LayerNorm
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.hidden_size // 2, self.output_size)
            ).to(self.device)
            
            # 初始化权重
            self._initialize_weights(model)
            return model
        
        # 替换方法
        ANNClassifier._build_model = patched_build_model
        print("✅ 运行时修复已应用")
        
        # 测试修复是否生效
        test_classifier = ANNClassifier(input_size=245, output_size=2)
        test_model = test_classifier._build_model()
        
        # 检查是否还有BatchNorm
        has_batchnorm = any(isinstance(m, nn.BatchNorm1d) for m in test_model.modules())
        has_layernorm = any(isinstance(m, nn.LayerNorm) for m in test_model.modules())
        
        print(f"修复后检查:")
        print(f"  包含BatchNorm1d: {has_batchnorm}")
        print(f"  包含LayerNorm: {has_layernorm}")
        
        if not has_batchnorm and has_layernorm:
            print("✅ 运行时修复成功!")
            return True
        else:
            print("❌ 运行时修复失败")
            return False
            
    except Exception as e:
        print(f"❌ 运行时修复失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fix_with_batch_size_1():
    """测试修复后的batch_size=1情况"""
    print("\n" + "=" * 60)
    print("测试batch_size=1的情况")
    print("=" * 60)
    
    try:
        import torch
        from graspdataprocessing.machine_learning_module.ANN import ANNClassifier
        
        # 创建分类器
        classifier = ANNClassifier(input_size=245, output_size=2)
        model = classifier._build_model()
        
        # 测试batch_size=1
        x = torch.randn(1, 245)
        
        # 尝试训练模式下的前向传播
        model.train()  # 设为训练模式
        output = model(x)
        
        print(f"✅ batch_size=1训练模式测试成功: 输入{x.shape} -> 输出{output.shape}")
        return True
        
    except Exception as e:
        print(f"❌ batch_size=1训练模式测试失败: {e}")
        if "Expected more than 1 value per channel" in str(e):
            print("   这是典型的BatchNorm错误!")
        return False

def main():
    """主函数"""
    print("BatchNorm运行时修复脚本")
    
    # 应用修复
    if apply_runtime_fix():
        # 测试修复
        test_fix_with_batch_size_1()
    else:
        print("修复失败，无法继续测试")

if __name__ == "__main__":
    main()