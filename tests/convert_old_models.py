#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
转换旧模型文件，将BatchNorm1d替换为LayerNorm
"""

import sys
import os
import pickle
import torch
import torch.nn as nn
from pathlib import Path
import argparse

def convert_batchnorm_to_layernorm(model):
    """递归地将模型中的BatchNorm1d替换为LayerNorm"""
    
    def replace_bn_with_ln(module):
        """替换单个模块"""
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm1d):
                # 获取BatchNorm的参数
                num_features = child.num_features
                eps = child.eps
                elementwise_affine = child.affine
                
                # 创建对应的LayerNorm
                new_layer = nn.LayerNorm(num_features, eps=eps, elementwise_affine=elementwise_affine)
                
                # 如果BatchNorm有权重和偏置，转移到LayerNorm
                if elementwise_affine and child.weight is not None:
                    new_layer.weight.data = child.weight.data.clone()
                if elementwise_affine and child.bias is not None:
                    new_layer.bias.data = child.bias.data.clone()
                
                # 替换模块
                setattr(module, name, new_layer)
                print(f"  替换 {name}: BatchNorm1d({num_features}) -> LayerNorm({num_features})")
            else:
                # 递归处理子模块
                replace_bn_with_ln(child)
    
    replace_bn_with_ln(model)
    return model

def convert_model_file(model_path, output_path=None):
    """转换单个模型文件"""
    model_path = Path(model_path)
    if output_path is None:
        output_path = model_path.with_suffix('.converted.pkl')
    else:
        output_path = Path(output_path)
    
    print(f"转换模型: {model_path}")
    
    try:
        # 加载模型
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        print(f"原始模型类型: {type(model)}")
        
        # 检查是否包含BatchNorm
        has_batchnorm = False
        if hasattr(model, 'model') and hasattr(model.model, 'modules'):
            # 如果是ANNClassifier对象
            for module in model.model.modules():
                if isinstance(module, nn.BatchNorm1d):
                    has_batchnorm = True
                    break
        elif hasattr(model, 'modules'):
            # 如果直接是PyTorch模型
            for module in model.modules():
                if isinstance(module, nn.BatchNorm1d):
                    has_batchnorm = True
                    break
        
        if not has_batchnorm:
            print("  模型中没有BatchNorm1d，无需转换")
            return False
        
        print("  发现BatchNorm1d，开始转换...")
        
        # 转换模型
        if hasattr(model, 'model'):
            # ANNClassifier对象
            model.model = convert_batchnorm_to_layernorm(model.model)
        else:
            # 直接的PyTorch模型
            model = convert_batchnorm_to_layernorm(model)
        
        # 验证转换结果
        has_batchnorm_after = False
        has_layernorm_after = False
        
        if hasattr(model, 'model'):
            for module in model.model.modules():
                if isinstance(module, nn.BatchNorm1d):
                    has_batchnorm_after = True
                elif isinstance(module, nn.LayerNorm):
                    has_layernorm_after = True
        else:
            for module in model.modules():
                if isinstance(module, nn.BatchNorm1d):
                    has_batchnorm_after = True
                elif isinstance(module, nn.LayerNorm):
                    has_layernorm_after = True
        
        if has_batchnorm_after:
            print("  ❌ 转换失败：仍然包含BatchNorm1d")
            return False
        
        if not has_layernorm_after:
            print("  ⚠️  转换后没有LayerNorm，可能有问题")
        
        # 保存转换后的模型
        with open(output_path, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"  ✅ 转换成功，保存到: {output_path}")
        return True
        
    except Exception as e:
        print(f"  ❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='转换旧模型文件中的BatchNorm为LayerNorm')
    parser.add_argument('model_path', help='模型文件路径或包含模型的目录')
    parser.add_argument('--output', '-o', help='输出路径（可选）')
    parser.add_argument('--inplace', action='store_true', help='就地替换原文件（备份原文件）')
    parser.add_argument('--backup-dir', default='backup_models', help='备份目录（默认：backup_models）')
    
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    
    print("BatchNorm转LayerNorm模型转换工具")
    print("=" * 50)
    
    if model_path.is_file():
        # 单个文件
        if args.inplace:
            # 就地替换
            backup_dir = Path(args.backup_dir)
            backup_dir.mkdir(exist_ok=True)
            backup_path = backup_dir / model_path.name
            
            # 备份原文件
            import shutil
            shutil.copy2(model_path, backup_path)
            print(f"备份原文件到: {backup_path}")
            
            # 转换并覆盖原文件
            success = convert_model_file(model_path, model_path)
        else:
            # 输出到新文件
            success = convert_model_file(model_path, args.output)
            
    elif model_path.is_dir():
        # 目录中的所有.pkl文件
        pkl_files = list(model_path.glob('*.pkl'))
        
        if not pkl_files:
            print(f"目录 {model_path} 中没有找到.pkl文件")
            return
        
        print(f"找到 {len(pkl_files)} 个模型文件")
        
        if args.inplace:
            backup_dir = Path(args.backup_dir)
            backup_dir.mkdir(exist_ok=True)
        
        success_count = 0
        for pkl_file in pkl_files:
            print(f"\n处理: {pkl_file.name}")
            
            if args.inplace:
                # 备份并就地替换
                backup_path = backup_dir / pkl_file.name
                import shutil
                shutil.copy2(pkl_file, backup_path)
                print(f"  备份到: {backup_path}")
                
                if convert_model_file(pkl_file, pkl_file):
                    success_count += 1
            else:
                # 输出到.converted.pkl文件
                if convert_model_file(pkl_file):
                    success_count += 1
        
        print(f"\n总结: {success_count}/{len(pkl_files)} 个文件转换成功")
        
    else:
        print(f"错误: {model_path} 既不是文件也不是目录")
        return
    
    print("\n转换完成!")

if __name__ == "__main__":
    main()