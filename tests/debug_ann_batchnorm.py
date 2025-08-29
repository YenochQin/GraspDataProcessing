#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
BatchNorm错误诊断脚本
用于排查"Expected more than 1 value per channel when training"错误
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

def check_ann_source():
    """检查ANN.py源码内容"""
    print("=" * 60)
    print("1. 检查ANN.py源码")
    print("=" * 60)
    
    try:
        import graspkit as gk
        from graspdataprocessing.machine_learning_module.ANN import ANNClassifier
        import inspect
        
        # 检查实际加载的ANN.py文件
        ann_file = inspect.getfile(ANNClassifier)
        print(f"实际使用的ANN.py文件: {ann_file}")
        
        # 读取并显示_build_model方法的内容
        with open(ann_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        print("\n_build_model方法内容:")
        print("-" * 40)
        in_build_method = False
        indent_level = None
        
        for i, line in enumerate(lines, 1):
            if 'def _build_model(self)' in line:
                in_build_method = True
                indent_level = len(line) - len(line.lstrip())
                print(f"{i:3d}: {line.rstrip()}")
            elif in_build_method:
                current_indent = len(line) - len(line.lstrip())
                # 如果缩进回到同级或更少，且不是空行，说明方法结束
                if line.strip() and current_indent <= indent_level:
                    break
                print(f"{i:3d}: {line.rstrip()}")
        
        return True
        
    except Exception as e:
        print(f"❌ 检查源码失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_creation():
    """测试模型创建和结构"""
    print("\n" + "=" * 60)
    print("2. 测试模型创建")
    print("=" * 60)
    
    try:
        from graspdataprocessing.machine_learning_module.ANN import ANNClassifier
        import torch.nn as nn
        
        # 创建分类器实例
        classifier = ANNClassifier(input_size=245, output_size=2)
        model = classifier._build_model()
        
        print("✅ 模型创建成功")
        print("\n模型结构:")
        print("-" * 40)
        for i, layer in enumerate(model):
            print(f"  {i}: {layer}")
        
        # 检查每个模块的类型
        has_batchnorm = False
        has_layernorm = False
        
        print("\n模块类型检查:")
        print("-" * 40)
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm1d):
                has_batchnorm = True
                print(f"❌ 发现BatchNorm1d: {name} -> {module}")
            elif isinstance(module, nn.LayerNorm):
                has_layernorm = True
                print(f"✅ 发现LayerNorm: {name} -> {module}")
        
        print(f"\n总结:")
        print(f"包含BatchNorm1d: {has_batchnorm}")
        print(f"包含LayerNorm: {has_layernorm}")
        
        if has_batchnorm:
            print("❌ 仍然包含BatchNorm1d，这会导致错误")
            return False
        else:
            print("✅ 没有BatchNorm1d，应该不会有问题")
            return True
        
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_import_paths():
    """检查Python导入路径"""
    print("\n" + "=" * 60)
    print("3. 检查Python导入路径")
    print("=" * 60)
    
    print("Python路径:")
    for i, path in enumerate(sys.path):
        print(f"  {i}: {path}")
    
    print(f"\n项目根目录: {project_root}")
    print(f"源码路径: {src_path}")
    
    # 检查graspdataprocessing导入
    try:
        import graspkit as gk
        print(f"\n✅ graspdataprocessing导入成功")
        print(f"   位置: {gk.__file__}")
        
        # 检查是否有多个graspdataprocessing
        import pkgutil
        grasp_modules = []
        for finder, name, ispkg in pkgutil.iter_modules():
            if 'grasp' in name.lower():
                grasp_modules.append(name)
        
        if grasp_modules:
            print(f"\n找到的相关模块: {grasp_modules}")
        
    except Exception as e:
        print(f"\n❌ graspdataprocessing导入失败: {e}")
        return False
    
    # 检查已安装的包
    try:
        import pkg_resources
        installed_grasp = []
        for pkg in pkg_resources.working_set:
            if 'grasp' in pkg.key.lower():
                installed_grasp.append(f"{pkg.key} - {pkg.location}")
        
        if installed_grasp:
            print(f"\n已安装的相关包:")
            for pkg in installed_grasp:
                print(f"  {pkg}")
        else:
            print(f"\n✅ 没有通过pip安装的grasp相关包")
            
    except Exception as e:
        print(f"\n检查已安装包时出错: {e}")
    
    return True

def check_cache_files():
    """检查和清理缓存文件"""
    print("\n" + "=" * 60)
    print("4. 检查和清理缓存文件")
    print("=" * 60)
    
    import glob
    
    # 查找.pyc文件
    pyc_files = []
    pycache_dirs = []
    
    for root, dirs, files in os.walk(project_root):
        for file in files:
            if file.endswith('.pyc'):
                pyc_files.append(os.path.join(root, file))
        for dir in dirs:
            if dir == '__pycache__':
                pycache_dirs.append(os.path.join(root, dir))
    
    if pyc_files:
        print(f"找到 {len(pyc_files)} 个.pyc文件:")
        for f in pyc_files[:10]:  # 只显示前10个
            print(f"  {f}")
        if len(pyc_files) > 10:
            print(f"  ... 还有 {len(pyc_files) - 10} 个文件")
    else:
        print("✅ 没有找到.pyc文件")
    
    if pycache_dirs:
        print(f"\n找到 {len(pycache_dirs)} 个__pycache__目录:")
        for d in pycache_dirs[:10]:  # 只显示前10个
            print(f"  {d}")
        if len(pycache_dirs) > 10:
            print(f"  ... 还有 {len(pycache_dirs) - 10} 个目录")
    else:
        print("✅ 没有找到__pycache__目录")
    
    # 提供清理选项
    if pyc_files or pycache_dirs:
        print(f"\n要清理这些缓存文件吗? [y/N]: ", end="")
        try:
            response = input().strip().lower()
            if response in ['y', 'yes']:
                # 删除.pyc文件
                for f in pyc_files:
                    try:
                        os.remove(f)
                    except Exception as e:
                        print(f"删除{f}失败: {e}")
                
                # 删除__pycache__目录
                import shutil
                for d in pycache_dirs:
                    try:
                        shutil.rmtree(d)
                    except Exception as e:
                        print(f"删除{d}失败: {e}")
                
                print("✅ 缓存清理完成")
            else:
                print("跳过缓存清理")
        except KeyboardInterrupt:
            print("\n跳过缓存清理")

def test_simple_forward():
    """测试简单的前向传播"""
    print("\n" + "=" * 60)
    print("5. 测试简单前向传播")
    print("=" * 60)
    
    try:
        from graspdataprocessing.machine_learning_module.ANN import ANNClassifier
        import torch
        
        # 创建分类器
        classifier = ANNClassifier(input_size=245, output_size=2)
        model = classifier._build_model()
        
        # 测试不同batch size的输入
        test_cases = [
            (1, "单样本 - 会触发BatchNorm错误"),
            (2, "两样本"),
            (32, "正常batch")
        ]
        
        for batch_size, description in test_cases:
            try:
                # 创建测试数据
                x = torch.randn(batch_size, 245)
                
                # 前向传播
                model.eval()  # 设为评估模式
                with torch.no_grad():
                    output = model(x)
                
                print(f"✅ {description}: 输入{x.shape} -> 输出{output.shape}")
                
            except Exception as e:
                print(f"❌ {description}: 输入({batch_size}, 245) 失败 - {e}")
                if batch_size == 1:
                    print("   这证实了BatchNorm问题!")
                    return False
        
        return True
        
    except Exception as e:
        print(f"❌ 前向传播测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("BatchNorm错误诊断脚本")
    print("用于排查'Expected more than 1 value per channel when training'错误")
    print("项目路径:", project_root)
    print("源码路径:", src_path)
    
    # 执行所有检查
    checks = [
        ("源码检查", check_ann_source),
        ("模型创建测试", test_model_creation),
        ("导入路径检查", check_import_paths),
        ("缓存文件检查", check_cache_files),
        ("前向传播测试", test_simple_forward)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ {name}执行失败: {e}")
            results.append((name, False))
    
    # 显示总结
    print("\n" + "=" * 60)
    print("诊断总结")
    print("=" * 60)
    
    all_passed = True
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 所有检查都通过了！BatchNorm问题应该已经解决。")
    else:
        print("\n⚠️  发现问题，需要进一步修复。")
        print("\n建议解决方案:")
        print("1. 清理所有缓存文件")
        print("2. 确保使用最新的源码")
        print("3. 检查是否有多个graspdataprocessing包")
        print("4. 如果问题持续，使用运行时修复方案")

if __name__ == "__main__":
    main()