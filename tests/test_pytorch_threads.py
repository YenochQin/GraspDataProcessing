#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
测试cpu_threads配置参数是否生效
"""

import sys
import os
from pathlib import Path

# 添加源码路径
script_dir = Path(__file__).parent
project_root = script_dir.parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

def test_cpu_threads_config():
    """测试cpu_threads配置"""
    print("=" * 60)
    print("测试PyTorch线程数配置")
    print("=" * 60)
    
    import torch
    import graspdataprocessing as gdp
    
    # 创建一个简单的配置对象测试
    class TestConfig:
        def __init__(self, cpu_threads=None):
            self.cpu_threads = cpu_threads
    
    # 测试不同的配置值
    test_cases = [
        (None, "未设置"),
        (16, "设置为16"),
        (8, "设置为8"),
        (64, "设置为64（超过系统核心数）"),
        ("invalid", "无效值"),
    ]
    
    for threads_value, description in test_cases:
        print(f"\n测试用例: {description}")
        print(f"配置值: {threads_value}")
        
        config = TestConfig(cpu_threads=threads_value)
        
        # 模拟train_model中的逻辑
        cpu_count = os.cpu_count() or 4
        
        config_threads = getattr(config, 'cpu_threads', None)
        if config_threads is not None:
            try:
                config_threads = int(config_threads)
                optimal_threads = min(config_threads, cpu_count)
                print(f"✅ 配置值有效，计算得出: {optimal_threads}")
            except (ValueError, TypeError):
                optimal_threads = min(32, cpu_count)
                print(f"❌ 配置值无效，使用默认: {optimal_threads}")
        else:
            optimal_threads = min(32, cpu_count)
            print(f"ℹ️  未设置配置，使用默认: {optimal_threads}")
        
        # 实际设置并验证
        torch.set_num_threads(optimal_threads)
        actual_threads = torch.get_num_threads()
        
        print(f"设置线程数: {optimal_threads}")
        print(f"实际线程数: {actual_threads}")
        print(f"系统核心数: {cpu_count}")
        
        if actual_threads == optimal_threads:
            print("✅ 设置成功")
        else:
            print("⚠️  设置可能未完全生效")

def test_with_real_config():
    """使用真实的config.toml测试"""
    print("\n" + "=" * 60)
    print("使用真实配置文件测试")
    print("=" * 60)
    
    import graspdataprocessing as gdp
    import torch
    
    # 查找config.toml文件
    config_paths = [
        Path.cwd() / 'config.toml',
        Path('/home/computer-0-2/4thdd/GdI/cv6odd1_j3as5/config.toml'),
        script_dir / 'ml_csf_choosing' / 'config.toml'
    ]
    
    config_file = None
    for path in config_paths:
        if path.exists():
            config_file = path
            break
    
    if config_file:
        print(f"找到配置文件: {config_file}")
        try:
            config = gdp.load_config(str(config_file))
            cpu_threads = getattr(config, 'cpu_threads', None)
            print(f"配置文件中的cpu_threads: {cpu_threads}")
            
            # 模拟设置过程
            cpu_count = os.cpu_count() or 4
            if cpu_threads is not None:
                try:
                    config_threads = int(cpu_threads)
                    optimal_threads = min(config_threads, cpu_count)
                    print(f"计算得出的optimal_threads: {optimal_threads}")
                except (ValueError, TypeError):
                    optimal_threads = min(32, cpu_count)
                    print(f"无效值，使用默认: {optimal_threads}")
            else:
                optimal_threads = min(32, cpu_count)
                print(f"未设置，使用默认: {optimal_threads}")
            
            # 实际设置
            torch.set_num_threads(optimal_threads)
            actual = torch.get_num_threads()
            print(f"设置后的实际线程数: {actual}")
            
        except Exception as e:
            print(f"加载配置文件失败: {e}")
    else:
        print("未找到配置文件")

if __name__ == "__main__":
    test_cpu_threads_config()
    test_with_real_config()
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)