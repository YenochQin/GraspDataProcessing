#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
测试mkdisks配置参数读取功能
"""

import sys
import os
from pathlib import Path

# 添加源码路径
script_dir = Path(__file__).parent
project_root = script_dir.parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

def test_config_reading():
    """测试配置文件读取"""
    print("=" * 60)
    print("测试mpi_tmp_path配置参数读取")
    print("=" * 60)
    
    # 测试不同的配置文件
    config_files = [
        project_root / 'scripts' / 'config.toml',
        project_root / 'tests' / 'ml_csf_choosing' / 'config.toml'
    ]
    
    import graspdataprocessing as gdp
    
    for config_file in config_files:
        if config_file.exists():
            print(f"\n测试配置文件: {config_file}")
            try:
                config = gdp.load_config(str(config_file))
                
                # 测试读取mpi_tmp_path
                mpi_tmp_path = getattr(config, 'mpi_tmp_path', None)
                print(f"  mpi_tmp_path: {mpi_tmp_path}")
                
                # 测试读取tasks_per_node（用于mkdisks的第一个参数）
                tasks_per_node = getattr(config, 'tasks_per_node', None)
                print(f"  tasks_per_node: {tasks_per_node}")
                
                # 模拟run_script.sh中的逻辑
                if mpi_tmp_path is not None and str(mpi_tmp_path) != "null":
                    print(f"  ✅ 模拟mkdisks调用: mkdisks {tasks_per_node} {mpi_tmp_path}")
                    print(f"     -> 将创建: '{mpi_tmp_path}/mpi_tmp' 路径")
                else:
                    print(f"  ✅ 模拟mkdisks调用: mkdisks {tasks_per_node}")
                    print(f"     -> 将使用默认路径（当前目录下的mpi_tmp）")
                
            except Exception as e:
                print(f"  ❌ 加载配置失败: {e}")
        else:
            print(f"\n⚠️ 配置文件不存在: {config_file}")

def test_mkdisks_scenarios():
    """测试不同的mkdisks调用场景"""
    print("\n" + "=" * 60)
    print("测试mkdisks不同调用场景")
    print("=" * 60)
    
    # 模拟不同的配置值
    test_cases = [
        (None, "未配置mpi_tmp_path"),
        ("", "空字符串"),
        ("null", "null值"),
        ("/tmp/test", "有效路径"),
        ("/home/workstation3/caltmp", "典型路径"),
    ]
    
    for mpi_tmp_path, description in test_cases:
        print(f"\n测试场景: {description}")
        print(f"  配置值: {mpi_tmp_path}")
        
        # 模拟run_script.sh中的判断逻辑
        processor = 46  # 示例处理器数量
        
        if mpi_tmp_path is not None and str(mpi_tmp_path) != "null" and mpi_tmp_path != "":
            print(f"  命令: mkdisks {processor} {mpi_tmp_path}")
            print(f"  结果: 使用指定路径 {mpi_tmp_path}/mpi_tmp")
        else:
            print(f"  命令: mkdisks {processor}")
            print(f"  结果: 使用默认路径（当前目录/mpi_tmp）")

def demonstrate_usage():
    """演示如何使用新功能"""
    print("\n" + "=" * 60)
    print("使用说明和示例")
    print("=" * 60)
    
    print("\n1. 在config.toml中添加mpi_tmp_path参数:")
    print("   ```toml")
    print("   # GRASP计算参数")
    print("   tasks_per_node = 46")
    print("   mpi_tmp_path = \"/home/workstation3/caltmp\"  # MPI临时文件存储路径")
    print("   ```")
    
    print("\n2. 不同服务器的配置示例:")
    print("   - workstation2: mpi_tmp_path = \"/home/workstation2/caltmp\"")
    print("   - workstation3: mpi_tmp_path = \"/home/workstation3/caltmp\"") 
    print("   - 本地测试:     mpi_tmp_path = \"/tmp/grasp_calc\"")
    print("   - 使用默认:     不设置mpi_tmp_path参数")
    
    print("\n3. mkdisks脚本的新行为:")
    print("   - 有配置: mkdisks 46 /home/workstation3/caltmp")
    print("   - 无配置: mkdisks 46")
    
    print("\n4. 生成的disks文件内容:")
    print("   ```")
    print("   '/path/to/working/directory'")
    print("   '/home/workstation3/caltmp/mpi_tmp'  # 或当前目录/mpi_tmp")
    print("   '/home/workstation3/caltmp/mpi_tmp'")
    print("   ...（重复46次）")
    print("   ```")

if __name__ == "__main__":
    test_config_reading()
    test_mkdisks_scenarios()
    demonstrate_usage()
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)