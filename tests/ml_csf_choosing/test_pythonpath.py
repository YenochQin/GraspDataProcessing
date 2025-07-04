#!/usr/bin/env python3
"""
测试脚本：验证 GraspDataProcessing 包的路径设置和导入
"""

import sys
import os
from pathlib import Path

def log_with_timestamp(message):
    """带时间戳的日志输出"""
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}")

def test_pythonpath_setup():
    """测试PYTHONPATH设置"""
    log_with_timestamp("开始测试 PYTHONPATH 设置...")
    
    # 1. 检查环境变量
    log_with_timestamp("=" * 50)
    log_with_timestamp("1. 检查环境变量")
    pythonpath = os.environ.get('PYTHONPATH', '')
    if pythonpath:
        log_with_timestamp(f"✅ PYTHONPATH 已设置: {pythonpath}")
        for path in pythonpath.split(os.pathsep):
            if path.strip():
                log_with_timestamp(f"   - {path}")
    else:
        log_with_timestamp("⚠️ PYTHONPATH 未设置")
    
    # 2. 检查sys.path
    log_with_timestamp("=" * 50)
    log_with_timestamp("2. 检查 sys.path")
    log_with_timestamp("Python 搜索路径:")
    for i, path in enumerate(sys.path, 1):
        log_with_timestamp(f"   {i:2}. {path}")
    
    # 3. 检查 GraspDataProcessing 源码路径
    log_with_timestamp("=" * 50)
    log_with_timestamp("3. 检查 GraspDataProcessing 源码路径")
    
    # 计算预期的源码路径
    current_dir = Path(__file__).parent.absolute()
    expected_src_path = current_dir / "../../../src"
    expected_src_path = expected_src_path.resolve()
    
    log_with_timestamp(f"当前脚本路径: {current_dir}")
    log_with_timestamp(f"预期源码路径: {expected_src_path}")
    
    if expected_src_path.exists():
        log_with_timestamp("✅ 源码目录存在")
        
        # 检查关键文件
        init_file = expected_src_path / "graspdataprocessing" / "__init__.py"
        if init_file.exists():
            log_with_timestamp("✅ graspdataprocessing/__init__.py 存在")
        else:
            log_with_timestamp("❌ graspdataprocessing/__init__.py 不存在")
            
        # 列出源码目录内容
        log_with_timestamp("源码目录内容:")
        try:
            for item in expected_src_path.iterdir():
                if item.is_dir():
                    log_with_timestamp(f"   📁 {item.name}/")
                else:
                    log_with_timestamp(f"   📄 {item.name}")
        except Exception as e:
            log_with_timestamp(f"❌ 无法读取目录内容: {e}")
    else:
        log_with_timestamp("❌ 源码目录不存在")
    
    # 4. 测试包导入
    log_with_timestamp("=" * 50)
    log_with_timestamp("4. 测试包导入")
    
    try:
        log_with_timestamp("尝试导入 graspdataprocessing...")
        import graspdataprocessing as gdp
        log_with_timestamp("✅ graspdataprocessing 导入成功")
        
        # 检查包的位置
        package_file = gdp.__file__
        log_with_timestamp(f"包文件位置: {package_file}")
        
        # 检查包的版本（如果有）
        if hasattr(gdp, '__version__'):
            log_with_timestamp(f"包版本: {gdp.__version__}")
        
        # 尝试导入子模块
        try:
            from graspdataprocessing.CSFs_choosing import csfs_choosing_class
            log_with_timestamp("✅ CSFs_choosing.csfs_choosing_class 导入成功")
        except ImportError as e:
            log_with_timestamp(f"⚠️ CSFs_choosing.csfs_choosing_class 导入失败: {e}")
        
        try:
            from graspdataprocessing.machine_learning_module import ml_model
            log_with_timestamp("✅ machine_learning_module.ml_model 导入成功")
        except ImportError as e:
            log_with_timestamp(f"⚠️ machine_learning_module.ml_model 导入失败: {e}")
            
    except ImportError as e:
        log_with_timestamp(f"❌ graspdataprocessing 导入失败: {e}")
        
        # 提供调试信息
        log_with_timestamp("调试信息:")
        log_with_timestamp(f"Python 版本: {sys.version}")
        log_with_timestamp(f"Python 可执行文件: {sys.executable}")
        
        return False
    
    # 5. 总结
    log_with_timestamp("=" * 50)
    log_with_timestamp("5. 测试总结")
    log_with_timestamp("✅ 所有测试完成！")
    return True

if __name__ == "__main__":
    success = test_pythonpath_setup()
    if success:
        log_with_timestamp("🎉 测试成功！PYTHONPATH 设置正常工作")
        sys.exit(0)
    else:
        log_with_timestamp("💥 测试失败！请检查 PYTHONPATH 设置")
        sys.exit(1) 