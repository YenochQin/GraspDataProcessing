#!/usr/bin/env python3
"""
安装验证脚本
运行此脚本来检查所有依赖是否正确安装
"""

import sys
import importlib
import platform

def check_package(package_name, import_name=None, version_attr='__version__'):
    """检查包是否安装并获取版本信息"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        if hasattr(module, version_attr):
            version = getattr(module, version_attr)
        else:
            version = "已安装 (版本未知)"
        return True, version
    except ImportError:
        return False, "未安装"

def check_pytorch_environment():
    """检查PyTorch环境配置"""
    try:
        import torch
        print(f"✅ PyTorch版本: {torch.__version__}")
        
        # 检查CUDA可用性
        if torch.cuda.is_available():
            print(f"🚀 GPU模式: 可用")
            print(f"   设备数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   设备 {i}: {torch.cuda.get_device_name(i)}")
            print(f"   CUDA版本: {torch.version.cuda}")
            print(f"   cuDNN版本: {torch.backends.cudnn.version()}")
        else:
            print(f"🖥️  CPU模式: PyTorch将使用CPU运行")
            
        return True
    except ImportError:
        print("❌ PyTorch未安装")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("🔍 GraspDataProcessing 安装验证")
    print("=" * 60)
    
    # 系统信息
    print(f"\n📋 系统信息:")
    print(f"   操作系统: {platform.system()} {platform.release()}")
    print(f"   Python版本: {sys.version.split()[0]}")
    print(f"   架构: {platform.machine()}")
    
    # 必需包列表
    required_packages = [
        ('numpy', 'numpy'),
        ('pandas', 'pandas'), 
        ('matplotlib', 'matplotlib'),
        ('tqdm', 'tqdm'),
        ('scikit-learn', 'sklearn'),
        ('imbalanced-learn', 'imblearn'),
        ('seaborn', 'seaborn'),
        ('joblib', 'joblib'),
        ('tabulate', 'tabulate'),
        ('pyyaml', 'yaml'),
    ]
    
    # 可选包列表
    optional_packages = [
        ('bitarray', 'bitarray'),
    ]
    
    print(f"\n📦 核心依赖检查:")
    all_required_ok = True
    
    for package_name, import_name in required_packages:
        is_installed, version = check_package(package_name, import_name)
        if is_installed:
            print(f"   ✅ {package_name}: {version}")
        else:
            print(f"   ❌ {package_name}: {version}")
            all_required_ok = False
    
    # PyTorch环境检查
    print(f"\n🔥 PyTorch环境检查:")
    pytorch_ok = check_pytorch_environment()
    
    # 可选包检查
    print(f"\n🔧 可选依赖检查:")
    for package_name, import_name in optional_packages:
        is_installed, version = check_package(package_name, import_name)
        status = "✅" if is_installed else "⚠️ "
        print(f"   {status} {package_name}: {version}")
    
    # 项目包检查
    print(f"\n📋 项目包检查:")
    project_ok, project_version = check_package('graspdataprocessing', 'graspdataprocessing')
    if project_ok:
        print(f"   ✅ graspdataprocessing: {project_version}")
    else:
        print(f"   ⚠️  graspdataprocessing: 未安装 (可能需要运行: pip install -e .)")
    
    # 总结
    print(f"\n" + "=" * 60)
    if all_required_ok and pytorch_ok:
        print("🎉 恭喜！所有核心依赖都已正确安装")
        print("📖 您可以开始使用GraspDataProcessing了")
    else:
        print("⚠️  某些依赖缺失，请参考以下建议:")
        if not all_required_ok:
            print("   - 安装缺失的核心依赖")
        if not pytorch_ok:
            print("   - 安装PyTorch (参考INSTALL.md)")
        print("   - 运行对应环境的requirements文件:")
        print("     pip install -r requirements-cpu.txt  # CPU环境")
        print("     pip install -r requirements-gpu.txt  # GPU环境")
    
    print("=" * 60)

if __name__ == "__main__":
    main() 