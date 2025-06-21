#!/usr/bin/env python3
"""
安装全局脚本工具
将get_config_value等工具安装到系统PATH中
"""

import os
import sys
import shutil
from pathlib import Path

def install_global_scripts():
    """安装全局脚本"""
    
    # 获取脚本目录
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # 确定安装位置
    if sys.platform == "win32":
        # Windows: 安装到用户目录
        install_dir = Path.home() / "AppData" / "Local" / "Programs" / "grasp-tools"
    else:
        # Unix/Linux/macOS: 安装到 ~/.local/bin
        install_dir = Path.home() / ".local" / "bin"
    
    # 创建安装目录
    install_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建可执行脚本
    script_content = f'''#!/bin/bash
# 获取TOML配置文件中的值
# 用法: get_config_value <key> [config_file]

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
PYTHON_SCRIPT="{project_root}/tests/ml_csf_choosing/get_config_value.py"

# 检查Python脚本是否存在
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "错误: Python脚本不存在: $PYTHON_SCRIPT" >&2
    exit 1
fi

# 调用Python脚本
python3 "$PYTHON_SCRIPT" "$@"
'''
    
    # 写入脚本文件
    script_path = install_dir / "get_config_value"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # 设置可执行权限
    os.chmod(script_path, 0o755)
    
    print(f"✅ 脚本已安装到: {script_path}")
    
    # 检查PATH
    path_dirs = os.environ.get('PATH', '').split(os.pathsep)
    if str(install_dir) not in path_dirs:
        print(f"⚠️  请将以下路径添加到PATH环境变量:")
        print(f"   {install_dir}")
        
        if sys.platform == "win32":
            print("\nWindows用户:")
            print(f"   1. 打开系统属性 -> 环境变量")
            print(f"   2. 在用户变量PATH中添加: {install_dir}")
        else:
            print("\nUnix/Linux/macOS用户:")
            print(f"   在 ~/.bashrc 或 ~/.zshrc 中添加:")
            print(f"   export PATH=\"$PATH:{install_dir}\"")
    
    print(f"\n🎉 安装完成！现在可以使用: get_config_value <key>")

if __name__ == "__main__":
    install_global_scripts() 