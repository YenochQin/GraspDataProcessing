#!/usr/bin/env python3
"""
安装GRASP工具到系统PATH
"""

import os
import sys
import shutil
from pathlib import Path

def install_tools():
    """安装工具到系统PATH"""
    
    # 获取当前脚本目录
    script_dir = Path(__file__).parent
    tools_script = script_dir / "grasp_tools.py"
    
    if not tools_script.exists():
        print(f"错误: 工具脚本不存在: {tools_script}")
        return False
    
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
    if sys.platform == "win32":
        # Windows批处理文件
        batch_content = f'''@echo off
python "{tools_script}" %*
'''
        script_path = install_dir / "grasp-tools.bat"
        with open(script_path, 'w') as f:
            f.write(batch_content)
    else:
        # Unix shell脚本
        shell_content = f'''#!/bin/bash
python3 "{tools_script}" "$@"
'''
        script_path = install_dir / "grasp-tools"
        with open(script_path, 'w') as f:
            f.write(shell_content)
        # 设置可执行权限
        os.chmod(script_path, 0o755)
    
    print(f"✅ 工具已安装到: {script_path}")
    
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
    
    print(f"\n🎉 安装完成！")
    print(f"使用方法:")
    print(f"  grasp-tools get <key>                    # 获取配置值")
    print(f"  grasp-tools set <key> <value>            # 设置配置值")
    print(f"  grasp-tools get continue_cal             # 获取continue_cal值")
    print(f"  grasp-tools set cal_loop_num 2           # 设置cal_loop_num为2")
    
    return True

if __name__ == "__main__":
    install_tools() 