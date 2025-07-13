#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- encoding: utf-8 -*-
'''
@Id :csfs_ml_choosing_config_load.py
@date :2025/06/22 13:25:05
@author :YenochQin (秦毅)
'''

"""
GRASP数据处理工具包
提供命令行工具用于处理GRASP项目相关任务
"""

'''
### 设置shebang 指向python环境
将本文件的第一行修改为：用户python环境路径
在激活的conda环境中执行：
which python
将输出结果复制到本文件的第一行'!'后

### 修改本文件为可执行程序
chmod +x csfs_ml_choosing_config_load.py

### 将本文件添加到PATH环境变量
export PATH=$PATH:{/path/to/csfs_ml_choosing_config_load.py}
#### 注意修改路径为实际路径

### 获取配置值

```bash
# 获取简单键值
csfs_ml_choosing_config_load get continue_cal

# 获取嵌套键值
csfs_ml_choosing_config_load get model_params.n_estimators

# 指定配置文件
csfs_ml_choosing_config_load get continue_cal -f /path/to/config.toml
```

### 设置配置值

```bash
# 设置简单键值
csfs_ml_choosing_config_load set cal_loop_num 2

# 设置嵌套键值
csfs_ml_choosing_config_load set model_params.random_state 42

# 指定配置文件
csfs_ml_choosing_config_load set continue_cal false -f /path/to/config.toml
```
'''

import sys
import argparse
from pathlib import Path

# 使用多层级TOML库支持，提供最大兼容性
TOML_SUPPORT = None

# 尝试不同的TOML库，按优先级排序
try:
    # 方案1: Python 3.11+ 标准库 tomllib (读) + toml (写)
    import tomllib
    import toml
    def load_toml(file_path):
        with open(file_path, 'rb') as f:
            return tomllib.load(f)
    def dump_toml(data, file_path):
        with open(file_path, 'w') as f:
            toml.dump(data, f)
    TOML_SUPPORT = "tomllib+toml"
except ImportError:
    try:
        # 方案2: 使用toml库
        import toml
        def load_toml(file_path):
            return toml.load(file_path)
        def dump_toml(data, file_path):
            with open(file_path, 'w') as f:
                toml.dump(data, f)
        TOML_SUPPORT = "toml"
    except ImportError:
        try:
            # 方案3: 使用tomli (读) + tomli_w (写)
            import tomli
            import tomli_w
            def load_toml(file_path):
                with open(file_path, 'rb') as f:
                    return tomli.load(f)
            def dump_toml(data, file_path):
                with open(file_path, 'wb') as f:
                    tomli_w.dump(data, f)
            TOML_SUPPORT = "tomli+tomli_w"
        except ImportError:
            print("错误: 需要安装TOML库支持TOML文件操作", file=sys.stderr)
            print("请选择以下任一方案安装:", file=sys.stderr)
            print("  方案1: pip install toml", file=sys.stderr)
            print("  方案2: pip install tomli tomli-w", file=sys.stderr)
            print("  方案3: 升级到Python 3.11+并安装: pip install toml", file=sys.stderr)
            sys.exit(1)

# 输出调试信息（可选，仅在详细模式下显示）
# if __debug__:
#     print(f"调试: 使用TOML支持方案: {TOML_SUPPORT}", file=sys.stderr)

def get_config_value(key, config_file="config.toml"):
    """从TOML配置文件中读取指定键的值"""
    try:
        config_path = Path(config_file)
        if not config_path.exists():
            print(f"错误: 配置文件 {config_file} 不存在", file=sys.stderr)
            sys.exit(1)
        
        config = load_toml(config_path)
        
        # 支持嵌套键，如 "model_params.n_estimators"
        keys = key.split('.')
        value = config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                print(f"错误: 键 '{key}' 不存在", file=sys.stderr)
                sys.exit(1)
        
        # 输出值（布尔值转换为小写字符串）
        if isinstance(value, bool):
            print(str(value).lower())
        else:
            print(value)
            
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)

def set_config_value(key, value, config_file="config.toml"):
    """设置TOML配置文件中的值"""
    try:
        config_path = Path(config_file)
        if not config_path.exists():
            print(f"错误: 配置文件 {config_file} 不存在", file=sys.stderr)
            sys.exit(1)
        
        config = load_toml(config_path)
        
        # 支持嵌套键
        keys = key.split('.')
        current = config
        
        # 导航到父级字典
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        # 设置值
        current[keys[-1]] = value
        
        # 保存配置
        dump_toml(config, config_path)
        
        print(f"✅ 已设置 {key} = {value}")
        
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="GRASP数据处理工具")
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # get命令
    get_parser = subparsers.add_parser('get', help='获取配置值')
    get_parser.add_argument('key', help='配置键名')
    get_parser.add_argument('-f', '--file', default='config.toml', help='配置文件路径')
    
    # set命令
    set_parser = subparsers.add_parser('set', help='设置配置值')
    set_parser.add_argument('key', help='配置键名')
    set_parser.add_argument('value', help='配置值')
    set_parser.add_argument('-f', '--file', default='config.toml', help='配置文件路径')
    
    args = parser.parse_args()
    
    if args.command == 'get':
        get_config_value(args.key, args.file)
    elif args.command == 'set':
        set_config_value(args.key, args.value, args.file)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 