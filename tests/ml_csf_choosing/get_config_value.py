#!/usr/bin/env python3
"""
读取TOML配置文件中的值
用法: python get_config_value.py <key>
"""

import sys
import toml
from pathlib import Path

def get_config_value(key, config_file="config.toml"):
    """从TOML配置文件中读取指定键的值"""
    try:
        config_path = Path(config_file)
        if not config_path.exists():
            print(f"错误: 配置文件 {config_file} 不存在", file=sys.stderr)
            sys.exit(1)
        
        config = toml.load(config_path)
        
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

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python get_config_value.py <key>", file=sys.stderr)
        sys.exit(1)
    
    key = sys.argv[1]
    get_config_value(key) 