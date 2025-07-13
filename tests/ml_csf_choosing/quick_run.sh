#!/bin/bash

# Quick ML Runner - 简化版本
# 用于快速运行ML相关程序

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

# 颜色
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== ML CSF Choosing Quick Runner ===${NC}"

# 检查配置文件
if [[ ! -f "config.toml" ]]; then
    echo -e "${RED}错误: 未找到 config.toml${NC}"
    echo "请在包含config.toml的目录中运行此脚本"
    exit 1
fi

# 显示选项
echo "请选择要运行的程序:"
echo "1) initial_csfs.py  - 初始化CSFs"
echo "2) choosing_csfs.py - 选择CSFs"  
echo "3) train.py         - 机器学习训练"
echo "4) 运行所有程序"
echo "q) 退出"

while true; do
    echo -n "输入选择 [1-4/q]: "
    read -r choice
    
    case $choice in
        1)
            echo -e "${YELLOW}运行 initial_csfs.py...${NC}"
            python "${SCRIPT_DIR}/initial_csfs.py"
            break
            ;;
        2)
            echo -e "${YELLOW}运行 choosing_csfs.py...${NC}"
            python "${SCRIPT_DIR}/choosing_csfs.py"
            break
            ;;
        3)
            echo -e "${YELLOW}运行 train.py...${NC}"
            python "${SCRIPT_DIR}/train.py"
            break
            ;;
        4)
            echo -e "${YELLOW}运行所有程序...${NC}"
            for script in "initial_csfs.py" "choosing_csfs.py" "train.py"; do
                echo -e "${GREEN}>>> 运行 $script${NC}"
                python "${SCRIPT_DIR}/$script"
                if [[ $? -ne 0 ]]; then
                    echo -e "${RED}$script 运行失败${NC}"
                    echo -n "是否继续? [y/N]: "
                    read -r cont
                    if [[ ! "$cont" =~ ^[Yy]$ ]]; then
                        break
                    fi
                fi
            done
            break
            ;;
        q|Q)
            echo "退出"
            exit 0
            ;;
        *)
            echo -e "${RED}无效选择，请输入 1-4 或 q${NC}"
            ;;
    esac
done