#!/bin/bash

# ML CSF Choosing Scripts Runner
# 用于运行机器学习CSF选择相关的Python程序

# 设置脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# 设置Python路径
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 显示使用说明
show_usage() {
    echo -e "${BLUE}ML CSF Choosing Scripts Runner${NC}"
    echo "用法: $0 [选项] [程序名]"
    echo ""
    echo "可用的程序:"
    echo "  1, initial     - 运行 initial_csfs.py (初始化CSFs)"
    echo "  2, choosing    - 运行 choosing_csfs.py (选择CSFs)" 
    echo "  3, train       - 运行 train.py (机器学习训练)"
    echo "  all           - 依次运行所有程序"
    echo ""
    echo "选项:"
    echo "  -h, --help    - 显示此帮助信息"
    echo "  -v, --verbose - 详细输出模式"
    echo "  -d, --dir DIR - 指定工作目录 (默认为当前目录)"
    echo ""
    echo "示例:"
    echo "  $0 train                    # 运行训练程序"
    echo "  $0 all                      # 运行所有程序"
    echo "  $0 -d /path/to/work train   # 在指定目录运行训练程序"
}

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查Python环境
check_python_env() {
    log_info "检查Python环境..."
    
    # 检查Python是否可用
    if ! command -v python &> /dev/null; then
        log_error "Python未找到，请确保Python已安装"
        exit 1
    fi
    
    # 检查graspdataprocessing模块
    if ! python -c "import graspdataprocessing" &> /dev/null; then
        log_error "graspdataprocessing模块未找到，请确保已正确安装"
        exit 1
    fi
    
    log_info "Python环境检查通过"
}

# 检查配置文件
check_config() {
    if [[ ! -f "config.toml" ]]; then
        log_error "配置文件 config.toml 未找到"
        log_info "请确保在正确的工作目录中运行脚本"
        exit 1
    fi
    log_info "找到配置文件: config.toml"
}

# 运行Python程序
run_python_script() {
    local script_name="$1"
    local script_path="${SCRIPT_DIR}/${script_name}"
    
    if [[ ! -f "$script_path" ]]; then
        log_error "脚本文件不存在: $script_path"
        return 1
    fi
    
    log_info "运行脚本: $script_name"
    log_info "工作目录: $(pwd)"
    log_info "命令: python $script_path"
    
    echo "----------------------------------------"
    
    # 运行脚本并捕获退出码
    python "$script_path"
    local exit_code=$?
    
    echo "----------------------------------------"
    
    if [[ $exit_code -eq 0 ]]; then
        log_info "$script_name 运行成功"
        return 0
    else
        log_error "$script_name 运行失败 (退出码: $exit_code)"
        return $exit_code
    fi
}

# 运行所有程序
run_all() {
    log_info "开始运行所有ML程序..."
    
    local scripts=("initial_csfs.py" "choosing_csfs.py" "train.py")
    local failed=0
    
    for script in "${scripts[@]}"; do
        if ! run_python_script "$script"; then
            ((failed++))
            log_error "程序 $script 执行失败"
            
            # 询问是否继续
            echo -n "是否继续运行下一个程序? [y/N]: "
            read -r response
            if [[ ! "$response" =~ ^[Yy]$ ]]; then
                log_warn "用户中止执行"
                break
            fi
        fi
    done
    
    if [[ $failed -eq 0 ]]; then
        log_info "所有程序运行成功！"
    else
        log_warn "有 $failed 个程序运行失败"
    fi
}

# 主函数
main() {
    local verbose=false
    local work_dir=""
    local program=""
    
    # 解析参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            -v|--verbose)
                verbose=true
                shift
                ;;
            -d|--dir)
                work_dir="$2"
                shift 2
                ;;
            1|initial)
                program="initial_csfs.py"
                shift
                ;;
            2|choosing)
                program="choosing_csfs.py"
                shift
                ;;
            3|train)
                program="train.py"
                shift
                ;;
            all)
                program="all"
                shift
                ;;
            *)
                log_error "未知参数: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # 如果没有指定程序，显示帮助
    if [[ -z "$program" ]]; then
        show_usage
        exit 1
    fi
    
    # 切换到工作目录
    if [[ -n "$work_dir" ]]; then
        if [[ ! -d "$work_dir" ]]; then
            log_error "工作目录不存在: $work_dir"
            exit 1
        fi
        cd "$work_dir" || exit 1
        log_info "切换到工作目录: $work_dir"
    fi
    
    # 详细模式
    if [[ "$verbose" == "true" ]]; then
        set -x
    fi
    
    # 执行前检查
    check_python_env
    check_config
    
    # 执行程序
    case "$program" in
        "all")
            run_all
            ;;
        *)
            run_python_script "$program"
            ;;
    esac
}

# 运行主函数
main "$@"