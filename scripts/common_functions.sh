#!/bin/bash

# Set proper locale and encoding for Chinese characters
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

# 公共shell函数库
# 消除重复代码，提供统一的日志和工具函数

# 带时间戳的日志函数
log_with_timestamp() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] $message"
}

# 带时间戳的错误日志函数
log_error_with_timestamp() {
    local message="$1" 
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] ERROR: $message" >&2
}

# 带时间戳的警告日志函数
log_warning_with_timestamp() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] WARNING: $message" >&2
}

# 阶段日志函数
log_stage() {
    local stage_name="$1"
    local stage_type="$2"  # START or END
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    if [ "$stage_type" = "START" ]; then
        echo "[$timestamp] 🔧 开始阶段: $stage_name"
    elif [ "$stage_type" = "END" ]; then
        echo "[$timestamp] ✅ 完成阶段: $stage_name"
    else
        echo "[$timestamp] 📊 阶段: $stage_name"
    fi
}

# 检查环境变量是否在SLURM中运行
is_slurm_environment() {
    if [ -n "$SLURM_JOB_ID" ] || [ -n "$SLURM_PROCID" ] || [ -n "$SLURM_LOCALID" ]; then
        return 0  # true
    else
        return 1  # false
    fi
}

# 检查是否在调试模式
is_debug_mode() {
    if [ "$DEBUG" = "1" ] || [ "$DEBUG" = "true" ] || [ "$PYTHON_DEBUG" = "1" ]; then
        return 0  # true
    else
        return 1  # false
    fi
}

# 环境感知的Python执行函数
run_python_with_env() {
    local python_script="$1"
    shift  # 移除第一个参数，剩下的作为脚本参数
    
    # 检查是否为配置读取脚本（不需要成功日志）
    local script_basename=$(basename "$python_script")
    local is_config_script=false
    if [[ "$script_basename" == "csfs_ml_choosing_config_load.py" ]]; then
        is_config_script=true
    fi
    
    # 设置环境变量
    if is_slurm_environment && ! is_debug_mode; then
        # SLURM生产环境：关闭进度条
        export PYTHONUNBUFFERED=1
        # 只为非配置脚本输出执行日志
        if [[ "$is_config_script" == "false" ]]; then
            log_with_timestamp "在SLURM环境中运行Python脚本（生产模式）: $python_script"
        fi
    else
        # 调试模式：启用进度条
        export DEBUG=1
        # 只为非配置脚本输出执行日志
        if [[ "$is_config_script" == "false" ]]; then
            log_with_timestamp "在调试模式中运行Python脚本: $python_script"
        fi
    fi
    
    # 执行Python脚本
    python "$python_script" "$@"
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        # 只为非配置脚本输出成功日志
        if [[ "$is_config_script" == "false" ]]; then
            log_with_timestamp "Python脚本执行成功: $python_script"
        fi
    else
        # 失败日志始终输出（包括配置脚本）
        log_error_with_timestamp "Python脚本执行失败: $python_script (退出码: $exit_code)"
    fi
    
    return $exit_code
}

# 安全的配置值读取函数（确保返回纯数值，失败时退出）
safe_get_config_value() {
    local config_file="$1"
    local key="$2"
    local description="$3"
    
    # 使用临时文件捕获输出和错误
    local temp_output=$(mktemp)
    local temp_error=$(mktemp)
    
    # 执行配置读取
    python "${GRASP_DATA_PROCESSING_ROOT}/scripts/csfs_ml_choosing_config_load.py" get "$key" -f "$config_file" > "$temp_output" 2> "$temp_error"
    local exit_code=$?
    
    # 读取结果
    local value=$(cat "$temp_output")
    local error_msg=$(cat "$temp_error")
    
    # 清理临时文件
    rm -f "$temp_output" "$temp_error"
    
    # 检查是否成功
    if [ $exit_code -ne 0 ] || [ -z "$value" ] || [[ "$value" == *"错误"* ]] || [[ "$value" == *"Error"* ]]; then
        log_error_with_timestamp "读取配置项 '$key' 失败: $description"
        if [ -n "$error_msg" ]; then
            log_error_with_timestamp "错误详情: $error_msg"
        fi
        if [ -n "$value" ]; then
            log_error_with_timestamp "返回值: $value"
        fi
        exit 1
    fi
    
    # 返回纯净的值
    echo "$value"
}

# 文件存在性检查函数
check_file_exists() {
    local file_path="$1"
    local description="$2"
    
    if [ -f "$file_path" ]; then
        log_with_timestamp "找到文件: $description ($file_path)"
        return 0
    else
        log_error_with_timestamp "未找到文件: $description ($file_path)"
        return 1
    fi
}

# 目录创建函数
ensure_directory() {
    local dir_path="$1"
    local description="$2"
    
    if [ ! -d "$dir_path" ]; then
        mkdir -p "$dir_path"
        log_with_timestamp "创建目录: $description ($dir_path)"
    else
        log_with_timestamp "目录已存在: $description ($dir_path)"
    fi
}

# 计算执行时间的函数
calculate_execution_time() {
    local start_time="$1"
    local end_time="$2"
    local execution_time=$((end_time - start_time))
    
    local hours=$((execution_time / 3600))
    local minutes=$(((execution_time % 3600) / 60))
    local seconds=$((execution_time % 60))
    
    if [ $hours -gt 0 ]; then
        echo "${hours}小时${minutes}分钟${seconds}秒"
    elif [ $minutes -gt 0 ]; then
        echo "${minutes}分钟${seconds}秒"
    else
        echo "${seconds}秒"
    fi
}

# 输出环境信息
print_environment_info() {
    log_with_timestamp "=== 环境信息 ==="
    log_with_timestamp "主机名: $(hostname)"
    log_with_timestamp "当前用户: $(whoami)"
    log_with_timestamp "工作目录: $(pwd)"
    
    if is_slurm_environment; then
        log_with_timestamp "运行环境: SLURM作业 (Job ID: ${SLURM_JOB_ID:-未知})"
    else
        log_with_timestamp "运行环境: 本地执行"
    fi
    
    if is_debug_mode; then
        log_with_timestamp "调试模式: 已启用"
    else
        log_with_timestamp "调试模式: 已禁用"
    fi
    
    log_with_timestamp "Python版本: $(python --version 2>&1)"
    log_with_timestamp "==================="
} 

# 根据程序名确定期望的输出文件
get_expected_files() {
    local program_name="$1"
    local conf="$2" 
    local loop="$3"
    
    case "$program_name" in
        "mkdisks")
            echo "disks"
            ;;
        "rangular_mpi")
            echo ""  # 没有文件输出
            ;;
        "rwfnestimate")
            echo "rwfn.inp"
            ;;
        "rmcdhf_mem_mpi"|"rmcdhf_mpi")
            echo "rwfn.out rmix.out"
            ;;
        "rsave")
            echo "${conf}_${loop}.w ${conf}_${loop}.c ${conf}_${loop}.m ${conf}_${loop}.sum ${conf}_${loop}.alog ${conf}_${loop}.log"
            ;;
        "jj2lsj")
            echo "${conf}_${loop}.lsj.lbl"
            ;;
        "rci_mpi")
            echo "${conf}_${loop}.cm"
            ;;
        "rlevels")
            echo "${conf}_${loop}.level"
            ;;
        "rnucleus")
            echo "isodata"
            ;;
        *)
            echo ""  # 未知程序，不检查文件
            ;;
    esac
}

# GRASP程序错误检查函数
check_grasp_errors() {
    local program_name="$1"
    local output_log="$2"
    local expected_files="$3"
    
    # 检查严重错误模式
    local error_patterns=(
        "Fortran runtime error"
        "Error termination"
        "does not exist, redo"
        "STOP.*ERROR"
        "ABORT"
        "Segmentation fault"
        "Bus error"
        "killed"
        "core dumped"
    )
    
    # 搜索错误模式
    for pattern in "${error_patterns[@]}"; do
        if grep -qi "$pattern" "$output_log"; then
            log_with_timestamp "❌ $program_name 检测到错误: $pattern"
            log_with_timestamp "错误上下文："
            grep -i -A2 -B2 "$pattern" "$output_log" | tail -10
            return 1
        fi
    done
    
    # 检查是否有预期的输出文件
    if [ -n "$expected_files" ]; then
        log_with_timestamp "📁 当前工作目录: $(pwd)"
        log_with_timestamp "📋 检查预期文件: $expected_files"
        
        # 对于 MPI 程序，等待一段时间确保文件完全写入
        if [[ "$program_name" == *"_mpi" ]]; then
            log_with_timestamp "⏱️ MPI程序检测，等待3秒确保文件写入完成..."
            sleep 3
        fi
        
        # 使用重试机制检查文件
        local max_retries=5
        local retry_count=0
        local all_files_exist=false
        
        # 将文件列表转换为数组进行处理
        local files_array
        if [[ -n "${ZSH_VERSION:-}" ]]; then
            # zsh 下使用特殊语法
            files_array=(${=expected_files})
        else
            # bash 下直接分割
            files_array=($expected_files)
        fi
        local file_count=${#files_array[@]}
        
        # 如果分割失败，尝试使用 read 命令
        if [ $file_count -eq 1 ] && [[ "$expected_files" == *" "* ]]; then
            log_with_timestamp "⚠️ 检测到文件列表可能未正确分割，尝试使用read命令..."
            files_array=()
            local IFS=' '
            read -r -a files_array <<< "$expected_files"
            file_count=${#files_array[@]}
        fi
        
        log_with_timestamp "🔍 预期文件数量: $file_count 个"
        local index=1
        for file in "${files_array[@]}"; do
            log_with_timestamp "  [$index]: '$file'"
            index=$((index + 1))
        done
        
        while [ $retry_count -lt $max_retries ] && [ "$all_files_exist" = false ]; do
            all_files_exist=true
            local missing_files=""
            
            # 使用数组进行文件检查
            for file in "${files_array[@]}"; do
                log_with_timestamp "🔍 检查文件: '$file'"
                
                if [ ! -f "$file" ]; then
                    missing_files="$missing_files $file"
                    all_files_exist=false
                elif [ ! -s "$file" ]; then
                    log_with_timestamp "❌ $program_name 生成的文件为空: $file"
                    all_files_exist=false
                    break
                fi
            done
            
            if [ "$all_files_exist" = false ]; then
                retry_count=$((retry_count + 1))
                if [ $retry_count -lt $max_retries ]; then
                    log_with_timestamp "⏳ 第 $retry_count 次重试，等待文件生成... 缺失文件:$missing_files"
                    sleep 2
                else
                    log_with_timestamp "❌ $program_name 未生成预期文件:$missing_files"
                    log_with_timestamp "📂 当前目录内容："
                    ls -la
                    log_with_timestamp "🔍 详细检查文件情况："
                    for file in "${files_array[@]}"; do
                        log_with_timestamp "检查文件: $file"
                        if [ -e "$file" ]; then
                            log_with_timestamp "  - 文件存在但可能有问题"
                            log_with_timestamp "  - 文件大小: $(du -h "$file" 2>/dev/null || echo "无法获取大小")"
                            log_with_timestamp "  - 文件权限: $(ls -l "$file" 2>/dev/null || echo "无法获取权限")"
                        else
                            log_with_timestamp "  - 文件不存在"
                            # 查找类似的文件名
                            local basename=$(basename "$file")
                            local similar_files=$(ls -1 | grep -i "${basename%.*}" 2>/dev/null || echo "无相似文件")
                            log_with_timestamp "  - 相似文件: $similar_files"
                        fi
                    done
                    return 1
                fi
            fi
        done
        
        if [ "$all_files_exist" = true ]; then
            log_with_timestamp "✅ 所有预期文件检查通过: $expected_files"
        fi
    fi
    
    return 0
}

# 安全执行GRASP程序的函数
safe_grasp_execute() {
    local program_name="$1"
    local input_commands="$2"
    shift 2
    
    log_with_timestamp "执行 $program_name..."
    
    # 动态获取期望的文件列表
    local expected_files=""
    if [[ "$program_name" == "rsave" ]]; then
        # 对于 rsave，我们需要从上下文获取 conf 和 loop 变量
        expected_files=$(get_expected_files "$program_name" "$conf" "$loop")
    else
        expected_files=$(get_expected_files "$program_name" "$conf" "$loop")
    fi
    
    log_with_timestamp "🎯 根据程序 $program_name 自动确定期望文件: $expected_files"
    
    # 创建临时日志文件
    local temp_log="/tmp/${program_name}_${SLURM_JOB_ID}_$$.log"
    
    # 执行程序并获取退出码
    local exit_code=0
    if [ -n "$input_commands" ]; then
        # 带输入的程序
        echo "$input_commands" | "$@" 2>&1 | tee "$temp_log"
        exit_code=${PIPESTATUS[0]:-$?}
    else
        # 不带输入的程序
        "$@" 2>&1 | tee "$temp_log"
        exit_code=$?
    fi
    
    # 确保退出码是数字
    if [ -z "$exit_code" ]; then
        exit_code=1
        log_with_timestamp "⚠️ 无法获取 $program_name 的退出码，假设为失败"
    elif ! [[ "$exit_code" =~ ^[0-9]+$ ]]; then
        exit_code=1
        log_with_timestamp "⚠️ $program_name 的退出码不是数字，假设为失败"
    fi
    
    # 检查退出码
    if [ "$exit_code" -ne 0 ]; then
        log_with_timestamp "❌ $program_name 非正常退出，退出码: $exit_code"
        log_with_timestamp "最后的输出："
        tail -20 "$temp_log"
        rm -f "$temp_log"
        exit 1
    fi
    
    # 检查GRASP特定错误
    check_grasp_errors "$program_name" "$temp_log" "$expected_files"
    local check_result=$?
    if [ "$check_result" -ne 0 ]; then
        rm -f "$temp_log"
        exit 1
    fi
    
    rm -f "$temp_log"
    log_with_timestamp "✅ $program_name 完成"
}

# =============================================================================
# 日志格式增强函数
# =============================================================================

# 颜色代码定义
readonly COLOR_RED='\033[0;31m'
readonly COLOR_GREEN='\033[0;32m'
readonly COLOR_YELLOW='\033[1;33m'
readonly COLOR_BLUE='\033[0;34m'
readonly COLOR_PURPLE='\033[0;35m'
readonly COLOR_CYAN='\033[0;36m'
readonly COLOR_WHITE='\033[1;37m'
readonly COLOR_BOLD='\033[1m'
readonly COLOR_RESET='\033[0m'

# 路径简化函数 - 去除root_path前缀，只显示相对路径
simplify_path() {
    local full_path="$1"
    local root_path="$2"
    
    # 如果没有提供root_path，尝试从config.toml中获取
    if [ -z "$root_path" ] && [ -f "config.toml" ]; then
        root_path=$(safe_get_config_value "config.toml" "root_path" "根目录路径" 2>/dev/null || echo "")
    fi
    
    # 如果root_path为空或者路径不包含root_path，返回原路径
    if [ -z "$root_path" ] || [[ "$full_path" != "$root_path"* ]]; then
        echo "$full_path"
        return
    fi
    
    # 移除root_path前缀
    local relative_path="${full_path#$root_path}"
    # 移除开头的斜杠
    relative_path="${relative_path#/}"
    
    # 如果简化后路径为空，表示就是root目录
    if [ -z "$relative_path" ]; then
        echo "."
    else
        echo "$relative_path"
    fi
}

# 数值高亮函数
highlight_number() {
    local text="$1"
    local color="${2:-$COLOR_CYAN}"
    
    # 使用颜色高亮数值
    echo -e "${color}${text}${COLOR_RESET}"
}

# 参数高亮函数
highlight_param() {
    local key="$1"
    local value="$2"
    local key_color="${3:-$COLOR_WHITE}"
    local value_color="${4:-$COLOR_CYAN}"
    
    echo -e "${key_color}${key}${COLOR_RESET}=$(highlight_number "$value" "$value_color")"
}

# 支持路径简化的日志函数
log_with_timestamp_and_path() {
    local message="$1"
    local path_to_simplify="$2"
    local root_path="$3"
    
    if [ -n "$path_to_simplify" ]; then
        local simplified_path=$(simplify_path "$path_to_simplify" "$root_path")
        message="${message}: ${simplified_path}"
    fi
    
    log_with_timestamp "$message"
}

# 增强的配置参数日志函数
log_config_params() {
    local atom="$1"
    local conf="$2" 
    local processor="$3"
    local active_space="$4"
    local cal_levels="$5"
    
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "[$timestamp] 配置参数: $(highlight_param "atom" "$atom") $(highlight_param "conf" "$conf") $(highlight_param "processor" "$processor" "$COLOR_WHITE" "$COLOR_GREEN")"
    echo -e "[$timestamp] 活性空间: $(highlight_number "$active_space" "$COLOR_YELLOW"), 计算能级: $(highlight_number "$cal_levels" "$COLOR_YELLOW")"
}