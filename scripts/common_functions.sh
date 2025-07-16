#!/bin/bash

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
    
    # 设置环境变量
    if is_slurm_environment && ! is_debug_mode; then
        # SLURM生产环境：关闭进度条
        export PYTHONUNBUFFERED=1
        log_with_timestamp "在SLURM环境中运行Python脚本（生产模式）: $python_script"
    else
        # 调试模式：启用进度条
        export DEBUG=1
        log_with_timestamp "在调试模式中运行Python脚本: $python_script"
    fi
    
    # 执行Python脚本
    python "$python_script" "$@"
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        log_with_timestamp "Python脚本执行成功: $python_script"
    else
        log_error_with_timestamp "Python脚本执行失败: $python_script (退出码: $exit_code)"
    fi
    
    return $exit_code
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