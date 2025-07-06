#!/bin/zsh
#SBATCH -J memory_monitor_example
#SBATCH -N 1
#SBATCH --ntasks-per-node=46
#SBATCH -p work3
#SBATCH --output=%j_%x.log
#SBATCH --error=%j_%x.log

# 添加时间戳函数
log_with_timestamp() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# 内存监控函数
start_memory_monitor() {
    local monitor_log="$1"
    local interval="${2:-5}"  # 默认5秒间隔
    
    log_with_timestamp "🔍 启动内存监控，日志文件: $monitor_log, 间隔: ${interval}秒"
    
    # 后台进程监控内存
    (
        echo "时间,总内存(GB),已用内存(GB),可用内存(GB),内存使用率(%),进程内存(MB),进程CPU(%)" > "$monitor_log"
        while true; do
            # 获取系统内存信息
            local mem_info=$(free -g | grep "Mem:")
            local total_mem=$(echo $mem_info | awk '{print $2}')
            local used_mem=$(echo $mem_info | awk '{print $3}')
            local avail_mem=$(echo $mem_info | awk '{print $7}')
            local mem_usage=$(echo "scale=1; $used_mem * 100 / $total_mem" | bc)
            
            # 获取当前作业的进程内存和CPU使用情况
            local job_processes=""
            local total_proc_mem=0
            local total_proc_cpu=0
            
            if [ -n "${SLURM_JOB_ID:-}" ]; then
                # 通过SLURM作业ID获取进程
                job_processes=$(pgrep -f "${SLURM_JOB_ID}" 2>/dev/null || echo "")
            fi
            
            # 如果没有找到SLURM进程，尝试查找常见的GRASP进程
            if [ -z "$job_processes" ]; then
                job_processes=$(pgrep -f "rmcdhf\|rci\|rangular\|python.*train.py\|python.*choosing_csfs.py" 2>/dev/null || echo "")
            fi
            
            if [ -n "$job_processes" ]; then
                for pid in $job_processes; do
                    if [ -d "/proc/$pid" ]; then
                        local proc_mem=$(ps -p $pid -o rss= 2>/dev/null | awk '{print $1/1024}' || echo "0")
                        local proc_cpu=$(ps -p $pid -o %cpu= 2>/dev/null || echo "0")
                        total_proc_mem=$(echo "$total_proc_mem + $proc_mem" | bc)
                        total_proc_cpu=$(echo "$total_proc_cpu + $proc_cpu" | bc)
                    fi
                done
            fi
            
            local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
            echo "$timestamp,$total_mem,$used_mem,$avail_mem,$mem_usage,$total_proc_mem,$total_proc_cpu" >> "$monitor_log"
            
            sleep $interval
        done
    ) &
    
    # 保存监控进程PID
    echo $! > "/tmp/memory_monitor_${SLURM_JOB_ID}.pid"
    log_with_timestamp "✅ 内存监控已启动，PID: $!"
}

# 停止内存监控
stop_memory_monitor() {
    local pid_file="/tmp/memory_monitor_${SLURM_JOB_ID}.pid"
    if [ -f "$pid_file" ]; then
        local monitor_pid=$(cat "$pid_file")
        if kill -0 "$monitor_pid" 2>/dev/null; then
            kill "$monitor_pid"
            log_with_timestamp "🛑 内存监控已停止，PID: $monitor_pid"
        fi
        rm -f "$pid_file"
    fi
}

# 生成内存使用报告
generate_memory_report() {
    local monitor_log="$1"
    local report_file="${monitor_log%.csv}_report.txt"
    
    if [ -f "$monitor_log" ]; then
        log_with_timestamp "📊 生成内存使用报告: $report_file"
        
        {
            echo "========== 内存使用报告 =========="
            echo "作业ID: ${SLURM_JOB_ID:-未知}"
            echo "作业名: ${SLURM_JOB_NAME:-未知}"
            echo "生成时间: $(date)"
            echo ""
            
            # 计算统计信息
            local max_mem=$(tail -n +2 "$monitor_log" | cut -d',' -f3 | sort -n | tail -1)
            local avg_mem=$(tail -n +2 "$monitor_log" | cut -d',' -f3 | awk '{sum+=$1} END {print sum/NR}')
            local max_proc_mem=$(tail -n +2 "$monitor_log" | cut -d',' -f6 | sort -n | tail -1)
            local avg_proc_mem=$(tail -n +2 "$monitor_log" | cut -d',' -f6 | awk '{sum+=$1} END {print sum/NR}')
            
            echo "系统内存统计："
            echo "  最大内存使用: ${max_mem} GB"
            echo "  平均内存使用: $(printf "%.2f" $avg_mem) GB"
            echo ""
            echo "进程内存统计："
            echo "  最大进程内存: $(printf "%.2f" $max_proc_mem) MB"
            echo "  平均进程内存: $(printf "%.2f" $avg_proc_mem) MB"
            echo ""
            echo "详细数据请查看: $monitor_log"
            echo "================================="
        } > "$report_file"
        
        log_with_timestamp "✅ 内存报告已生成: $report_file"
    fi
}

# 主程序开始
log_with_timestamp "========== 开始执行 sbatch 脚本 =========="

# 设置内存监控
MEMORY_LOG="${SLURM_JOB_ID}_memory_usage.csv"
start_memory_monitor "$MEMORY_LOG" 10  # 每10秒记录一次

# 设置陷阱，确保脚本退出时停止监控
trap 'stop_memory_monitor; generate_memory_report "$MEMORY_LOG"' EXIT

# 这里是您的实际计算代码
log_with_timestamp "开始执行计算任务..."

# 示例：模拟一些计算任务
for i in {1..5}; do
    log_with_timestamp "执行任务 $i/5..."
    # 模拟内存密集型任务
    python -c "
import numpy as np
import time
print('创建大型数组...')
arr = np.random.random((1000, 1000, 10))
print(f'数组大小: {arr.nbytes / 1024 / 1024:.2f} MB')
time.sleep(20)  # 保持20秒
print('任务完成')
" &
    
    # 等待任务完成
    wait
    log_with_timestamp "任务 $i 完成"
done

log_with_timestamp "========== sbatch 脚本执行完成 ==========" 