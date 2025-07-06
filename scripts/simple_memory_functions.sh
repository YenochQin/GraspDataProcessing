# ============= 简单内存监控函数集合 =============
# 可以直接复制这些函数到您的现有sbatch脚本中使用

# 简单的内存监控函数
simple_memory_monitor() {
    local log_file="${1:-memory.log}"
    local interval="${2:-30}"
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🔍 启动内存监控: $log_file (间隔${interval}秒)"
    
    # 后台监控进程
    (
        echo "时间,系统内存使用(GB),GRASP进程内存(MB)" > "$log_file"
        while true; do
            # 获取系统内存使用
            local sys_mem=$(free -g | grep "Mem:" | awk '{print $3}')
            
            # 获取GRASP相关进程内存
            local grasp_mem=0
            local grasp_pids=$(pgrep -f "rmcdhf\|rci\|rangular\|python.*train\|python.*choosing" 2>/dev/null)
            if [ -n "$grasp_pids" ]; then
                for pid in $grasp_pids; do
                    if [ -d "/proc/$pid" ]; then
                        local proc_mem=$(ps -p $pid -o rss= 2>/dev/null | awk '{print $1/1024}' || echo "0")
                        grasp_mem=$(echo "$grasp_mem + $proc_mem" | bc 2>/dev/null || echo "$grasp_mem")
                    fi
                done
            fi
            
            local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
            echo "$timestamp,$sys_mem,$grasp_mem" >> "$log_file"
            
            sleep $interval
        done
    ) &
    
    # 保存监控进程PID
    echo $! > "/tmp/simple_monitor_$$.pid"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ 内存监控已启动，PID: $!"
}

# 停止内存监控
stop_simple_monitor() {
    local pid_file="/tmp/simple_monitor_$$.pid"
    if [ -f "$pid_file" ]; then
        local monitor_pid=$(cat "$pid_file")
        if kill -0 "$monitor_pid" 2>/dev/null; then
            kill "$monitor_pid"
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🛑 内存监控已停止"
        fi
        rm -f "$pid_file"
    fi
}

# 记录单个程序的内存使用
log_program_memory() {
    local program_name="$1"
    local current_mem=$(free -m | grep "Mem:" | awk '{print $3}')
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 📊 $program_name - 系统内存: ${current_mem}MB"
}

# 生成简单的内存报告
simple_memory_report() {
    local log_file="${1:-memory.log}"
    
    if [ -f "$log_file" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 📊 生成内存报告..."
        
        local max_sys=$(tail -n +2 "$log_file" | cut -d',' -f2 | sort -n | tail -1)
        local avg_sys=$(tail -n +2 "$log_file" | cut -d',' -f2 | awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print 0}')
        local max_grasp=$(tail -n +2 "$log_file" | cut -d',' -f3 | sort -n | tail -1)
        
        echo "========== 内存使用摘要 =========="
        echo "最大系统内存: ${max_sys} GB"
        echo "平均系统内存: $(printf "%.1f" $avg_sys) GB"
        echo "最大GRASP内存: $(printf "%.1f" $max_grasp) MB"
        echo "详细数据: $log_file"
        echo "================================"
    fi
}

# ============= 使用示例 =============
# 将以下代码添加到您的sbatch脚本中：

# # 在脚本开始处启动监控
# simple_memory_monitor "${SLURM_JOB_ID}_memory.log" 30
# 
# # 设置退出陷阱
# trap 'stop_simple_monitor; simple_memory_report "${SLURM_JOB_ID}_memory.log"' EXIT
# 
# # 在关键程序执行前后记录内存
# log_program_memory "开始rmcdhf计算"
# # ... 您的rmcdhf计算代码 ...
# log_program_memory "完成rmcdhf计算"
# 
# # 脚本结束时会自动生成报告 