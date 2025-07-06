#!/bin/zsh
#SBATCH -J slurm_memory_monitor
#SBATCH -N 1
#SBATCH --ntasks-per-node=46
#SBATCH -p work3
#SBATCH --output=%j_%x.log
#SBATCH --error=%j_%x.log
#SBATCH --mem=64G                    # 申请内存限制
#SBATCH --time=24:00:00             # 时间限制

# 添加时间戳函数
log_with_timestamp() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# SLURM内存监控函数
monitor_slurm_memory() {
    local interval="${1:-30}"  # 默认30秒间隔
    local log_file="${SLURM_JOB_ID}_slurm_memory.log"
    
    log_with_timestamp "🔍 启动SLURM内存监控，间隔: ${interval}秒"
    
    # 后台监控进程
    (
        echo "时间,最大RSS(MB),平均RSS(MB),最大虚拟内存(MB),页面错误数" > "$log_file"
        while true; do
            if [ -n "${SLURM_JOB_ID:-}" ]; then
                # 使用sstat获取作业统计信息
                local stats=$(sstat -j ${SLURM_JOB_ID} --format=MaxRSS,AveRSS,MaxVMSize,MinCPU,AveCPU --parsable2 --noheader 2>/dev/null | head -1)
                
                if [ -n "$stats" ]; then
                    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
                    # 解析sstat输出
                    local max_rss=$(echo "$stats" | cut -d'|' -f1 | sed 's/K$//' | awk '{print $1/1024}')
                    local ave_rss=$(echo "$stats" | cut -d'|' -f2 | sed 's/K$//' | awk '{print $1/1024}')
                    local max_vmsize=$(echo "$stats" | cut -d'|' -f3 | sed 's/K$//' | awk '{print $1/1024}')
                    
                    echo "$timestamp,$max_rss,$ave_rss,$max_vmsize,0" >> "$log_file"
                    
                    log_with_timestamp "📊 当前内存使用 - 最大RSS: ${max_rss}MB, 平均RSS: ${ave_rss}MB"
                fi
            fi
            sleep $interval
        done
    ) &
    
    echo $! > "/tmp/slurm_monitor_${SLURM_JOB_ID}.pid"
    log_with_timestamp "✅ SLURM内存监控已启动"
}

# 停止SLURM监控
stop_slurm_monitor() {
    local pid_file="/tmp/slurm_monitor_${SLURM_JOB_ID}.pid"
    if [ -f "$pid_file" ]; then
        local monitor_pid=$(cat "$pid_file")
        if kill -0 "$monitor_pid" 2>/dev/null; then
            kill "$monitor_pid"
            log_with_timestamp "🛑 SLURM内存监控已停止"
        fi
        rm -f "$pid_file"
    fi
}

# 生成最终的内存报告
generate_final_report() {
    local job_id="${SLURM_JOB_ID}"
    local report_file="${job_id}_final_memory_report.txt"
    
    log_with_timestamp "📊 生成最终内存报告..."
    
    # 等待作业完成后获取完整统计
    sleep 5
    
    {
        echo "========== SLURM 作业内存报告 =========="
        echo "作业ID: $job_id"
        echo "作业名: ${SLURM_JOB_NAME:-未知}"
        echo "生成时间: $(date)"
        echo ""
        
        # 获取作业完整统计信息
        echo "=== 作业完整统计信息 ==="
        sacct -j $job_id --format=JobID,JobName,MaxRSS,AveRSS,MaxVMSize,ElapsedTime,State --units=M 2>/dev/null || echo "无法获取sacct信息"
        echo ""
        
        # 获取作业效率信息
        echo "=== 作业效率信息 ==="
        seff $job_id 2>/dev/null || echo "无法获取seff信息"
        echo ""
        
        echo "详细监控数据请查看: ${job_id}_slurm_memory.log"
        echo "======================================="
    } > "$report_file"
    
    log_with_timestamp "✅ 最终报告已生成: $report_file"
}

# 主程序
log_with_timestamp "========== 开始执行 SLURM 内存监控脚本 =========="
log_with_timestamp "作业ID: ${SLURM_JOB_ID}"
log_with_timestamp "分配内存: ${SLURM_MEM_PER_NODE:-未知}MB"

# 启动监控
monitor_slurm_memory 60  # 每60秒监控一次

# 设置退出陷阱
trap 'stop_slurm_monitor; generate_final_report' EXIT

# 您的实际计算代码
log_with_timestamp "开始执行计算任务..."

# 示例计算任务
python -c "
import numpy as np
import time
print('开始内存密集型计算...')
for i in range(5):
    print(f'步骤 {i+1}/5')
    # 创建大型数组
    arr = np.random.random((2000, 2000, 5))
    print(f'数组大小: {arr.nbytes / 1024 / 1024:.2f} MB')
    # 进行一些计算
    result = np.sum(arr ** 2)
    print(f'计算结果: {result:.2e}')
    time.sleep(30)  # 保持30秒
    del arr
print('计算完成')
"

log_with_timestamp "========== SLURM 脚本执行完成 ==========" 