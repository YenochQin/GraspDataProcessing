#!/bin/zsh
#SBATCH -J Gd_IoddImlcij2as4_with_memory
#SBATCH -N 1
#SBATCH --ntasks-per-node=46
#SBATCH -p work3
#SBATCH --output=%j_%x.log
#SBATCH --error=%j_%x.log
#SBATCH --mem=64G                    # 建议添加内存限制
. /usr/share/Modules/init/zsh

# 添加时间戳函数
log_with_timestamp() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# ============= 内存监控功能 =============
# 启动内存监控
start_memory_monitor() {
    local monitor_log="$1"
    local interval="${2:-30}"  # 默认30秒间隔
    
    log_with_timestamp "🔍 启动内存监控，日志: $monitor_log, 间隔: ${interval}秒"
    
    # 后台进程监控内存
    (
        echo "时间,系统总内存(GB),系统已用内存(GB),系统可用内存(GB),系统内存使用率(%),GRASP进程内存(MB),GRASP进程CPU(%),Python进程内存(MB)" > "$monitor_log"
        while true; do
            # 获取系统内存信息
            local mem_info=$(free -g | grep "Mem:")
            local total_mem=$(echo $mem_info | awk '{print $2}')
            local used_mem=$(echo $mem_info | awk '{print $3}')
            local avail_mem=$(echo $mem_info | awk '{print $7}')
            local mem_usage=$(echo "scale=1; $used_mem * 100 / $total_mem" | bc 2>/dev/null || echo "0")
            
            # 监控GRASP相关进程
            local grasp_processes=$(pgrep -f "rmcdhf\|rci\|rangular\|mkdisks\|rsave\|jj2lsj\|rlevels" 2>/dev/null || echo "")
            local total_grasp_mem=0
            local total_grasp_cpu=0
            
            if [ -n "$grasp_processes" ]; then
                for pid in $grasp_processes; do
                    if [ -d "/proc/$pid" ]; then
                        local proc_mem=$(ps -p $pid -o rss= 2>/dev/null | awk '{print $1/1024}' || echo "0")
                        local proc_cpu=$(ps -p $pid -o %cpu= 2>/dev/null || echo "0")
                        total_grasp_mem=$(echo "$total_grasp_mem + $proc_mem" | bc 2>/dev/null || echo "$total_grasp_mem")
                        total_grasp_cpu=$(echo "$total_grasp_cpu + $proc_cpu" | bc 2>/dev/null || echo "$total_grasp_cpu")
                    fi
                done
            fi
            
            # 监控Python进程（机器学习训练）
            local python_processes=$(pgrep -f "python.*train.py\|python.*choosing_csfs.py\|python.*initial_csfs.py" 2>/dev/null || echo "")
            local total_python_mem=0
            
            if [ -n "$python_processes" ]; then
                for pid in $python_processes; do
                    if [ -d "/proc/$pid" ]; then
                        local proc_mem=$(ps -p $pid -o rss= 2>/dev/null | awk '{print $1/1024}' || echo "0")
                        total_python_mem=$(echo "$total_python_mem + $proc_mem" | bc 2>/dev/null || echo "$total_python_mem")
                    fi
                done
            fi
            
            local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
            echo "$timestamp,$total_mem,$used_mem,$avail_mem,$mem_usage,$total_grasp_mem,$total_grasp_cpu,$total_python_mem" >> "$monitor_log"
            
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
            echo "========== GRASP计算内存使用报告 =========="
            echo "作业ID: ${SLURM_JOB_ID:-未知}"
            echo "作业名: ${SLURM_JOB_NAME:-未知}"
            echo "生成时间: $(date)"
            echo ""
            
            # 计算统计信息
            local max_sys_mem=$(tail -n +2 "$monitor_log" | cut -d',' -f3 | sort -n | tail -1)
            local avg_sys_mem=$(tail -n +2 "$monitor_log" | cut -d',' -f3 | awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print 0}')
            local max_grasp_mem=$(tail -n +2 "$monitor_log" | cut -d',' -f6 | sort -n | tail -1)
            local avg_grasp_mem=$(tail -n +2 "$monitor_log" | cut -d',' -f6 | awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print 0}')
            local max_python_mem=$(tail -n +2 "$monitor_log" | cut -d',' -f8 | sort -n | tail -1)
            local avg_python_mem=$(tail -n +2 "$monitor_log" | cut -d',' -f8 | awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print 0}')
            
            echo "系统内存统计："
            echo "  最大系统内存使用: ${max_sys_mem} GB"
            echo "  平均系统内存使用: $(printf "%.2f" $avg_sys_mem) GB"
            echo ""
            echo "GRASP程序内存统计："
            echo "  最大GRASP内存: $(printf "%.2f" $max_grasp_mem) MB"
            echo "  平均GRASP内存: $(printf "%.2f" $avg_grasp_mem) MB"
            echo ""
            echo "Python程序内存统计："
            echo "  最大Python内存: $(printf "%.2f" $max_python_mem) MB"
            echo "  平均Python内存: $(printf "%.2f" $avg_python_mem) MB"
            echo ""
            echo "内存使用峰值时间点："
            grep "$(tail -n +2 "$monitor_log" | cut -d',' -f3 | sort -n | tail -1)" "$monitor_log" | head -1
            echo ""
            echo "详细数据请查看: $monitor_log"
            echo "可以使用以下命令分析数据："
            echo "  - 查看内存趋势: cut -d',' -f1,3 $monitor_log"
            echo "  - 查看GRASP内存: cut -d',' -f1,6 $monitor_log"
            echo "  - 查看Python内存: cut -d',' -f1,8 $monitor_log"
            echo "================================="
        } > "$report_file"
        
        log_with_timestamp "✅ 内存报告已生成: $report_file"
    fi
}

# 记录程序开始时的内存状态
log_program_memory() {
    local program_name="$1"
    local phase="$2"  # start 或 end
    
    local current_mem=$(free -m | grep "Mem:" | awk '{print $3}')
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    if [ "$phase" = "start" ]; then
        log_with_timestamp "🚀 开始执行 $program_name - 当前系统内存使用: ${current_mem}MB"
    else
        log_with_timestamp "✅ 完成执行 $program_name - 当前系统内存使用: ${current_mem}MB"
    fi
}

# ============= 原有的GRASP功能 =============
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

# GRASP程序错误检查函数（保持原有逻辑）
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
            files_array=(${=expected_files})
        else
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

# 安全执行GRASP程序的函数（增强版）
safe_grasp_execute() {
    local program_name="$1"
    local input_commands="$2"
    shift 2
    
    log_program_memory "$program_name" "start"
    
    # 动态获取期望的文件列表
    local expected_files=""
    if [[ "$program_name" == "rsave" ]]; then
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
    log_program_memory "$program_name" "end"
    log_with_timestamp "✅ $program_name 完成"
}

# ============= 主程序开始 =============
log_with_timestamp "========== 开始执行带内存监控的 GRASP 脚本 =========="
log_with_timestamp "作业名: ${SLURM_JOB_NAME:-未设置}"
log_with_timestamp "作业编号: ${SLURM_JOB_ID:-未设置}"

# 设置内存监控
MEMORY_LOG="${SLURM_JOB_ID}_grasp_memory_usage.csv"
start_memory_monitor "$MEMORY_LOG" 20  # 每20秒记录一次

# 设置陷阱，确保脚本退出时停止监控并生成报告
trap 'stop_memory_monitor; generate_memory_report "$MEMORY_LOG"' EXIT

###########################################
# mpi run CPU core
processor=46
###########################################
## module load
log_with_timestamp "加载必要的模块..."
module load mpi/openmpi-x86_64-gcc
module load openblas/0.3.28-gcc-11.4.1
module load grasp/grasp_openblas
###########################################
# ⚠️ 关键修改：确保正确加载 Conda（zsh 需要手动初始化）
log_with_timestamp "初始化 Conda 环境..."
source /home/workstation3/AppFiles/miniconda3/etc/profile.d/conda.sh  || {
    log_with_timestamp "❌ 加载 Conda 失败！请检查路径是否正确。"
    exit 1
}
conda activate grasp-env || {
    log_with_timestamp "❌ 激活环境失败！请确认环境名是否正确。"
    exit 1
}
log_with_timestamp "✅ Conda 环境激活成功"
###########################################
## configuration
atom=Gd_I
conf="cv6odd1_j2as4"
loop1_rwfn_file="mJ-1-90chosenas3_odd2.w"
rwfnestimate_file=cv6odd1_j2as4_1.w
Active_space="10s,9p,8d,7f,6g"
cal_levels="1-3"
log_with_timestamp "配置参数: atom=$atom, conf=$conf, processor=$processor"
###########################################
# 检查 Python 路径
log_with_timestamp "检查 Python 环境..."
which python
python --version

# 自动设置 GraspDataProcessing 包路径
GRASP_DATA_PROCESSING_ROOT="/home/workstation3/AppFiles/GraspDataProcessing"
export PYTHONPATH="${GRASP_DATA_PROCESSING_ROOT}/src:${PYTHONPATH}"
log_with_timestamp "设置 PYTHONPATH: $PYTHONPATH"
###########################################
log_with_timestamp "获取计算目录..."
cal_dir=${PWD}
csfs_ml_choosing_config_load.py set root_path ${cal_dir} 2>&1
log_with_timestamp "计算目录: $cal_dir"
###########################################
log_with_timestamp "复制python文件到计算目录..."
cp ${GRASP_DATA_PROCESSING_ROOT}/tests/ml_csf_choosing/*.py ${cal_dir}
log_with_timestamp "✅ 复制python文件到计算目录完成"
###########################################

# 主计算循环
while true
do
###########################################
log_with_timestamp "获取循环计数..."
loop=$(csfs_ml_choosing_config_load.py get cal_loop_num 2>&1)
log_with_timestamp "当前循环: $loop"

if [ $loop -eq 1 ]; then
    # 初始化必要csfs文件数据
    log_with_timestamp "================初始化必要csfs文件数据================"
    log_program_memory "initial_csfs.py" "start"
    python initial_csfs.py 
    log_program_memory "initial_csfs.py" "end"
fi
###########################################
log_with_timestamp "检查计算状态..."
cal_status=$(csfs_ml_choosing_config_load.py get continue_cal 2>&1)
log_with_timestamp "计算状态: $cal_status"

if [[ "$cal_status" == "false" ]]; then
    log_with_timestamp "================计算终止================"
    break
fi
###########################################
## 组态选择处理
log_with_timestamp "================执行组态选择================"
log_program_memory "choosing_csfs.py" "start"
python choosing_csfs.py 2>&1
if [ $? -ne 0 ]; then
    log_with_timestamp "❌ 组态选择失败!"
    exit 1
fi
log_program_memory "choosing_csfs.py" "end"
log_with_timestamp "✅ 组态选择完成"
###########################################
## grasp calculation routine

log_with_timestamp "进入计算目录: ${conf}_${loop}"
cd ${conf}_${loop}

safe_grasp_execute "mkdisks" "" mkdisks ${processor} caltmp

### rcsf
log_with_timestamp "准备 rcsf 输入文件..."
cp ${conf}_${loop}.c rcsf.inp # rmcdhf
cp ../isodata .

if [ $loop -le 3 ]; then
log_with_timestamp "================第一次循环，使用${loop1_rwfn_file}================"
cp ../${loop1_rwfn_file} ${conf}.w
orbital_params=${Active_space}

### rnucleus - 设置原子核参数
log_with_timestamp "设置原子核参数..."
input_commands="64
157
n
157
1
1
1"
safe_grasp_execute "rnucleus" "$input_commands" rnucleus
log_with_timestamp "✅ 原子核参数设置完成"

### rangular
safe_grasp_execute "rangular_mpi" "y" mpirun -np ${processor} rangular_mpi

### rwfnestimate
input_commands="y
1
${conf}.w
*
2
*
3
*"
safe_grasp_execute "rwfnestimate" "$input_commands" rwfnestimate

### rmcdhf
input_commands="y
${cal_levels}
5
${orbital_params}

100"
safe_grasp_execute "rmcdhf_mem_mpi" "$input_commands" mpirun -np ${processor} rmcdhf_mem_mpi

### rsave
safe_grasp_execute "rsave" "" rsave ${conf}_${loop}

cp ${conf}_${loop}.w ..

### jj2lsj rmcdhf
input_commands="${conf}_${loop}
n
y
y"
safe_grasp_execute "jj2lsj_rmcdhf" "$input_commands" jj2lsj

# 生成能级数据文件
safe_grasp_execute "rlevels_rmcdhf" "" bash -c "rlevels ${conf}_${loop}.m > ${conf}_${loop}.level"

else
log_with_timestamp "================第${loop}次循环，使用${rwfnestimate_file}================"
cp ../${rwfnestimate_file} ${conf}_${loop}.w

# rci
input_commands="y
${conf}_${loop}
y
y
1.d-6
y
n
n
y
5
${cal_levels}"
safe_grasp_execute "rci_mpi" "$input_commands" mpirun -np ${processor} rci_mpi

### jj2lsj rci
input_commands="${conf}_${loop}
y
y
y"
safe_grasp_execute "jj2lsj_rci" "$input_commands" jj2lsj

# 生成能级数据文件
safe_grasp_execute "rlevels_rci" "" bash -c "rlevels ${conf}_${loop}.cm > ${conf}_${loop}.level"

fi

# 清理临时文件夹
if [ -d "mpi_tmp" ]; then
    log_with_timestamp "发现临时文件夹 mpi_tmp，正在清理..."
    rm -rf mpi_tmp
    if [ $? -eq 0 ]; then
        log_with_timestamp "✅ 临时文件夹 mpi_tmp 清理完成"
    else
        log_with_timestamp "⚠️ 临时文件夹 mpi_tmp 清理失败"
    fi
else
    log_with_timestamp "未发现临时文件夹 mpi_tmp"
fi

log_with_timestamp "返回上级目录..."
cd ..

## 机器学习训练
log_with_timestamp "================执行机器学习训练================"
log_program_memory "train.py" "start"
python train.py 2>&1
if [ $? -ne 0 ]; then
    log_with_timestamp "❌ 机器学习训练失败!"
    exit 1
fi
log_program_memory "train.py" "end"
log_with_timestamp "✅ 机器学习训练完成"

log_with_timestamp "循环 $loop 完成，准备下一次迭代..."
done

log_with_timestamp "========== 带内存监控的 GRASP 脚本执行完成 ==========" 