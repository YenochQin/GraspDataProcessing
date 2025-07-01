#!/bin/zsh
#SBATCH -J GdIoddImlcias4_odd2
#SBATCH -N 1
#SBATCH --ntasks-per-node=46
#SBATCH -p work3
#SBATCH --output=%j_%x.log
#SBATCH --error=%j_%x.log
. /usr/share/Modules/init/zsh

# 添加时间戳函数
log_with_timestamp() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
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
        for file in $expected_files; do
            if [ ! -f "$file" ]; then
                log_with_timestamp "❌ $program_name 未生成预期文件: $file"
                return 1
            fi
            # 检查文件是否为空
            if [ ! -s "$file" ]; then
                log_with_timestamp "❌ $program_name 生成的文件为空: $file"
                return 1
            fi
        done
    fi
    
    return 0
}

# 安全执行GRASP程序的函数
safe_grasp_execute() {
    local program_name="$1"
    local expected_files="$2"
    local input_commands="$3"
    shift 3
    
    log_with_timestamp "执行 $program_name..."
    
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

log_with_timestamp "========== 开始执行 sbatch 脚本 =========="
log_with_timestamp "作业名: ${SLURM_JOB_NAME:-未设置}"
log_with_timestamp "作业编号: ${SLURM_JOB_ID:-未设置}"

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
# conda init zsh  # 如果未初始化过 Conda for zsh，运行一次（本地终端）
conda activate grasp-env || {
    log_with_timestamp "❌ 激活环境失败！请确认环境名是否正确。"
    exit 1
}
log_with_timestamp "✅ Conda 环境激活成功"
###########################################
## configuration
atom=GdI
conf="cv4odd1as4_odd2"
loop1_rwfn_file="mJ-1-90chosenas3_odd2.w"
rwfnestimate_file=${conf}_1.w
Active_space="10s,9p,8d,7f,6g"
cal_levels="1-4"
log_with_timestamp "配置参数: atom=$atom, conf=$conf, processor=$processor"
###########################################
# 检查 Python 路径
log_with_timestamp "检查 Python 环境..."
which python
python --version
###########################################
while true
do
###########################################
log_with_timestamp "获取循环计数..."
loop=$(csfs_ml_choosing_config_load.py get cal_loop_num 2>&1)
log_with_timestamp "当前循环: $loop"

if [ $loop -eq 1 ]; then
    # 初始化必要csfs文件数据
    log_with_timestamp "================初始化必要csfs文件数据================"
    python initial_csfs.py 
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
python choosing_csfs.py 2>&1
if [ $? -ne 0 ]; then
    log_with_timestamp "❌ 组态选择失败!"
    exit 1
fi
log_with_timestamp "✅ 组态选择完成"
###########################################
## grasp calculation routine

log_with_timestamp "进入计算目录: ${conf}_${loop}"
cd ${conf}_${loop}

safe_grasp_execute "mkdisks" "" "" mkdisks ${processor} caltmp

### rcsf
log_with_timestamp "准备 rcsf 输入文件..."
cp ${conf}_${loop}.c rcsf.inp # rmcdhf
cp ../isodata .

if [ $loop -eq 1 ]; then
log_with_timestamp "================第一次循环，使用${loop1_rwfn_file}================"
cp ../${loop1_rwfn_file} ${conf}.w
orbital_params=${Active_space}

### rangular
safe_grasp_execute "rangular_mpi" "" "y" mpirun -np ${processor} rangular_mpi

### rwfnestimate
input_commands="y
1
${conf}.w
*
2
*
3
*"
safe_grasp_execute "rwfnestimate" "rwfn.inp" "$input_commands" rwfnestimate

### rmcdhf
input_commands="y
${cal_levels}
5
${orbital_params}

100"
safe_grasp_execute "rmcdhf_mem_mpi" "rwfn.out rmix.out rmcdhf.sum rmcdhf.log" "$input_commands" mpirun -np ${processor} rmcdhf_mem_mpi

### rsave
safe_grasp_execute "rsave" "${conf}_${loop}.w ${conf}_${loop}.m" "" rsave ${conf}_${loop}

cp ${conf}_${loop}.w ..

### jj2lsj rmcdhf
input_commands="${conf}_${loop}
n
y
y"
safe_grasp_execute "jj2lsj_rmcdhf" "${conf}_${loop}.lsj.lbl" "$input_commands" jj2lsj

# 生成能级数据文件
safe_grasp_execute "rlevels_rmcdhf" "${conf}_${loop}.level" "" bash -c "rlevels ${conf}_${loop}.m > ${conf}_${loop}.level"

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
safe_grasp_execute "rci_mpi" "${conf}_${loop}.cm" "$input_commands" mpirun -np ${processor} rci_mpi

### jj2lsj rci
input_commands="${conf}_${loop}
y
y
y"
safe_grasp_execute "jj2lsj_rci" "${conf}_${loop}.lsj.lbl" "$input_commands" jj2lsj

# 生成能级数据文件
safe_grasp_execute "rlevels_rci" "${conf}_${loop}.level" "" bash -c "rlevels ${conf}_${loop}.cm > ${conf}_${loop}.level"

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
python train.py 2>&1
if [ $? -ne 0 ]; then
    log_with_timestamp "❌ 机器学习训练失败!"
    exit 1
fi
log_with_timestamp "✅ 机器学习训练完成"

log_with_timestamp "循环 $loop 完成，准备下一次迭代..."
done

log_with_timestamp "========== sbatch 脚本执行完成 =========="