#!/bin/zsh
#SBATCH -J Gd_I_ml_cv6odd1_j3as5
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

log_with_timestamp "========== 开始执行 sbatch 脚本 =========="
log_with_timestamp "作业名: ${SLURM_JOB_NAME:-未设置}"
log_with_timestamp "作业编号: ${SLURM_JOB_ID:-未设置}"

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
# 自动设置 GraspDataProcessing 包路径和工具脚本路径
GRASP_DATA_PROCESSING_ROOT="/home/workstation3/AppFiles/GraspDataProcessing"
export PYTHONPATH="${GRASP_DATA_PROCESSING_ROOT}/src:${PYTHONPATH}"
export PATH="${GRASP_DATA_PROCESSING_ROOT}/scripts:${PATH}"
log_with_timestamp "设置 PYTHONPATH: $PYTHONPATH"
log_with_timestamp "设置 PATH: $PATH"
###########################################
# 检查 Python 路径和配置工具
log_with_timestamp "检查 Python 环境..."
which python
python --version
which csfs_ml_choosing_config_load.py
###########################################
## 从config.toml读取配置参数
log_with_timestamp "从config.toml读取配置参数..."
atom=$(python "${GRASP_DATA_PROCESSING_ROOT}/scripts/csfs_ml_choosing_config_load.py" get atom 2>&1)
conf=$(python "${GRASP_DATA_PROCESSING_ROOT}/scripts/csfs_ml_choosing_config_load.py" get conf 2>&1)
processor=$(python "${GRASP_DATA_PROCESSING_ROOT}/scripts/csfs_ml_choosing_config_load.py" get tasks_per_node 2>&1)
Active_space=$(python "${GRASP_DATA_PROCESSING_ROOT}/scripts/csfs_ml_choosing_config_load.py" get active_space 2>&1)
cal_levels=$(python "${GRASP_DATA_PROCESSING_ROOT}/scripts/csfs_ml_choosing_config_load.py" get cal_levels 2>&1)
selected_csfs_file=$(python "${GRASP_DATA_PROCESSING_ROOT}/scripts/csfs_ml_choosing_config_load.py" get selected_csfs_file 2>&1)

# 生成派生文件名
loop1_rwfn_file=$(basename "$selected_csfs_file" .c).w
rwfnestimate_file="${conf}_1.w"

log_with_timestamp "配置参数: atom=$atom, conf=$conf, processor=$processor"
log_with_timestamp "活性空间: $Active_space, 计算能级: $cal_levels"
log_with_timestamp "初始波函数文件: $loop1_rwfn_file"
###########################################
log_with_timestamp "获取计算目录..."
cal_dir=${PWD}
python "${GRASP_DATA_PROCESSING_ROOT}/scripts/csfs_ml_choosing_config_load.py" set root_path ${cal_dir} 2>&1
log_with_timestamp "计算目录: $cal_dir"
###########################################
log_with_timestamp "设置Python程序绝对路径..."
ML_PYTHON_DIR="${GRASP_DATA_PROCESSING_ROOT}/tests/ml_csf_choosing"
log_with_timestamp "✅ Python程序路径设置完成: $ML_PYTHON_DIR"

###########################################
# 读取步骤控制参数
log_with_timestamp "读取步骤控制配置..."
enable_step_control=$(python "${GRASP_DATA_PROCESSING_ROOT}/scripts/csfs_ml_choosing_config_load.py" get step_control.enable_step_control 2>&1)
target_loop=$(python "${GRASP_DATA_PROCESSING_ROOT}/scripts/csfs_ml_choosing_config_load.py" get step_control.target_loop 2>&1)
start_step=$(python "${GRASP_DATA_PROCESSING_ROOT}/scripts/csfs_ml_choosing_config_load.py" get step_control.start_step 2>&1)
end_step=$(python "${GRASP_DATA_PROCESSING_ROOT}/scripts/csfs_ml_choosing_config_load.py" get step_control.end_step 2>&1)
skip_completed_steps=$(python "${GRASP_DATA_PROCESSING_ROOT}/scripts/csfs_ml_choosing_config_load.py" get step_control.skip_completed_steps 2>&1)

log_with_timestamp "步骤控制配置:"
log_with_timestamp "  启用步骤控制: $enable_step_control"
log_with_timestamp "  目标循环: $target_loop"
log_with_timestamp "  起始步骤: $start_step"
log_with_timestamp "  结束步骤: $end_step"
log_with_timestamp "  跳过已完成步骤: $skip_completed_steps"

# 步骤检查函数
check_step_should_run() {
    local current_step="$1"
    local current_loop="$2"
    
    # 如果未启用步骤控制，总是执行
    if [[ "$enable_step_control" != "true" ]]; then
        return 0
    fi
    
    # 检查目标循环
    if [[ "$target_loop" != "0" && "$current_loop" != "$target_loop" ]]; then
        return 1  # 跳过不是目标循环的步骤
    fi
    
    # 检查起始步骤
    if [[ "$start_step" != "auto" ]]; then
        case "$current_step" in
            "initial_csfs")
                if [[ "$start_step" != "initial_csfs" ]]; then return 1; fi
                ;;
            "choosing_csfs")
                case "$start_step" in
                    "initial_csfs"|"choosing_csfs") ;;
                    *) return 1 ;;
                esac
                ;;
            "mkdisks")
                case "$start_step" in
                    "initial_csfs"|"choosing_csfs"|"mkdisks") ;;
                    *) return 1 ;;
                esac
                ;;
            "rangular")
                case "$start_step" in
                    "initial_csfs"|"choosing_csfs"|"mkdisks"|"rangular") ;;
                    *) return 1 ;;
                esac
                ;;
            "rwfnestimate")
                case "$start_step" in
                    "initial_csfs"|"choosing_csfs"|"mkdisks"|"rangular"|"rwfnestimate") ;;
                    *) return 1 ;;
                esac
                ;;
            "rmcdhf"|"rci")
                case "$start_step" in
                    "initial_csfs"|"choosing_csfs"|"mkdisks"|"rangular"|"rwfnestimate"|"rmcdhf"|"rci") ;;
                    *) return 1 ;;
                esac
                ;;
            "rsave")
                case "$start_step" in
                    "initial_csfs"|"choosing_csfs"|"mkdisks"|"rangular"|"rwfnestimate"|"rmcdhf"|"rci"|"rsave") ;;
                    *) return 1 ;;
                esac
                ;;
            "jj2lsj")
                case "$start_step" in
                    "initial_csfs"|"choosing_csfs"|"mkdisks"|"rangular"|"rwfnestimate"|"rmcdhf"|"rci"|"rsave"|"jj2lsj") ;;
                    *) return 1 ;;
                esac
                ;;
            "rlevels")
                case "$start_step" in
                    "initial_csfs"|"choosing_csfs"|"mkdisks"|"rangular"|"rwfnestimate"|"rmcdhf"|"rci"|"rsave"|"jj2lsj"|"rlevels") ;;
                    *) return 1 ;;
                esac
                ;;
            "train")
                # train总是可以执行（除非明确跳过）
                ;;
        esac
    fi
    
    # 检查结束步骤
    if [[ "$end_step" != "auto" ]]; then
        case "$current_step" in
            "initial_csfs")
                if [[ "$end_step" == "initial_csfs" ]]; then
                    log_with_timestamp "🎯 达到结束步骤: $end_step，将在此步骤后停止"
                fi
                ;;
            "choosing_csfs")
                if [[ "$end_step" == "choosing_csfs" ]]; then
                    log_with_timestamp "🎯 达到结束步骤: $end_step，将在此步骤后停止"
                fi
                ;;
            "mkdisks")
                if [[ "$end_step" == "mkdisks" ]]; then
                    log_with_timestamp "🎯 达到结束步骤: $end_step，将在此步骤后停止"
                fi
                ;;
            "rangular")
                if [[ "$end_step" == "rangular" ]]; then
                    log_with_timestamp "🎯 达到结束步骤: $end_step，将在此步骤后停止"
                fi
                ;;
            "rwfnestimate")
                if [[ "$end_step" == "rwfnestimate" ]]; then
                    log_with_timestamp "🎯 达到结束步骤: $end_step，将在此步骤后停止"
                fi
                ;;
            "rmcdhf"|"rci")
                if [[ "$end_step" == "rmcdhf" || "$end_step" == "rci" ]]; then
                    log_with_timestamp "🎯 达到结束步骤: $end_step，将在此步骤后停止"
                fi
                ;;
            "rsave")
                if [[ "$end_step" == "rsave" ]]; then
                    log_with_timestamp "🎯 达到结束步骤: $end_step，将在此步骤后停止"
                fi
                ;;
            "jj2lsj")
                if [[ "$end_step" == "jj2lsj" ]]; then
                    log_with_timestamp "🎯 达到结束步骤: $end_step，将在此步骤后停止"
                fi
                ;;
            "rlevels")
                if [[ "$end_step" == "rlevels" ]]; then
                    log_with_timestamp "🎯 达到结束步骤: $end_step，将在此步骤后停止"
                fi
                ;;
            "train")
                if [[ "$end_step" == "train" ]]; then
                    log_with_timestamp "🎯 达到结束步骤: $end_step，将在此步骤后停止"
                fi
                ;;
        esac
    fi
    
    return 0
}

# 检查步骤完成后是否应该停止
check_should_stop_after_step() {
    local current_step="$1"
    
    if [[ "$enable_step_control" != "true" ]]; then
        return 1  # 不停止
    fi
    
    if [[ "$end_step" != "auto" && "$current_step" == "$end_step" ]]; then
        return 0  # 应该停止
    fi
    
    return 1  # 不停止
}

# 文件存在性检查函数（用于跳过已完成步骤）
check_step_completed() {
    local step_name="$1"
    local loop_num="$2"
    local conf_name="$3"
    
    if [[ "$skip_completed_steps" != "true" ]]; then
        return 1  # 不跳过
    fi
    
    case "$step_name" in
        "mkdisks")
            if [[ -f "disks" ]]; then
                log_with_timestamp "⏭️ 跳过已完成的步骤: $step_name (发现文件: disks)"
                return 0
            fi
            ;;
        "rangular")
            # rangular没有直接输出文件，检查后续步骤的前提文件
            # 这里可以根据实际情况调整判断逻辑
            ;;
        "rwfnestimate")
            if [[ -f "rwfn.inp" ]]; then
                log_with_timestamp "⏭️ 跳过已完成的步骤: $step_name (发现文件: rwfn.inp)"
                return 0
            fi
            ;;
        "rmcdhf"|"rci")
            if [[ -f "rwfn.out" && -f "rmix.out" ]] || [[ -f "${conf_name}_${loop_num}.cm" ]]; then
                log_with_timestamp "⏭️ 跳过已完成的步骤: $step_name (发现输出文件)"
                return 0
            fi
            ;;
        "rsave")
            if [[ -f "${conf_name}_${loop_num}.w" && -f "${conf_name}_${loop_num}.c" && -f "${conf_name}_${loop_num}.m" ]]; then
                log_with_timestamp "⏭️ 跳过已完成的步骤: $step_name (发现保存文件)"
                return 0
            fi
            ;;
        "jj2lsj")
            if [[ -f "${conf_name}_${loop_num}.lsj.lbl" ]]; then
                log_with_timestamp "⏭️ 跳过已完成的步骤: $step_name (发现文件: ${conf_name}_${loop_num}.lsj.lbl)"
                return 0
            fi
            ;;
        "rlevels")
            if [[ -f "${conf_name}_${loop_num}.level" ]]; then
                log_with_timestamp "⏭️ 跳过已完成的步骤: $step_name (发现文件: ${conf_name}_${loop_num}.level)"
                return 0
            fi
            ;;
    esac
    
    return 1  # 未完成，不跳过
}
###########################################
### rnucleus - 设置原子核参数
log_with_timestamp "设置原子核参数..."
atomic_number=$(python "${GRASP_DATA_PROCESSING_ROOT}/scripts/csfs_ml_choosing_config_load.py" get atomic_number 2>&1)
mass_number=$(python "${GRASP_DATA_PROCESSING_ROOT}/scripts/csfs_ml_choosing_config_load.py" get mass_number 2>&1)
atomic_mass=$(python "${GRASP_DATA_PROCESSING_ROOT}/scripts/csfs_ml_choosing_config_load.py" get atomic_mass 2>&1)
nuclear_spin=$(python "${GRASP_DATA_PROCESSING_ROOT}/scripts/csfs_ml_choosing_config_load.py" get nuclear_spin 2>&1)
nuclear_dipole=$(python "${GRASP_DATA_PROCESSING_ROOT}/scripts/csfs_ml_choosing_config_load.py" get nuclear_dipole 2>&1)
nuclear_quadrupole=$(python "${GRASP_DATA_PROCESSING_ROOT}/scripts/csfs_ml_choosing_config_load.py" get nuclear_quadrupole 2>&1)

input_commands="$atomic_number
$mass_number
n
$atomic_mass
$nuclear_spin
$nuclear_dipole
$nuclear_quadrupole"
safe_grasp_execute "rnucleus" "$input_commands" rnucleus
log_with_timestamp "✅ 原子核参数设置完成"
###########################################
while true
do
###########################################
log_with_timestamp "获取循环计数..."
loop=$(python "${GRASP_DATA_PROCESSING_ROOT}/scripts/csfs_ml_choosing_config_load.py" get cal_loop_num 2>&1)
log_with_timestamp "当前循环: $loop"

if [ $loop -eq 1 ]; then
    # 初始化必要csfs文件数据
    if check_step_should_run "initial_csfs" "$loop"; then
        log_with_timestamp "================初始化必要csfs文件数据================"
        python "${ML_PYTHON_DIR}/initial_csfs.py"
        
        # 检查是否应该在此步骤后停止
        if check_should_stop_after_step "initial_csfs"; then
            log_with_timestamp "🛑 根据配置在initial_csfs步骤后停止执行"
            exit 0
        fi
    else
        log_with_timestamp "⏭️ 跳过步骤: initial_csfs (根据步骤控制配置)"
    fi
fi
###########################################
log_with_timestamp "检查计算状态..."
cal_status=$(python "${GRASP_DATA_PROCESSING_ROOT}/scripts/csfs_ml_choosing_config_load.py" get continue_cal 2>&1)
log_with_timestamp "计算状态: $cal_status"

if [[ "$cal_status" == "false" ]]; then
    log_with_timestamp "================计算终止================"
    break
fi
###########################################
## 组态选择处理
if check_step_should_run "choosing_csfs" "$loop"; then
    log_with_timestamp "================执行组态选择================"
    python "${ML_PYTHON_DIR}/choosing_csfs.py" 2>&1
    if [ $? -ne 0 ]; then
        log_with_timestamp "❌ 组态选择失败!"
        exit 1
    fi
    log_with_timestamp "✅ 组态选择完成"
    
    # 检查是否应该在此步骤后停止
    if check_should_stop_after_step "choosing_csfs"; then
        log_with_timestamp "🛑 根据配置在choosing_csfs步骤后停止执行"
        exit 0
    fi
else
    log_with_timestamp "⏭️ 跳过步骤: choosing_csfs (根据步骤控制配置)"
fi
###########################################
## grasp calculation routine

log_with_timestamp "进入计算目录: ${conf}_${loop}"
cd ${conf}_${loop}

# mkdisks步骤
if check_step_should_run "mkdisks" "$loop"; then
    if ! check_step_completed "mkdisks" "$loop" "$conf"; then
        # 读取mpi_tmp_path配置参数
        mpi_tmp_path=$(python "${GRASP_DATA_PROCESSING_ROOT}/scripts/csfs_ml_choosing_config_load.py" get mpi_tmp_path 2>&1)
        
        # 检查是否成功读取到参数
        if [[ -n "$mpi_tmp_path" && "$mpi_tmp_path" != "null" && ! "$mpi_tmp_path" =~ ^ERROR: ]]; then
            log_with_timestamp "使用配置的mpi_tmp路径: $mpi_tmp_path"
            safe_grasp_execute "mkdisks" "" mkdisks ${processor} "$mpi_tmp_path"
        else
            log_with_timestamp "未配置mpi_tmp_path或读取失败，使用默认路径（当前目录）"
            safe_grasp_execute "mkdisks" "" mkdisks ${processor}
        fi
    fi
    
    # 检查是否应该在此步骤后停止
    if check_should_stop_after_step "mkdisks"; then
        log_with_timestamp "🛑 根据配置在mkdisks步骤后停止执行"
        exit 0
    fi
else
    log_with_timestamp "⏭️ 跳过步骤: mkdisks (根据步骤控制配置)"
fi

### rcsf
log_with_timestamp "准备 rcsf 输入文件..."
cp ${conf}_${loop}.c rcsf.inp # rmcdhf
cp ../isodata .

if [ $loop -eq 1 ]; then
log_with_timestamp "================第一次循环，使用${loop1_rwfn_file}================"
cp ../${loop1_rwfn_file} ${conf}.w
orbital_params=${Active_space}
cal_method='rmcdhf'

### rangular
if check_step_should_run "rangular" "$loop"; then
    safe_grasp_execute "rangular_mpi" "y" mpirun -np ${processor} rangular_mpi
    
    # 检查是否应该在此步骤后停止
    if check_should_stop_after_step "rangular"; then
        log_with_timestamp "🛑 根据配置在rangular步骤后停止执行"
        exit 0
    fi
else
    log_with_timestamp "⏭️ 跳过步骤: rangular (根据步骤控制配置)"
fi

### rwfnestimate
if check_step_should_run "rwfnestimate" "$loop"; then
    if ! check_step_completed "rwfnestimate" "$loop" "$conf"; then
        input_commands="y
1
${conf}.w
*
2
*
3
*"
        safe_grasp_execute "rwfnestimate" "$input_commands" rwfnestimate
    fi
    
    # 检查是否应该在此步骤后停止
    if check_should_stop_after_step "rwfnestimate"; then
        log_with_timestamp "🛑 根据配置在rwfnestimate步骤后停止执行"
        exit 0
    fi
else
    log_with_timestamp "⏭️ 跳过步骤: rwfnestimate (根据步骤控制配置)"
fi

### rmcdhf
if check_step_should_run "rmcdhf" "$loop"; then
    if ! check_step_completed "rmcdhf" "$loop" "$conf"; then
        input_commands="y
${cal_levels}
5
${orbital_params}

100"
        safe_grasp_execute "rmcdhf_mem_mpi" "$input_commands" mpirun -np ${processor} rmcdhf_mem_mpi
    fi
    
    # 检查是否应该在此步骤后停止
    if check_should_stop_after_step "rmcdhf"; then
        log_with_timestamp "🛑 根据配置在rmcdhf步骤后停止执行"
        exit 0
    fi
else
    log_with_timestamp "⏭️ 跳过步骤: rmcdhf (根据步骤控制配置)"
fi

### rsave
if check_step_should_run "rsave" "$loop"; then
    if ! check_step_completed "rsave" "$loop" "$conf"; then
        safe_grasp_execute "rsave" "" rsave ${conf}_${loop}
        cp ${conf}_${loop}.w ..
    fi
    
    # 检查是否应该在此步骤后停止
    if check_should_stop_after_step "rsave"; then
        log_with_timestamp "🛑 根据配置在rsave步骤后停止执行"
        exit 0
    fi
else
    log_with_timestamp "⏭️ 跳过步骤: rsave (根据步骤控制配置)"
fi

### jj2lsj rmcdhf
if check_step_should_run "jj2lsj" "$loop"; then
    if ! check_step_completed "jj2lsj" "$loop" "$conf"; then
        input_commands="${conf}_${loop}
n
y
y"
        safe_grasp_execute "jj2lsj_rmcdhf" "$input_commands" jj2lsj
    fi
    
    # 检查是否应该在此步骤后停止
    if check_should_stop_after_step "jj2lsj"; then
        log_with_timestamp "🛑 根据配置在jj2lsj步骤后停止执行"
        exit 0
    fi
else
    log_with_timestamp "⏭️ 跳过步骤: jj2lsj (根据步骤控制配置)"
fi

# 生成能级数据文件
if check_step_should_run "rlevels" "$loop"; then
    if ! check_step_completed "rlevels" "$loop" "$conf"; then
        safe_grasp_execute "rlevels_rmcdhf" "" bash -c "rlevels ${conf}_${loop}.m > ${conf}_${loop}.level"
    fi
    
    # 检查是否应该在此步骤后停止
    if check_should_stop_after_step "rlevels"; then
        log_with_timestamp "🛑 根据配置在rlevels步骤后停止执行"
        exit 0
    fi
else
    log_with_timestamp "⏭️ 跳过步骤: rlevels (根据步骤控制配置)"
fi

else
log_with_timestamp "================第${loop}次循环，使用${rwfnestimate_file}================"
cp ../${rwfnestimate_file} ${conf}_${loop}.w
cal_method='rci'

# rci
if check_step_should_run "rci" "$loop"; then
    if ! check_step_completed "rci" "$loop" "$conf"; then
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
    fi
    
    # 检查是否应该在此步骤后停止
    if check_should_stop_after_step "rci"; then
        log_with_timestamp "🛑 根据配置在rci步骤后停止执行"
        exit 0
    fi
else
    log_with_timestamp "⏭️ 跳过步骤: rci (根据步骤控制配置)"
fi

### jj2lsj rci
if check_step_should_run "jj2lsj" "$loop"; then
    if ! check_step_completed "jj2lsj" "$loop" "$conf"; then
        input_commands="${conf}_${loop}
y
y
y"
        safe_grasp_execute "jj2lsj_rci" "$input_commands" jj2lsj
    fi
    
    # 检查是否应该在此步骤后停止
    if check_should_stop_after_step "jj2lsj"; then
        log_with_timestamp "🛑 根据配置在jj2lsj步骤后停止执行"
        exit 0
    fi
else
    log_with_timestamp "⏭️ 跳过步骤: jj2lsj (根据步骤控制配置)"
fi

# 生成能级数据文件
if check_step_should_run "rlevels" "$loop"; then
    if ! check_step_completed "rlevels" "$loop" "$conf"; then
        safe_grasp_execute "rlevels_rci" "" bash -c "rlevels ${conf}_${loop}.cm > ${conf}_${loop}.level"
    fi
    
    # 检查是否应该在此步骤后停止
    if check_should_stop_after_step "rlevels"; then
        log_with_timestamp "🛑 根据配置在rlevels步骤后停止执行"
        exit 0
    fi
else
    log_with_timestamp "⏭️ 跳过步骤: rlevels (根据步骤控制配置)"
fi

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
python "${GRASP_DATA_PROCESSING_ROOT}/scripts/csfs_ml_choosing_config_load.py" set cal_method ${cal_method} 2>&1
## 机器学习训练
if check_step_should_run "train" "$loop"; then
    log_with_timestamp "================执行机器学习训练================"
    
    # 捕获输出和退出码
    train_output=$(python "${ML_PYTHON_DIR}/train.py" 2>&1)
    train_exit_code=$?
    
    # 显示输出
    echo "$train_output"
    
    # 检查退出码
    if [ $train_exit_code -ne 0 ]; then
        log_with_timestamp "❌ 机器学习训练失败! 退出码: $train_exit_code"
        exit 1
    fi
    
    # 检查输出中是否包含错误信息
    if echo "$train_output" | grep -q "程序执行失败"; then
        log_with_timestamp "❌ 机器学习训练失败! 检测到错误消息"
        exit 1
    fi
    
    log_with_timestamp "✅ 机器学习训练完成"
    
    # 检查是否应该在此步骤后停止
    if check_should_stop_after_step "train"; then
        log_with_timestamp "🛑 根据配置在train步骤后停止执行"
        exit 0
    fi
else
    log_with_timestamp "⏭️ 跳过步骤: train (根据步骤控制配置)"
fi

log_with_timestamp "循环 $loop 完成，准备下一次迭代..."
done

log_with_timestamp "========== sbatch 脚本执行完成 =========="