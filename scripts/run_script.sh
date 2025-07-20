#!/bin/zsh
#SBATCH -J Gd_I_ml_cv6odd1_j3as5
#SBATCH -N 1
#SBATCH --ntasks-per-node=46
#SBATCH -p work3
#SBATCH --output=%j_%x.log
#SBATCH --error=%j_%x.log
. /usr/share/Modules/init/zsh

# Set proper locale and encoding for Chinese characters
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

# 设置 GraspDataProcessing 根目录路径（便于脚本移动到其他目录使用）
GRASP_DATA_PROCESSING_ROOT="/home/workstation3/AppFiles/GraspDataProcessing"
export PYTHONPATH="${GRASP_DATA_PROCESSING_ROOT}/src:${PYTHONPATH}"
export PATH="${GRASP_DATA_PROCESSING_ROOT}/scripts:${PATH}"

# 加载公共函数库（使用绝对路径，消除重复代码）
source "${GRASP_DATA_PROCESSING_ROOT}/scripts/common_functions.sh"


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

# 输出环境信息（使用正确的Python版本）
print_environment_info

###########################################
# GraspDataProcessing 包路径和工具脚本路径已在脚本开头设置
log_with_timestamp "设置 PYTHONPATH: $PYTHONPATH"
log_with_timestamp "设置 PATH: $PATH"
###########################################
# 检查 Python 路径和配置工具
log_with_timestamp "检查 Python 环境..."
which python
python --version
which csfs_ml_choosing_config_load.py
###########################################
## 从config.toml读取配置参数（在计算根目录中）
log_with_timestamp "从config.toml读取配置参数..."
cal_dir=${PWD}
config_file="${cal_dir}/config.toml"

# 检查配置文件是否存在
if [ ! -f "$config_file" ]; then
    log_error_with_timestamp "配置文件不存在: $config_file"
    exit 1
fi

atom=$(safe_get_config_value "${config_file}" "atom" "原子符号")
conf=$(safe_get_config_value "${config_file}" "conf" "组态名称")
processor=$(safe_get_config_value "${config_file}" "tasks_per_node" "处理器核数")
Active_space=$(safe_get_config_value "${config_file}" "active_space" "活性空间")
cal_levels=$(safe_get_config_value "${config_file}" "cal_levels" "计算能级")
selected_csfs_file=$(safe_get_config_value "${config_file}" "selected_csfs_file" "初筛CSFs文件")

# 读取mpi_tmp_path配置参数（在进入子目录之前读取）
mpi_tmp_path=$(safe_get_config_value "${config_file}" "mpi_tmp_path" "MPI临时路径")
log_with_timestamp "MPI临时文件路径配置: $mpi_tmp_path"

# 生成派生文件名
loop1_rwfn_file=$(basename "$selected_csfs_file" .c).w
rwfnestimate_file="${conf}_1.w"

log_config_params "$atom" "$conf" "$processor" "$Active_space" "$cal_levels"
log_with_timestamp "初始波函数文件: $loop1_rwfn_file"
# 更新配置文件中的root_path
run_python_with_env "${GRASP_DATA_PROCESSING_ROOT}/scripts/csfs_ml_choosing_config_load.py" set root_path ${cal_dir} -f "${config_file}"
log_with_timestamp_and_path "计算目录" "$cal_dir"
###########################################
log_with_timestamp "设置Python程序绝对路径..."
ML_PYTHON_DIR="${GRASP_DATA_PROCESSING_ROOT}/tests/ml_csf_choosing"
log_with_timestamp "✅ Python程序路径设置完成: $ML_PYTHON_DIR"

###########################################
# 读取步骤控制参数
log_with_timestamp "读取步骤控制配置..."
enable_step_control=$(safe_get_config_value "${config_file}" "step_control.enable_step_control" "启用步骤控制")
target_loop=$(safe_get_config_value "${config_file}" "step_control.target_loop" "目标循环")
start_step=$(safe_get_config_value "${config_file}" "step_control.start_step" "起始步骤")
end_step=$(safe_get_config_value "${config_file}" "step_control.end_step" "结束步骤")
skip_completed_steps=$(safe_get_config_value "${config_file}" "step_control.skip_completed_steps" "跳过已完成步骤")

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
atomic_number=$(safe_get_config_value "${config_file}" "atomic_number" "原子序数")
mass_number=$(safe_get_config_value "${config_file}" "mass_number" "质量数")
atomic_mass=$(safe_get_config_value "${config_file}" "atomic_mass" "原子质量")
nuclear_spin=$(safe_get_config_value "${config_file}" "nuclear_spin" "核自旋量子数")
nuclear_dipole=$(safe_get_config_value "${config_file}" "nuclear_dipole" "核偶极矩")
nuclear_quadrupole=$(safe_get_config_value "${config_file}" "nuclear_quadrupole" "核四极矩")

# 验证数值的有效性
local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
echo -e "[$timestamp] 原子核参数: $(highlight_param "Z" "$atomic_number") $(highlight_param "A" "$mass_number") $(highlight_param "质量" "$atomic_mass")"
echo -e "[$timestamp] 核性质: $(highlight_param "I" "$nuclear_spin") $(highlight_param "μ" "$nuclear_dipole") $(highlight_param "Q" "$nuclear_quadrupole")"

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
# 断点重启完成后自动重置步骤控制设置（避免无限循环）
reset_step_control_if_needed() {
    # 检查是否启用了步骤控制且当前不是默认设置
    if [[ "$enable_step_control" == "true" && "$start_step" != "auto" ]]; then
        log_with_timestamp "🔄 检测到断点重启模式，检查是否需要重置步骤控制..."
        
        # 如果已经完成了指定的断点重启步骤，重置为正常模式
        local should_reset=false
        
        # 检查各种重置条件
        if [[ "$end_step" != "auto" ]]; then
            log_with_timestamp "⚠️ 检测到指定结束步骤($end_step)，完成后将重置步骤控制"
            should_reset=true
        elif [[ "$start_step" == "train" ]]; then
            log_with_timestamp "⚠️ 检测到从train步骤开始，完成train后将重置步骤控制"
            should_reset=true
        fi
        
        if [[ "$should_reset" == "true" ]]; then
            # 设置一个标记，表示需要在适当时机重置
            export SHOULD_RESET_STEP_CONTROL="true"
            log_with_timestamp "📋 已标记：将在完成当前断点重启后自动重置步骤控制"
        fi
    fi
}

# 执行步骤控制重置
do_step_control_reset() {
    if [[ "$SHOULD_RESET_STEP_CONTROL" == "true" ]]; then
        log_with_timestamp "🔄 断点重启完成，重置步骤控制设置为正常模式..."
        
        # 重置步骤控制设置
        run_python_with_env "${GRASP_DATA_PROCESSING_ROOT}/scripts/csfs_ml_choosing_config_load.py" set step_control.start_step "auto" -f "${config_file}"
        run_python_with_env "${GRASP_DATA_PROCESSING_ROOT}/scripts/csfs_ml_choosing_config_load.py" set step_control.end_step "auto" -f "${config_file}"
        run_python_with_env "${GRASP_DATA_PROCESSING_ROOT}/scripts/csfs_ml_choosing_config_load.py" set step_control.enable_step_control "false" -f "${config_file}"
        
        # 更新本地变量
        start_step="auto"
        end_step="auto"
        enable_step_control="false"
        
        # 清除重置标记
        export SHOULD_RESET_STEP_CONTROL=""
        
        log_with_timestamp "✅ 步骤控制已重置，后续循环将正常执行所有步骤"
    fi
}

# 在主循环开始前检查是否需要重置
reset_step_control_if_needed

while true
do
###########################################
log_with_timestamp "获取循环计数..."
loop=$(safe_get_config_value "${config_file}" "cal_loop_num" "循环计数")
log_with_timestamp "当前循环: $(highlight_number "$loop" "$COLOR_CYAN")"

if [ $loop -eq 1 ]; then
    # 初始化必要csfs文件数据
    if check_step_should_run "initial_csfs" "$loop"; then
        log_stage "初始化必要csfs文件数据" "START"
        run_python_with_env "${ML_PYTHON_DIR}/initial_csfs.py"
        
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
cal_status=$(safe_get_config_value "${config_file}" "continue_cal" "计算继续状态")
log_with_timestamp "计算状态: $cal_status"

if [[ "$cal_status" == "false" ]]; then
    log_with_timestamp "================计算终止================"
    break
fi
###########################################
## 组态选择处理
if check_step_should_run "choosing_csfs" "$loop"; then
    log_stage "执行组态选择" "START"
    run_python_with_env "${ML_PYTHON_DIR}/choosing_csfs.py"
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

log_with_timestamp_and_path "进入计算目录" "${conf}_${loop}"
cd ${conf}_${loop}

# mkdisks步骤
if check_step_should_run "mkdisks" "$loop"; then
    if ! check_step_completed "mkdisks" "$loop" "$conf"; then
        # 使用循环外已读取的mpi_tmp_path配置参数
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

cp ../isodata .

if [ $loop -eq 1 ]; then
log_with_timestamp "================第一次循环，使用${loop1_rwfn_file}================"
log_with_timestamp "准备 rcsf 输入文件..."
cp ${conf}_${loop}.c rcsf.inp # rmcdhf
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
        safe_grasp_execute "jj2lsj" "$input_commands" jj2lsj
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
        safe_grasp_execute "rlevels" "${conf}_${loop}" bash -c "rlevels ${conf}_${loop}.m > ${conf}_${loop}.level"
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
        safe_grasp_execute "jj2lsj" "$input_commands" jj2lsj
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
        safe_grasp_execute "rlevels" "${conf}_${loop}" bash -c "rlevels ${conf}_${loop}.cm > ${conf}_${loop}.level"
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
run_python_with_env "${GRASP_DATA_PROCESSING_ROOT}/scripts/csfs_ml_choosing_config_load.py" set cal_method ${cal_method} -f "${config_file}"
## 机器学习训练
if check_step_should_run "train" "$loop"; then
    log_stage "执行机器学习训练" "START"
    
    # 直接执行，让输出实时显示（run_python_with_env已包含错误处理）
    run_python_with_env "${ML_PYTHON_DIR}/train.py"
    
    log_with_timestamp "✅ 机器学习训练完成"
    
    # 检查是否应该在此步骤后停止
    if check_should_stop_after_step "train"; then
        log_with_timestamp "🛑 根据配置在train步骤后停止执行"
        exit 0
    fi
    
    # 如果完成了train步骤，检查是否需要重置步骤控制
    do_step_control_reset
else
    log_with_timestamp "⏭️ 跳过步骤: train (根据步骤控制配置)"
fi

log_with_timestamp "循环 $loop 完成，准备下一次迭代..."
done

log_with_timestamp "========== sbatch 脚本执行完成 =========="