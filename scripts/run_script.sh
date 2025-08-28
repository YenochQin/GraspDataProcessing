#!/bin/zsh
#SBATCH -J cv1even1j0_2
#SBATCH -N 1
#SBATCH --ntasks-per-node=46
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH -p batch
#SBATCH --output=%j_%x.log
#SBATCH --error=%j_%x.log
. /usr/share/Modules/init/zsh

# Set GraspDataProcessing root directory path for script portability
GRASP_DATA_PROCESSING_ROOT="/home/computer-0-2/AppFiles/GraspDataProcessing"
export PYTHONPATH="${GRASP_DATA_PROCESSING_ROOT}/src:${PYTHONPATH}"
export PATH="${GRASP_DATA_PROCESSING_ROOT}/scripts:${PATH}"

# Load common function library (absolute path to eliminate code duplication)
source "${GRASP_DATA_PROCESSING_ROOT}/scripts/common_functions.sh"


log_with_timestamp "========== Starting sbatch script execution ========="
log_with_timestamp "Job name: ${SLURM_JOB_NAME:-Not set}"
log_with_timestamp "Job ID: ${SLURM_JOB_ID:-Not set}"

###########################################
## module load
log_with_timestamp "Loading required modules..."
module load mpi/openmpi-x86_64
module load grasp/grasp_openblas
###########################################
# Critical fix: Ensure proper Conda loading (zsh requires manual initialization)
log_with_timestamp "Initializing Conda environment..."
source /home/computer-0-2/AppFiles/miniconda3/etc/profile.d/conda.sh  || {
    log_with_timestamp "ERROR: Failed to load Conda! Please check if path is correct."
    exit 1
}
conda activate grasp-env || {
    log_with_timestamp "ERROR: Failed to activate environment! Please confirm environment name is correct."
    exit 1
}
log_with_timestamp "SUCCESS: Conda environment activated successfully"

# Print environment information (using correct Python version)
print_environment_info

###########################################
# GraspDataProcessing package path and tool script paths set at script beginning
log_with_timestamp "Set PYTHONPATH: $PYTHONPATH"
log_with_timestamp "Set PATH: $PATH"
###########################################
# Check Python path and configuration tools
log_with_timestamp "Checking Python environment..."
which python
python --version
which csfs_ml_choosing_config_load.py
###########################################
## Read configuration parameters from config.toml (in calculation root directory)
log_with_timestamp "Reading configuration parameters from config.toml..."
cal_dir=${PWD}
config_file="${cal_dir}/config.toml"

# Check if configuration file exists
if [ ! -f "$config_file" ]; then
    log_error_with_timestamp "Configuration file does not exist: $config_file"
    exit 1
fi

atom=$(safe_get_config_value "${config_file}" "atom" "atomic symbol")
conf=$(safe_get_config_value "${config_file}" "conf" "configuration name")
processor=$(safe_get_config_value "${config_file}" "tasks_per_node" "processor count")
Active_space=$(safe_get_config_value "${config_file}" "active_space" "active space")
cal_levels=$(safe_get_config_value "${config_file}" "cal_levels" "calculation levels")
selected_csfs_file=$(safe_get_config_value "${config_file}" "selected_csfs_file" "initial CSFs file")

# Read mpi_tmp_path configuration parameter (before entering subdirectory)
mpi_tmp_path=$(safe_get_config_value "${config_file}" "mpi_tmp_path" "MPI temp path")
log_with_timestamp "MPI temporary file path configuration: $mpi_tmp_path"

# Generate derived filenames
loop1_rwfn_file=$(safe_get_config_value "${config_file}" "loop1_rwfn_file" "previous got RWFN file")
rwfnestimate_file="${conf}_1.w"

log_config_params "$atom" "$conf" "$processor" "$Active_space" "$cal_levels"
log_with_timestamp "Initial wavefunction file: $loop1_rwfn_file"
###########################################
# Update root_path in configuration file
run_python_with_env "${GRASP_DATA_PROCESSING_ROOT}/scripts/csfs_ml_choosing_config_load.py" set root_path ${cal_dir} -f "${config_file}"
log_with_timestamp_and_path "Calculation directory" "$cal_dir"
###########################################
log_with_timestamp "Setting absolute paths for Python programs..."
ML_PYTHON_DIR="${GRASP_DATA_PROCESSING_ROOT}/tests/ml_csf_choosing"
log_with_timestamp "SUCCESS: Python program path setup complete: $ML_PYTHON_DIR"

###########################################
# Read step control parameters
log_with_timestamp "Reading step control configuration..."
enable_step_control=$(safe_get_config_value "${config_file}" "step_control.enable_step_control" "Enable step control")
target_loop=$(safe_get_config_value "${config_file}" "step_control.target_loop" "Target loop")
start_step=$(safe_get_config_value "${config_file}" "step_control.start_step" "Start step")
end_step=$(safe_get_config_value "${config_file}" "step_control.end_step" "End step")
skip_completed_steps=$(safe_get_config_value "${config_file}" "step_control.skip_completed_steps" "Skip completed steps")

log_with_timestamp "Step control configuration:"
log_with_timestamp "  Enable step control: $enable_step_control"
log_with_timestamp "  Target loop: $target_loop"
log_with_timestamp "  Start step: $start_step"
log_with_timestamp "  End step: $end_step"
log_with_timestamp "  Skip completed steps: $skip_completed_steps"

# Step check function
check_step_should_run() {
    local current_step="$1"
    local current_loop="$2"
    
    # If step control is not enabled, always execute
    if [[ "$enable_step_control" != "true" ]]; then
        return 0
    fi
    
    # Check target loop
    if [[ "$target_loop" != "0" && "$current_loop" != "$target_loop" ]]; then
        return 1  # Skip steps not in target loop
    fi
    
    # Check start step
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
                # train can always be executed (unless explicitly skipped)
                ;;
        esac
    fi
    
    # Check end step
    if [[ "$end_step" != "auto" ]]; then
        case "$current_step" in
            "initial_csfs")
                if [[ "$end_step" == "initial_csfs" ]]; then
                    log_with_timestamp "[TARGET] Reached ending step: $end_step, will stop after this step"
                fi
                ;;
            "choosing_csfs")
                if [[ "$end_step" == "choosing_csfs" ]]; then
                    log_with_timestamp "[TARGET] Reached ending step: $end_step, will stop after this step"
                fi
                ;;
            "mkdisks")
                if [[ "$end_step" == "mkdisks" ]]; then
                    log_with_timestamp "[TARGET] Reached ending step: $end_step, will stop after this step"
                fi
                ;;
            "rangular")
                if [[ "$end_step" == "rangular" ]]; then
                    log_with_timestamp "[TARGET] Reached ending step: $end_step, will stop after this step"
                fi
                ;;
            "rwfnestimate")
                if [[ "$end_step" == "rwfnestimate" ]]; then
                    log_with_timestamp "[TARGET] Reached ending step: $end_step, will stop after this step"
                fi
                ;;
            "rmcdhf"|"rci")
                if [[ "$end_step" == "rmcdhf" || "$end_step" == "rci" ]]; then
                    log_with_timestamp "[TARGET] Reached ending step: $end_step, will stop after this step"
                fi
                ;;
            "rsave")
                if [[ "$end_step" == "rsave" ]]; then
                    log_with_timestamp "[TARGET] Reached ending step: $end_step, will stop after this step"
                fi
                ;;
            "jj2lsj")
                if [[ "$end_step" == "jj2lsj" ]]; then
                    log_with_timestamp "[TARGET] Reached ending step: $end_step, will stop after this step"
                fi
                ;;
            "rlevels")
                if [[ "$end_step" == "rlevels" ]]; then
                    log_with_timestamp "[TARGET] Reached ending step: $end_step, will stop after this step"
                fi
                ;;
            "train")
                if [[ "$end_step" == "train" ]]; then
                    log_with_timestamp "[TARGET] Reached ending step: $end_step, will stop after this step"
                fi
                ;;
        esac
    fi
    
    return 0
}

# Check if should stop after step completion
check_should_stop_after_step() {
    local current_step="$1"
    
    if [[ "$enable_step_control" != "true" ]]; then
        return 1  # Don't stop
    fi
    
    if [[ "$end_step" != "auto" && "$current_step" == "$end_step" ]]; then
        return 0  # Should stop
    fi
    
    return 1  # Don't stop
}

# File existence check function (for skipping completed steps)
check_step_completed() {
    local step_name="$1"
    local loop_num="$2"
    local conf_name="$3"
    
    if [[ "$skip_completed_steps" != "true" ]]; then
        return 1  # Don't skip
    fi
    
    case "$step_name" in
        "mkdisks")
            if [[ -f "disks" ]]; then
                log_with_timestamp "[SKIP] Skip completed step: $step_name (found file: disks)"
                return 0
            fi
            ;;
        "rangular")
            # rangular has no direct output files, check prerequisite files for subsequent steps
            # The judgment logic can be adjusted according to actual conditions
            ;;
        "rwfnestimate")
            if [[ -f "rwfn.inp" ]]; then
                log_with_timestamp "[SKIP] Skip completed step: $step_name (found file: rwfn.inp)"
                return 0
            fi
            ;;
        "rmcdhf"|"rci")
            if [[ -f "rwfn.out" && -f "rmix.out" ]] || [[ -f "${conf_name}_${loop_num}.cm" ]]; then
                log_with_timestamp "[SKIP] Skip completed step: $step_name (found output files)"
                return 0
            fi
            ;;
        "rsave")
            if [[ -f "${conf_name}_${loop_num}.w" && -f "${conf_name}_${loop_num}.c" && -f "${conf_name}_${loop_num}.m" ]]; then
                log_with_timestamp "[SKIP] Skip completed step: $step_name (found saved files)"
                return 0
            fi
            ;;
        "jj2lsj")
            if [[ -f "${conf_name}_${loop_num}.lsj.lbl" ]]; then
                log_with_timestamp "[SKIP] Skip completed step: $step_name (found file: ${conf_name}_${loop_num}.lsj.lbl)"
                return 0
            fi
            ;;
        "rlevels")
            if [[ -f "${conf_name}_${loop_num}.level" ]]; then
                log_with_timestamp "[SKIP] Skip completed step: $step_name (found file: ${conf_name}_${loop_num}.level)"
                return 0
            fi
            ;;
    esac
    
    return 1  # Not completed, do not skip
}
###########################################
### rnucleus - Setting nuclear parameters
log_with_timestamp "Setting nuclear parameters..."
atomic_number=$(safe_get_config_value "${config_file}" "atomic_number" "atomic number")
mass_number=$(safe_get_config_value "${config_file}" "mass_number" "mass number")
atomic_mass=$(safe_get_config_value "${config_file}" "atomic_mass" "atomic mass")
nuclear_spin=$(safe_get_config_value "${config_file}" "nuclear_spin" "nuclear spin quantum number")
nuclear_dipole=$(safe_get_config_value "${config_file}" "nuclear_dipole" "nuclear dipole moment")
nuclear_quadrupole=$(safe_get_config_value "${config_file}" "nuclear_quadrupole" "nuclear quadrupole moment")

# Verify the validity of numeric values
local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
echo -e "[$timestamp] Nuclear parameters: $(highlight_param "Z" "$atomic_number") $(highlight_param "A" "$mass_number") $(highlight_param "mass" "$atomic_mass")"
echo -e "[$timestamp] Nuclear properties: $(highlight_param "I" "$nuclear_spin") $(highlight_param "mu" "$nuclear_dipole") $(highlight_param "Q" "$nuclear_quadrupole")"

input_commands="$atomic_number
$mass_number
n
$atomic_mass
$nuclear_spin
$nuclear_dipole
$nuclear_quadrupole"
safe_grasp_execute "rnucleus" "$input_commands" rnucleus
log_with_timestamp "[SUCCESS] Nuclear parameter setup completed"
###########################################
# Automatically reset step control settings after breakpoint restart (to avoid infinite loops)
reset_step_control_if_needed() {
    # Check if step control is enabled and current settings are not default
    if [[ "$enable_step_control" == "true" && "$start_step" != "auto" ]]; then
        log_with_timestamp "[RESTART] Detected breakpoint restart mode, checking if step control reset is needed..."
        
        # If the specified breakpoint restart step has been completed, reset to normal mode
        local should_reset=false
        
        # Check various reset conditions
        if [[ "$end_step" != "auto" ]]; then
            log_with_timestamp "[WARN] Detected specified end step ($end_step), will reset step control after completion"
            should_reset=true
        elif [[ "$start_step" == "train" ]]; then
            log_with_timestamp "[WARN] Detected start from train step, will reset step control after train completion"
            should_reset=true
        fi
        
        if [[ "$should_reset" == "true" ]]; then
            # Set a flag indicating reset is needed at appropriate time
            export SHOULD_RESET_STEP_CONTROL="true"
            log_with_timestamp "[MARK] Marked: Will automatically reset step control after completing current breakpoint restart"
        fi
    fi
}

# Execute step control reset
do_step_control_reset() {
    if [[ "$SHOULD_RESET_STEP_CONTROL" == "true" ]]; then
        log_with_timestamp "[RESTART] Breakpoint restart completed, resetting step control settings to normal mode..."
        
        # Reset step control settings
        run_python_with_env "${GRASP_DATA_PROCESSING_ROOT}/scripts/csfs_ml_choosing_config_load.py" set step_control.start_step "auto" -f "${config_file}"
        run_python_with_env "${GRASP_DATA_PROCESSING_ROOT}/scripts/csfs_ml_choosing_config_load.py" set step_control.end_step "auto" -f "${config_file}"
        run_python_with_env "${GRASP_DATA_PROCESSING_ROOT}/scripts/csfs_ml_choosing_config_load.py" set step_control.enable_step_control "false" -f "${config_file}"
        
        # Update local variables
        start_step="auto"
        end_step="auto"
        enable_step_control="false"
        
        # Clear reset flag
        export SHOULD_RESET_STEP_CONTROL=""
        
        log_with_timestamp "[SUCCESS] Step control has been reset, subsequent loops will execute all steps normally"
    fi
}

# Check if reset is needed before main loop starts
reset_step_control_if_needed

while true
do
###########################################
log_with_timestamp "Getting loop count..."
loop=$(safe_get_config_value "${config_file}" "cal_loop_num" "loop count")
log_with_timestamp "Current loop: $(highlight_number "$loop" "$COLOR_CYAN")"

# Check for backward loop needed flag
backward_loop_needed=$(safe_get_config_value "${config_file}" "backward_loop_needed" "backward loop needed")
if [[ "$backward_loop_needed" == "true" ]]; then
    target_backward_loop=$(safe_get_config_value "${config_file}" "target_backward_loop" "target backward loop")
    cal_error_num=$(safe_get_config_value "${config_file}" "cal_error_num" "calculation error count")
    log_with_timestamp "🔄 检测到回退标志: 需要回退到第 $target_backward_loop 轮重新训练"
    
    # 重命名当前轮次目录为错误目录
    current_dir="${conf}_${loop}"
    error_dir="${conf}_${loop}_err_${cal_error_num}"
    
    if [[ -d "$current_dir" ]]; then
        log_with_timestamp "📁 重命名当前计算目录: $current_dir -> $error_dir"
        mv "$current_dir" "$error_dir"
        if [[ $? -eq 0 ]]; then
            log_with_timestamp "✅ 目录重命名成功"
        else
            log_with_timestamp "❌ 目录重命名失败"
            exit 1
        fi
    fi
    
    # 验证回退的合理性
    if [[ "$loop" == "$target_backward_loop" ]]; then
        log_with_timestamp "✅ 已回退到目标轮次 $target_backward_loop，开始重新执行train.py"
        
        # 清除回退标志
        run_python_with_env "${GRASP_DATA_PROCESSING_ROOT}/scripts/csfs_ml_choosing_config_load.py" set backward_loop_needed false -f "${config_file}"
        run_python_with_env "${GRASP_DATA_PROCESSING_ROOT}/scripts/csfs_ml_choosing_config_load.py" set target_backward_loop 0 -f "${config_file}"
        
        # 直接跳转到train.py执行
        log_with_timestamp "🚀 跳转到第 $loop 轮train.py重新训练..."
        goto_train_step=true
    else
        log_with_timestamp "❌ 错误: 当前轮次 $loop 与目标回退轮次 $target_backward_loop 不匹配"
        log_with_timestamp "这可能是配置错误，停止执行以防止无限循环"
        exit 1
    fi
else
    goto_train_step=false
fi

if [ $loop -eq 1 ]; then
    # Initialize necessary CSF file data
    if check_step_should_run "initial_csfs" "$loop"; then
        log_stage "Initialize necessary CSF file data" "START"
        run_python_with_env "${ML_PYTHON_DIR}/initial_csfs.py"
        
        # Check if should stop after this step
        if check_should_stop_after_step "initial_csfs"; then
            log_with_timestamp "[STOP] Stop execution after initial_csfs step according to configuration"
            exit 0
        fi
    else
        log_with_timestamp "[SKIP] Skip step: initial_csfs (according to step control configuration)"
    fi
fi
###########################################
log_with_timestamp "Checking calculation status..."
cal_status=$(safe_get_config_value "${config_file}" "continue_cal" "calculation continue status")
log_with_timestamp "Calculation status: $cal_status"

if [[ "$cal_status" == "false" ]]; then
    log_with_timestamp "================ Calculation terminated ================"
    break
fi
###########################################
## Configuration selection processing
if check_step_should_run "choosing_csfs" "$loop" && [[ "$goto_train_step" != "true" ]]; then
    log_stage "Execute configuration selection" "START"
    run_python_with_env "${ML_PYTHON_DIR}/choosing_csfs.py"
    log_with_timestamp "[SUCCESS] Configuration selection completed"
    
    # Check if should stop after this step
    if check_should_stop_after_step "choosing_csfs"; then
        log_with_timestamp "[STOP] Stop execution after choosing_csfs step according to configuration"
        exit 0
    fi
else
    if [[ "$goto_train_step" == "true" ]]; then
        log_with_timestamp "[SKIP] Skip step: choosing_csfs (backward loop mode - jumping to train)"
    else
        log_with_timestamp "[SKIP] Skip step: choosing_csfs (according to step control configuration)"
    fi
fi
###########################################
## grasp calculation routine

if [[ "$goto_train_step" != "true" ]]; then
    log_with_timestamp_and_path "Enter calculation directory" "${conf}_${loop}"
    cd ${conf}_${loop}

# mkdisks step
if check_step_should_run "mkdisks" "$loop"; then
    if ! check_step_completed "mkdisks" "$loop" "$conf"; then
        # Use mpi_tmp_path configuration parameter read outside loop
        if [[ -n "$mpi_tmp_path" && "$mpi_tmp_path" != "null" && ! "$mpi_tmp_path" =~ ^ERROR: ]]; then
            log_with_timestamp "Use configured mpi_tmp path: $mpi_tmp_path"
            safe_grasp_execute "mkdisks" "" mkdisks ${processor} "$mpi_tmp_path"
        else
            log_with_timestamp "mpi_tmp_path not configured or failed to read, using default path (current directory)"
            safe_grasp_execute "mkdisks" "" mkdisks ${processor}
        fi
    fi
    
    # Check if should stop after this step
    if check_should_stop_after_step "mkdisks"; then
        log_with_timestamp "[STOP] Stop execution after mkdisks step according to configuration"
        exit 0
    fi
else
    log_with_timestamp "[SKIP] Skip step: mkdisks (according to step control configuration)"
fi

### rcsf

cp ../isodata .
cp ${conf}_${loop}.c rcsf.inp
if [ $loop -eq 1 ]; then
log_with_timestamp "================第一次循环，使用${loop1_rwfn_file}================"
cp ../${loop1_rwfn_file} ${conf}.w
orbital_params=${Active_space}
cal_method='rmcdhf'

# rangular step
if check_step_should_run "rangular" "$loop"; then
    if ! check_step_completed "rangular" "$loop" "$conf"; then
        safe_grasp_execute "rangular_mpi" "y" srun --mpi=pmix --cpu-bind=hwthread --ntasks=${processor} rangular_mpi
    fi
    
    # Check if should stop after this step
    if check_should_stop_after_step "rangular"; then
        log_with_timestamp "[STOP] Stop execution after rangular step according to configuration"
        exit 0
    fi
else
    log_with_timestamp "[SKIP] Skip step: rangular (according to step control configuration)"
fi

# rwfnestimate step (loop 1)
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
    
    # Check if should stop after this step
    if check_should_stop_after_step "rwfnestimate"; then
        log_with_timestamp "[STOP] Stop execution after rwfnestimate step according to configuration"
        exit 0
    fi
else
    log_with_timestamp "[SKIP] Skip step: rwfnestimate (according to step control configuration)"
fi

# rmcdhf step
if check_step_should_run "rmcdhf" "$loop"; then
    if ! check_step_completed "rmcdhf" "$loop" "$conf"; then
        input_commands="y
${cal_levels}
5
${orbital_params}

100"
        safe_grasp_execute "rmcdhf_mem_mpi" "$input_commands" srun --mpi=pmix --cpu-bind=hwthread --ntasks=${processor} rmcdhf_mem_mpi
    fi
    
    # Check if should stop after this step
    if check_should_stop_after_step "rmcdhf"; then
        log_with_timestamp "[STOP] Stop execution after rmcdhf step according to configuration"
        exit 0
    fi
else
    log_with_timestamp "[SKIP] Skip step: rmcdhf (according to step control configuration)"
fi

# rsave step
if check_step_should_run "rsave" "$loop"; then
    if ! check_step_completed "rsave" "$loop" "$conf"; then
        safe_grasp_execute "rsave" "" rsave ${conf}_${loop}
    fi
    
    # Check if should stop after this step
    if check_should_stop_after_step "rsave"; then
        log_with_timestamp "[STOP] Stop execution after rsave step according to configuration"
        exit 0
    fi
else
    log_with_timestamp "[SKIP] Skip step: rsave (according to step control configuration)"
fi

# Copy wavefunction file if rsave step was executed
if check_step_should_run "rsave" "$loop"; then
    cp ${conf}_${loop}.w ..
fi

# jj2lsj step (loop 1)
if check_step_should_run "jj2lsj" "$loop"; then
    if ! check_step_completed "jj2lsj" "$loop" "$conf"; then
        input_commands="${conf}_${loop}
n
y
y"
        safe_grasp_execute "jj2lsj" "$input_commands" jj2lsj
    fi
    
    # Check if should stop after this step
    if check_should_stop_after_step "jj2lsj"; then
        log_with_timestamp "[STOP] Stop execution after jj2lsj step according to configuration"
        exit 0
    fi
else
    log_with_timestamp "[SKIP] Skip step: jj2lsj (according to step control configuration)"
fi

# rlevels step (loop 1)
if check_step_should_run "rlevels" "$loop"; then
    if ! check_step_completed "rlevels" "$loop" "$conf"; then
        safe_grasp_execute "rlevels" "${conf}_${loop}" bash -c "rlevels ${conf}_${loop}.m > ${conf}_${loop}.level"
    fi
    
    # Check if should stop after this step
    if check_should_stop_after_step "rlevels"; then
        log_with_timestamp "[STOP] Stop execution after rlevels step according to configuration"
        exit 0
    fi
else
    log_with_timestamp "[SKIP] Skip step: rlevels (according to step control configuration)"
fi

else
log_with_timestamp "================第${loop}次循环，使用${rwfnestimate_file}================"
cp ../${rwfnestimate_file} .

### rwfnestimate
if check_step_should_run "rwfnestimate" "$loop"; then
    if ! check_step_completed "rwfnestimate" "$loop" "$conf"; then
        input_commands="y
1
${rwfnestimate_file}
*
2
*
3
*"
        safe_grasp_execute "rwfnestimate" "$input_commands" rwfnestimate
    fi
    
    # Check if should stop after this step
    if check_should_stop_after_step "rwfnestimate"; then
        log_with_timestamp "[STOP] Stop execution after rwfnestimate step according to configuration"
        exit 0
    fi
else
    log_with_timestamp "[SKIP] Skip step: rwfnestimate (according to step control configuration)"
fi
rm rcsf.inp
cp rwfn.inp ${conf}_${loop}.w

cal_method='rci'

# rci step
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
        safe_grasp_execute "rci_mpi" "$input_commands" srun --mpi=pmix --cpu-bind=hwthread --ntasks=${processor} rci_mpi
    fi
    
    # Check if should stop after this step
    if check_should_stop_after_step "rci"; then
        log_with_timestamp "[STOP] Stop execution after rci step according to configuration"
        exit 0
    fi
else
    log_with_timestamp "[SKIP] Skip step: rci (according to step control configuration)"
fi

# jj2lsj step (non-loop-1)
if check_step_should_run "jj2lsj" "$loop"; then
    if ! check_step_completed "jj2lsj" "$loop" "$conf"; then
        input_commands="${conf}_${loop}
y
y
y"
        safe_grasp_execute "jj2lsj" "$input_commands" jj2lsj
    fi
    
    # Check if should stop after this step
    if check_should_stop_after_step "jj2lsj"; then
        log_with_timestamp "[STOP] Stop execution after jj2lsj step according to configuration"
        exit 0
    fi
else
    log_with_timestamp "[SKIP] Skip step: jj2lsj (according to step control configuration)"
fi

# rlevels step (non-loop-1)
if check_step_should_run "rlevels" "$loop"; then
    if ! check_step_completed "rlevels" "$loop" "$conf"; then
        safe_grasp_execute "rlevels" "${conf}_${loop}" bash -c "rlevels ${conf}_${loop}.cm > ${conf}_${loop}.level"
    fi
    
    # Check if should stop after this step
    if check_should_stop_after_step "rlevels"; then
        log_with_timestamp "[STOP] Stop execution after rlevels step according to configuration"
        exit 0
    fi
else
    log_with_timestamp "[SKIP] Skip step: rlevels (according to step control configuration)"
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
else
    log_with_timestamp "🔄 跳过所有GRASP计算步骤 (backward loop mode - jumping to train)"
fi
## 机器学习训练
if check_step_should_run "train" "$loop" || [[ "$goto_train_step" == "true" ]]; then
    log_stage "执行机器学习训练" "START"
    
    if [[ "$goto_train_step" == "true" ]]; then
        log_with_timestamp "🔄 执行回退模式训练 - 重新训练第 $loop 轮模型"
    fi
    
    # 执行训练（run_python_with_env函数会自动处理错误）
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
