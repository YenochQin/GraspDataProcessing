#!/bin/zsh

# Common shell function library
# Eliminate duplicate code, provide unified logging and utility functions

# Logging function with timestamp
log_with_timestamp() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] $message"
}

# Error logging function with timestamp
log_error_with_timestamp() {
    local message="$1" 
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] ERROR: $message" >&2
}

# Warning logging function with timestamp
log_warning_with_timestamp() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] WARNING: $message" >&2
}

# Stage logging function
log_stage() {
    local stage_name="$1"
    local stage_type="$2"  # START or END
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    if [ "$stage_type" = "START" ]; then
        echo "[$timestamp] [START] Starting stage: $stage_name"
    elif [ "$stage_type" = "END" ]; then
        echo "[$timestamp] [DONE] Completed stage: $stage_name"
    else
        echo "[$timestamp] [INFO] Stage: $stage_name"
    fi
}

# Check if running in SLURM environment
is_slurm_environment() {
    if [ -n "$SLURM_JOB_ID" ] || [ -n "$SLURM_PROCID" ] || [ -n "$SLURM_LOCALID" ]; then
        return 0  # true
    else
        return 1  # false
    fi
}

# Check if in debug mode
is_debug_mode() {
    if [ "$DEBUG" = "1" ] || [ "$DEBUG" = "true" ] || [ "$PYTHON_DEBUG" = "1" ]; then
        return 0  # true
    else
        return 1  # false
    fi
}

# Environment-aware Python execution function
run_python_with_env() {
    local python_script="$1"
    shift  # Remove first parameter, remaining parameters serve as script arguments
    
    # Check if it's a config reading script (no success log needed)
    local script_basename=$(basename "$python_script")
    local is_config_script=false
    if [[ "$script_basename" == "csfs_ml_choosing_config_load.py" ]]; then
        is_config_script=true
    fi
    
    # Set environment variables
    if is_slurm_environment && ! is_debug_mode; then
        # SLURM production environment: disable progress bar
        export PYTHONUNBUFFERED=1
        # Only output execution log for non-config scripts
        if [[ "$is_config_script" == "false" ]]; then
            log_with_timestamp "Running Python script in SLURM environment (production mode): $python_script"
        fi
    else
        # Debug mode: enable progress bar
        export DEBUG=1
        # Only output execution log for non-config scripts
        if [[ "$is_config_script" == "false" ]]; then
            log_with_timestamp "Running Python script in debug mode: $python_script"
        fi
    fi
    
    # Execute Python script
    python "$python_script" "$@"
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        # Only output success log for non-config scripts
        if [[ "$is_config_script" == "false" ]]; then
            log_with_timestamp "Python script executed successfully: $python_script"
        fi
    else
        # Failure log is always output (including config scripts)
        log_error_with_timestamp "Python script execution failed: $python_script (exit code: $exit_code)"
        
        # For critical scripts, exit immediately on failure
        if [[ "$is_config_script" == "true" || "$script_basename" == "initial_csfs.py" || "$script_basename" == "choosing_csfs.py" || "$script_basename" == "train.py" ]]; then
            log_error_with_timestamp "Critical Python script failed, terminating execution!"
            exit $exit_code
        fi
    fi
    
    return $exit_code
}

# Safe configuration value reading function (ensures pure numeric return, exits on failure)
safe_get_config_value() {
    local config_file="$1"
    local key="$2"
    local description="$3"
    
    # Use temporary files to capture output and errors
    local temp_output=$(mktemp)
    local temp_error=$(mktemp)
    
    # Execute configuration reading
    python "${GRASP_DATA_PROCESSING_ROOT}/scripts/csfs_ml_choosing_config_load.py" get "$key" -f "$config_file" > "$temp_output" 2> "$temp_error"
    local exit_code=$?
    
    # Read results
    local value=$(cat "$temp_output")
    local error_msg=$(cat "$temp_error")
    
    # Clean up temporary files
    rm -f "$temp_output" "$temp_error"
    
    # Check if successful (allow empty values)
    if [ $exit_code -ne 0 ] || [[ "$value" == *"Error"* ]] || [[ "$error_msg" == *"错误"* ]]; then
        log_error_with_timestamp "Failed to read configuration item '$key': $description"
        if [ -n "$error_msg" ]; then
            log_error_with_timestamp "Error details: $error_msg"
        fi
        if [ -n "$value" ]; then
            log_error_with_timestamp "Return value: $value"
        fi
        exit 1
    fi
    
    # Return clean value (allow empty strings)
    echo "$value"
}

# File existence check function
check_file_exists() {
    local file_path="$1"
    local description="$2"
    
    if [ -f "$file_path" ]; then
        log_with_timestamp "Found file: $description ($file_path)"
        return 0
    else
        log_error_with_timestamp "File not found: $description ($file_path)"
        return 1
    fi
}

# Directory creation function
ensure_directory() {
    local dir_path="$1"
    local description="$2"
    
    if [ ! -d "$dir_path" ]; then
        mkdir -p "$dir_path"
        log_with_timestamp "Created directory: $description ($dir_path)"
    else
        log_with_timestamp "Directory already exists: $description ($dir_path)"
    fi
}

# Function to calculate execution time
calculate_execution_time() {
    local start_time="$1"
    local end_time="$2"
    local execution_time=$((end_time - start_time))
    
    local hours=$((execution_time / 3600))
    local minutes=$(((execution_time % 3600) / 60))
    local seconds=$((execution_time % 60))
    
    if [ $hours -gt 0 ]; then
        echo "${hours} hours ${minutes} minutes ${seconds} seconds"
    elif [ $minutes -gt 0 ]; then
        echo "${minutes} minutes ${seconds} seconds"
    else
        echo "${seconds} seconds"
    fi
}

# Print environment information
print_environment_info() {
    log_with_timestamp "=== Environment Information ==="
    log_with_timestamp "Hostname: $(hostname)"
    log_with_timestamp "Current user: $(whoami)"
    log_with_timestamp "Working directory: $(pwd)"
    
    if is_slurm_environment; then
        log_with_timestamp "Runtime environment: SLURM job (Job ID: ${SLURM_JOB_ID:-Unknown})"
    else
        log_with_timestamp "Runtime environment: Local execution"
    fi
    
    if is_debug_mode; then
        log_with_timestamp "Debug mode: Enabled"
    else
        log_with_timestamp "Debug mode: Disabled"
    fi
    
    log_with_timestamp "Python version: $(python --version 2>&1)"
    log_with_timestamp "==================="
} 

# Determine expected output files based on program name
get_expected_files() {
    local program_name="$1"
    local conf="$2" 
    local loop="$3"
    
    case "$program_name" in
        "mkdisks")
            echo "disks"
            ;;
        "rangular_mpi")
            echo ""  # No file output
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
            echo ""  # Unknown program, no file check
            ;;
    esac
}

# GRASP program error checking function
check_grasp_errors() {
    local program_name="$1"
    local output_log="$2"
    local expected_files="$3"
    
    # Check for serious error patterns
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
    
    # Search for error patterns
    for pattern in "${error_patterns[@]}"; do
        if grep -qi "$pattern" "$output_log"; then
            log_with_timestamp "[ERROR] $program_name detected error: $pattern"
            log_with_timestamp "Error context:"
            grep -i -A2 -B2 "$pattern" "$output_log" | tail -10
            return 1
        fi
    done
    
    # Check if expected output files exist
    if [ -n "$expected_files" ]; then
        log_with_timestamp "[INFO] Current working directory: $(pwd)"
        log_with_timestamp "[CHECK] Check expected files: $expected_files"
        
        # For MPI programs, wait a while to ensure files are fully written
        if [[ "$program_name" == *"_mpi" ]]; then
            log_with_timestamp "[WAIT] MPI program detected, waiting 3 seconds to ensure file writing is complete..."
            sleep 3
        fi
        
        # Use retry mechanism to check files
        local max_retries=5
        local retry_count=0
        local all_files_exist=false
        
        # Convert file list to array for processing
        local files_array
        # Use zsh special syntax for parameter expansion
        files_array=(${=expected_files})
        local file_count=${#files_array[@]}
        
        # If splitting fails, try using read command
        if [ $file_count -eq 1 ] && [[ "$expected_files" == *" "* ]]; then
            log_with_timestamp "[WARN] Detected file list may not be correctly split, trying read command..."
            files_array=()
            local IFS=' '
            read -A files_array <<< "$expected_files"
            file_count=${#files_array[@]}
        fi
        
        log_with_timestamp "[INFO] Expected file count: $file_count items"
        local index=1
        for file in "${files_array[@]}"; do
            log_with_timestamp "  [$index]: '$file'"
            index=$((index + 1))
        done
        
        while [ $retry_count -lt $max_retries ] && [ "$all_files_exist" = false ]; do
            all_files_exist=true
            local missing_files=""
            
            # Use array for file checking
            for file in "${files_array[@]}"; do
                log_with_timestamp "[CHECK] Check file: '$file'"
                
                if [ ! -f "$file" ]; then
                    missing_files="$missing_files $file"
                    all_files_exist=false
                elif [ ! -s "$file" ]; then
                    log_with_timestamp "[ERROR] $program_name generated empty file: $file"
                    all_files_exist=false
                    break
                fi
            done
            
            if [ "$all_files_exist" = false ]; then
                retry_count=$((retry_count + 1))
                if [ $retry_count -lt $max_retries ]; then
                    log_with_timestamp "[RETRY] Retry $retry_count, waiting for file generation... Missing files:$missing_files"
                    sleep 2
                else
                    log_with_timestamp "[ERROR] $program_name did not generate expected files:$missing_files"
                    log_with_timestamp "[INFO] Current directory contents:"
                    ls -la
                    log_with_timestamp "[CHECK] Detailed file status check:"
                    for file in "${files_array[@]}"; do
                        log_with_timestamp "Check file: $file"
                        if [ -e "$file" ]; then
                            log_with_timestamp "  - File exists but may have issues"
                            log_with_timestamp "  - File size: $(du -h "$file" 2>/dev/null || echo "Cannot get size")"
                            log_with_timestamp "  - File permissions: $(ls -l "$file" 2>/dev/null || echo "Cannot get permissions")"
                        else
                            log_with_timestamp "  - File does not exist"
                            # Look for similar file names
                            local basename=$(basename "$file")
                            local similar_files=$(ls -1 | grep -i "${basename%.*}" 2>/dev/null || echo "No similar files")
                            log_with_timestamp "  - Similar files: $similar_files"
                        fi
                    done
                    return 1
                fi
            fi
        done
        
        if [ "$all_files_exist" = true ]; then
            log_with_timestamp "[SUCCESS] All expected file checks passed: $expected_files"
        fi
    fi
    
    return 0
}

# Function to safely execute GRASP programs
safe_grasp_execute() {
    local program_name="$1"
    local input_commands="$2"
    shift 2
    
    log_with_timestamp "Execute $program_name..."
    
    # Dynamically get expected file list
    local expected_files=""
    if [[ "$program_name" == "rsave" ]]; then
        # For rsave, we need to get conf and loop variables from context
        expected_files=$(get_expected_files "$program_name" "$conf" "$loop")
    else
        expected_files=$(get_expected_files "$program_name" "$conf" "$loop")
    fi
    
    log_with_timestamp "[AUTO] Automatically determine expected files based on program $program_name: $expected_files"
    
    # Create temporary log file
    local temp_log="/tmp/${program_name}_${SLURM_JOB_ID}_$$.log"
    
    # Execute program and get exit code
    local exit_code=0
    if [ -n "$input_commands" ]; then
        # Program with input
        echo "$input_commands" | "$@" 2>&1 | tee "$temp_log"
        exit_code=${PIPESTATUS[0]:-$?}
    else
        # Program without input
        "$@" 2>&1 | tee "$temp_log"
        exit_code=$?
    fi
    
    # Ensure exit code is numeric
    if [ -z "$exit_code" ]; then
        exit_code=1
        log_with_timestamp "[WARN] Cannot get exit code for $program_name, assuming failure"
    elif ! [[ "$exit_code" =~ ^[0-9]+$ ]]; then
        exit_code=1
        log_with_timestamp "[WARN] Exit code for $program_name is not numeric, assuming failure"
    fi
    
    # Check exit code
    if [ "$exit_code" -ne 0 ]; then
        log_with_timestamp "[ERROR] $program_name abnormal exit, exit code: $exit_code"
        log_with_timestamp "Last output:"
        tail -20 "$temp_log"
        rm -f "$temp_log"
        exit 1
    fi
    
    # Check GRASP-specific errors
    check_grasp_errors "$program_name" "$temp_log" "$expected_files"
    local check_result=$?
    if [ "$check_result" -ne 0 ]; then
        rm -f "$temp_log"
        exit 1
    fi
    
    rm -f "$temp_log"
    log_with_timestamp "[SUCCESS] $program_name completed"
}

# =============================================================================
# Log format enhancement functions
# =============================================================================

# Color code definitions
readonly COLOR_RED='\033[0;31m'
readonly COLOR_GREEN='\033[0;32m'
readonly COLOR_YELLOW='\033[1;33m'
readonly COLOR_BLUE='\033[0;34m'
readonly COLOR_PURPLE='\033[0;35m'
readonly COLOR_CYAN='\033[0;36m'
readonly COLOR_WHITE='\033[1;37m'
readonly COLOR_BOLD='\033[1m'
readonly COLOR_RESET='\033[0m'

# Path simplification function - remove root_path prefix, show only relative path
simplify_path() {
    local full_path="$1"
    local root_path="$2"
    
    # If root_path not provided, try to get from config.toml
    if [ -z "$root_path" ] && [ -f "config.toml" ]; then
        root_path=$(safe_get_config_value "config.toml" "root_path" "Root directory path" 2>/dev/null || echo "")
    fi
    
    # If root_path is empty or path doesn't contain root_path, return original path
    if [ -z "$root_path" ] || [[ "$full_path" != "$root_path"* ]]; then
        echo "$full_path"
        return
    fi
    
    # Remove root_path prefix
    local relative_path="${full_path#$root_path}"
    # Remove leading slash
    relative_path="${relative_path#/}"
    
    # If simplified path is empty, means it's the root directory
    if [ -z "$relative_path" ]; then
        echo "."
    else
        echo "$relative_path"
    fi
}

# Numeric highlighting function
highlight_number() {
    local text="$1"
    local color="${2:-$COLOR_CYAN}"
    
    # Handle empty or null values
    if [ -z "$text" ] || [ "$text" = "null" ]; then
        if is_slurm_environment; then
            printf "(empty)"
        else
            printf "%s(empty)%s" "$COLOR_YELLOW" "$COLOR_RESET"
        fi
    else
        # In SLURM environment, disable colors to avoid escape sequence clutter in logs
        if is_slurm_environment; then
            printf "%s" "$text"
        else
            # Use color to highlight numbers in interactive mode
            printf "%s%s%s" "$color" "$text" "$COLOR_RESET"
        fi
    fi
}

# Parameter highlighting function
highlight_param() {
    local key="$1"
    local value="$2"
    local key_color="${3:-$COLOR_WHITE}"
    local value_color="${4:-$COLOR_CYAN}"
    
    # In SLURM environment, disable colors to avoid escape sequence clutter in logs
    if is_slurm_environment; then
        printf "%s=%s" "$key" "$(highlight_number "$value" "$value_color")"
    else
        printf "%s%s%s=%s" "$key_color" "$key" "$COLOR_RESET" "$(highlight_number "$value" "$value_color")"
    fi
}

# Log function with path simplification support
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

# Enhanced configuration parameter logging function
log_config_params() {
    local atom="$1"
    local conf="$2" 
    local processor="$3"
    local active_space="$4"
    local cal_levels="$5"
    
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    # Use printf instead of echo -e for better compatibility and encoding handling
    printf "[%s] Configuration parameters: %s %s %s\n" \
        "$timestamp" \
        "$(highlight_param "atom" "$atom")" \
        "$(highlight_param "conf" "$conf")" \
        "$(highlight_param "processor" "$processor" "$COLOR_WHITE" "$COLOR_GREEN")"
    
    printf "[%s] Active space: %s, Calculation levels: %s\n" \
        "$timestamp" \
        "$(highlight_number "$active_space" "$COLOR_YELLOW")" \
        "$(highlight_number "$cal_levels" "$COLOR_YELLOW")"
}