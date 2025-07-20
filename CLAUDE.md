# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Python package for data collection and processing of results from GRASP (General-purpose Relativistic Atomic Structure Package). This tool enhances GRASP's built-in data handling capabilities with more flexible Python-based processing, machine learning optimization, and automated workflow management.

## Core Architecture

### Package Structure
- **graspdataprocessing/** - Main Python package (in `src/`)
  - **CSFs_choosing/** - Configuration State Function selection with ML
  - **data_IO/** - Data input/output, plotting, and visualization  
  - **machine_learning_module/** - ANN models and training infrastructure
  - **processing/** - ASF/transition data collection and analysis
  - **utils/** - Utilities, environment config, progress management

### Key Components

1. **ML-driven CSF Selection Pipeline** - Uses machine learning to optimize Configuration State Function selection for quantum mechanical calculations
2. **GRASP Integration** - Automated workflow management for GRASP calculations via shell scripts
3. **Data Processing** - Comprehensive tools for atomic physics data analysis and visualization

## Development Commands

### Environment Setup
```bash
# Choose appropriate environment
pip install -r requirements-cpu.txt    # CPU environment
pip install -r requirements-gpu.txt    # GPU environment

# Development installation
pip install -e .

# Build package
python -m build
```

### Common Workflows

#### Running ML CSF Selection
```bash
# Navigate to work directory containing config.toml
cd /path/to/calculation/directory

# Interactive mode (recommended)
/path/to/GraspDataProcessing/tests/ml_csf_choosing/quick_run.sh

# Command line mode
/path/to/GraspDataProcessing/tests/ml_csf_choosing/run_ml.sh train
/path/to/GraspDataProcessing/tests/ml_csf_choosing/run_ml.sh all
```

#### Key Programs
- **initial_csfs.py** - Initialize CSF configurations and descriptors
- **choosing_csfs.py** - Strategy-based CSF selection for next calculation round  
- **train.py** - ML model training and intelligent CSF selection

### Configuration Management

#### Main Config (config.toml)
The `config.toml` file controls all aspects of calculations:
- Atomic parameters (atom, conf, active_space, cal_levels)
- ML parameters (expansion_ratio, chosen_ratio, model_params)
- Step control for checkpoint/restart functionality
- CPU optimization settings

#### Step-Level Control
Advanced checkpoint/restart system allows granular control:
```toml
[step_control]
enable_step_control = true
target_loop = 3              # Specific loop to control
start_step = "rmcdhf"        # Start from specific step
end_step = "rmcdhf"          # Stop after specific step
skip_completed_steps = true  # Auto-skip completed steps
```

Available steps: `initial_csfs`, `choosing_csfs`, `mkdisks`, `rangular`, `rwfnestimate`, `rmcdhf`, `rci`, `rsave`, `jj2lsj`, `rlevels`, `train`

### GRASP Integration

The system integrates with GRASP2018 through automated shell scripts:
- **run_script.sh** - Main SLURM job script with comprehensive error handling
- **common_functions.sh** - Shared utilities for logging and configuration
- Supports MPI parallel execution with configurable thread counts
- Automated error detection and file validation

### Build System

- **Package Manager**: Hatchling (modern Python packaging)
- **Dependencies**: PyTorch, scikit-learn, pandas, numpy 2.0+
- **Python Version**: Requires 3.12+
- **Linting**: Ruff with NumPy 2.0 compatibility rules

### File Organization Patterns

#### ML Training Data Flow
1. GRASP calculations produce `.level` files with energy data
2. CSF selection tools process `.c` and `.cm` files  
3. ML models train on energy convergence patterns
4. New CSF selections written for next iteration

#### Configuration Precedence
1. Command line arguments override config.toml
2. config.toml overrides package defaults
3. Environment variables for system paths (PYTHONPATH, conda environments)

## Important Notes

### Environment Requirements
- Must run in correct conda environment with GRASP modules loaded
- Requires MPI (OpenMPI) for parallel GRASP calculations
- Python environment must include all scientific computing dependencies

### Workflow Dependencies  
- Each calculation step depends on previous step outputs
- Step control system allows selective re-execution but requires understanding of dependencies
- ML training requires completed GRASP calculations with energy level data

### Performance Considerations
- PyTorch thread count configurable via `cpu_config.pytorch_threads`
- MPI temporary file paths configurable to avoid I/O bottlenecks
- Large CSF sets require careful memory management