# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## AI Assistant Core Rules

### Three-Stage Workflow

#### Stage 1: Analyze Problems

**Declaration Format**: `【分析问题】`

**Purpose**
Since there may be multiple possible solutions, to make the right decision, sufficient evidence is needed.

**Must Do**:
- Understand my intent, ask me if there are ambiguities
- Search all relevant code
- Identify the root cause of problems

**Proactively Discover Issues**
- Find duplicate code
- Identify unreasonable naming
- Find redundant code, classes
- Find possibly outdated designs
- Find overly complex designs, calls
- Find inconsistent type definitions
- Further search code to see if there are similar issues in a larger scope

After completing the above tasks, you can ask me questions.

**Absolutely Forbidden**:
- ❌ Modify any code
- ❌ Rush to give solutions
- ❌ Skip search and understanding steps
- ❌ Recommend solutions without analysis

**Stage Transition Rules**
In this stage you need to ask me questions.
If there are multiple solutions you cannot choose from, ask me as part of your questioning.
If there's nothing you need to ask me, proceed directly to the next stage.

#### Stage 2: Formulate Plan
**Declaration Format**: `【制定方案】`

**Prerequisites**:
- I have clearly answered key technical decisions.

**Must Do**:
- List files to be changed (added, modified, deleted), briefly describe changes for each file
- Eliminate duplicate logic: if duplicate code is found, it must be eliminated through reuse or abstraction
- Ensure modified code follows DRY principles and good architectural design

If new key decisions that need to be collected from me are discovered in this stage, you can continue to ask me until there are no unclear issues, then this stage ends.
This stage is not allowed to automatically switch to the next stage.

#### Stage 3: Execute Plan
**Declaration Format**: `【执行方案】`

**Must Do**:
- Strictly implement according to the selected plan
- Run type checking after modifications

**Absolutely Forbidden**:
- ❌ Commit code (unless user explicitly requests)
- ❌ Start development server

If you discover uncertain issues in this stage, please ask me.

When receiving user messages, generally start from the 【分析问题】 stage, unless the user explicitly specifies a stage name.

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