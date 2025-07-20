# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python package for post-processing GRASP (General-purpose Relativistic Atomic Structure Package) calculation data with machine learning-enhanced CSF (Configuration State Function) selection. The package provides tools for data extraction, processing, visualization, and intelligent CSF optimization using neural networks.

## Installation and Environment Setup

### Environment Selection
Choose the appropriate environment based on your hardware:

```bash
# CPU-only environment (faster setup, broader compatibility)
pip install -r requirements-cpu.txt

# GPU environment (for deep learning acceleration)
pip install -r requirements-gpu.txt
```

### Package Installation
```bash
# Development installation (recommended for working with source)
pip install -e .

# Production installation from PyPI
pip install -i https://test.pypi.org/simple/ grasp-data-processing

# Manual build and install
python -m build
pip install dist/grasp_data_processing-*.whl
```

### Environment Verification
```bash
# Check installation status
python check_installation.py

# Manual verification
python -c "import graspdataprocessing; print(graspdataprocessing.__version__)"
```

## Core Development Commands

### Machine Learning CSF Selection Scripts
The primary workflow uses interactive scripts in `tests/ml_csf_choosing/`:

```bash
# Interactive quick runner (recommended for beginners)
./tests/ml_csf_choosing/quick_run.sh

# Command-line runner with options
./tests/ml_csf_choosing/run_ml.sh train        # Train ML model
./tests/ml_csf_choosing/run_ml.sh initial      # Initialize CSFs
./tests/ml_csf_choosing/run_ml.sh choosing     # Select CSFs
./tests/ml_csf_choosing/run_ml.sh all          # Run complete pipeline

# With options
./tests/ml_csf_choosing/run_ml.sh -d /path/to/work -v train
```

### GRASP Integration Workflow
The complete computational workflow via SLURM:

```bash
# Main computation script with integrated ML optimization
sbatch scripts/run_script.sh

# Step-level control for debugging/resuming
# Modify config.toml [step_control] section, then:
sbatch scripts/run_script.sh
```

### Configuration Management
```bash
# Load/modify configuration parameters
python scripts/csfs_ml_choosing_config_load.py set atom "Gd_I" -f config.toml
python scripts/csfs_ml_choosing_config_load.py set root_path /path/to/calc -f config.toml
```

### Testing and Development
```bash
# Run development tests
python tests/train.py
python tests/debug_ann_batchnorm.py

# Check specific components
python tests/read_level.py
python tests/test_pytorch_threads.py
```

## Architecture Overview

### Core Package Structure (`src/graspdataprocessing/`)

**Data I/O Module (`data_IO/`)**:
- `grasp_raw_data_load.py`: GRASP output file parsers (levels, transitions, wavefunctions)
- `processing_data_load.py`: Configuration and metadata loaders
- `produced_data_write.py`: CSF file writers and data persistence
- `radial_wavefunction_plot.py`: Visualization tools

**CSF Selection Module (`CSFs_choosing/`)**:
- `CSFs_choosing.py`: Core CSF selection algorithms based on mixing coefficients
- `CSFs_compress_extract.py`: CSF descriptor generation and J-coupling analysis

**Machine Learning Module (`machine_learning_module/`)**:
- `ANN.py`: Neural network classifier with batch normalization
- `machine_learning_initialization.py`: ML pipeline setup and validation
- `machine_learning_training.py`: Training, evaluation, and convergence checking

**Data Processing Module (`processing/`)**:
- `ASF_data_collection.py`: Atomic State Function energy and composition analysis
- `transition_data_collection.py`: Transition probability calculations

**Utilities Module (`utils/`)**:
- `data_modules.py`: Core data structures (CSFs, MixCoefficientData)
- `environment_config.py`: SLURM/local environment detection
- `progress_manager.py`: Progress tracking for long calculations
- `tool_function.py`: Quantum mechanics utilities (J-coupling, energy conversions)

### External Scripts and Tools

**Configuration Scripts (`scripts/`)**:
- `csfs_ml_choosing_config_load.py`: TOML configuration management
- `run_script.sh`: Main SLURM computation orchestrator
- `common_functions.sh`: Shell utilities and GRASP execution wrappers

**ML Pipeline Scripts (`tests/ml_csf_choosing/`)**:
- `initial_csfs.py`: CSF initialization and descriptor generation
- `choosing_csfs.py`: CSF selection strategy implementation
- `train.py`: ML model training and evaluation
- Interactive runners: `quick_run.sh`, `run_ml.sh`

## Key Configuration Files

### Main Configuration (`config.toml`)
Controls all aspects of the calculation:
```toml
atom = "Gd_I"                    # Target atom
conf = "cv6odd1_j3as5"           # Configuration name
cal_loop_num = 4                 # Current iteration
chosen_ratio = 0.085             # CSF selection ratio
active_space = "11s,10p,9d,8f,7g,6h"
cal_method = "rci"               # Current calculation method

[step_control]                   # Step-level execution control
enable_step_control = false
start_step = "auto"
end_step = "auto"

[ml_config]                      # ML model parameters
expansion_ratio = 2
high_prob_percentile = 95
overfitting_threshold = 0.1
```

### Step Control System
For debugging and resuming interrupted calculations:
```toml
[step_control]
enable_step_control = true
target_loop = 3                  # Apply control to specific loop
start_step = "rmcdhf"           # Resume from specific step
end_step = "rmcdhf"             # Stop after specific step
skip_completed_steps = true      # Skip steps with existing output
```

Available steps: `initial_csfs`, `choosing_csfs`, `mkdisks`, `rangular`, `rwfnestimate`, `rmcdhf`/`rci`, `rsave`, `jj2lsj`, `rlevels`, `train`

## Development Workflow

### Typical CSF Optimization Cycle
1. **Initialize**: `initial_csfs.py` - Generate initial CSF pool and descriptors
2. **GRASP Calculation**: Automated via `run_script.sh` (mkdisks → rangular → rwfnestimate → rmcdhf/rci → rsave → jj2lsj → rlevels)
3. **CSF Selection**: `choosing_csfs.py` - Select next round CSFs based on mixing coefficients
4. **ML Training**: `train.py` - Train neural network, evaluate convergence, suggest next ratio
5. **Iterate**: Repeat until convergence or maximum loops

### Environment Detection
The package automatically detects execution environment:
- **SLURM**: Full parallel execution with MPI programs
- **Local**: CPU-only mode with simplified parallelization
- **Debug**: Enhanced logging and intermediate file preservation

### File Naming Conventions
- Configuration files: `{conf}_{loop}.c`, `{conf}_{loop}.w`, `{conf}_{loop}.cm`
- Level data: `{conf}_{loop}.level`
- ML data: `descriptors_{loop}.pkl`, `csf_metadata_{loop}.pkl`
- Models: `model_loop_{loop}.pkl`, `training_results.csv`

## Integration with GRASP

### GRASP Program Integration
The package orchestrates standard GRASP workflow:
- `rcsfgenerate90` → `rwfnestimate` → `rmcdhf_mpi`/`rci_mpi` → `rsave` → `jj2lsj` → `rlevels`
- MPI execution: `mpirun -np {processor} {program}`
- Error checking: Automated output validation and file existence checks

### Data Flow
1. **Input**: Initial CSF list, atomic parameters, active space definition
2. **GRASP**: Quantum mechanical calculations producing energy levels and mixing coefficients
3. **Analysis**: Mixing coefficient analysis and descriptor generation
4. **ML**: Neural network training for CSF importance prediction
5. **Selection**: Next iteration CSF pool optimization
6. **Output**: Optimized CSF sets, convergence metrics, performance plots

## Performance Considerations

### Parallel Execution
- **MPI**: GRASP calculations scale with processor count (typically 16-64 cores)
- **PyTorch**: CPU threading control via `pytorch_threads` configuration
- **OpenMP**: Not used in this package (GRASP programs may use it)

### Memory Management
- Large CSF sets (>50,000) require careful memory management
- Descriptors cached as binary files for fast reloading
- Progress tracking prevents memory leaks in long calculations

### File I/O Optimization
- Binary serialization for large datasets (`*.pkl` files)
- Sparse matrix handling for large Hamiltonian matrices
- Temporary file cleanup after each iteration

## Debugging and Troubleshooting

### Common Issues
- **BatchNorm errors**: Use `debug_ann_batchnorm.py` to diagnose
- **PyTorch threading**: Configure via `pytorch_threads` in config.toml
- **GRASP execution**: Check step control logs for program-specific errors
- **Missing files**: Use step control to resume from specific points

### Log Analysis
```bash
# Check step execution status
grep "步骤控制配置" *.log
grep "⏭️" *.log          # Skipped steps
grep "🛑" *.log          # Stop points
grep "✅" *.log          # Completed steps
grep "❌" *.log          # Failed steps
```

### Environment Variables
- `PYTHONPATH`: Automatically set by run scripts
- `OMP_NUM_THREADS`: Controlled via PyTorch configuration
- `SLURM_*`: Automatically detected for parallel execution