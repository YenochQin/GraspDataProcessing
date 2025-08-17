# GraspDataProcessing Scripts and Auxiliary Files

This directory contains auxiliary scripts, test files, and documentation that have been separated from the main GraspDataProcessing Python package. These files support the operation and usage of the core GraspDataProcessing library but are not part of the main package itself.

## 🎯 Purpose

The separation provides:
- **Clean package structure**: Core library separated from operational scripts
- **Flexible deployment**: Scripts can be customized for different environments
- **Clear responsibilities**: Main package for library functionality, scripts for runtime operations

## 📁 Directory Structure

### `scripts/` - Runtime Scripts & Configuration
Essential scripts for GRASP workflow execution:

| File | Purpose |
|------|---------|
| `run_script.sh` | Main SLURM job script orchestrating GRASP calculations |
| `common_functions.sh` | Shared utilities for logging, configuration, and error handling |
| `csfs_ml_choosing_config_load.py` | Configuration management via TOML files |
| `config.toml` | Sample configuration template |
| `disks`, `mkdisks` | GRASP utility scripts |
| `grasp_dual_generator.html` | Web-based configuration generator |

### `tests/` - Examples & Validation
Comprehensive testing and examples:

| Directory | Contents |
|-----------|----------|
| `ml_csf_choosing/` | Machine learning CSF selection pipeline examples |
| `level_plot/` | Data visualization and plotting examples |
| `cpp_parallel_csf/` | C++ descriptor parallelization tests |
| Root files | Utility test scripts for data validation |

### `modify_logs/` - Development Documentation
Complete development history:
- Feature implementation logs
- Bug fix documentation
- Performance optimization records
- Bilingual documentation (English/Chinese)

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Install main package
cd /path/to/GraspDataProcessing
pip install -e .

# Verify installation
python -c "import graspdataprocessing; print('✅ Package installed successfully')"
```

### 2. Script Configuration
Update paths in `scripts/run_script.sh`:

```bash
# Before: (original system path)
GRASP_DATA_PROCESSING_ROOT="/home/computer-0-2/AppFiles/GraspDataProcessing"

# After: (your system path)
GRASP_DATA_PROCESSING_ROOT="/your/path/to/GraspDataProcessing"
```

### 3. Test Run
```bash
# Navigate to test directory
cd tests/ml_csf_choosing

# Run example configuration
python initial_csfs.py
```

## ⚙️ Configuration Guide

### Path Variables to Update

| Variable | Location | Purpose |
|----------|----------|---------|
| `GRASP_DATA_PROCESSING_ROOT` | `run_script.sh:11` | Main package location |
| `PYTHONPATH` | `run_script.sh:12` | Python module search path |
| `conda` paths | `run_script.sh:30-38` | Conda environment activation |

### Example Configuration
```bash
# Typical HPC environment setup
export GRASP_DATA_PROCESSING_ROOT="/home/user/GraspDataProcessing"
export PYTHONPATH="${GRASP_DATA_PROCESSING_ROOT}/src:${PYTHONPATH}"
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate grasp-env
```

## 📊 Usage Examples

### Machine Learning CSF Selection
```bash
# Interactive mode
cd tests/ml_csf_choosing
./quick_run.sh

# Command line training
./run_ml.sh train

# Full pipeline
./run_ml.sh all
```

### Data Processing Pipeline
```bash
# Process GRASP output
cd /your/calculation/directory
python /path/to/scripts/csfs_ml_choosing_config_load.py set atom "Gd" -f config.toml
```

## 🔧 Development Notes

### Adding New Scripts
1. Place new scripts in appropriate subdirectory
2. Update this README with script documentation
3. Include usage examples in `tests/` if applicable

### Environment Compatibility
- **Python**: 3.12+ (requirements in main package)
- **MPI**: OpenMPI for parallel GRASP calculations
- **Conda**: Recommended for dependency management
- **GRASP**: GRASP2018 installation required

### Common Issues

| Issue | Solution |
|-------|----------|
| Import errors | Verify PYTHONPATH includes main src directory |
| Conda activation | Check conda.sh path in run_script.sh |
| GRASP commands | Ensure GRASP binaries are in PATH |
| Configuration errors | Validate config.toml syntax |

## 📚 Additional Resources

- **Main Package**: `/Users/yiqin/Documents/PythonProjects/GraspDataProcessing/`
- **Documentation**: See main package README.md for detailed API documentation
- **Issues**: Report problems to project maintainers
- **Contributing**: Follow development patterns in modify_logs/

## 🔄 Migration Notes

This structure was created by separating:
- **Core library** (remaining in main directory)
- **Operational scripts** (moved to this directory)
- **Test examples** (moved to tests/)
- **Development logs** (moved to modify_logs/)

No functional changes were made to the code itself - only reorganization for better modularity.