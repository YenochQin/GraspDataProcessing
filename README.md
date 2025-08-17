# GraspDataProcessing Scripts and Auxiliary Files

This directory contains auxiliary scripts, test files, and documentation that have been separated from the main GraspDataProcessing Python package. These files support the operation and usage of the core GraspDataProcessing library but are not part of the main package itself.

## 🎯 Purpose

The separation provides:
- **Clean package structure**: Core library separated from operational scripts
- **Flexible deployment**: Scripts can be customized for different environments
- **Web-based configuration**: Simplified setup through web interface
- **Clear responsibilities**: Main package for library functionality, scripts for runtime operations

## 📁 Directory Structure

### `scripts/` - Runtime Scripts & Configuration
Essential scripts for GRASP workflow execution:

| File | Purpose |
|------|---------|
| `grasp_dual_generator.html` | **Web-based configuration generator** - primary interface for ML CSF selection |
| `run_script.sh` | Main SLURM job script orchestrating GRASP calculations |
| `common_functions.sh` | Shared utilities for logging, configuration, and error handling |
| `csfs_ml_choosing_config_load.py` | Configuration management via TOML files |
| `config.toml` | Sample configuration template |
| `disks`, `mkdisks` | GRASP utility scripts |

### `tests/` - Examples & Validation
Comprehensive testing and examples:

| Directory | Contents |
|-----------|----------|
| `ml_csf_choosing/` | Machine learning CSF selection pipeline examples |
| `level_plot/` | Data visualization and plotting examples |
| `cpp_parallel_csf/` | C++ descriptor parallelization tests |
| Root files | Utility test scripts for data validation |

### `modify_logs/` - Development Documentation
Complete development history with bilingual documentation.

## 🚀 Installation Guide

### Method 1: GitHub Release (Recommended)
Download the latest release from GitHub:
```bash
# Download from GitHub releases page
wget https://github.com/YenochQin/graspdataprocessing/releases/latest/download/grasp-data-processing-x.x.x-py3-none-any.whl

# Install the package
pip install grasp-data-processing-x.x.x-py3-none-any.whl
```

### Method 2: Test PyPI Installation
Install from Test PyPI repository:
```bash
pip install -i https://test.pypi.org/simple/ grasp-data-processing
```

### Method 3: Development Installation
For development or custom modifications:
```bash
git clone https://github.com/YenochQin/graspdataprocessing.git
cd graspdataprocessing
pip install -e .
```

## 📦 System Requirements

### Python Dependencies
Install these Python packages before running:
```bash
# Core scientific computing
pip install numpy>=2.0.0 pandas>=2.2.2 matplotlib>=3.8.4

# Machine learning
pip install torch>=2.0.0 scikit-learn>=1.3.0 imbalanced-learn>=0.11.0

# Additional utilities
pip install seaborn>=0.11.0 joblib>=1.1.0 pyyaml>=6.0 rtoml>=0.9.0
```

### External Dependencies

#### GRASP Library
**Required**: GRASP2018 installation
- Download from: https://grasp.wp.st-andrews.ac.uk/
- Ensure GRASP binaries are in system PATH
- Verify installation: `which rangular_mpi` should return path

#### CSFs_2_descripors Library
**Required**: CSFs descriptor generation library
- Repository: https://github.com/YenochQin/CSFs_2_descripors
- Installation:
  ```bash
  git clone https://github.com/YenochQin/CSFs_2_descripors.git
  cd CSFs_2_descripors
  # Follow installation instructions in README
  ```

## 🤖 Machine Learning CSF Selection

### Web-Based Configuration (Recommended)

The primary interface for machine learning CSF selection is the web-based generator:

#### Step 1: Open Web Interface
```bash
# Navigate to scripts directory
cd scripts/

# Open web interface in browser
open grasp_dual_generator.html  # macOS
xdg-open grasp_dual_generator.html  # Linux
# Or simply double-click grasp_dual_generator.html
```

#### Step 2: Configure Parameters
In the web interface:
1. **Atomic Parameters**: Select target atom and configuration
2. **Active Space**: Define orbital space for CSF selection
3. **ML Parameters**: Set expansion ratio, chosen ratio, and model parameters
4. **Calculation Levels**: Specify energy levels to calculate
5. **Advanced Options**: Configure convergence criteria and checkpoints

#### Step 3: Generate Configuration
The web interface will:
- Validate input parameters
- Generate `config.toml` file
- Create job submission scripts
- Provide next-step instructions

#### Step 4: Submit Job
```bash
# Copy generated files to calculation directory
cp config.toml run_script.sh /your/calculation/directory/
cd /your/calculation/directory

# Submit to SLURM
sbatch run_script.sh
```

### Manual Configuration (Advanced)
For advanced users or custom workflows:
```bash
# Edit configuration file
cp scripts/config.toml ./my_config.toml
nano my_config.toml

# Run configuration validation
python scripts/csfs_ml_choosing_config_load.py validate -f my_config.toml
```

## ⚙️ System Configuration

### Environment Variables
```bash
# Add to ~/.bashrc or ~/.zshrc
export GRASP_PATH="/path/to/grasp2018"
export CSFS_DESCRIPTORS_PATH="/path/to/CSFs_2_descripors"
export PYTHONPATH="/path/to/GraspDataProcessing/src:$PYTHONPATH"
```

### SLURM Configuration
Update `scripts/run_script.sh` for your cluster:
```bash
# SLURM parameters
#SBATCH -J your_job_name
#SBATCH -N 1
#SBATCH --ntasks-per-node=your_core_count
#SBATCH -p your_partition
#SBATCH --output=%j_%x.log

# Paths
GRASP_DATA_PROCESSING_ROOT="/your/path/to/GraspDataProcessing"
source /your/path/to/miniconda3/etc/profile.d/conda.sh
conda activate your_env_name
```

## 🔧 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| **Import errors** | Verify PYTHONPATH includes main src directory |
| **GRASP commands not found** | Add GRASP binaries to PATH or use full paths |
| **CSFs_2_descripors missing** | Install from GitHub repository |
| **Web interface not opening** | Use file browser or direct browser navigation |
| **SLURM job fails** | Check log files and verify GRASP installation |
| **ML model errors** | Ensure all Python dependencies are installed |

### Verification Steps
```bash
# Test Python package
python -c "import graspdataprocessing; print('✅ Package OK')"

# Test GRASP installation
which rangular_mpi && echo "✅ GRASP OK"

# Test CSFs descriptors
python -c "from CSFs_processing import CSFs_descriptor_parallel; print('✅ Descriptors OK')"

# Test web interface
ls scripts/grasp_dual_generator.html && echo "✅ Web interface OK"
```

## 📚 Additional Resources

- **Main Package**: `/Users/yiqin/Documents/PythonProjects/GraspDataProcessing/`
- **CSFs_2_descripors**: https://github.com/YenochQin/CSFs_2_descripors
- **GRASP Official**: https://grasp.wp.st-andrews.ac.uk/
- **Issues**: Report problems to project GitHub issues
- **Documentation**: See main package README.md for detailed API documentation

## 🔄 Migration Notes

This structure was created by separating:
- **Core library**: Scientific computing and ML functionality
- **Operational scripts**: Runtime and configuration tools
- **Web interface**: User-friendly configuration generator
- **Development logs**: Complete project history

No functional changes were made - only reorganization for better modularity and user experience.