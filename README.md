# Grasp Data Processing

A simple data collection & processing tool for grasp2018.

Coding in python3.12 Numpy, Pandas and Matplotlib are needed.

Although GRASP has some original tools to handle the data, the programs written in fortran are less convenient to operate and the operations are fixed, which is not conducive to subsequent processing.

---

Examples given in test folder. The only thing need to do is change the data file location, data file parameters and `calculation_parameters'.

---

## Installation

### 🚀 快速安装

```bash
# 根据您的环境选择：
pip install -r requirements-cpu.txt    # CPU环境
pip install -r requirements-gpu.txt    # GPU环境
```

📖 **详细安装指南**: 请查看 [INSTALL.md](INSTALL.md) 了解不同环境的安装选项和故障排除。

🔍 **验证安装**: 运行 `python check_installation.py` 检查所有依赖是否正确安装。

### 手动安装

```bash
# 1. 安装依赖
pip install -r requirements-cpu[gpu].txt  # 或选择对应环境的依赖文件

# 2. 构建包
python -m build

# 3. 安装
pip install dist/grasp_data_processing-*.whl
```

### 使用pip安装

```bash
pip install -i https://test.pypi.org/simple/ grasp-data-processing
```
