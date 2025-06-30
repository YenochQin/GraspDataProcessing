# 安装指南

本项目提供了针对不同运行环境的依赖文件配置。

## 环境选择

### 🖥️ CPU环境安装
适用于：
- 没有GPU的机器
- 不需要GPU加速的场景
- 快速测试和开发
- 资源受限的环境

```bash
pip install -r requirements-cpu.txt
```

**优势**：
- 安装包更小，下载更快
- 启动速度更快
- 兼容性更好

### 🚀 GPU环境安装  
适用于：
- 有NVIDIA GPU的机器
- 需要深度学习加速的场景
- 大规模模型训练
- 高性能计算需求

```bash
pip install -r requirements-gpu.txt
```

**前提条件**：
- 安装了NVIDIA GPU驱动
- 安装了合适版本的CUDA (推荐11.8+)
- 确认GPU可用：`nvidia-smi`

## 特定CUDA版本安装

如果需要特定的CUDA版本，可以修改 `requirements-gpu.txt` 中的PyTorch安装源：

### CUDA 11.8
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### CUDA 12.1
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 验证安装

### 🔍 快速验证脚本

我们提供了一个自动化验证脚本，可以检查所有依赖是否正确安装：

```bash
python check_installation.py
```

此脚本会：
- 检查所有必需和可选依赖的安装状态
- 显示各个包的版本信息
- 检测PyTorch环境 (CPU/GPU)
- 提供具体的安装建议

### 📝 手动验证

如果您想手动验证，可以运行以下代码：

```python
import torch
import numpy as np
import pandas as pd
import sklearn

# 检查PyTorch版本
print(f"PyTorch版本: {torch.__version__}")

# 检查GPU可用性
if torch.cuda.is_available():
    print(f"GPU可用: {torch.cuda.get_device_name(0)}")
    print(f"CUDA版本: {torch.version.cuda}")
else:
    print("使用CPU模式")

# 检查其他主要库
print(f"NumPy版本: {np.__version__}")
print(f"Pandas版本: {pd.__version__}")
print(f"Scikit-learn版本: {sklearn.__version__}")
```

## 开发模式安装

如果要进行项目开发，推荐使用可编辑安装：

```bash
# CPU环境
pip install -r requirements-cpu.txt
pip install -e .

# GPU环境  
pip install -r requirements-gpu.txt
pip install -e .
```

## 虚拟环境推荐

建议使用虚拟环境隔离依赖：

```bash
# 创建虚拟环境
python -m venv grasp_env

# 激活环境 (Windows)
grasp_env\Scripts\activate

# 激活环境 (Linux/Mac)
source grasp_env/bin/activate

# 安装依赖
pip install -r requirements-cpu.txt  # 或 requirements-gpu.txt
```

## 故障排除

### 常见问题

1. **PyTorch GPU版本无法使用GPU**
   - 检查CUDA驱动是否正确安装
   - 确认PyTorch版本与CUDA版本兼容

2. **安装过程中出现依赖冲突**
   - 尝试在全新的虚拟环境中安装
   - 升级pip：`pip install --upgrade pip`

3. **某些包安装失败**
   - 确保有足够的磁盘空间
   - 尝试使用镜像源：`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements-xxx.txt`

### 获取帮助

如果遇到安装问题，请提供以下信息：
- 操作系统版本
- Python版本
- 错误信息完整输出
- 使用的requirements文件 