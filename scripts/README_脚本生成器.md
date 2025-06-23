# GRASP计算脚本生成器

## 概述

这个脚本生成器可以帮助您快速生成GRASP原子结构计算所需的配置文件和运算脚本，支持机器学习方法进行组态选择。

## 功能特性

- 🎯 **自动生成配置文件** (`config.toml`)
- 🔧 **生成SLURM计算脚本** (`csfs_choosing_SCF_cal_ml_choosing.sh`)
- 📝 **生成初始化脚本** (`initial_csfs.py`)
- 📦 **生成依赖文件** (`requirements.txt`)
- 📖 **生成项目说明** (`README.md`)
- 🚀 **支持批量生成** 多个原子的计算脚本

## 文件结构

```
GraspDataProcessing/scripts/
├── generate_calculation_scripts.py  # 主生成器脚本
├── example_usage.py                 # 使用示例
└── README_脚本生成器.md             # 本说明文档
```

## 快速开始

### 1. 基本使用

```bash
# 进入脚本目录
cd GraspDataProcessing/scripts

# 生成默认配置（GdI原子）
python generate_calculation_scripts.py

# 生成自定义配置
python generate_calculation_scripts.py \
    --atom Ce \
    --conf cv4odd1as2_odd1 \
    --spectral-term "5s(2).4d(10)1S0_1S.5p(6).6s(2).4f(2)3H_3H.5d_4D" \
    --root-path "/home/workstation3/caldata/Ce/cvodd1/as2_odd1" \
    --job-name "CeoddImlcias2_odd1" \
    --output-dir "./ce_calculation"
```

### 2. 运行示例

```bash
# 运行所有示例
python example_usage.py

# 查看生成的示例文件
ls -la gdI_example/
ls -la custom_atom_example/
ls -la batch_example/
```

## 命令行参数

| 参数 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `--output-dir` | `-o` | `.` | 输出目录 |
| `--atom` | - | `GdI` | 原子符号 |
| `--conf` | - | `cv4odd1as3_odd1` | 组态名称 |
| `--spectral-term` | - | `5s(2).4d(10)1S0_1S.5p(6).6s(2).4f(7)8S0_8S.5d_7D` | 光谱项 |
| `--root-path` | - | `/home/workstation3/caldata/GdI/cvodd1/as3_odd1` | 根路径 |
| `--job-name` | - | `GdIoddImlcias4_odd2` | SLURM作业名称 |
| `--tasks-per-node` | - | `46` | 每节点任务数 |
| `--partition` | - | `work3` | SLURM分区 |
| `--project-name` | - | `GRASP计算项目` | 项目名称 |

## 生成的文件说明

### 1. config.toml
主要的配置文件，包含：
- 原子和组态信息
- 计算参数设置
- 机器学习模型参数
- 收敛性检查参数

### 2. csfs_choosing_SCF_cal_ml_choosing.sh
SLURM计算脚本，包含：
- SLURM作业配置
- GRASP环境设置
- 计算流程控制
- 错误处理机制

### 3. initial_csfs.py
CSFs初始化脚本，用于：
- 数据预处理
- 二进制文件生成
- 描述符计算

### 4. requirements.txt
Python依赖包列表

### 5. README.md
项目说明文档

## 高级用法

### 1. 批量生成

```python
from generate_calculation_scripts import CalculationScriptGenerator

# 定义多个原子配置
atoms_config = [
    {'atom': 'GdI', 'conf': 'cv4odd1as3_odd1', ...},
    {'atom': 'Ce', 'conf': 'cv4odd1as2_odd1', ...},
    {'atom': 'Pr', 'conf': 'cv4odd1as2_odd1', ...}
]

# 批量生成
for config in atoms_config:
    generator = CalculationScriptGenerator(f"./output/{config['atom']}")
    files = generator.generate_all_files(config_params, script_params)
```

### 2. 自定义配置模板

```python
# 自定义配置参数
config_params = {
    'atom': 'CustomAtom',
    'conf': 'custom_config',
    'spetral_term': ['custom_spectral_term'],
    'chosen_ratio': 0.15,  # 自定义初始比例
    'cutoff_value': 1e-10,  # 自定义截断值
    'model_params': {
        'n_estimators': 2000,  # 自定义模型参数
        'class_weight': {'0': 1, '1': 5}
    }
}
```

## 配置参数详解

### 计算参数
- `chosen_ratio`: 初始CSFs选择比例 (0.05-0.2)
- `cutoff_value`: 混合系数截断值 (1e-6 到 1e-12)
- `cal_loop_num`: 计算循环次数 (通常10-20)
- `std_threshold`: 收敛性检查标准差阈值

### 机器学习参数
- `n_estimators`: 随机森林树的数量 (500-2000)
- `class_weight`: 类别权重，平衡正负样本
- `random_state`: 随机种子，确保结果可重现

### SLURM参数
- `tasks_per_node`: 每节点任务数，根据计算资源调整
- `partition`: SLURM分区名称
- `job_name`: 作业名称，便于识别

## 使用流程

1. **生成脚本**
   ```bash
   python generate_calculation_scripts.py --atom YourAtom --conf YourConfig
   ```

2. **检查配置**
   ```bash
   # 检查生成的配置文件
   cat config.toml
   ```

3. **修改参数**
   ```bash
   # 根据需要修改配置文件
   vim config.toml
   ```

4. **初始化数据**
   ```bash
   python initial_csfs.py
   ```

5. **提交计算**
   ```bash
   sbatch csfs_choosing_SCF_cal_ml_choosing.sh
   ```

## 注意事项

### 环境要求
- Python 3.7+
- GRASP环境正确配置
- SLURM作业调度系统
- 必要的Python包（见requirements.txt）

### 文件路径
- 确保所有输入文件路径正确
- 检查GRASP环境变量设置
- 验证conda环境路径

### 计算资源
- 根据实际硬件调整`tasks_per_node`
- 监控内存使用情况
- 合理设置计算时间限制

## 故障排除

### 常见问题

1. **模块导入错误**
   ```bash
   # 检查Python路径
   export PYTHONPATH="/path/to/GraspDataProcessing/src:$PYTHONPATH"
   ```

2. **GRASP环境问题**
   ```bash
   # 检查GRASP模块
   module list
   module load grasp/grasp_openblas
   ```

3. **SLURM提交失败**
   ```bash
   # 检查SLURM配置
   sinfo
   squeue
   ```

### 调试建议
- 先在小规模数据上测试
- 检查日志文件输出
- 验证输入文件格式
- 监控计算收敛性

## 更新日志

- **2025-01-27**: 初始版本发布
  - 支持基本脚本生成
  - 添加批量生成功能
  - 完善错误处理机制

