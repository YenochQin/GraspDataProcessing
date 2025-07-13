# ML CSF Choosing Scripts 使用指南

本目录提供了两个便于运行机器学习CSF选择程序的脚本。

## 脚本说明

### 1. `quick_run.sh` - 交互式快速运行器（推荐）

最简单的使用方式，提供交互式菜单：

```bash
./quick_run.sh
```

特点：
- 交互式菜单选择
- 自动检查环境和配置
- 简单易用
- 支持运行单个程序或所有程序

### 2. `run_ml.sh` - 命令行运行器（高级）

提供命令行参数控制：

```bash
# 显示帮助
./run_ml.sh --help

# 运行特定程序
./run_ml.sh train           # 运行训练程序
./run_ml.sh initial         # 运行初始化程序
./run_ml.sh choosing        # 运行选择程序

# 运行所有程序
./run_ml.sh all

# 在指定目录运行
./run_ml.sh -d /path/to/work train

# 详细输出模式
./run_ml.sh -v train
```

## 程序说明

1. **`initial_csfs.py`** - 初始化CSFs
   - 准备初始的CSF配置
   - 生成描述符数据

2. **`choosing_csfs.py`** - 选择CSFs
   - 根据策略选择重要的CSF
   - 准备下一轮计算数据

3. **`train.py`** - 机器学习训练
   - 训练机器学习模型
   - 评估模型性能
   - 智能选择下一轮CSF

## 使用前提

1. **环境要求**：
   - Python 3.12+
   - 已安装 graspdataprocessing 包
   - 相关依赖包（numpy, pandas, scikit-learn等）

2. **工作目录要求**：
   - 包含 `config.toml` 配置文件
   - 包含必要的数据文件

3. **运行位置**：
   - 必须在包含 `config.toml` 的目录中运行脚本
   - 例如：`/home/computer-0-2/4thdd/GdI/cv6odd1_j3as5/`

## 使用示例

### 快速开始（推荐）

```bash
# 1. 进入工作目录
cd /home/computer-0-2/4thdd/GdI/cv6odd1_j3as5/

# 2. 运行快速脚本
/home/computer-0-2/AppFiles/GraspDataProcessing/tests/ml_csf_choosing/quick_run.sh

# 3. 选择要运行的程序（1-4）
```

### 命令行方式

```bash
# 进入工作目录
cd /home/computer-0-2/4thdd/GdI/cv6odd1_j3as5/

# 运行训练程序
/home/computer-0-2/AppFiles/GraspDataProcessing/tests/ml_csf_choosing/run_ml.sh train

# 运行所有程序
/home/computer-0-2/AppFiles/GraspDataProcessing/tests/ml_csf_choosing/run_ml.sh all
```

### 创建别名（可选）

为了更方便使用，可以在 `~/.bashrc` 中添加别名：

```bash
# 添加到 ~/.bashrc
alias ml-quick='/home/computer-0-2/AppFiles/GraspDataProcessing/tests/ml_csf_choosing/quick_run.sh'
alias ml-run='/home/computer-0-2/AppFiles/GraspDataProcessing/tests/ml_csf_choosing/run_ml.sh'

# 重新加载配置
source ~/.bashrc

# 使用别名
ml-quick        # 交互式运行
ml-run train    # 直接运行训练
```

## 故障排除

1. **权限错误**：
   ```bash
   chmod +x /home/computer-0-2/AppFiles/GraspDataProcessing/tests/ml_csf_choosing/*.sh
   ```

2. **Python模块未找到**：
   - 确保在正确的Python环境中
   - 检查 graspdataprocessing 包是否已安装

3. **配置文件未找到**：
   - 确保在包含 `config.toml` 的目录中运行脚本

4. **路径问题**：
   - 脚本会自动设置 PYTHONPATH
   - 如果仍有问题，请检查项目结构

## 注意事项

- 运行前确保所有依赖文件存在
- 建议在运行前检查 `config.toml` 配置
- 长时间运行的程序建议使用 `nohup` 或 `screen`
- 运行日志会显示在终端，建议保存重要输出
