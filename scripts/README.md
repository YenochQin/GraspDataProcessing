# GRASP数据处理工具

这个目录包含了用于GRASP数据处理项目的命令行工具。

## 安装

### 方法1：使用安装脚本（推荐）

```bash
# 进入scripts目录
cd GraspDataProcessing/scripts

# 运行安装脚本
python install_tools.py
```

### 方法2：手动安装

1. 将 `grasp_tools.py` 复制到系统PATH中的目录
2. 创建可执行脚本调用该Python文件

## 使用方法

安装完成后，你可以在任何目录使用以下命令：

### 获取配置值

```bash
# 获取简单键值
grasp-tools get continue_cal

# 获取嵌套键值
grasp-tools get model_params.n_estimators

# 指定配置文件
grasp-tools get continue_cal -f /path/to/config.toml
```

### 设置配置值

```bash
# 设置简单键值
grasp-tools set cal_loop_num 2

# 设置嵌套键值
grasp-tools set model_params.random_state 42

# 指定配置文件
grasp-tools set continue_cal false -f /path/to/config.toml
```

## 在Shell脚本中使用

```bash
#!/bin/bash

# 获取配置值到变量
cal_status=$(grasp-tools get continue_cal)
loop_num=$(grasp-tools get cal_loop_num)

echo "继续计算: $cal_status"
echo "循环次数: $loop_num"

# 根据配置值做判断
if [ "$cal_status" = "true" ]; then
    echo "继续计算..."
    grasp-tools set cal_loop_num $((loop_num + 1))
else
    echo "停止计算"
fi
```

## 环境变量配置

安装后，如果工具无法直接使用，需要将安装目录添加到PATH环境变量：

### macOS/Linux

在 `~/.bashrc` 或 `~/.zshrc` 中添加：

```bash
export PATH="$PATH:$HOME/.local/bin"
```

然后重新加载配置：

```bash
source ~/.bashrc  # 或 source ~/.zshrc
```

### Windows

1. 打开系统属性 -> 环境变量
2. 在用户变量PATH中添加：`%USERPROFILE%\AppData\Local\Programs\grasp-tools`

## 故障排除

1. **命令未找到**：检查PATH环境变量是否正确设置
2. **权限错误**：确保脚本有执行权限
3. **Python路径错误**：确保使用正确的Python解释器

## 工具特性

- ✅ 支持嵌套键访问（如 `model_params.n_estimators`）
- ✅ 自动处理布尔值转换
- ✅ 支持自定义配置文件路径
- ✅ 跨平台兼容（Windows/macOS/Linux）
- ✅ 友好的错误提示 