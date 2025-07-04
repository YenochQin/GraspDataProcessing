# PYTHONPATH 设置测试脚本

本目录包含用于测试 GraspDataProcessing 包路径设置的脚本。

## 文件说明

- `test_env_setup.sh` - Shell脚本，模拟sbatch环境设置
- `test_pythonpath.py` - Python测试脚本，验证包导入
- `test_README.md` - 本说明文件

## 使用方法

### 方法1：完整测试（推荐）
```bash
# 在 ml_csf_choosing 目录下运行
chmod +x test_env_setup.sh
./test_env_setup.sh
```

### 方法2：仅测试Python部分
```bash
# 手动设置环境变量
export PYTHONPATH="$(pwd)/../../../src:$PYTHONPATH"

# 运行Python测试
python test_pythonpath.py
```

### 方法3：快速验证
```bash
# 一行命令测试
PYTHONPATH="$(pwd)/../../../src:$PYTHONPATH" python -c "import graspdataprocessing; print('✅ 导入成功')"
```

## 测试内容

测试脚本会检查：

1. **环境变量** - PYTHONPATH是否正确设置
2. **路径验证** - 源码目录是否存在
3. **文件检查** - 关键包文件是否存在
4. **包导入** - graspdataprocessing及其子模块是否能正常导入
5. **路径解析** - Python搜索路径是否包含正确目录

## 预期输出

成功时会显示：
```
[2024-01-01 12:00:00] 🎉 测试成功！PYTHONPATH 设置正常工作
```

失败时会显示详细的错误信息和调试信息。

## 故障排除

如果测试失败，请检查：

1. **目录结构** - 确保 `../../../src/graspdataprocessing/` 目录存在
2. **文件权限** - 确保有读取源码目录的权限
3. **Python环境** - 确保使用正确的Python环境
4. **相对路径** - 确保在正确的目录下运行测试脚本

## 与sbatch脚本的关系

这些测试脚本模拟了 `csfs_choosing_SCF_cal_ml_choosing.sh` 中的环境设置：

```bash
# sbatch脚本中的设置
GRASP_DATA_PROCESSING_ROOT="${cal_dir}/../../../src"
export PYTHONPATH="${GRASP_DATA_PROCESSING_ROOT}:${PYTHONPATH}"
```

测试通过后，sbatch脚本中的Python脚本应该能正常导入包。 