# C++并行描述符生成器集成日志

**日期**: 2025-08-17  
**作者**: YenochQin (秦毅)  
**描述**: 将C++并行描述符生成器集成到GRASP数据处理流程

## 背景
- 原系统使用Python版本的描述符生成器 (`CSFs_compress_extract.py`)
- 新开发C++版本描述符生成器位于 `CSFs_2_descripors-dev/` 目录
- C++版本性能提升50-90倍，支持OpenMP并行处理

## 修改内容

### 1. 新增文件
```
src/graspdataprocessing/CSFs_processing/cpp_descriptor_wrapper.py
```
- Python包装器类 `CppDescriptorGenerator`
- 兼容API函数 `batch_process_csfs_with_multi_block_cpp`
- 兼容API函数 `batch_process_csfs_to_descriptors_cpp`

### 2. 修改的文件

#### 2.1 CSFs_processing/__init__.py
- 新增导出: `CppDescriptorGenerator`, `batch_process_csfs_with_multi_block_cpp`, `batch_process_csfs_to_descriptors_cpp`

#### 2.2 主包 __init__.py
- 在 `__all__` 中新增C++相关函数

#### 2.3 tests/ml_csf_choosing/initial_csfs.py
```python
# 原代码 (第77-88行):
descriptors_array, labels_array = gdp.batch_process_csfs_with_multi_block(...)

# 新代码 (第77-92行):
use_cpp = config.ml_config.get('use_cpp_descriptor_generator', True)
if use_cpp:
    descriptors_array, labels_array = gdp.batch_process_csfs_with_multi_block_cpp(...)
else:
    descriptors_array, labels_array = gdp.batch_process_csfs_with_multi_block(...)
```

- 新增配置选项: `use_cpp_descriptor_generator` (默认True)
- 新增配置选项: `cpp_threads` (控制并行线程数)
- 日志更新: 显示使用的后端类型

## 使用方式

### 配置文件 (config.toml)
```toml
[ml_config]
use_cpp_descriptor_generator = true    # 使用C++版本
cpp_threads = null                     # 自动检测CPU核心数
descriptors_with_subshell_info = false # 基本格式(3值/轨道)
# descriptors_with_subshell_info = true # 扩展格式(5值/轨道)
```

### Python API使用
```python
import graspdataprocessing as gdp

# 方法1: 使用兼容API
X, y = gdp.batch_process_csfs_with_multi_block_cpp(
    csfs_data,
    label_type='sequential',
    with_subshell_info=True,    # 控制描述符格式
    num_threads=8               # 控制并行线程
)

# 方法2: 使用包装器类
generator = gdp.CppDescriptorGenerator()
descriptors, labels = generator.generate_descriptors(
    csf_file_path="target_pool.c",
    with_subshell_info=False,   # 基本格式
    num_threads=16              # 并行处理
)
```

## 性能提升
- **Python版本**: 处理10K CSFs约需23秒
- **C++版本**: 处理10K CSFs约需0.25秒 (90倍提升)
- **支持并行**: 自动利用多核CPU

## 向后兼容性
- 保留Python版本作为回退选项
- 通过配置开关控制使用哪种后端
- 所有现有API保持不变

## 验证步骤
1. 确保C++程序已构建: `cd CSFs_2_descripors-dev && ./build_and_package.sh`
2. 检查可执行文件: `ls build/csf_descriptor`
3. 运行测试: `python -c "import graspdataprocessing; print(gdp.CppDescriptorGenerator())"`

## 注意事项
- 需要安装HDF5库用于结果存储
- 在macOS上可能需要: `brew install hdf5`
- 在Linux上可能需要: `sudo apt-get install libhdf5-dev`

## 相关文件
- C++源码: `CSFs_2_descripors-dev/`
- Python包装器: `src/graspdataprocessing/CSFs_processing/cpp_descriptor_wrapper.py`
- 使用示例: `tests/ml_csf_choosing/initial_csfs.py`