# MPI临时文件路径配置功能

## 功能说明

新增了`mpi_tmp_path`配置参数，允许在`config.toml`中指定MPI临时文件的存储路径，无需在每台服务器上单独修改`mkdisks`脚本。

## 修改内容

### 1. 增强的mkdisks脚本

**文件**: `scripts/mkdisks`

**新功能**:
- 支持两个参数: `mkdisks <processor_count> [mpi_tmp_path]`
- 第二个参数可选，指定mpi_tmp的基础路径
- 自动创建目录、清理旧文件、详细的日志输出

**使用示例**:
```bash
# 使用默认路径（当前目录下的mpi_tmp）
./mkdisks 4

# 使用指定路径
./mkdisks 4 /home/workstation3/caltmp

# 使用其他服务器路径
./mkdisks 8 /tmp/grasp_calc
```

### 2. 配置文件参数

**文件**: `config.toml`

**新增参数**:
```toml
# GRASP计算参数
tasks_per_node = 46
mpi_tmp_path = "/home/workstation3/caltmp"  # MPI临时文件存储路径，如果不设置则使用当前目录
```

### 3. 自动化脚本集成

**文件**: `scripts/run_script.sh`

**修改的mkdisks调用逻辑**:
```bash
# 在循环外读取mpi_tmp_path配置参数（第292行）
mpi_tmp_path=$(python "${GRASP_DATA_PROCESSING_ROOT}/scripts/csfs_ml_choosing_config_load.py" get mpi_tmp_path 2>&1)

# 在每个计算循环中使用已读取的配置参数
if [[ -n "$mpi_tmp_path" && "$mpi_tmp_path" != "null" && ! "$mpi_tmp_path" =~ ^ERROR: ]]; then
    log_with_timestamp "使用配置的mpi_tmp路径: $mpi_tmp_path"
    safe_grasp_execute "mkdisks" "" mkdisks ${processor} "$mpi_tmp_path"
else
    log_with_timestamp "未配置mpi_tmp_path或读取失败，使用默认路径（当前目录）"
    safe_grasp_execute "mkdisks" "" mkdisks ${processor}
fi
```

## 使用方法

### 配置不同服务器

**workstation2**:
```toml
mpi_tmp_path = "/home/workstation2/caltmp"
```

**workstation3**:
```toml
mpi_tmp_path = "/home/workstation3/caltmp"
```

**本地测试**:
```toml
mpi_tmp_path = "/tmp/grasp_calc"
```

**使用默认（当前目录）**:
```toml
# 不设置mpi_tmp_path参数，或设置为空
```

### 生成的disks文件

**有配置时** (`mpi_tmp_path = "/home/workstation3/caltmp"`):
```
'/path/to/working/directory'
'/home/workstation3/caltmp/mpi_tmp'
'/home/workstation3/caltmp/mpi_tmp'
'/home/workstation3/caltmp/mpi_tmp'
'/home/workstation3/caltmp/mpi_tmp'
```

**无配置时**（使用默认）:
```
'/path/to/working/directory'
'/path/to/working/directory/mpi_tmp'
'/path/to/working/directory/mpi_tmp'
'/path/to/working/directory/mpi_tmp'
'/path/to/working/directory/mpi_tmp'
```

## 优势

1. **服务器无关**: 无需为每台服务器单独修改脚本
2. **配置集中**: 所有路径配置都在config.toml中
3. **向下兼容**: 未配置时自动使用默认行为
4. **错误处理**: 自动创建目录、清理旧文件
5. **详细日志**: 显示使用的路径和操作过程
6. **循环兼容**: 每次计算循环都正确执行mkdisks，避免配置读取位置错误

## 测试方法

使用提供的测试脚本验证配置读取功能:

```bash
python tests/test_mkdisks_config.py
```

## 故障排除

### 常见问题

1. **配置参数未生效**:
   - 检查config.toml中的参数名称是否正确：`mpi_tmp_path`
   - 确认路径格式正确，使用绝对路径

2. **目录权限问题**:
   - 确保指定的基础路径有写权限
   - mkdisks会自动创建不存在的目录

3. **路径不存在**:
   - mkdisks会自动创建基础目录
   - 检查日志输出确认创建是否成功

### 调试方法

1. **手动测试mkdisks**:
   ```bash
   cd /path/to/working/directory
   ./mkdisks 4 /your/test/path
   cat disks  # 检查生成的内容
   ```

2. **检查配置读取**:
   ```bash
   python scripts/csfs_ml_choosing_config_load.py get mpi_tmp_path
   ```

3. **查看run_script.sh日志**:
   ```bash
   grep "mpi_tmp路径" *.log
   ```

4. **验证配置读取位置**:
   - 确保mpi_tmp_path在循环外读取（第292行）
   - 确保每个计算循环都使用正确的配置变量

## 兼容性说明

- **向下兼容**: 旧的config.toml文件无需修改即可正常工作
- **脚本兼容**: 可以同时手动调用新版mkdisks脚本
- **路径灵活**: 支持任意有效的绝对路径

## 最佳实践

1. **路径选择**: 使用快速存储设备（如SSD）作为临时文件路径
2. **空间管理**: 确保路径有足够空间存储MPI临时文件
3. **清理策略**: mkdisks会自动清理旧的mpi_tmp内容
4. **权限设置**: 确保计算用户对指定路径有读写权限