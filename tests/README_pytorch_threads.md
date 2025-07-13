# PyTorch线程数配置修复

## 问题描述

之前的代码中存在一个bug：虽然`config.toml`文件中设置了`pytorch_threads`参数，但训练代码没有读取这个配置，而是硬编码使用了默认的线程数计算逻辑。

## 修复内容

修改了`src/graspdataprocessing/machine_learning_module/machine_learning_training.py`文件中的线程数设置逻辑：

### 修复前（第115-132行）：
```python
# CPU训练优化配置
if not torch.cuda.is_available():
    # 获取系统CPU核心数
    cpu_count = os.cpu_count() or 4
    optimal_threads = min(32, cpu_count)  # 硬编码为32
    
    # 设置PyTorch线程数
    torch.set_num_threads(optimal_threads)
    # ...
```

### 修复后：
```python
# CPU训练优化配置
if not torch.cuda.is_available():
    # 获取系统CPU核心数
    cpu_count = os.cpu_count() or 4
    
    # 从配置文件读取PyTorch线程数，如果未设置则使用默认值
    config_threads = getattr(config, 'pytorch_threads', None)
    if config_threads is not None:
        try:
            config_threads = int(config_threads)
            optimal_threads = min(config_threads, cpu_count)  # 不超过系统核心数
            logger.info(f"使用配置文件中的PyTorch线程数: {config_threads}")
        except (ValueError, TypeError):
            logger.warning(f"配置文件中的pytorch_threads值无效: {config_threads}，使用默认值")
            optimal_threads = min(32, cpu_count)
    else:
        optimal_threads = min(32, cpu_count)  # 默认最多使用32线程
        logger.info(f"配置文件中未设置pytorch_threads，使用默认值")
    
    # 设置PyTorch线程数
    torch.set_num_threads(optimal_threads)
    # ...
```

## 修复后的行为

1. **读取配置**：程序会首先尝试从`config.toml`中读取`pytorch_threads`参数
2. **值验证**：确保配置值是有效的整数
3. **限制检查**：确保设置的线程数不超过系统CPU核心数
4. **详细日志**：显示配置值、计算过程和最终使用的线程数
5. **向下兼容**：如果配置文件中没有设置该参数，使用默认逻辑

## 新的日志输出

修复后，你会看到更详细的线程配置信息：

```
启用CPU多线程优化:
- 系统CPU核心数: 48
- 配置的线程数: 16
- 实际PyTorch线程数: 16
- 建议配置: max_epochs=150, batch_size=4096, hidden_size=96
```

## 配置示例

在`config.toml`中设置：

```toml
# 设置PyTorch使用的CPU线程数
pytorch_threads = 16

# 如果你有48核CPU，可以设置更高的值
pytorch_threads = 32

# 如果设置的值超过系统核心数，会自动限制到系统核心数
pytorch_threads = 64  # 在48核系统上会自动限制为48
```

## 测试方法

1. **修改配置文件**：
   ```bash
   # 编辑config.toml
   vim config.toml
   # 修改 pytorch_threads = 16
   ```

2. **运行训练**：
   ```bash
   python train.py
   ```

3. **观察日志输出**：
   ```
   - 配置的线程数: 16
   - 实际PyTorch线程数: 16
   ```

## 验证修复

使用提供的测试脚本验证修复是否生效：

```bash
python tests/test_pytorch_threads.py
```

该脚本会测试各种配置值并验证设置是否正确应用。

## 注意事项

1. **线程数限制**：设置的线程数不会超过系统CPU核心数
2. **无效值处理**：如果配置值无效（如非数字），会使用默认值并记录警告
3. **环境变量**：除了PyTorch线程数，还会设置相关的环境变量（OMP_NUM_THREADS等）
4. **GPU模式**：如果检测到GPU，这些CPU优化配置不会生效

## 性能建议

- **一般情况**：设置为CPU核心数的50-75%
- **高内存系统**：可以设置为CPU核心数
- **共享服务器**：设置为较低值避免影响其他用户
- **调试模式**：设置为较低值（如4-8）便于调试