# 步骤级断点重启使用指南

## 快速开始

### 1. 启用步骤控制
在`config.toml`中设置：
```toml
[step_control]
enable_step_control = true
```

### 2. 常用配置示例

#### 重新运行失败的步骤
```toml
[step_control]
enable_step_control = true
target_loop = 3              # 在第3次循环
start_step = "rmcdhf"        # 从rmcdhf开始
end_step = "rmcdhf"          # 只运行rmcdhf
skip_completed_steps = false # 强制重新运行
```

#### 只运行机器学习训练
```toml
[step_control]
enable_step_control = true
target_loop = 0              # 当前循环
start_step = "train"         # 只运行train
end_step = "train"
skip_completed_steps = false
```

#### 跳过已完成步骤继续计算
```toml
[step_control]
enable_step_control = true
target_loop = 2
start_step = "auto"          # 从头开始
end_step = "auto"            # 到最后
skip_completed_steps = true  # 自动跳过已完成步骤
```

## 所有可用步骤

### 完整步骤列表
1. `initial_csfs` - 初始化CSFs文件数据
2. `choosing_csfs` - 组态选择处理  
3. `mkdisks` - 创建计算磁盘
4. `rangular` - 角系数计算
5. `rwfnestimate` - 波函数估计
6. `rmcdhf` - 自洽场计算（第1次循环）
7. `rci` - 组态相互作用计算（第2+次循环）
8. `rsave` - 保存计算结果
9. `jj2lsj` - jj到LSJ耦合转换
10. `rlevels` - 能级数据生成
11. `train` - 机器学习训练

### 步骤依赖关系
```
initial_csfs → choosing_csfs → mkdisks → rangular → rwfnestimate 
    ↓
rmcdhf/rci → rsave → jj2lsj → rlevels → train
```

## 配置参数详解

### `enable_step_control`
- **true**: 启用步骤级控制
- **false**: 正常执行（默认）

### `target_loop`
- **0**: 所有循环正常执行（默认）
- **>0**: 只在指定循环中应用步骤控制

### `start_step`
- **"auto"**: 从循环开始执行（默认）
- **步骤名**: 从指定步骤开始

### `end_step`
- **"auto"**: 执行到循环结束（默认）
- **步骤名**: 执行到指定步骤后停止

### `skip_completed_steps`
- **true**: 自动跳过已完成步骤（默认）
- **false**: 强制重新执行所有步骤

## 实际使用场景

### 场景1: rmcdhf计算失败重启
```bash
# 1. 修改config.toml
[step_control]
enable_step_control = true
target_loop = 2
start_step = "rmcdhf"
end_step = "rmcdhf"
skip_completed_steps = false

# 2. 重新运行脚本
sbatch run_script.sh
```

### 场景2: 跳过GRASP计算，只做机器学习
```bash
# 适用于GRASP计算已完成，只想重新训练模型
[step_control]
enable_step_control = true
target_loop = 3
start_step = "train"
end_step = "train"
skip_completed_steps = false
```

### 场景3: 智能断点续算
```bash
# 计算中断后，自动跳过已完成步骤继续
[step_control]
enable_step_control = true
target_loop = 0
start_step = "auto"
end_step = "auto"
skip_completed_steps = true
```

## 日志查看

### 查看步骤控制状态
```bash
grep "步骤控制配置" *.log
grep "⏭️" *.log          # 查看跳过的步骤
grep "🛑" *.log          # 查看停止点
```

### 查看步骤执行情况
```bash
grep "================" *.log    # 查看步骤开始
grep "✅" *.log                  # 查看完成步骤
grep "❌" *.log                  # 查看失败步骤
```

## 注意事项

### ⚠️ 重要提醒
1. **步骤依赖**: 跳过某些步骤可能导致后续步骤失败
2. **数据一致性**: 强制重新执行可能覆盖现有结果
3. **文件检查**: 完成检测基于输出文件，可能不够精确

### 💡 最佳实践
1. **测试模式**: 先在小范围测试步骤控制
2. **备份数据**: 使用前备份重要计算结果
3. **日志监控**: 密切关注日志输出确认步骤执行
4. **逐步调试**: 一次只控制一个步骤进行调试

## 故障排除

### 步骤没有跳过？
- 检查`enable_step_control = true`
- 确认`target_loop`设置正确
- 查看日志中的步骤控制配置

### 意外停止？
- 检查`end_step`配置
- 查看日志中的停止消息

### 步骤重复执行？
- 设置`skip_completed_steps = true`
- 检查输出文件是否存在

## 恢复正常模式

当调试完成后，恢复正常执行：
```toml
[step_control]
enable_step_control = false
```

或者直接注释掉整个配置块。