# 步骤级断点重启机制实现

**修改日期：** 2025-07-13  
**修改人员：** Claude Code Assistant  
**修改类型：** 功能增强和断点重启机制

## 修改概览

实现了细粒度的步骤级断点重启机制，允许用户指定在特定`cal_loop_num`下执行特定的计算步骤，实现了：
1. 步骤级精确控制
2. 智能步骤跳过机制
3. 已完成步骤检测
4. 灵活的起始和结束步骤配置
5. 目标循环指定功能

## 计算步骤结构分析

### 主要计算步骤
1. **initial_csfs** - 初始化CSFs文件数据
2. **choosing_csfs** - 组态选择处理
3. **mkdisks** - 创建计算磁盘
4. **rangular** - 角系数计算
5. **rwfnestimate** - 波函数估计
6. **rmcdhf/rci** - 自洽场/组态相互作用计算
7. **rsave** - 保存计算结果
8. **jj2lsj** - jj到LSJ耦合转换
9. **rlevels** - 能级数据生成
10. **train** - 机器学习训练

### 步骤依赖关系
```
initial_csfs → choosing_csfs → mkdisks → rangular → rwfnestimate 
    ↓
rmcdhf/rci → rsave → jj2lsj → rlevels → train
```

## 新增配置参数

### config.toml中的步骤控制配置
```toml
# 步骤级断点重启控制
[step_control]
enable_step_control = false      # 是否启用步骤级控制
target_loop = 0                  # 目标循环编号（0表示使用当前cal_loop_num）
start_step = "auto"              # 起始步骤
end_step = "auto"                # 结束步骤（auto表示执行到最后）
skip_completed_steps = true      # 是否跳过已完成的步骤
```

### 参数说明

#### `enable_step_control`
- **类型：** boolean
- **默认值：** false
- **说明：** 是否启用步骤级精确控制。设为false时按原有逻辑执行

#### `target_loop`
- **类型：** integer
- **默认值：** 0
- **说明：** 
  - 0：使用当前cal_loop_num，正常执行所有循环
  - >0：只执行指定的循环编号，其他循环跳过

#### `start_step`
- **类型：** string
- **可选值：** auto, initial_csfs, choosing_csfs, mkdisks, rangular, rwfnestimate, rmcdhf, rci, rsave, jj2lsj, rlevels, train
- **默认值：** auto
- **说明：** 
  - auto：从当前循环的第一步开始
  - 其他值：从指定步骤开始执行

#### `end_step`
- **类型：** string
- **可选值：** auto, initial_csfs, choosing_csfs, mkdisks, rangular, rwfnestimate, rmcdhf, rci, rsave, jj2lsj, rlevels, train
- **默认值：** auto
- **说明：** 
  - auto：执行到当前循环的最后一步
  - 其他值：执行到指定步骤后停止

#### `skip_completed_steps`
- **类型：** boolean
- **默认值：** true
- **说明：** 是否检查并跳过已完成的步骤（基于输出文件检测）

## 实现的核心功能

### 1. 步骤执行控制函数

#### `check_step_should_run(current_step, current_loop)`
检查当前步骤是否应该执行：
- 验证目标循环匹配
- 检查起始步骤约束
- 标记结束步骤提示

#### `check_should_stop_after_step(current_step)`
检查步骤完成后是否应该停止：
- 在指定的end_step后停止执行
- 优雅退出并记录日志

#### `check_step_completed(step_name, loop_num, conf_name)`
检查步骤是否已完成：
- 基于输出文件检测步骤完成状态
- 支持智能跳过已完成步骤

### 2. 文件完成状态检测

每个步骤的完成检测逻辑：

| 步骤 | 检测文件 | 检测逻辑 |
|------|----------|----------|
| mkdisks | disks | 文件存在 |
| rwfnestimate | rwfn.inp | 文件存在 |
| rmcdhf/rci | rwfn.out, rmix.out 或 *.cm | 任一文件存在 |
| rsave | *.w, *.c, *.m | 所有文件存在 |
| jj2lsj | *.lsj.lbl | 文件存在 |
| rlevels | *.level | 文件存在 |

### 3. 智能步骤跳转

步骤控制逻辑的优先级：
1. **步骤启用检查**：如果未启用步骤控制，正常执行
2. **循环匹配检查**：只在目标循环中执行步骤控制
3. **起始步骤检查**：跳过start_step之前的所有步骤
4. **完成状态检查**：如果启用，跳过已完成的步骤
5. **结束步骤检查**：在end_step后停止执行

## 使用场景和示例

### 场景1：重新运行失败的rmcdhf步骤
```toml
[step_control]
enable_step_control = true
target_loop = 3
start_step = "rmcdhf"
end_step = "rmcdhf"
skip_completed_steps = true
```

### 场景2：只执行机器学习训练
```toml
[step_control]
enable_step_control = true
target_loop = 0    # 当前循环
start_step = "train"
end_step = "train"
skip_completed_steps = false
```

### 场景3：跳过GRASP计算，只做后处理
```toml
[step_control]
enable_step_control = true
target_loop = 2
start_step = "jj2lsj"
end_step = "train"
skip_completed_steps = true
```

### 场景4：测试单个GRASP步骤
```toml
[step_control]
enable_step_control = true
target_loop = 1
start_step = "rangular"
end_step = "rangular"
skip_completed_steps = false
```

### 场景5：部分重新计算（从rwfnestimate开始）
```toml
[step_control]
enable_step_control = true
target_loop = 4
start_step = "rwfnestimate"
end_step = "auto"
skip_completed_steps = false
```

## 日志输出增强

### 步骤控制状态日志
```
[2025-07-13 10:30:00] 步骤控制配置:
[2025-07-13 10:30:00]   启用步骤控制: true
[2025-07-13 10:30:00]   目标循环: 3
[2025-07-13 10:30:00]   起始步骤: rmcdhf
[2025-07-13 10:30:00]   结束步骤: rsave
[2025-07-13 10:30:00]   跳过已完成步骤: true
```

### 步骤跳过日志
```
[2025-07-13 10:30:05] ⏭️ 跳过步骤: mkdisks (根据步骤控制配置)
[2025-07-13 10:30:05] ⏭️ 跳过已完成的步骤: rwfnestimate (发现文件: rwfn.inp)
```

### 步骤停止日志
```
[2025-07-13 10:45:30] 🎯 达到结束步骤: rsave，将在此步骤后停止
[2025-07-13 10:45:35] 🛑 根据配置在rsave步骤后停止执行
```

## 技术实现细节

### 1. 配置参数读取
使用`csfs_ml_choosing_config_load.py`动态读取步骤控制配置：
```bash
enable_step_control=$(csfs_ml_choosing_config_load.py get step_control.enable_step_control 2>&1)
target_loop=$(csfs_ml_choosing_config_load.py get step_control.target_loop 2>&1)
```

### 2. 步骤包装模式
每个计算步骤都被包装在控制检查中：
```bash
if check_step_should_run "rmcdhf" "$loop"; then
    if ! check_step_completed "rmcdhf" "$loop" "$conf"; then
        # 实际的计算步骤
        safe_grasp_execute "rmcdhf_mem_mpi" "$input_commands" ...
    fi
    
    # 检查是否应该停止
    if check_should_stop_after_step "rmcdhf"; then
        log_with_timestamp "🛑 根据配置在rmcdhf步骤后停止执行"
        exit 0
    fi
else
    log_with_timestamp "⏭️ 跳过步骤: rmcdhf (根据步骤控制配置)"
fi
```

### 3. 循环级和步骤级控制结合
- 外层循环控制（原有）：基于`continue_cal`的循环断点重启
- 内层步骤控制（新增）：循环内的细粒度步骤控制

## 向后兼容性

### 完全兼容
- **默认行为不变**：`enable_step_control = false`时完全按原逻辑执行
- **配置文件兼容**：新增字段有默认值，现有配置无需修改
- **脚本接口不变**：运行方式和参数完全相同

### 性能影响
- **最小开销**：禁用时仅增加几个布尔检查
- **文件检测优化**：只在需要时进行文件存在性检查
- **日志增强**：提供更详细的执行状态信息

## 错误处理和安全性

### 错误处理
- **无效步骤名**：自动忽略并记录警告
- **文件访问错误**：优雅降级，不跳过步骤
- **配置读取失败**：使用默认值继续执行

### 安全性考虑
- **步骤依赖检查**：不强制检查步骤依赖（用户自行负责）
- **数据一致性**：跳过步骤可能导致数据不一致，需用户谨慎使用
- **备份建议**：建议在使用前备份重要数据

## 维护和扩展

### 添加新步骤
1. 在`check_step_should_run`函数中添加新步骤的case
2. 在`check_step_completed`函数中添加完成检测逻辑
3. 在实际步骤执行处添加控制包装
4. 更新文档中的步骤列表

### 自定义完成检测
可以根据实际需要修改`check_step_completed`函数中的文件检测逻辑。

## 文件修改清单

### 修改的文件：
- ✅ `/scripts/config.toml` - 添加步骤控制配置区块
- ✅ `/scripts/run_script.sh` - 实现步骤级控制逻辑

### 新增功能模块：
- **步骤控制配置读取** - 动态读取步骤控制参数
- **步骤执行检查函数** - 核心控制逻辑
- **步骤完成检测机制** - 基于文件的智能跳过
- **增强日志系统** - 详细的步骤状态记录

### 代码行数统计：
- **新增代码：** ~200行（包括注释和空行）
- **修改代码：** ~50行（步骤包装）
- **配置参数：** 5个新字段

---

**修复状态：** ✅ 已完成  
**测试状态：** ⏳ 待验证  
**部署状态：** ✅ 可立即使用  
**向后兼容：** ✅ 完全兼容现有工作流程  
**功能复杂度：** 🔧 中等（需要理解步骤依赖关系）