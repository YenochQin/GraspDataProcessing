#!/bin/bash

# 测试脚本：模拟 sbatch 环境设置并验证 PYTHONPATH 配置

# 添加时间戳函数
log_with_timestamp() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log_with_timestamp "========== 开始环境设置测试 =========="

# 模拟 sbatch 脚本中的关键步骤
log_with_timestamp "1. 获取计算目录..."
cal_dir=${PWD}
log_with_timestamp "计算目录: $cal_dir"

log_with_timestamp "2. 设置 GraspDataProcessing 包路径..."
GRASP_DATA_PROCESSING_ROOT="${cal_dir}/../../../src"
export PYTHONPATH="${GRASP_DATA_PROCESSING_ROOT}:${PYTHONPATH}"
log_with_timestamp "设置 PYTHONPATH: $PYTHONPATH"

# 验证路径存在
log_with_timestamp "3. 验证路径是否存在..."
if [ -d "$GRASP_DATA_PROCESSING_ROOT" ]; then
    log_with_timestamp "✅ 源码目录存在: $GRASP_DATA_PROCESSING_ROOT"
    
    # 列出目录内容
    log_with_timestamp "目录内容:"
    ls -la "$GRASP_DATA_PROCESSING_ROOT"
else
    log_with_timestamp "❌ 源码目录不存在: $GRASP_DATA_PROCESSING_ROOT"
    exit 1
fi

# 检查关键文件
log_with_timestamp "4. 检查关键文件..."
PACKAGE_INIT="${GRASP_DATA_PROCESSING_ROOT}/graspdataprocessing/__init__.py"
if [ -f "$PACKAGE_INIT" ]; then
    log_with_timestamp "✅ 包初始化文件存在: $PACKAGE_INIT"
else
    log_with_timestamp "❌ 包初始化文件不存在: $PACKAGE_INIT"
    log_with_timestamp "请检查 graspdataprocessing 包的目录结构"
    exit 1
fi

# 运行 Python 测试脚本
log_with_timestamp "5. 运行 Python 测试脚本..."
python test_pythonpath.py
TEST_RESULT=$?

# 检查测试结果
log_with_timestamp "6. 检查测试结果..."
if [ $TEST_RESULT -eq 0 ]; then
    log_with_timestamp "🎉 测试成功！环境设置正确"
else
    log_with_timestamp "💥 测试失败！退出码: $TEST_RESULT"
    exit 1
fi

log_with_timestamp "========== 环境设置测试完成 ===========" 