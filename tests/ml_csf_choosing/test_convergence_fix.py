#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
测试修改后的收敛性检查功能
"""

import sys
import logging
from pathlib import Path

# 添加模块路径
sys.path.append('/home/workstation3/AppFiles/GraspDataProcessing/src')

try:
    import graspdataprocessing as gdp
    from types import SimpleNamespace
    
    # 创建模拟配置
    config = SimpleNamespace()
    config.cal_loop_num = 48  # 测试第48轮
    config.conf = "cv5odd1_j2as5"
    config.root_path = Path("/home/workstation3/caldata/GdI/cv5odd1/as5/j2")
    config.energy_std_threshold = 1e-5
    config.csfs_num_relative_std_threshold = 1e-3
    
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    print("测试1: 不传递current_calculation_csfs参数（原有逻辑）")
    try:
        result1 = gdp.check_grasp_cal_convergence(config, logger)
        print(f"结果1: {result1}")
    except Exception as e:
        print(f"错误1: {e}")
    
    print("\n测试2: 传递current_calculation_csfs参数（新逻辑）")
    try:
        # 模拟当前轮CSFs数量
        current_calculation_csfs = 400000  # 假设值
        result2 = gdp.check_grasp_cal_convergence(config, logger, current_calculation_csfs)
        print(f"结果2: {result2}")
    except Exception as e:
        print(f"错误2: {e}")
    
    print("\n测试完成")
    
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保graspdataprocessing模块可用")
except Exception as e:
    print(f"测试过程中发生错误: {e}") 