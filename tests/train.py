#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Id :train.py
@date :2025/05/25 13:53:10
@author :YenochQin (秦毅)
'''
import argparse
import logging
from types import SimpleNamespace

from pathlib import Path

import sys
import math
import numpy as np
import pandas as pd
import time
from tabulate import tabulate

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
# sys.path.append('/Users/yiqin/Documents/PythonProjects/GraspDataProcessing/src')
# sys.path.append('D:\\PythonPrograms\\GraspDataProcessing\\src')
sys.path.append('D:\\PythonProjects\\GraspDataProcessing\\src')
try:
    import graspdataprocessing as gdp
except ImportError:
    print("错误: 无法导入 graspdataprocessing 模块")
    sys.exit(1)


def main(config):
    """主程序逻辑"""
    config.file_name = f'{config.conf}_{config.cal_loop_num}'
    logger = gdp.setup_logging(config)
    logger.info("机器学习训练程序启动")
    execution_time = time.time()

    gdp.setup_directories(config)
    # 初始化结果文件
    gdp.initialize_results_file(config, logger)

    # 验证初始文件
    gdp.validate_initial_files(config, logger)

    logger.info(f"初始比例: {config.initial_ratio}")
    logger.info(f"光谱项: {config.spetral_term}")
    
    try:
        # 加载数据文件
        energy_level_data_pd, rmix_file_data, target_pool_csfs_data, raw_csfs_descriptors, cal_csfs_data, caled_csfs_indices_dict = gdp.load_data_files(config, logger)

    except Exception as e:
            logger.error(f"数据文件加载失败: {str(e)}")
            raise

    # 检查组态耦合
    cal_result = gdp.check_configuration_coupling(config, energy_level_data_pd, logger)
    logger.info("************************************************")

    if cal_result:
        # 记录能量信息
        logger.info("能级数据表格：\n%s", 
           tabulate(energy_level_data_pd, headers='keys', tablefmt='fancy_grid', showindex=False))
        logger.info("耦合正确")
        logger.info("************************************************")

        # 提取特征
        logger.info("             数据预处理")
        caled_csfs_descriptors = gdp.generate_chosen_csfs_descriptors(config, caled_csfs_indices_dict, raw_csfs_descriptors, rmix_file_data, logger)
        stay_csfs_descriptors = gdp.get_stay_descriptors(raw_csfs_descriptors, caled_csfs_indices_dict)
        X_stay = stay_csfs_descriptors.copy()
        logger.info("             特征提取完成")

        # 训练模型
        model, X_train, X_test, y_train, y_test, training_time, weight = gdp.train_model(config, caled_csfs_descriptors, rmix_file_data, logger)
        
        
        # 评估模型
        evaluation_results = gdp.evaluate_model(
            model, X_train, X_test, y_train, y_test, X_stay, config, logger
        )
        
        # 选择组态
        selection_results = gdp.select_configurations(
            config, unique_indices, evaluation_results['y_pred_other'], 
            raw_csf_data, indices_temp, logger
        )
        
        # 写入组态文件
        write_configuration_files(
            selection_results['chosen_index'], raw_csf_data, config, root_path, indices_temp
        )
        
        # 保存结果
        save_iteration_results(
            config, training_time, evaluation_results['eval_time'], 
            execution_time, evaluation_results, selection_results, weight, logger
        )
        
        # 检查收敛
        converged = check_convergence(config, sum_num_list, logger)
        if not converged:
            # 更新循环计数
            config.cal_loop_num += 1
            update_config(f'{config.root_path}/config.yaml', {'cal_loop_num': config.cal_loop_num})
    else:
        # 处理计算错误
        handle_calculation_error(config, indices_temp, raw_csf_data, root_path, logger)


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='机器学习训练程序')
    parser.add_argument('--config', type=str, default='config.toml', help='配置文件路径')
    args = parser.parse_args()
    
    # 加载配置
    try:
        cfg = gdp.load_config(args.config)
        main(cfg)
    except FileNotFoundError:
        print(f"错误: 配置文件 {args.config} 不存在")
    except Exception as e:
        print(f"程序执行失败: {str(e)}")