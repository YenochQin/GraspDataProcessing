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
import os
from pathlib import Path
import csv
import sys
import math
import numpy as np
import pandas as pd
import time
import joblib
import json 
from tabulate import tabulate 
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

sys.path.append('/home/workstation3/AppFiles/GraspDataProcessing/src')
try:
    import graspdataprocessing as gdp
except ImportError:
    print("错误: 无法导入 graspdataprocessing 模块")
    sys.exit(1)

important_config_count_history = []

def main(config):
    """主程序逻辑"""
    config.file_name = f'{config.conf}_{config.cal_loop_num}'
    logger = gdp.setup_logging(config)
    
    config_file_path = config.root_path / 'config.toml'
    logger.info("机器学习训练程序启动")
    execution_time = time.time()

    gdp.setup_directories(config)

    # 验证初始文件
    gdp.validate_initial_files(config, logger)

    logger.info(f"初始比例: {config.chosen_ratio}")
    logger.info(f"光谱项: {config.spetral_term}")

    try:
        # 加载数据文件
        energy_level_data_pd, rmix_file_data, target_pool_csfs_data, raw_csfs_descriptors, cal_csfs_data, caled_csfs_indices_dict, unselected_csfs_indices_dict = gdp.load_data_files(config, logger)
        
        # 检查组态耦合
        cal_result, asfs_position = gdp.check_configuration_coupling(config, energy_level_data_pd, logger)
        logger.info
        ("************************************************")

    except Exception as e:
            logger.error(f"程序执行过程中发生错误: {str(e)}")
            raise
    # 选择asfs_position索引对应的行
    selected_energy_data = energy_level_data_pd.iloc[asfs_position]
    # 保存正确的能级数据为CSV
    correct_levels_csv_path = config.scf_cal_path / f'{config.conf}_{config.cal_loop_num}_correct_levels.csv'
    selected_energy_data.to_csv(correct_levels_csv_path, index=False)
    logger.info(f"选择的能级数据已保存到: {correct_levels_csv_path}")

    should_continue = True
    if config.cal_loop_num >= 3:
        # 检查收敛性
        should_continue = gdp.check_grasp_cal_convergence(config, logger)
        logger.info(f"检查收敛性结果: {not should_continue}")

    if cal_result and should_continue:
        # 记录能量信息
        logger.info("能级数据表格：\n%s", 
           tabulate(energy_level_data_pd, headers='keys', tablefmt='fancy_grid', showindex=False, 
                   floatfmt=('d', 'd', 'd', 's', '.7f', '.2f', '.2f', 's')))
        logger.info("耦合正确")
        logger.info("************************************************")

        # 提取特征
        logger.info("             数据预处理")
        caled_csfs_descriptors = gdp.generate_chosen_csfs_descriptors(config, caled_csfs_indices_dict, raw_csfs_descriptors, rmix_file_data, asfs_position, logger)
        unselected_csfs_descriptors = gdp.get_unselected_descriptors(raw_csfs_descriptors, caled_csfs_indices_dict)
        X_unselected = unselected_csfs_descriptors.copy()
        logger.info("             特征提取完成")

        # 训练模型
        model, X_train, X_test, y_train, y_test, training_time = gdp.train_model(config, caled_csfs_descriptors, rmix_file_data, logger)

        # 评估模型
        evaluation_results = gdp.evaluate_model(
            model, X_train, X_test, y_train, y_test, X_unselected, config, logger
        )

        # 访问结果
        test_predictions = evaluation_results['predictions']['y_prediction_test']
        test_probabilities = evaluation_results['probabilities']['y_probability_test']
        test_f1 = evaluation_results['test_metrics']['f1']
        train_f1 = evaluation_results['train_metrics']['f1']

        overfitting_check = test_f1 - train_f1  # 如果差异过大说明过拟合
        logger.info(f'             {overfitting_check:}')
        logger.info("             模型推理")
        start_time = time.time()
        X_unselected = unselected_csfs_descriptors.copy()
        y_unselected_prediction = model.predict(X_unselected)
        y_unselected_probability = model.predict_probability(X_unselected)[:, 1]
        eval_time = time.time() - start_time
        logger.info(f"             模型推理时间:{eval_time}")
        
        y_prediction = evaluation_results['predictions']['y_prediction_test']
        y_probability = evaluation_results['probabilities']['y_probability_test']
        y_probability_all = evaluation_results['probabilities']['y_probability_all']
        y_probability_other = evaluation_results['probabilities']['y_probability_other']
        
        # 使用标准化的保存和绘图函数
        saved_files = gdp.save_and_plot_results(
            evaluation_results=evaluation_results,
            model=model,
            config=config,
            save_model=True,
            save_data=True,
            plot_curves=True,
            logger=logger
        )
        logger.info(f"             预测结果与模型保存成功")
        logger.info(f"             保存的文件: {list(saved_files.keys())}")

        csfs_above_threshold_indices = np.where(np.any(rmix_file_data.mix_coefficient_List[0][asfs_position]**2 >= np.float64(config.cutoff_value), axis = 0))[0]
        high_prob_threshold = np.percentile(y_probability_all[:, 1], 90)  # 取90分位数作为高概率阈值
        logger.info(f"             高于90分位数作为高概率阈值: {high_prob_threshold}")
        promising_unselected_CSFs_indices = unselected_csfs_indices_dict[0][y_probability_other > high_prob_threshold]
        logger.info(f"             ml预测的组态数: {promising_unselected_CSFs_indices.shape}")
        filtered_chosen_indices = caled_csfs_indices_dict[0][csfs_above_threshold_indices]
        logger.info(f"             本轮计算重要组态数: {filtered_chosen_indices.shape}")
        all_chosen_indices = np.union1d(filtered_chosen_indices, promising_unselected_CSFs_indices)
        logger.info(f"             本轮选择的组态总数: {all_chosen_indices.shape}")
        important_config_count = all_chosen_indices.shape[0]
        important_config_count_history.append(important_config_count)
        logger.info(f"             本轮重要组态数量: {important_config_count}")
        logger.info(f"             迭代重要组态数量历史: {important_config_count_history}")
        ml_chosen_indices_dict = {0 : all_chosen_indices}
        
        ml_chosen_indices_dict_path = config.root_path / 'results' / f'{config.conf}_{config.cal_loop_num}_ml_chosen_indices.pkl'
        gdp.csfs_index_storange(ml_chosen_indices_dict, ml_chosen_indices_dict_path)
        logger.info(f"             本轮选择的组态索引保存到: {ml_chosen_indices_dict_path}")

        # 保存迭代结果
        selection_results = {
            'indexs_import_temp': [],  # 如果需要可以添加实际值
            'indexs_import_stay_temp': [],  # 如果需要可以添加实际值
            'chosen_index': all_chosen_indices.tolist()
        }
        
        # 计算总执行时间
        total_execution_time = time.time() - execution_time
        
        gdp.save_iteration_results(
            config=config,
            training_time=training_time,
            eval_time=eval_time,
            execution_time=total_execution_time,
            evaluation_results=evaluation_results,
            selection_results=selection_results,
            logger=logger
        )
        
        gdp.update_config(config_file_path, {'continue_cal': True})
        gdp.update_config(config_file_path, {'cal_error_num': 0})
        gdp.update_config(config_file_path, {'cal_loop_num': config.cal_loop_num + 1})
        # gdp.continue_calculate(config.root_path, True)

        # 计算动态选择率
        dynamic_ratio = gdp.calculate_dynamic_chosen_ratio(
            config, 
            all_chosen_indices, 
            target_pool_csfs_data, 
            y_probability_other, 
            evaluation_results, 
            energy_level_data_pd, 
            logger
        )
        config.chosen_ratio = dynamic_ratio
        gdp.update_config(config_file_path, {'chosen_ratio': dynamic_ratio})

    elif not should_continue:
        logger.info("************************************************")
        logger.info("计算收敛，停止计算")
        gdp.update_config(config_file_path, {'continue_cal': False})

    else:
        logger.info("************************************************")
        gdp.handle_calculation_error(config, logger)


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