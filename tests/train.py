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
import joblib
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
        energy_level_data_pd, rmix_file_data, target_pool_csfs_data, raw_csfs_descriptors, cal_csfs_data, caled_csfs_indices_dict, unselected_csfs_indices_dict = gdp.load_data_files(config, logger)
        
        asfs_position = []
        # 检查组态耦合
        cal_result, asfs_position = gdp.check_configuration_coupling(config, energy_level_data_pd, logger)
        logger.info("************************************************")
        
        
        

    except Exception as e:
            logger.error(f"程序执行过程中发生错误: {str(e)}")
            raise

    # 初始化迭代结果文件
    gdp.initialize_iteration_results_csv(config, logger)

    if cal_result:
        # 记录能量信息
        logger.info("能级数据表格：\n%s", 
           tabulate(energy_level_data_pd, headers='keys', tablefmt='fancy_grid', showindex=False))
        logger.info("耦合正确")
        logger.info("************************************************")

        # 提取特征
        logger.info("             数据预处理")
        caled_csfs_descriptors = gdp.generate_chosen_csfs_descriptors(config, caled_csfs_indices_dict, raw_csfs_descriptors, rmix_file_data, asfs_position, logger)
        unselected_csfs_descriptors = gdp.get_unselected_descriptors(raw_csfs_descriptors, caled_csfs_indices_dict)
        X_unselected = unselected_csfs_descriptors.copy()
        logger.info("             特征提取完成")

        # 训练模型
        model, X_train, X_test, y_train, y_test, training_time, weight = gdp.train_model(config, caled_csfs_descriptors, rmix_file_data, logger)

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
        result_file_path = config.root_path / 'test_data' / f'{config.conf}_{config.cal_loop_num}.csv'
        pd.DataFrame({"y_test": y_test, "y_prediction": y_prediction, "y_probability": y_probability}).to_csv(result_file_path, index=False)
        model_file_path = config.root_path / 'models' / f'{config.conf}_{config.cal_loop_num}.pkl'
        joblib.dump(model, model_file_path)
        logger.info(f"             预测结果与模型保存成功")

        csfs_above_threshold_indices = np.where(np.any(rmix_file_data.mix_coefficient_List[0][asfs_position]**2 >= np.float64(config.cutoff_value), axis = 0))[0]
        high_prob_threshold = np.percentile(y_probability_all[:, 1], 90)  # 取90分位数作为高概率阈值
        logger.info(f"             高于90分位数作为高概率阈值: {high_prob_threshold}")
        promising_unselected_CSFs_indices = unselected_csfs_indices_dict[0][y_probability_other > high_prob_threshold]
        logger.info(f"             ml预测的组态数: {promising_unselected_CSFs_indices.shape}")
        filtered_chosen_indices = caled_csfs_indices_dict[0][csfs_above_threshold_indices]
        logger.info(f"             本轮计算重要组态数: {filtered_chosen_indices.shape}")
        all_chosen_indices = np.union1d(filtered_chosen_indices, promising_unselected_CSFs_indices)
        logger.info(f"             本轮选择的组态总数: {all_chosen_indices.shape}")
        ml_chosen_indices_dict = {0 : all_chosen_indices}
        
        ml_chosen_indices_dict_path = config.root_path / 'results' / f'{config.conf}_{config.cal_loop_num}_ml_chosen_indices.pkl'
        gdp.csfs_index_storange(ml_chosen_indices_dict, ml_chosen_indices_dict_path)
        logger.info(f"             本轮选择的组态索引保存到: {ml_chosen_indices_dict_path}")

    else:
        logger.info("************************************************")
        logger.info(f"             第{config.cal_loop_num}次迭代开始")
        
        

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