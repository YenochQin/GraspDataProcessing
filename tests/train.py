#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Id :train.py
@date :2025/05/25 13:53:10
@author :YenochQin (秦毅)
'''
import yaml
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
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# sys.path.append('/home/workstation2/AppFiles/GraspDataProcessing/src')
sys.path.append('D:\\PythonProjects\\GraspDataProcessing\\src')
try:
    import graspdataprocessing as gdp
except ImportError:
    print("警告: 无法导入 graspdataprocessing 模块")
    gdp = None


def main(config):
    """主程序逻辑"""
    config.file_name = f'{config.conf}_{config.cal_loop_num}'
    logger = gdp.setup_logging(config)
    logger.info("机器学习训练程序启动")
    execution_time = time.time()

    # 设置目录结构
    root_path = gdp.setup_directories(config.root_path)
    
    # 初始化结果文件
    if config.cal_loop_num == 1:
        result_csv_path = root_path / 'results/results.csv'
        gdp.initialize_results_file(result_csv_path, logger)
    
    # 验证初始文件
    gdp.validate_initial_files(config, logger)
    
    logger.info(f"初始比例: {config.initial_ratio}")
    logger.info(f"光谱项: {config.spetral_term}")
    
    try:
        # 加载数据文件
        energy_level_data_pd, rmix_file_data, target_pool_csfs_data, raw_csfs_descriptors, cal_csfs_data, caled_csfs_indices_dict = gdp.load_data_files(config)

        # 检查组态耦合
        cal_result = gdp.check_configuration_coupling(config, energy_level_data_pd, logger)
        
        caled_csfs_descriptors = gdp.generate_chosen_csfs_descriptors(config, caled_csfs_indices_dict, raw_csfs_descriptors, rmix_file_data, logger)
        
        
        logger.info("             特征提取完成")
        logger.info("             数据预处理")

        X = caled_csfs_descriptors[:, :-1]
        y = caled_csfs_descriptors[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_resampled, y_resampled = X_train, y_train
        model = gdp.ANNClassifier(input_size=X_train.shape[1], hidden_size=128)
        
        
        # Model training
        logger.info("             训练模型")
        start_time = time.time()
        model.fit(X_resampled, y_resampled)
        training_time = time.time() - start_time

        # Model evaluation
        logger.info("             预测与评估")
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred_train = model.predict(X_train)
        y_proba_train = model.predict_proba(X_train)[:, 1]
        y_proba_all = model.predict_proba(X)
        print(y_proba_all[:, 1].shape)
        roc_auc, pr_auc = gdp.ANNClassifier.plot_curve(rmix_file_data.mix_coefficient_List[0], y_proba_all, y_test, y_proba, root_path.joinpath('roc_auc.png'))
        f1, roc_auc, accuracy, precision, recall = gdp.ANNClassifier.model_evaluation(y_test, y_pred, y_proba)
        logger.info ("测试集预测结果:")
        logger.info (f"AUC:{roc_auc}, pr_auc:{pr_auc}, f1:{f1}, accuracy:{accuracy}, precision:{precision}, recall:{recall}")
        # Overfitting and underfitting monitoring
        f1_train, roc_auc_train, accuracy_train, precision_train, recall_train = gdp.ANNClassifier.model_evaluation(y_train, y_pred_train, y_proba_train)
        logger.info (f"训练集预测结果:")
        logger.info (f"AUC:{roc_auc_train}, f1:{f1_train}, accuracy:{accuracy_train}, precision:{precision_train}, recall:{recall_train}")
        
        
        
        if cal_result:
            # 记录能量信息
            for level in rmix_file_data.level_list:
                logger.info(f"迭代能量：{level}")
            logger.info("耦合正确")
            logger.info("************************************************")
            
            # 提取特征
            unique_indices = extract_features(config, rmix_file_data, logger)
            
            # 训练模型
            model, X_train, X_test, y_train, y_test, X_stay, training_time, weight = train_model(config, logger)
            
            # 评估模型
            evaluation_results = evaluate_model(
                model, X_train, X_test, y_train, y_test, X_stay, config, logger
            )
            
            # 选择组态
            selection_results = select_configurations(
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
            
    except Exception as e:
        logger.error(f"程序执行过程中发生错误: {str(e)}")
        raise

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='机器学习训练程序')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    args = parser.parse_args()
    
    # 加载配置
    try:
        cfg = gdp.load_config(args.config)
        main(cfg)
    except FileNotFoundError:
        print(f"错误: 配置文件 {args.config} 不存在")
    except yaml.YAMLError as e:
        print(f"错误: 配置文件解析失败 - {str(e)}")
    except Exception as e:
        print(f"程序执行失败: {str(e)}")