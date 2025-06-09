#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Id :machine_learning_traning.py
@date :2025/06/09 15:58:42
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
from typing import Dict, Tuple, List, Optional

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from .CSFs_choosing import batch_asfs_mix_square_above_threshold
from .ANN import ANNClassifier
from .data_modules import MixCoefficientData




def train_model(config, caled_csfs_descriptors: np.ndarray, rmix_file_data: MixCoefficientData,logger):
    """训练机器学习模型"""
    
    X = caled_csfs_descriptors[:, :-1]
    y = caled_csfs_descriptors[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_resampled, y_resampled = X_train, y_train
    model = ANNClassifier(input_size=X_train.shape[1], hidden_size=128)

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
    roc_auc, pr_auc = ANNClassifier.plot_curve(rmix_file_data.mix_coefficient_List[0], y_proba_all, y_test, y_proba, config.cal_path.joinpath('roc_auc.png'))
    f1, roc_auc, accuracy, precision, recall = ANNClassifier.model_evaluation(y_test, y_pred, y_proba)
    logger.info ("测试集预测结果:")
    logger.info (f"AUC:{roc_auc}, pr_auc:{pr_auc}, f1:{f1}, accuracy:{accuracy}, precision:{precision}, recall:{recall}")
    
    # Overfitting and underfitting monitoring
    f1_train, roc_auc_train, accuracy_train, precision_train, recall_train = ANNClassifier.model_evaluation(y_train, y_pred_train, y_proba_train)
    logger.info (f"训练集预测结果:")
    logger.info (f"AUC:{roc_auc_train}, f1:{f1_train}, accuracy:{accuracy_train}, precision:{precision_train}, recall:{recall_train}")
    # 初始化或加载模型
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    if config.cal_loop_num == 1:
        model = ANNClassifier(input_size=X_train.shape[1], hidden_size=128)
    else:
        model_path = models_dir / f"{config.conf}_{config.cal_loop_num-1}.pkl"
        if model_path.exists():
            model = joblib.load(model_path)
        else:
            model = ANNClassifier(input_size=X_train.shape[1], hidden_size=128)
    
    # 设置权重
    weight = [1, max(1, 12 - 2*config.cal_loop_num)]
    logger.info(f"权重: {weight}")
    
    # 重采样和训练
    X_resampled, y_resampled = model.resampling(X_train, y_train, weight)
    start_time = time.time()
    model.fit(X_resampled, y_resampled)
    training_time = time.time() - start_time
    
    return model, X_train, X_test, y_train, y_test, training_time, weight

def evaluate_model(model, X_train, X_test, y_train, y_test, config, logger):
    """评估模型性能"""
    
    logger.info("开始预测与评估")
    
    # 预测
    start_time = time.time()
    y_pred = model.predict(X_test)
    y_pred_other = model.predict(X_stay)
    eval_time = time.time() - start_time
    
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred_train = model.predict(X_train)
    y_proba_train = model.predict_proba(X_train)[:, 1]
    
    # 评估
    roc_auc, pr_auc = gdp.ANNClassifier.plot_curve(y_test, y_proba, config.file_name)
    f1, roc_auc, accuracy, precision, recall = gdp.ANNClassifier.model_evaluation(
        y_test, y_pred, y_proba
    )
    
    # 训练集评估（过拟合监控）
    f1_train, roc_auc_train, accuracy_train, precision_train, recall_train = gdp.ANNClassifier.model_evaluation(
        y_train, y_pred_train, y_proba_train
    )
    
    # 保存结果
    test_data_dir = Path("test_data")
    test_data_dir.mkdir(exist_ok=True)
    result_file = test_data_dir / f"{config.file_name}.csv"
    pd.DataFrame({
        "y_test": y_test, 
        "y_pred": y_pred, 
        "y_proba": y_proba
    }).to_csv(result_file, index=False)
    
    # 保存模型
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    model_file = models_dir / f"{config.file_name}.pkl"
    joblib.dump(model, model_file)
    logger.info("预测结果与模型保存成功")
    
    return {
        'y_pred_other': y_pred_other,
        'eval_time': eval_time,
        'metrics': {
            'f1': f1, 'roc_auc': roc_auc, 'accuracy': accuracy,
            'precision': precision, 'recall': recall,
            'f1_train': f1_train, 'roc_auc_train': roc_auc_train,
            'accuracy_train': accuracy_train, 'precision_train': precision_train,
            'recall_train': recall_train
        }
    }



def select_configurations(config, unique_indices, y_pred_other, raw_csf_data, indices_temp, logger):
    """选择重要组态"""
    if gdp is None:
        raise ImportError("graspdataprocessing 模块未正确导入")
    
    # 计算最小组态数
    sum_num_min = round(math.ceil(raw_csf_data.CSFs_block_length[0] * config.initial_ratio))
    sum_num = len(unique_indices)
    
    if sum_num <= sum_num_min:
        sum_num = sum_num_min
    
    # 获取重要组态索引
    indexs_import = unique_indices.tolist()
    indexs_import_temp = [indices_temp[i] for i in indexs_import]
    indexs_import_stay = np.where(y_pred_other == 1)[0].tolist()
    
    # 计算stay索引 (假设所有索引都在stay中)
    stay_indices_all = list(range(len(y_pred_other)))
    indexs_import_stay_temp = [stay_indices_all[i] for i in indexs_import_stay]
    
    # 保存重要组态索引
    np.save(f"results/indexs_import_ab{config.cal_loop_num}.npy", indexs_import_temp)
    np.save(f"results/indexs_import_ml{config.cal_loop_num}.npy", indexs_import_stay_temp)
    
    logger.info(f"开始选择组态，当前重要组态数为：{len(indexs_import_temp)}")
    
    # 选择新增组态
    if len(indexs_import_stay) >= config.expansion_ratio * sum_num:
        ml_add_csfs = np.random.choice(
            indexs_import_stay_temp,
            size=config.expansion_ratio * sum_num,
            replace=False
        ).tolist()
        mc_add_csfs = None
        new_add_csfs = ml_add_csfs
    else:
        stay_index = gdp.pop_other_ci(indices_temp, indexs_import_stay_temp + indexs_import_temp)
        ml_add_csfs = indexs_import_stay_temp
        mc_add_csfs = np.random.choice(
            stay_index,
            size=config.expansion_ratio * sum_num - len(indexs_import_stay_temp),
            replace=False
        ).tolist()
        new_add_csfs = ml_add_csfs + mc_add_csfs
    
    chosen_index = np.sort(np.array(indexs_import_temp + new_add_csfs))
    logger.info(f"下一步计算组态数为：{len(chosen_index)}")
    
    return {
        'chosen_index': chosen_index,
        'indexs_import_temp': indexs_import_temp,
        'indexs_import_stay_temp': indexs_import_stay_temp,
        'sum_num': sum_num,
        'ml_add_csfs': ml_add_csfs,
        'new_add_csfs': new_add_csfs
    }

def write_configuration_files(chosen_index, raw_csf_data, config, root_path, indices_temp):
    """写入组态文件"""
    if gdp is None:
        raise ImportError("graspdataprocessing 模块未正确导入")
    
    # 写入选中的组态
    chosen_csfs_data = [
        csf for i in chosen_index 
        for csf in raw_csf_data.CSFs_block_data[0][i]
    ]
    
    gdp.write_sorted_CSFs_to_cfile(
        raw_csf_data.CSFs_file_info,
        chosen_csfs_data,
        root_path / f'{config.conf}_{config.cal_loop_num+1}.c'
    )
    
    # 写入未选中的组态
    stay_indices = np.array(gdp.pop_other_ci(indices_temp, chosen_index))
    unchosen_csfs_data = [
        csf for i in stay_indices 
        for csf in raw_csf_data.CSFs_block_data[0][i]
    ]
    
    gdp.write_sorted_CSFs_to_cfile(
        raw_csf_data.CSFs_file_info,
        unchosen_csfs_data,
        root_path / f'{config.conf}_{config.cal_loop_num+1}_stay.c'
    )

def save_iteration_results(config, training_time, eval_time, execution_time, 
                          evaluation_results, selection_results, weight, logger):
    """保存迭代结果"""
    all_time = execution_time + training_time + eval_time
    metrics = evaluation_results['metrics']
    
    # 保存到CSV文件
    with open(f'{config.root_path}/results/iteration_results.csv', mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            training_time, eval_time, execution_time, all_time,
            metrics['f1'], metrics['roc_auc'], metrics['accuracy'],
            metrics['precision'], metrics['recall'],
            0,  # Es placeholder
            len(selection_results['indexs_import_temp']),
            len(selection_results['indexs_import_stay_temp']),
            None,  # MLsampling_ratio placeholder
            len(selection_results['chosen_index']),
            weight,
            metrics['f1_train'], metrics['roc_auc_train'], metrics['accuracy_train'],
            metrics['precision_train'], metrics['recall_train']
        ])

def check_convergence(config, sum_num_list, logger):
    """检查收敛性"""
    # 这里需要Es_term的历史数据，暂时先跳过收敛检查
    # 在实际使用中，需要维护能量项的历史记录
    logger.info("收敛检查功能需要能量历史数据，当前跳过")
    return False

def handle_calculation_error(config, indices_temp, raw_csf_data, root_path, logger):
    """处理计算错误的情况"""
    if gdp is None:
        raise ImportError("graspdataprocessing 模块未正确导入")
    
    cal_error_num = getattr(config, 'cal_error_num', 0) + 1
    gdp.update_config(f'{config.root_path}/config.yaml', {'cal_error_num': cal_error_num})
    
    if cal_error_num >= 3:
        logger.info("连续三次波函数未改进，迭代收敛，退出筛选程序")
        with open(f'{config.root_path}/run.input', 'w') as file:
            file.write('False')
        return
    
    logger.info('组态选择出现问题，正在重选组态')
    
    # 加载前一次的重要组态
    prev_file = f"results/indexs_import_ab{config.cal_loop_num-1}.npy"
    if os.path.exists(prev_file):
        indexs_import_temp = np.load(prev_file)
    else:
        indexs_import_temp = []
    
    stay_indices = gdp.pop_other_ci(indices_temp, indexs_import_temp)
    
    # 随机选择组态
    sum_num_min = round(math.ceil(raw_csf_data.CSFs_block_length[0] * config.initial_ratio))
    mc_add_csfs = np.random.choice(
        stay_indices,
        size=config.expansion_ratio * sum_num_min,
        replace=False
    ).tolist()
    
    chosen_index = np.sort(np.array(list(indexs_import_temp) + mc_add_csfs))
    
    # 写入组态文件
    write_configuration_files(chosen_index, raw_csf_data, config, root_path, indices_temp)

def get_stay_descriptors(raw_csfs_descriptors: np.ndarray, chosen_csfs_indices_dict: Dict[int, List[int]]) -> np.ndarray:
    """
    找出不在chosen_csfs_indices_dict索引中的描述符
    
    Args:
        raw_csfs_descriptors: 原始CSFs描述符数组
        chosen_csfs_indices_dict: 已选择的CSFs索引字典，格式为{block_index: [indices]}
        
    Returns:
        np.ndarray: 不在chosen_csfs_indices_dict中的描述符数组
    """
    # 获取所有已选择的索引
    chosen_indices = []
    for block_indices in chosen_csfs_indices_dict.values():
        chosen_indices.extend(block_indices)
    chosen_indices = set(chosen_indices)
    
    # 获取所有可能的索引
    all_indices = set(range(len(raw_csfs_descriptors)))
    
    # 找出不在chosen_indices中的索引
    stay_indices = list(all_indices - chosen_indices)
    
    # 返回对应的描述符
    return raw_csfs_descriptors[stay_indices]
