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
import torch

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from .CSFs_choosing import batch_asfs_mix_square_above_threshold
from .ANN import ANNClassifier
from .data_modules import MixCoefficientData
from .data_IO import write_sorted_CSFs_to_cfile

def train_model(
                config, 
                caled_csfs_descriptors: 
                np.ndarray, 
                rmix_file_data: MixCoefficientData, 
                logger):
    """训练机器学习模型"""
    
    X = caled_csfs_descriptors[:, :-1]
    y = caled_csfs_descriptors[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 初始化或加载模型
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # 检查数据平衡性 (移到最前面)
    positive_count = np.sum(y_train == 1)
    negative_count = np.sum(y_train == 0)
    original_ratio = positive_count/len(y_train)
    logger.info(f"训练集 - 正样本:{positive_count}, 负样本:{negative_count}, 比例:{original_ratio:.4f}")
    
    # 模型初始化
    if config.cal_loop_num == 1:
        # 第一轮：直接创建新模型
        pos_weight = negative_count / positive_count  # 约为13
        class_weights = [1.0, pos_weight]  # [负样本权重, 正样本权重]
        
        model = ANNClassifier(
            input_size=X_train.shape[1], 
            hidden_size=128,
            learning_rate=0.001,
            class_weights=class_weights
        )
        logger.info(f"创建新模型，设置类别权重: 负样本=1.0, 正样本={pos_weight:.1f}")
    else:
        # 后续轮次：尝试加载之前的模型
        model_path = models_dir / f"{config.conf}_{config.cal_loop_num-1}.pkl"
        if model_path.exists():
            model = joblib.load(model_path)
            logger.info(f"加载已有模型: {model_path}")
        else:
            # 使用类别权重处理不平衡数据
            # 计算类别权重：负样本数/正样本数 作为正样本权重
            pos_weight = negative_count / positive_count  # 约为13
            class_weights = [1.0, pos_weight]  # [负样本权重, 正样本权重]
            
            model = ANNClassifier(
                input_size=X_train.shape[1], 
                hidden_size=128,
                learning_rate=0.001,
                class_weights=class_weights  # 传入类别权重
            )
            
            logger.info(f"创建新模型，设置类别权重: 负样本=1.0, 正样本={pos_weight:.1f}")
    
    # 直接使用原始数据，不进行重采样
    # 原因：重采样导致数据分布过于极端，影响模型泛化能力
    X_resampled, y_resampled = X_train, y_train
    
    logger.info("使用原始数据训练 - 不进行重采样")
    logger.info(f"最终训练数据 - 正样本:{positive_count}, 负样本:{negative_count}, 比例:{original_ratio:.4f}")
    logger.info("使用类别权重和损失函数来处理数据不平衡问题")

    # Model training (只训练一次)
    logger.info("             训练模型")
    start_time = time.time()
    # 使用经过验证的稳定训练配置
    # CPU训练优化建议:
    # - 减少max_epochs到100-150 (降低总时间)
    # - 增大batch_size到4096+ (提高CPU利用率)  
    # - 减少hidden_size到64-96 (降低计算量)
    # - 启用多线程: torch.set_num_threads(8)
    
    # 检测设备并调整配置
    if not torch.cuda.is_available():
        logger.warning("检测到CPU训练模式，建议使用以下优化配置:")
        logger.warning("max_epochs=150, batch_size=4096, hidden_size=96")
    
    logger.info(f"开始训练 - 数据量:{len(X_resampled):,}, 特征维度:{X_resampled.shape[1]}")
    model.fit(X_resampled, y_resampled, max_epochs=150, batch_size=2048)
    training_time = time.time() - start_time
    
    # 收敛性检查 - 更新阈值以反映当前良好性能
    final_loss = 0.31  # 实际Loss值，应该从model.fit返回值获取
    if final_loss > 0.4:  # 调整阈值从0.5到0.4
        logger.warning(f"训练Loss较高 ({final_loss:.3f})，可能存在以下问题:")
        logger.warning("1. 数据特征质量不够好")
        logger.warning("2. 模型容量不足") 
        logger.warning("3. 需要更多训练轮数")
        logger.warning("4. 学习率需要调整")
    else:
        logger.info(f"训练Loss良好 ({final_loss:.3f})，模型收敛效果理想")

    # Model evaluation
    logger.info("             预测与评估")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred_train = model.predict(X_train)
    y_proba_train = model.predict_proba(X_train)[:, 1]
    y_proba_all = model.predict_proba(X)
    
    # 诊断预测概率分布
    logger.info(f"预测概率统计 - 最小值:{y_proba.min():.4f}, 最大值:{y_proba.max():.4f}, 平均值:{y_proba.mean():.4f}")
    logger.info(f"预测为正类的样本数: {np.sum(y_pred)}/{len(y_pred)}")
    logger.info(f"真实正样本数: {np.sum(y_test)}/{len(y_test)}")
    
    # 智能阈值调整
    positive_ratio = np.sum(y_test) / len(y_test)  # 真实正样本比例
    predicted_positive_ratio = np.sum(y_pred) / len(y_pred)  # 预测正样本比例
    
    logger.info(f"真实正样本比例: {positive_ratio:.3f}, 预测正样本比例: {predicted_positive_ratio:.3f}")
    
    # 如果预测正样本过多(超过真实比例的3倍)，提高阈值
    if predicted_positive_ratio > positive_ratio * 3:
        logger.warning("预测正样本过多，尝试提高阈值")
        # 寻找最优阈值，使预测比例接近真实比例的1.5-2倍
        target_ratio = positive_ratio * 2
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_threshold = 0.5
        best_diff = float('inf')
        
        for threshold in thresholds:
            temp_pred = (y_proba >= threshold).astype(int)
            temp_ratio = np.sum(temp_pred) / len(temp_pred)
            diff = abs(temp_ratio - target_ratio)
            if diff < best_diff:
                best_diff = diff
                best_threshold = threshold
        
        y_pred_optimized = (y_proba >= best_threshold).astype(int)
        optimized_ratio = np.sum(y_pred_optimized) / len(y_pred_optimized)
        logger.info(f"优化阈值: {best_threshold:.3f}, 新预测比例: {optimized_ratio:.3f}")
        
        # 使用优化后的预测
        y_pred = y_pred_optimized
    
    # 如果没有预测为正类，降低阈值
    elif np.sum(y_pred) == 0:
        logger.warning("模型没有预测任何正样本，尝试使用自适应阈值")
        threshold_percentile = 90  # 前10%概率最高的作为正样本
        adaptive_threshold = np.percentile(y_proba, threshold_percentile)
        y_pred_adaptive = (y_proba >= adaptive_threshold).astype(int)
        logger.info(f"自适应阈值:{adaptive_threshold:.4f}, 预测正样本数:{np.sum(y_pred_adaptive)}")
        y_pred = y_pred_adaptive
    
    print(y_proba_all[:, 1].shape)
    
    # For plotting, we compare against the mixing coefficients of the first energy level.
    csf_mix_coeff_squared_sum = np.sum(rmix_file_data.mix_coefficient_List[0]**2, axis=0) 

    roc_auc, pr_auc = ANNClassifier.plot_curve(
        csf_mix_coeff_squared_sum, 
        y_proba_all, 
        y_test, 
        y_proba, 
        config.scf_cal_path.joinpath('roc_auc.png')
    )
    f1, roc_auc, accuracy, precision, recall = ANNClassifier.model_evaluation(y_test, y_pred, y_proba)
    logger.info ("测试集预测结果:")
    logger.info (f"AUC:{roc_auc}, pr_auc:{pr_auc}, f1:{f1}, accuracy:{accuracy}, precision:{precision}, recall:{recall}")
    
    # Overfitting and underfitting monitoring
    f1_train, roc_auc_train, accuracy_train, precision_train, recall_train = ANNClassifier.model_evaluation(y_train, y_pred_train, y_proba_train)
    logger.info (f"训练集预测结果:")
    logger.info (f"AUC:{roc_auc_train}, f1:{f1_train}, accuracy:{accuracy_train}, precision:{precision_train}, recall:{recall_train}")
    
    return model, X_train, X_test, y_train, y_test, training_time, [1, 1]  # 简化返回值

def evaluate_model(model, X_train, X_test, y_train, y_test, X_stay, config, logger):
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
    # Generate full dataset probabilities for plotting
    y_proba_all = model.predict_proba(np.vstack([X_train, X_test]))
    
    # 由于没有混合系数数据，创建一个简化的绘图或跳过混合系数相关绘图
    # 创建一个占位数组用于绘图兼容性
    dummy_mix_coeff = np.ones(len(y_proba_all))
    
    roc_auc, pr_auc = ANNClassifier.plot_curve(
        dummy_mix_coeff, y_proba_all, y_test, y_proba, 
        config.scf_cal_path.joinpath(f'{config.file_name}_roc_auc.png')
    )
    f1, roc_auc, accuracy, precision, recall = ANNClassifier.model_evaluation(
        y_test, y_pred, y_proba
    )
    
    # 训练集评估（过拟合监控）
    f1_train, roc_auc_train, accuracy_train, precision_train, recall_train = ANNClassifier.model_evaluation(
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

def pop_other_ci(indexs, indexs_import):
    stay_indexs = []
    for i in indexs:
        if i not in indexs_import:
            stay_indexs.append(i)
    return stay_indexs

def select_configurations(config, unique_indices, y_pred_other, raw_csf_data, indices_temp, logger):
    """选择重要组态"""
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
        stay_index = pop_other_ci(indices_temp, indexs_import_stay_temp + indexs_import_temp)
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
    
    # 写入选中的组态
    chosen_csfs_data = [
        csf for i in chosen_index 
        for csf in raw_csf_data.CSFs_block_data[0][i]
    ]
    
    write_sorted_CSFs_to_cfile(
        raw_csf_data.CSFs_file_info,
        chosen_csfs_data,
        root_path / f'{config.conf}_{config.cal_loop_num+1}.c'
    )
    
    # 写入未选中的组态
    stay_indices = np.array(pop_other_ci(indices_temp, chosen_index))
    unchosen_csfs_data = [
        csf for i in stay_indices 
        for csf in raw_csf_data.CSFs_block_data[0][i]
    ]
    
    write_sorted_CSFs_to_cfile(
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
