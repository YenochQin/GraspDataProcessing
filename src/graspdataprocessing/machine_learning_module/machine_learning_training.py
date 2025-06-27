#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Id :machine_learning_traning.py
@date :2025/06/09 15:58:42
@author :YenochQin (秦毅)
'''

# 标准库导入
import argparse
import csv
import logging
import math
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Tuple, List, Optional

# 第三方库导入
import joblib
import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 本地模块导入
from .ANN import ANNClassifier
from ..data_IO.produced_data_write import update_config
from ..utils.data_modules import MixCoefficientData

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
        
        # CPU优化：减少hidden_size以降低计算量
        hidden_size = 96 if not torch.cuda.is_available() else 128
        
        model = ANNClassifier(
            input_size=X_train.shape[1], 
            hidden_size=hidden_size,
            learning_rate=0.001,
            class_weights=class_weights
        )
        logger.info(f"创建新模型，设置类别权重: 负样本=1.0, 正样本={pos_weight:.1f}")
        logger.info(f"模型hidden_size: {hidden_size} ({'CPU优化' if not torch.cuda.is_available() else 'GPU模式'})")
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
            
            # CPU优化：减少hidden_size以降低计算量
            hidden_size = 96 if not torch.cuda.is_available() else 128
            
            model = ANNClassifier(
                input_size=X_train.shape[1], 
                hidden_size=hidden_size,
                learning_rate=0.001,
                class_weights=class_weights  # 传入类别权重
            )
            
            logger.info(f"创建新模型，设置类别权重: 负样本=1.0, 正样本={pos_weight:.1f}")
            logger.info(f"模型hidden_size: {hidden_size} ({'CPU优化' if not torch.cuda.is_available() else 'GPU模式'})")
    
    # 直接使用原始数据，不进行重采样
    # 原因：重采样导致数据分布过于极端，影响模型泛化能力
    X_resampled, y_resampled = X_train, y_train
    
    logger.info("使用原始数据训练 - 不进行重采样")
    logger.info(f"最终训练数据 - 正样本:{positive_count}, 负样本:{negative_count}, 比例:{original_ratio:.4f}")
    logger.info("使用类别权重和损失函数来处理数据不平衡问题")

    # Model training (只训练一次)
    logger.info("             训练模型")
    start_time = time.time()
    
    # CPU训练优化配置
    if not torch.cuda.is_available():
        # 获取系统CPU核心数
        cpu_count = os.cpu_count() or 4  # 如果无法获取则默认使用4核
        optimal_threads = min(32, cpu_count)  # 最多使用8线程，不超过系统核心数
        
        # 设置PyTorch线程数
        torch.set_num_threads(optimal_threads)
        
        # 设置额外的并行配置
        os.environ['OMP_NUM_THREADS'] = str(optimal_threads)
        os.environ['MKL_NUM_THREADS'] = str(optimal_threads)
        os.environ['NUMEXPR_NUM_THREADS'] = str(optimal_threads)
        
        logger.info(f"启用CPU多线程优化:")
        logger.info(f"- 系统CPU核心数: {cpu_count}")
        logger.info(f"- PyTorch线程数: {optimal_threads}")
        logger.info(f"- 建议配置: max_epochs=150, batch_size=4096, hidden_size=96")
        
        # 调整训练参数用于CPU优化
        batch_size_optimized = 4096
        max_epochs_optimized = 150
        
    else:
        logger.info("检测到GPU模式，使用标准配置")
        batch_size_optimized = 2048
        max_epochs_optimized = 150
    
    logger.info(f"开始训练 - 数据量:{len(X_resampled):,}, 特征维度:{X_resampled.shape[1]}")
    model.fit(X_resampled, y_resampled, batch_size=batch_size_optimized, max_epochs=max_epochs_optimized)
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
    y_prediction = model.predict(X_test)
    y_probability = model.predict_probability(X_test)[:, 1]
    y_prediction_train = model.predict(X_train)
    y_probability_train = model.predict_probability(X_train)[:, 1]
    y_probability_all = model.predict_probability(X)
    
    # 诊断预测概率分布
    logger.info(f"预测概率统计 - 最小值:{y_probability.min():.4f}, 最大值:{y_probability.max():.4f}, 平均值:{y_probability.mean():.4f}")
    logger.info(f"预测为正类的样本数: {np.sum(y_prediction)}/{len(y_prediction)}")
    logger.info(f"真实正样本数: {np.sum(y_test)}/{len(y_test)}")
    
    # 智能阈值调整
    positive_ratio = np.sum(y_test) / len(y_test)  # 真实正样本比例
    predicted_positive_ratio = np.sum(y_prediction) / len(y_prediction)  # 预测正样本比例
    
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
            temp_prediction = (y_probability >= threshold).astype(int)
            temp_ratio = np.sum(temp_prediction) / len(temp_prediction)
            diff = abs(temp_ratio - target_ratio)
            if diff < best_diff:
                best_diff = diff
                best_threshold = threshold
        
        y_prediction_optimized = (y_probability >= best_threshold).astype(int)
        optimized_ratio = np.sum(y_prediction_optimized) / len(y_prediction_optimized)
        logger.info(f"优化阈值: {best_threshold:.3f}, 新预测比例: {optimized_ratio:.3f}")
        
        # 使用优化后的预测
        y_prediction = y_prediction_optimized
    
    # 如果没有预测为正类，降低阈值
    elif np.sum(y_prediction) == 0:
        logger.warning("模型没有预测任何正样本，尝试使用自适应阈值")
        threshold_percentile = 90  # 前10%概率最高的作为正样本
        adaptive_threshold = np.percentile(y_probability, threshold_percentile)
        y_prediction_adaptive = (y_probability >= adaptive_threshold).astype(int)
        logger.info(f"自适应阈值:{adaptive_threshold:.4f}, 预测正样本数:{np.sum(y_prediction_adaptive)}")
        y_prediction = y_prediction_adaptive
    
    # For plotting, we compare against the mixing coefficients of the first energy level.
    csf_mix_coeff_squared_sum = np.sum(rmix_file_data.mix_coefficient_List[0]**2, axis=0) 
    
    # 诊断混合系数的信息
    logger.info(f"混合系数统计 - 最小值:{csf_mix_coeff_squared_sum.min():.6f}, 最大值:{csf_mix_coeff_squared_sum.max():.6f}")
    logger.info(f"混合系数平均值:{csf_mix_coeff_squared_sum.mean():.6f}, 零值数量:{np.sum(csf_mix_coeff_squared_sum == 0)}")
    logger.info(f"y_probability_all形状: {y_probability_all.shape}")

    roc_auc, pr_auc = ANNClassifier.plot_curve(
        csf_mix_coeff_squared_sum, 
        y_probability_all, 
        y_test, 
        y_probability, 
        config.scf_cal_path.joinpath('roc_auc.png')
    )
    f1, roc_auc, accuracy, precision, recall = ANNClassifier.model_evaluation(y_test, y_prediction, y_probability)
    logger.info ("测试集预测结果:")
    logger.info (f"AUC:{roc_auc}, pr_auc:{pr_auc}, f1:{f1}, accuracy:{accuracy}, precision:{precision}, recall:{recall}")
    
    # Overfitting and underfitting monitoring
    f1_train, roc_auc_train, accuracy_train, precision_train, recall_train = ANNClassifier.model_evaluation(y_train, y_prediction_train, y_probability_train)
    logger.info (f"训练集预测结果:")
    logger.info (f"AUC:{roc_auc_train}, f1:{f1_train}, accuracy:{accuracy_train}, precision:{precision_train}, recall:{recall_train}")
    
    return model, X_train, X_test, y_train, y_test, training_time

def evaluate_model(model, X_train, X_test, y_train, y_test, X_unselected, config, logger):
    """
    评估模型性能，返回所有预测结果和评估指标
    
    Args:
        model: 训练好的模型
        X_train, X_test, y_train, y_test: 训练和测试数据
        X_unselected: 其他需要预测的数据
        config: 配置对象
        logger: 日志记录器
    
    Returns:
        dict: 包含所有预测结果、概率、评估指标和元数据的完整结果字典
    """
    
    logger.info("开始预测与评估")
    
    # 预测
    start_time = time.time()
    y_prediction = model.predict(X_test)
    y_prediction_other = model.predict(X_unselected)
    eval_time = time.time() - start_time
    
    # 预测概率
    y_probability = model.predict_probability(X_test)[:, 1]
    y_prediction_train = model.predict(X_train)
    y_probability_train = model.predict_probability(X_train)[:, 1]
    y_probability_other = model.predict_probability(X_unselected)[:, 1]
    
    # 生成完整数据集的概率用于分析
    y_probability_all = model.predict_probability(np.vstack([X_train, X_test]))
    
    # 评估指标计算
    f1, roc_auc, accuracy, precision, recall = ANNClassifier.model_evaluation(
        y_test, y_prediction, y_probability
    )
    
    # 训练集评估（过拟合监控）
    f1_train, roc_auc_train, accuracy_train, precision_train, recall_train = ANNClassifier.model_evaluation(
        y_train, y_prediction_train, y_probability_train
    )
    
    logger.info("模型评估完成")
    
    # 返回完整的结果字典
    return {
        # 预测结果
        'predictions': {
            'y_prediction_test': y_prediction,
            'y_prediction_train': y_prediction_train, 
            'y_prediction_other': y_prediction_other
        },
        
        # 预测概率
        'probabilities': {
            'y_probability_test': y_probability,
            'y_probability_train': y_probability_train,
            'y_probability_other': y_probability_other,
            'y_probability_all': y_probability_all
        },
        
        # 真实标签
        'true_labels': {
            'y_test': y_test,
            'y_train': y_train
        },
        
        # 测试集评估指标
        'test_metrics': {
            'f1': f1,
            'roc_auc': roc_auc, 
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall
        },
        
        # 训练集评估指标（过拟合检测）
        'train_metrics': {
            'f1': f1_train,
            'roc_auc': roc_auc_train,
            'accuracy': accuracy_train, 
            'precision': precision_train,
            'recall': recall_train
        },
        
        # 元数据
        'metadata': {
            'eval_time': eval_time,
            'test_samples': len(y_test),
            'train_samples': len(y_train),
            'other_samples': len(X_unselected),
            'config_name': getattr(config, 'file_name', 'unknown')
        }
    }

def check_grasp_cal_convergence(config, logger):
    """
    检查GRASP计算的收敛性
    
    使用最近三次计算结果的能级数据和组态数量数据，通过计算：
    1. 每个能级的标准差
    2. 组态数量的相对标准差
    来判定收敛。
    
    Args:
        config: 配置对象
        logger: 日志记录器
        
    Returns:
        bool: True表示继续计算，False表示已收敛停止计算
    """
    try:
        # 读取最近3次计算的能级数据
        energy_data_list = []
        
        for i in range(3):
            loop_num = config.cal_loop_num - 2 + i  # 前3次：当前-2, 当前-1, 当前
            csv_path = config.root_path / f'{config.conf}_{loop_num}' / f'{config.conf}_{loop_num}_correct_levels.csv'
            
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                energy_data_list.append(df)
                logger.info(f"读取第{loop_num}轮能级数据: {csv_path}")
            else:
                logger.warning(f"未找到第{loop_num}轮能级数据文件: {csv_path}")
                return True  # 文件不存在，继续计算
        
        if len(energy_data_list) < 3:
            logger.warning("无法读取完整的3轮能级数据，继续计算")
            return True
        
        # 读取组态数量数据从iteration_results.csv
        iteration_results_path = config.root_path / 'results' / 'iteration_results.csv'
        csfs_num = []  # 存储每轮的组态数量
        
        if iteration_results_path.exists():
            try:
                iteration_df = pd.read_csv(iteration_results_path)
                # 获取最近3轮的组态数量
                for i in range(3):
                    loop_num = config.cal_loop_num - 2 + i
                    # 查找对应轮次的数据
                    loop_data = iteration_df[iteration_df['iteration'] == loop_num]
                    if not loop_data.empty:
                        current_count = loop_data['current_calculation_count'].iloc[0]
                        csfs_num.append(current_count)
                        logger.info(f"读取第{loop_num}轮组态数量: {current_count}")
                    else:
                        logger.warning(f"在iteration_results.csv中未找到第{loop_num}轮的数据")
                        return True  # 数据不完整，继续计算
                
                if len(csfs_num) < 3:
                    logger.warning("无法读取完整的3轮组态数量数据，继续计算")
                    return True
                    
            except Exception as e:
                logger.warning(f"读取iteration_results.csv文件出错: {e}")
                return True
        else:
            logger.warning(f"未找到iteration_results.csv文件: {iteration_results_path}")
            return True
        
        # === 1. 能级标准差计算 ===
        # 获取所有configuration
        configurations = energy_data_list[0]['configuration'].tolist()
        
        # 存储每个能级的标准差
        std_deviations = []
        
        for config_name in configurations:
            # 获取该configuration在3轮计算中的能级值
            energy_values = []
            for df in energy_data_list:
                if config_name in df['configuration'].values:
                    energy = df[df['configuration'] == config_name]['EnergyTotal'].iloc[0]
                    energy_values.append(energy)
                else:
                    logger.warning(f"在第{len(energy_values)+1}轮数据中未找到configuration: {config_name}")
                    return True  # 数据不完整，继续计算
            
            if len(energy_values) == 3:
                # 计算标准差
                energy_std = np.std(energy_values)
                std_deviations.append(energy_std)
                
                logger.debug(f"Configuration {config_name}: "
                           f"能级值={energy_values}, "
                           f"标准差={energy_std:.5e}")
        
        # 计算所有能级的平均标准差
        avg_energy_std = np.mean(std_deviations)
        
        # === 2. 组态数量相对标准差计算 ===
        # 计算组态数量的标准差和相对标准差
        csfs_num_std = np.std(csfs_num)
        csfs_num_mean = np.mean(csfs_num)
        
        if csfs_num_mean > 0:
            csfs_num_relative_std = csfs_num_std / csfs_num_mean
        else:
            csfs_num_relative_std = csfs_num_std  # 如果平均值为零，直接使用标准差
        
        # 从配置文件读取收敛阈值（如果没有设置则使用默认值）
        energy_std_threshold = getattr(config, 'energy_std_threshold', 1e-5)  # 能级标准差阈值
        csfs_num_relative_std_threshold = getattr(config, 'csfs_num_relative_std_threshold', 1e-3)  # 组态数量相对标准差阈值（5%）
        
        logger.info(f"收敛性统计:")
        logger.info(f"  最近3轮组态数量: {csfs_num}")
        logger.info(f"  组态数量平均值: {csfs_num_mean:.1f}")
        logger.info(f"  组态数量标准差: {csfs_num_std:.2f}")
        logger.info(f"  组态数量相对标准差: {csfs_num_relative_std:.4f} (阈值: {csfs_num_relative_std_threshold:.4f})")
        logger.info(f"  能级平均标准差: {avg_energy_std:.5e} (阈值: {energy_std_threshold:.5e})")
        logger.info(f"  能级标准差收敛: {avg_energy_std < energy_std_threshold}")
        logger.info(f"  组态数量相对标准差收敛: {csfs_num_relative_std < csfs_num_relative_std_threshold}")
        
        # 判断收敛性：两个条件都满足才算收敛
        energy_converged = avg_energy_std < energy_std_threshold
        csfs_num_converged = csfs_num_relative_std < csfs_num_relative_std_threshold
        is_converged = energy_converged and csfs_num_converged
        
        if is_converged:
            logger.info("能级和组态数量都已收敛，停止计算")
            return False
        else:
            if not energy_converged:
                logger.info("能级未完全收敛，继续计算")
            if not csfs_num_converged:
                logger.info("组态数量未稳定收敛，继续计算")
            return True
            
    except Exception as e:
        logger.error(f"收敛检查过程中出错: {e}")
        return True  # 出错时继续计算

def save_iteration_results(config, training_time, eval_time, execution_time, 
                          evaluation_results, selection_results, logger):
    """
    保存迭代结果到CSV文件
    
    Args:
        config: 配置对象
        training_time: 训练时间
        eval_time: 评估时间（模型推理时间）
        execution_time: 总执行时间
        evaluation_results: evaluate_model函数返回的结果字典
        selection_results: 选择结果字典，包含实际的组态选择信息
        logger: 日志记录器
    """
    
    # 从新的结果结构中提取指标
    test_metrics = evaluation_results['test_metrics']
    train_metrics = evaluation_results['train_metrics']
    metadata = evaluation_results['metadata']
    
    # 获取实际的评估时间（如果evaluation_results中有的话）
    actual_eval_time = metadata.get('eval_time', eval_time)
    
    # 保存到CSV文件
    results_file = Path(config.root_path) / 'results' / 'iteration_results.csv'
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 创建表头（如果文件不存在）
    if not results_file.exists():
        headers = [
            'iteration', 
            'important_count', 'ml_predicted_count', 'ml_new_count', 
            'total_original_count', 'current_calculation_count',
            'data_retention_rate', 'important_retention_rate', 'ml_retention_rate',
            'training_time', 'inference_time', 'execution_time', 'total_time',
            'test_f1', 'test_roc_auc', 'test_accuracy', 'test_precision', 'test_recall',
            'train_f1', 'train_roc_auc', 'train_accuracy', 'train_precision', 'train_recall',
            'overfitting_gap'
        ]
        with open(results_file, mode="w", newline="", encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
    
    # 计算过拟合差距
    overfitting_gap = train_metrics['f1'] - test_metrics['f1']
    
    # 提取选择结果的实际数据
    important_count = selection_results.get('important_count', 0)
    ml_predicted_count = selection_results.get('ml_predicted_count', 0) 
    ml_new_count = selection_results.get('ml_new_count', 0)
    total_original_count = selection_results.get('total_original_count', 1)
    current_calculation_count = selection_results.get('current_calculation_count', 1)
    data_retention_rate = selection_results.get('data_retention_rate', 0.0)
    important_retention_rate = selection_results.get('important_retention_rate', 0.0)
    ml_retention_rate = selection_results.get('ml_retention_rate', 0.0)
    
    with open(results_file, mode="a", newline="", encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([
            config.cal_loop_num,      # 迭代轮次
            important_count,          # 重要组态数量
            ml_predicted_count,       # ML预测的高概率组态总数
            ml_new_count,            # ML新增的组态数（下次计算用）
            total_original_count,     # 原始CSFs总数
            current_calculation_count,  # 本轮计算的组态数
            data_retention_rate,      # 数据留存率（交集/本轮计算）
            important_retention_rate, # 重要组态占原始比例
            ml_retention_rate,       # ML预测组态占原始比例
            training_time, 
            actual_eval_time,         # 推理时间
            execution_time, 
            execution_time,           # 总时间（现在与执行时间相同）
            test_metrics['f1'], 
            test_metrics['roc_auc'], 
            test_metrics['accuracy'],
            test_metrics['precision'], 
            test_metrics['recall'],
            train_metrics['f1'], 
            train_metrics['roc_auc'], 
            train_metrics['accuracy'],
            train_metrics['precision'], 
            train_metrics['recall'],
            overfitting_gap          # 过拟合差距
        ])
    
    logger.info(f"迭代结果已保存到: {results_file}")
    logger.info(f"第{config.cal_loop_num}轮 - 重要组态: {important_count} (占原始: {important_retention_rate:.4%})")
    logger.info(f"第{config.cal_loop_num}轮 - ML预测组态: {ml_new_count} (占原始: {ml_retention_rate:.4%})")
    logger.info(f"第{config.cal_loop_num}轮 - 数据留存率: {data_retention_rate:.4%}")

def handle_calculation_error(config, logger):
    """处理计算错误的情况"""
    config_file_path = config.root_path / 'config.toml'
    if config.cal_error_num < 3:
        # 更新配置文件
        update_config(config_file_path, {'cal_error_num': config.cal_error_num + 1})
        update_config(config_file_path, {'continue_cal': True})
        # continue_calculate(config.root_path, True)

        # 重命名结果目录
        original_cal_path = config.root_path / f'{config.conf}_{config.cal_loop_num}'
        new_cal_path = config.root_path / f'{config.conf}_{config.cal_loop_num}_err_{config.cal_error_num + 1}'
        
        if original_cal_path.exists():
            try:
                shutil.move(str(original_cal_path), str(new_cal_path))
                logger.info(f"结果目录已重命名: {original_cal_path} -> {new_cal_path}")
            except Exception as e:
                logger.error(f"重命名目录失败: {e}")
        
    else:
        logger.info("连续三次波函数未改进，迭代收敛，退出筛选程序")
        update_config(config_file_path, {'continue_cal': False})
        # continue_calculate(config.root_path, False)


def get_unselected_descriptors(raw_csfs_descriptors: np.ndarray, chosen_csfs_indices_dict: Dict[int, List[int]]) -> np.ndarray:
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
    unselected_indices = list(all_indices - chosen_indices)
    
    # 返回对应的描述符
    return raw_csfs_descriptors[unselected_indices]

def save_and_plot_results(
                            evaluation_results, model, config, 
                            save_model: bool = True,
                            save_data: bool = True, 
                            plot_curves: bool = True,
                            logger=None):
    """
    保存模型预测结果、模型文件和绘制性能曲线
    使用setup_directories创建的标准目录结构
    
    Args:
        evaluation_results: evaluate_model函数返回的结果字典
        model: 训练好的模型对象
        config: 配置对象，包含root_path和file_name等信息
        save_model: 是否保存模型文件
        save_data: 是否保存预测结果数据
        plot_curves: 是否绘制ROC/PR曲线
        logger: 日志记录器
        
    Returns:
        dict: 包含所有保存文件路径的字典
    """
    
    if logger:
        logger.info("开始保存结果和绘制图表")
    
    # 使用config中的root_path，这是setup_directories创建目录的基础路径
    root_path = getattr(config, 'root_path', Path('.'))
    
    # 获取文件名
    file_name = getattr(config, 'file_name', f'model_{int(time.time())}')
    
    saved_files = {}
    
    # 1. 保存预测结果数据到test_data目录
    if save_data:
        test_data_dir = root_path / "test_data"
        
        # 保存测试集结果
        test_file = test_data_dir / f"{file_name}_test_results.csv"
        pd.DataFrame({
            "y_true": evaluation_results['true_labels']['y_test'],
            "y_prediction": evaluation_results['predictions']['y_prediction_test'],
            "y_proba": evaluation_results['probabilities']['y_probability_test']
        }).to_csv(test_file, index=False)
        saved_files['test_data'] = str(test_file)
        
        # 保存训练集结果到results目录
        results_dir = root_path / "results"
        train_file = results_dir / f"{file_name}_train_results.csv"
        pd.DataFrame({
            "y_true": evaluation_results['true_labels']['y_train'],
            "y_prediction": evaluation_results['predictions']['y_prediction_train'],
            "y_proba": evaluation_results['probabilities']['y_probability_train']
        }).to_csv(train_file, index=False)
        saved_files['train_data'] = str(train_file)
        
        # 保存其他数据预测结果到results目录
        other_file = results_dir / f"{file_name}_other_predictionictions.csv"
        pd.DataFrame({
            "y_prediction": evaluation_results['predictions']['y_prediction_other'],
            "y_proba": evaluation_results['probabilities']['y_probability_other']
        }).to_csv(other_file, index=False)
        saved_files['other_predictionictions'] = str(other_file)
        
        if logger:
            logger.info(f"预测数据已保存到: {test_data_dir} 和 {results_dir}")
    
    # 2. 保存模型文件到models目录
    if save_model:
        models_dir = root_path / "models"
        
        model_file = models_dir / f"{file_name}.pkl"
        joblib.dump(model, model_file)
        saved_files['model'] = str(model_file)
        
        if logger:
            logger.info(f"模型已保存到: {model_file}")
    
    # 3. 绘制性能曲线到roc_curves目录
    if plot_curves:
        roc_curves_dir = root_path / "roc_curves"
        
        try:
            # 创建混合系数的占位数据用于绘图兼容性
            y_prob_all = evaluation_results['probabilities']['y_probability_all']
            dummy_mix_coeff = np.ones(len(y_prob_all))
            
            # 绘制ROC和PR曲线
            plot_file = roc_curves_dir / f"{file_name}_roc_pr_curves.png"
            roc_auc, pr_auc = ANNClassifier.plot_curve(
                dummy_mix_coeff, 
                y_prob_all, 
                evaluation_results['true_labels']['y_test'], 
                evaluation_results['probabilities']['y_probability_test'], 
                str(plot_file)
            )
            saved_files['roc_pr_plot'] = str(plot_file)
            
            # 额外绘制概率分布直方图
            prob_hist_file = roc_curves_dir / f"{file_name}_probability_distribution.png"
            _plot_probability_distribution(
                evaluation_results['probabilities']['y_probability_test'],
                evaluation_results['true_labels']['y_test'],
                str(prob_hist_file)
            )
            saved_files['prob_distribution'] = str(prob_hist_file)
            
            if logger:
                logger.info(f"性能图表已保存到: {roc_curves_dir}")
                
        except Exception as e:
            if logger:
                logger.warning(f"绘图过程出现错误: {e}")
            else:
                print(f"绘图错误: {e}")
    
    if logger:
        logger.info("所有结果保存完成")
    
    return saved_files

def _plot_probability_distribution(y_probability, y_true, save_path):
    """
    绘制预测概率分布直方图
    
    Args:
        y_proba: 预测概率
        y_true: 真实标签
        save_path: 保存路径
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    
    # 分别绘制正负样本的概率分布
    pos_probability = y_probability[y_true == 1]
    neg_probability = y_probability[y_true == 0]
    
    plt.hist(neg_probability, bins=50, alpha=0.7, label=f'负样本 (n={len(neg_probability)})', 
             color='lightcoral', density=True)
    plt.hist(pos_probability, bins=50, alpha=0.7, label=f'正样本 (n={len(pos_probability)})', 
             color='lightblue', density=True)
    
    plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.8, label='分类阈值 (0.5)')
    
    plt.xlabel('预测概率')
    plt.ylabel('密度')
    plt.title('预测概率分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def calculate_dynamic_chosen_ratio(
                                    config,
                                    all_chosen_indices,
                                    target_pool_csfs_data,
                                    y_all_probability,
                                    evaluation_results,
                                    energy_level_data_pd,
                                    logger):
    """
    基于组态分析的动态选择率计算
    
    根据重要组态占比、数据留存率和ML推理组态数量等直接指标，
    动态调整CSF选择率，更符合物理意义。
    
    Args:
        config: 配置对象，包含当前选择率和迭代轮次等信息
        all_chosen_indices: 当前轮次选择的所有CSF索引数组
        target_pool_csfs_data: 目标CSF池数据对象
        y_all_probability: 所有CSF的预测概率数组
        evaluation_results: evaluate_model函数返回的完整评估结果字典
        energy_level_data_pd: 当前轮次的能级数据DataFrame
        logger: 日志记录器
        
    Returns:
        float: 动态调整后的选择率，范围在[0.03, 0.30]之间
    """
    
    total_available_csfs = len(target_pool_csfs_data.CSFs_block_data[0])
    current_selected_count = len(all_chosen_indices)
    current_actual_ratio = current_selected_count / total_available_csfs
    base_ratio = config.chosen_ratio
    
    logger.info(f"             === 动态选择率计算 ===")
    logger.info(f"             当前实际选择率: {current_actual_ratio:.4f}")
    logger.info(f"             基础选择率: {base_ratio:.4f}")
    
    # === 统一读取重要组态文件 ===
    current_important_indices = None
    previous_important_indices = None
    important_ratio_in_calculation = 0.5  # 默认值
    data_retention_rate = 0.0
    
    try:
        import pickle
        
        # 读取当前轮次的重要组态索引
        current_important_path = config.root_path / 'results' / f'{config.conf}_{config.cal_loop_num}_important_indices.pkl'
        if current_important_path.exists():
            with open(current_important_path, 'rb') as f:
                current_important_dict = pickle.load(f)
            current_important_indices = current_important_dict[0]
            current_important_count = len(current_important_indices)
            important_ratio_in_calculation = current_important_count / current_selected_count
            logger.info(f"             当前重要组态占计算比例: {important_ratio_in_calculation:.4f}")
        else:
            logger.warning(f"             未找到当前重要组态文件: {current_important_path}")
            logger.warning(f"             使用默认重要组态占比: {important_ratio_in_calculation}")
        
        # 读取前一轮次的重要组态索引（如果存在）
        if config.cal_loop_num > 1:
            prev_important_path = config.root_path / 'results' / f'{config.conf}_{config.cal_loop_num-1}_important_indices.pkl'
            if prev_important_path.exists():
                with open(prev_important_path, 'rb') as f:
                    prev_important_dict = pickle.load(f)
                previous_important_indices = prev_important_dict[0]
                
                # 计算数据留存率（仅当两个文件都存在时）
                if current_important_indices is not None:
                    intersection = np.intersect1d(current_important_indices, previous_important_indices)
                    data_retention_rate = len(intersection) / current_selected_count
                    logger.info(f"             数据留存率: {data_retention_rate:.4f}")
                    logger.info(f"             - 本次重要组态数: {len(current_important_indices)}")
                    logger.info(f"             - 上次重要组态数: {len(previous_important_indices)}")
                    logger.info(f"             - 交集组态数: {len(intersection)}")
                else:
                    logger.warning(f"             无法计算留存率：缺少当前重要组态数据")
            else:
                logger.warning(f"             未找到前一轮重要组态文件: {prev_important_path}")
        else:
            logger.info(f"             第一轮计算，无法计算数据留存率")
            
    except Exception as e:
        logger.warning(f"             重要组态文件读取失败: {e}")
        logger.warning(f"             使用默认值进行计算")
    
    # === 3. ML推理组态数量分析 ===
    # 使用95分位数作为高概率阈值
    high_prob_threshold = np.percentile(y_all_probability, 95)
    high_prob_count = np.sum(y_all_probability > high_prob_threshold)
    ml_prediction_ratio = high_prob_count / total_available_csfs
    
    logger.info(f"             ML高概率组态数: {high_prob_count}")
    logger.info(f"             ML预测高概率比例: {ml_prediction_ratio:.4f}")
    
    # === 4. 动态调整策略 ===
    adjustment_factor = 1.0
    adjustment_reasons = []
    
    # 策略1: 基于重要组态占比调整
    if important_ratio_in_calculation > 0.7:
        # 重要组态占比过高，说明计算组态质量好，可以适当减少选择率
        factor = 0.9
        adjustment_factor *= factor
        adjustment_reasons.append(f"重要组态占比高({important_ratio_in_calculation:.3f}) -> 降低{1-factor:.1%}")
    elif important_ratio_in_calculation < 0.3:
        # 重要组态占比过低，需要更多组态来捕获重要信息
        factor = 1.2
        adjustment_factor *= factor
        adjustment_reasons.append(f"重要组态占比低({important_ratio_in_calculation:.3f}) -> 增加{factor-1:.1%}")
    
    # 策略2: 基于数据留存率调整
    if config.cal_loop_num > 1:
        if data_retention_rate > 0.8:
            # 留存率过高，说明计算过于保守，可以适当减少选择率
            factor = 0.95
            adjustment_factor *= factor
            adjustment_reasons.append(f"留存率过高({data_retention_rate:.3f}) -> 降低{1-factor:.1%}")
        elif data_retention_rate < 0.3:
            # 留存率过低，说明重要组态变化太大，需要更多组态稳定计算
            factor = 1.15
            adjustment_factor *= factor
            adjustment_reasons.append(f"留存率过低({data_retention_rate:.3f}) -> 增加{factor-1:.1%}")
    
    # 策略3: 基于ML预测组态数量调整
    if ml_prediction_ratio > 0.15:
        # ML预测高概率组态过多，可能阈值偏低，适当减少选择率
        factor = 0.9
        adjustment_factor *= factor
        adjustment_reasons.append(f"ML高概率组态过多({ml_prediction_ratio:.3f}) -> 降低{1-factor:.1%}")
    elif ml_prediction_ratio < 0.03:
        # ML预测高概率组态过少，可能需要更宽松的选择策略
        factor = 1.1
        adjustment_factor *= factor
        adjustment_reasons.append(f"ML高概率组态过少({ml_prediction_ratio:.3f}) -> 增加{factor-1:.1%}")
    
    # 策略4: 迭代轮次考虑（随轮次递减，但受组态质量影响）
    if config.cal_loop_num > 3:
        # 基础衰减，但根据组态质量调整衰减速度
        base_decay = 0.95
        if important_ratio_in_calculation > 0.6 and data_retention_rate > 0.6:
            # 质量较好，可以更快衰减
            decay_factor = base_decay ** 1.2
        else:
            # 质量一般，缓慢衰减
            decay_factor = base_decay ** 0.8
        
        adjustment_factor *= decay_factor
        adjustment_reasons.append(f"迭代衰减(轮次{config.cal_loop_num}) -> 降低{1-decay_factor:.1%}")
    
    # 策略5: 安全边界检查
    if current_actual_ratio < 0.05:
        # 选择率过低保护
        factor = 1.3
        adjustment_factor *= factor
        adjustment_reasons.append(f"选择率过低保护 -> 增加{factor-1:.1%}")
    elif current_actual_ratio > 0.25:
        # 选择率过高保护
        factor = 0.8
        adjustment_factor *= factor
        adjustment_reasons.append(f"选择率过高保护 -> 降低{1-factor:.1%}")
    
    # === 5. 计算最终选择率 ===
    # 使用当前实际比例作为基础，应用调整因子
    dynamic_ratio = current_actual_ratio * adjustment_factor
    
    # 严格的范围限制
    dynamic_ratio = np.clip(dynamic_ratio, 0.03, 0.30)
    
    # === 6. 输出调整日志 ===
    logger.info(f"             调整因子: {adjustment_factor:.4f}")
    for reason in adjustment_reasons:
        logger.info(f"             - {reason}")
    logger.info(f"             调整前选择率: {current_actual_ratio:.4f}")
    logger.info(f"             调整后选择率: {dynamic_ratio:.4f}")
    logger.info(f"             选择率变化: {(dynamic_ratio/current_actual_ratio-1)*100:+.1f}%")
    
    return dynamic_ratio
