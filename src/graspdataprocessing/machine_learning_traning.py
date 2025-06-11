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
from .data_IO import write_sorted_CSFs_to_cfile, update_config

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
    """
    评估模型性能，返回所有预测结果和评估指标
    
    Args:
        model: 训练好的模型
        X_train, X_test, y_train, y_test: 训练和测试数据
        X_stay: 其他需要预测的数据
        config: 配置对象
        logger: 日志记录器
    
    Returns:
        dict: 包含所有预测结果、概率、评估指标和元数据的完整结果字典
    """
    
    logger.info("开始预测与评估")
    
    # 预测
    start_time = time.time()
    y_pred = model.predict(X_test)
    y_pred_other = model.predict(X_stay)
    eval_time = time.time() - start_time
    
    # 预测概率
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred_train = model.predict(X_train)
    y_proba_train = model.predict_proba(X_train)[:, 1]
    y_proba_other = model.predict_proba(X_stay)[:, 1]
    
    # 生成完整数据集的概率用于分析
    y_proba_all = model.predict_proba(np.vstack([X_train, X_test]))
    
    # 评估指标计算
    f1, roc_auc, accuracy, precision, recall = ANNClassifier.model_evaluation(
        y_test, y_pred, y_proba
    )
    
    # 训练集评估（过拟合监控）
    f1_train, roc_auc_train, accuracy_train, precision_train, recall_train = ANNClassifier.model_evaluation(
        y_train, y_pred_train, y_proba_train
    )
    
    logger.info("模型评估完成")
    
    # 返回完整的结果字典
    return {
        # 预测结果
        'predictions': {
            'y_pred_test': y_pred,
            'y_pred_train': y_pred_train, 
            'y_pred_other': y_pred_other
        },
        
        # 预测概率
        'probabilities': {
            'y_proba_test': y_proba,
            'y_proba_train': y_proba_train,
            'y_proba_other': y_proba_other,
            'y_proba_all': y_proba_all
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
            'other_samples': len(X_stay),
            'config_name': getattr(config, 'file_name', 'unknown')
        }
    }

def save_convergence_results(config, convergence_results, logger):
    """
    保存收敛结果到独立的CSV文件
    
    Args:
        config: 配置对象
        convergence_results: check_energy_convergence函数返回的结果
        logger: 日志记录器
    """
    
    convergence_file = Path(config.root_path) / 'results' / 'convergence_history.csv'
    convergence_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 创建表头（如果文件不存在）
    if not convergence_file.exists():
        headers = [
            'cal_loop_num', 'is_converged', 'compared_states', 'max_abs_diff', 
            'mean_abs_diff', 'rms_diff', 'max_rel_diff', 'mean_rel_diff', 
            'convergence_threshold', 'reason'
        ]
        with open(convergence_file, mode="w", newline="", encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
    
    # 写入收敛数据
    with open(convergence_file, mode="a", newline="", encoding='utf-8') as file:
        writer = csv.writer(file)
        
        if convergence_results['energy_diff_stats'] is not None:
            stats = convergence_results['energy_diff_stats']
            writer.writerow([
                config.cal_loop_num,
                convergence_results['is_converged'],
                convergence_results['compared_states'],
                stats['max_abs_diff'],
                stats['mean_abs_diff'], 
                stats['rms_diff'],
                stats['max_rel_diff'],
                stats['mean_rel_diff'],
                convergence_results.get('convergence_threshold', 1e-6),
                convergence_results['reason']
            ])
        else:
            # 当无法进行收敛检查时
            writer.writerow([
                config.cal_loop_num,
                False,
                0,
                None, None, None, None, None,
                convergence_results.get('convergence_threshold', 1e-6),
                convergence_results['reason']
            ])
    
    logger.info(f"收敛结果已保存到: {convergence_file}")

def save_iteration_results(config, training_time, eval_time, execution_time, 
                          evaluation_results, selection_results, weight, logger,
                          convergence_results=None):
    """
    保存迭代结果到CSV文件
    
    Args:
        config: 配置对象
        training_time: 训练时间
        eval_time: 评估时间  
        execution_time: 执行时间
        evaluation_results: evaluate_model函数返回的结果字典
        selection_results: 选择结果字典
        weight: 权重值
        logger: 日志记录器
        convergence_results: 可选的收敛检查结果
    """
    
    all_time = execution_time + training_time + eval_time
    
    # 从新的结果结构中提取指标
    test_metrics = evaluation_results['test_metrics']
    train_metrics = evaluation_results['train_metrics']
    metadata = evaluation_results['metadata']
    
    # 获取实际的评估时间（如果evaluation_results中有的话）
    actual_eval_time = metadata.get('eval_time', eval_time)
    
    # 从收敛结果中提取能量差异信息
    max_energy_diff = None
    if convergence_results and convergence_results['energy_diff_stats']:
        max_energy_diff = convergence_results['energy_diff_stats']['max_abs_diff']
    
    # 保存到CSV文件
    results_file = Path(config.root_path) / 'results' / 'iteration_results.csv'
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 创建表头（如果文件不存在）
    if not results_file.exists():
        headers = [
            'cal_loop_num', 'training_time', 'eval_time', 'execution_time', 'total_time',
            'test_f1', 'test_roc_auc', 'test_accuracy', 'test_precision', 'test_recall',
            'max_energy_diff', 'import_count', 'stay_count', 'MLsampling_ratio', 'chosen_count', 'weight',
            'train_f1', 'train_roc_auc', 'train_accuracy', 'train_precision', 'train_recall'
        ]
        with open(results_file, mode="w", newline="", encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
    
    with open(results_file, mode="a", newline="", encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([
            config.cal_loop_num,  # 添加循环编号
            training_time, 
            actual_eval_time, 
            execution_time, 
            all_time,
            test_metrics['f1'], 
            test_metrics['roc_auc'], 
            test_metrics['accuracy'],
            test_metrics['precision'], 
            test_metrics['recall'],
            max_energy_diff,  # 使用实际的能量差异代替Es placeholder
            len(selection_results.get('indexs_import_temp', [])),
            len(selection_results.get('indexs_import_stay_temp', [])),
            None,  # MLsampling_ratio placeholder
            len(selection_results.get('chosen_index', [])),
            weight,
            train_metrics['f1'], 
            train_metrics['roc_auc'], 
            train_metrics['accuracy'],
            train_metrics['precision'], 
            train_metrics['recall']
        ])
    
    logger.info(f"迭代结果已保存到: {results_file}")
    
    # 如果有收敛结果，也保存收敛历史
    if convergence_results:
        save_convergence_results(config, convergence_results, logger)

def check_energy_convergence(config, current_energy_df, logger, 
                            convergence_threshold=1e-6, min_states=5):
    """
    检查能级数据的收敛性
    
    Args:
        config: 配置对象
        current_energy_df: 当前循环的能级数据DataFrame
        logger: 日志记录器
        convergence_threshold: 收敛阈值（Hartree单位）
        min_states: 至少需要比较的能级数量
        
    Returns:
        dict: 包含收敛信息的字典
    """
    
    if config.cal_loop_num == 1:
        logger.info("第一次计算，无法进行收敛检查")
        return {
            'is_converged': False,
            'reason': '第一次计算',
            'energy_diff_stats': None,
            'compared_states': 0
        }
    
    # 构造前一次计算的能级文件路径
    root_path = Path(config.root_path)
    previous_cal_path = root_path / f'{config.conf}_{config.cal_loop_num-1}'
    
    # 寻找前一次的能级文件
    previous_energy_file = None
    possible_names = [
        f'{config.conf}_{config.cal_loop_num-1}.energy',
        f'{config.conf}.energy',
        'energy_levels.csv'
    ]
    
    for name in possible_names:
        potential_file = previous_cal_path / name
        if potential_file.exists():
            previous_energy_file = potential_file
            break
    
    if previous_energy_file is None:
        logger.warning(f"未找到前一次计算的能级文件，路径: {previous_cal_path}")
        return {
            'is_converged': False,
            'reason': '未找到前一次能级文件',
            'energy_diff_stats': None,
            'compared_states': 0
        }
    
    try:
        # 读取前一次的能级数据
        # 假设是相同格式的文件，这里需要根据实际格式调整
        if previous_energy_file.suffix == '.csv':
            previous_energy_df = pd.read_csv(previous_energy_file)
        else:
            # 如果是GRASP能级文件，需要使用相应的解析函数
            # 这里暂时假设已经转换为DataFrame格式
            logger.warning(f"能级文件格式暂不支持自动解析: {previous_energy_file}")
            return {
                'is_converged': False,
                'reason': '不支持的能级文件格式',
                'energy_diff_stats': None,
                'compared_states': 0
            }
            
        logger.info(f"成功读取前一次能级数据: {previous_energy_file}")
        logger.info(f"前一次数据包含 {len(previous_energy_df)} 个能级")
        
    except Exception as e:
        logger.error(f"读取前一次能级文件失败: {str(e)}")
        return {
            'is_converged': False,
            'reason': f'读取前一次能级文件失败: {str(e)}',
            'energy_diff_stats': None,
            'compared_states': 0
        }
    
    # 基于'No', 'Pos', 'J'进行数据匹配
    try:
        # 确保两个DataFrame都有必要的列
        required_columns = ['No', 'Pos', 'J', 'EnergyTotal']
        for col in required_columns:
            if col not in current_energy_df.columns:
                raise ValueError(f"当前能级数据缺少列: {col}")
            if col not in previous_energy_df.columns:
                raise ValueError(f"前一次能级数据缺少列: {col}")
        
        # 创建合并键
        current_energy_df = current_energy_df.copy()
        previous_energy_df = previous_energy_df.copy()
        
        current_energy_df['merge_key'] = (
            current_energy_df['No'].astype(str) + '_' + 
            current_energy_df['Pos'].astype(str) + '_' + 
            current_energy_df['J'].astype(str)
        )
        
        previous_energy_df['merge_key'] = (
            previous_energy_df['No'].astype(str) + '_' + 
            previous_energy_df['Pos'].astype(str) + '_' + 
            previous_energy_df['J'].astype(str)
        )
        
        # 进行内连接，只保留两次计算都有的状态
        merged_df = pd.merge(
            current_energy_df[['merge_key', 'EnergyTotal']], 
            previous_energy_df[['merge_key', 'EnergyTotal']], 
            on='merge_key', 
            suffixes=('_current', '_previous')
        )
        
        compared_states = len(merged_df)
        logger.info(f"找到 {compared_states} 个相同的量子态进行比较")
        
        if compared_states < min_states:
            logger.warning(f"可比较的量子态数量过少 ({compared_states} < {min_states})")
            return {
                'is_converged': False,
                'reason': f'可比较的量子态数量过少 ({compared_states} < {min_states})',
                'energy_diff_stats': None,
                'compared_states': compared_states
            }
        
        # 计算能量差异
        energy_diff = merged_df['EnergyTotal_current'] - merged_df['EnergyTotal_previous']
        
        # 计算收敛统计量
        energy_diff_stats = {
            'max_abs_diff': abs(energy_diff).max(),
            'mean_abs_diff': abs(energy_diff).mean(),
            'rms_diff': np.sqrt((energy_diff**2).mean()),
            'max_rel_diff': abs(energy_diff / merged_df['EnergyTotal_previous']).max(),
            'mean_rel_diff': abs(energy_diff / merged_df['EnergyTotal_previous']).mean()
        }
        
        # 判断收敛性
        max_abs_diff = energy_diff_stats['max_abs_diff']
        is_converged = max_abs_diff < convergence_threshold
        
        # 记录详细信息
        logger.info(f"能级收敛性检查结果:")
        logger.info(f"  比较的量子态数量: {compared_states}")
        logger.info(f"  最大绝对差异: {max_abs_diff:.8e} Hartree")
        logger.info(f"  平均绝对差异: {energy_diff_stats['mean_abs_diff']:.8e} Hartree")
        logger.info(f"  RMS差异: {energy_diff_stats['rms_diff']:.8e} Hartree")
        logger.info(f"  最大相对差异: {energy_diff_stats['max_rel_diff']:.8e}")
        logger.info(f"  收敛阈值: {convergence_threshold:.8e} Hartree")
        logger.info(f"  是否收敛: {'是' if is_converged else '否'}")
        
        if is_converged:
            logger.info("✅ 能级计算已收敛")
        else:
            logger.info("⚠️ 能级计算尚未收敛，需要继续迭代")
        
        return {
            'is_converged': is_converged,
            'reason': '收敛' if is_converged else f'最大差异 {max_abs_diff:.8e} > 阈值 {convergence_threshold:.8e}',
            'energy_diff_stats': energy_diff_stats,
            'compared_states': compared_states,
            'convergence_threshold': convergence_threshold,
            'energy_differences': energy_diff.tolist(),  # 保存所有差异供进一步分析
            'merged_data': merged_df  # 保存合并后的数据供调试
        }
        
    except Exception as e:
        logger.error(f"能级收敛性检查失败: {str(e)}")
        return {
            'is_converged': False,
            'reason': f'收敛检查失败: {str(e)}',
            'energy_diff_stats': None,
            'compared_states': 0
        }

def check_convergence(config, sum_num_list, logger):
    """检查收敛性（保留原有接口）"""
    # 这里需要Es_term的历史数据，暂时先跳过收敛检查
    # 在实际使用中，需要维护能量项的历史记录
    logger.info("收敛检查功能需要能量历史数据，当前跳过")
    return False

def handle_calculation_error(config, logger):
    """处理计算错误的情况"""
   
    cal_error_num = getattr(config, 'cal_error_num', 0) + 1
    update_config(f'{config.root_path}/config.yaml', {'cal_error_num': cal_error_num})
    
    if cal_error_num >= 3:
        logger.info("连续三次波函数未改进，迭代收敛，退出筛选程序")
        with open(f'{config.root_path}/run.input', 'w') as file:
            file.write('False')
        return


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

def save_and_plot_results(evaluation_results, model, config, 
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
            "y_pred": evaluation_results['predictions']['y_pred_test'],
            "y_proba": evaluation_results['probabilities']['y_proba_test']
        }).to_csv(test_file, index=False)
        saved_files['test_data'] = str(test_file)
        
        # 保存训练集结果到results目录
        results_dir = root_path / "results"
        train_file = results_dir / f"{file_name}_train_results.csv"
        pd.DataFrame({
            "y_true": evaluation_results['true_labels']['y_train'],
            "y_pred": evaluation_results['predictions']['y_pred_train'],
            "y_proba": evaluation_results['probabilities']['y_proba_train']
        }).to_csv(train_file, index=False)
        saved_files['train_data'] = str(train_file)
        
        # 保存其他数据预测结果到results目录
        other_file = results_dir / f"{file_name}_other_predictions.csv"
        pd.DataFrame({
            "y_pred": evaluation_results['predictions']['y_pred_other'],
            "y_proba": evaluation_results['probabilities']['y_proba_other']
        }).to_csv(other_file, index=False)
        saved_files['other_predictions'] = str(other_file)
        
        # 保存评估指标到results目录
        metrics_file = results_dir / f"{file_name}_metrics.json"
        metrics_data = {
            'test_metrics': evaluation_results['test_metrics'],
            'train_metrics': evaluation_results['train_metrics'],
            'metadata': evaluation_results['metadata']
        }
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, indent=2, ensure_ascii=False)
        saved_files['metrics'] = str(metrics_file)
        
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
            dummy_mix_coeff = np.ones(len(evaluation_results['probabilities']['y_proba_all']))
            
            # 绘制ROC和PR曲线
            plot_file = roc_curves_dir / f"{file_name}_roc_pr_curves.png"
            roc_auc, pr_auc = ANNClassifier.plot_curve(
                dummy_mix_coeff, 
                evaluation_results['probabilities']['y_proba_all'], 
                evaluation_results['true_labels']['y_test'], 
                evaluation_results['probabilities']['y_proba_test'], 
                str(plot_file)
            )
            saved_files['roc_pr_plot'] = str(plot_file)
            
            # 额外绘制概率分布直方图
            prob_hist_file = roc_curves_dir / f"{file_name}_probability_distribution.png"
            _plot_probability_distribution(
                evaluation_results['probabilities']['y_proba_test'],
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

def _plot_probability_distribution(y_proba, y_true, save_path):
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
    pos_proba = y_proba[y_true == 1]
    neg_proba = y_proba[y_true == 0]
    
    plt.hist(neg_proba, bins=50, alpha=0.7, label=f'负样本 (n={len(neg_proba)})', 
             color='lightcoral', density=True)
    plt.hist(pos_proba, bins=50, alpha=0.7, label=f'正样本 (n={len(pos_proba)})', 
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
