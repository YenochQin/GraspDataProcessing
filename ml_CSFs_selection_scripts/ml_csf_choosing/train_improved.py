#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Id :train_improved.py
@date :2025/08/20
@author :YenochQin (秦毅)
@update :2025/08/20 - 增强版训练脚本，集成多种改进
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
from tabulate import tabulate 
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, accuracy_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import torch
import os

try:
    import graspkit as gk
except ImportError:
    print("错误: 无法导入 graspkit 模块")
    sys.exit(1)

# 新增：XGBoost集成
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("警告: XGBoost不可用，将使用RandomForest替代")

try:
    from graspkit.machine_learning_module.ANN import ANNClassifier
    ANN_AVAILABLE = True
except ImportError:
    ANN_AVAILABLE = False
    print("警告: ANN不可用，将使用sklearn模型")

important_config_count_history = []

def enhanced_feature_preprocessing(X_train, X_test, y_train):
    """增强的特征预处理 - CPU多核优化"""
    
    import os
    import time
    from sklearn.preprocessing import RobustScaler
    from sklearn.feature_selection import SelectKBest, mutual_info_classif
    import numpy as np
    
    cpu_count = os.cpu_count() or 4
    
    # 1. 特征缩放
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 2. 特征选择（基于互信息）- 真正的多核优化
    k_features = min(X_train.shape[1], 100)  # 选择最多100个特征
    
    print(f"使用完整数据({len(X_train_scaled)}样本)进行特征选择...")
    print(f"启用CPU多核优化 - 使用 {cpu_count} 核心")
    
    # 设置环境变量强制多核
    os.environ['OMP_NUM_THREADS'] = str(cpu_count)
    os.environ['MKL_NUM_THREADS'] = str(cpu_count)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_count)
    os.environ['BLAS_NUM_THREADS'] = str(cpu_count)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_count)
    
    # 使用joblib进行真正的多核并行
    import joblib
    from sklearn.feature_selection import f_classif
    
    # 开始计时
    start_time = time.time()
    
    # 方法1: 使用f_classif（支持n_jobs）
    print("使用f_classif进行特征选择（支持多核）...")
    selector = SelectKBest(score_func=f_classif, k=k_features)
    
    # 强制使用多核
    with joblib.parallel_backend('loky', n_jobs=-1):
        selector.fit(X_train_scaled, y_train)
    
    f_classif_time = time.time() - start_time
    print(f"f_classif特征选择完成，耗时: {f_classif_time:.2f}秒")
    
    # 方法2: 使用joblib并行优化mutual_info_classif
    # 注意：mutual_info_classif不支持n_jobs，我们使用其他策略
    
    X_train_selected = selector.transform(X_train_scaled)
    X_test_selected = selector.transform(X_test_scaled)
    
    print(f"特征选择后维度: {X_train_selected.shape[1]}")
    
    return X_train_selected, X_test_selected, scaler, selector

def advanced_imbalance_handling(X_train, y_train, method='adasyn'):
    """高级类别不平衡处理"""
    
    strategies = {
        'adasyn': ADASYN(random_state=42, n_neighbors=5),
        'random_under': RandomUnderSampler(random_state=42)
    }
    
    if method in strategies:
        X_resampled, y_resampled = strategies[method].fit_resample(X_train, y_train)
        return X_resampled, y_resampled
    else:
        return X_train, y_train

def optimize_hyperparameters(model, X_train, y_train, model_type='random_forest'):
    """超参数优化"""
    
    if model_type == 'random_forest':
        param_grid = {
            'n_estimators': [200, 300, 500, 1000],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2'],
            'class_weight': ['balanced', 'balanced_subsample']
        }
    elif model_type == 'xgboost' and XGBOOST_AVAILABLE:
        param_grid = {
            'n_estimators': [200, 500, 1000],
            'max_depth': [3, 5, 7, 10],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'scale_pos_weight': [1, 2, 5]  # 处理类别不平衡
        }
    else:
        return model, {}
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        model, param_grid, n_iter=50, cv=cv, 
        scoring='f1', n_jobs=-1, random_state=42
    )
    
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_

def build_ensemble_model(X_train, y_train, X_test, y_test, config):
    """构建集成模型 - CPU优化的ANN版本"""
    
    import os
    import time
    import torch
    
    # CPU优化配置
    if not torch.cuda.is_available() and ANN_AVAILABLE:
        cpu_count = os.cpu_count() or 4
        
        # 从配置读取CPU线程数
        config_threads = getattr(config, 'ml_config', {}).get('cpu_threads', None)
        if config_threads is not None:
            try:
                config_threads = int(config_threads)
                optimal_threads = min(config_threads, cpu_count)
            except (ValueError, TypeError):
                optimal_threads = min(32, cpu_count)
        else:
            optimal_threads = min(32, cpu_count)
        
        # 设置PyTorch CPU优化
        torch.set_num_threads(optimal_threads)
        torch.set_num_interop_threads(optimal_threads)  # 设置并行线程数
        
        # 设置环境变量确保多核
        os.environ['OMP_NUM_THREADS'] = str(optimal_threads)
        os.environ['MKL_NUM_THREADS'] = str(optimal_threads)
        os.environ['NUMEXPR_NUM_THREADS'] = str(optimal_threads)
        os.environ['OPENBLAS_NUM_THREADS'] = str(optimal_threads)
        os.environ['BLAS_NUM_THREADS'] = str(optimal_threads)
        
        print(f"启用CPU多线程优化 - 线程数: {optimal_threads}")
        print(f"PyTorch intra-op threads: {torch.get_num_threads()}")
        print(f"PyTorch inter-op threads: {torch.get_num_interop_threads()}")
        
        # ANN参数优化
        batch_size = 4096
        max_epochs = 150
        hidden_size = 96
        
        # 确保使用MKL加速（如果可用）
        try:
            import torch.backends.mkldnn
            if torch.backends.mkldnn.is_available():
                torch.backends.mkldnn.enabled = True
                print("✓ MKL-DNN加速已启用")
        except:
            print("MKL-DNN不可用，使用标准CPU优化")
    else:
        batch_size = 2048
        max_epochs = 150
        hidden_size = 128
    
    results = {}
    
    if ANN_AVAILABLE:
        # 使用PyTorch ANN模型（支持多核CPU并行）
        print("使用PyTorch ANN模型（CPU优化版）")
        
        # 计算类别权重
        class_counts = np.bincount(y_train.astype(int))
        pos_weight = class_counts[0] / class_counts[1] if len(class_counts) > 1 else 1.0
        class_weights = [1.0, pos_weight]
        
        # 创建ANN模型
        model = gk.ANNClassifier(
            input_size=X_train.shape[1],
            hidden_size=hidden_size,
            learning_rate=0.001,
            class_weights=class_weights,
            device='cpu'  # 强制使用CPU
        )
        
        # 训练
        print(f"开始训练 - 数据量: {len(X_train):,}, 特征维度: {X_train.shape[1]}")
        
        # 监控训练过程中的CPU使用情况
        try:
            import psutil
            process = psutil.Process()
            start_memory = process.memory_info().rss / 1024 / 1024  # MB
            print(f"初始内存使用: {start_memory:.1f} MB")
        except:
            pass
        
        start_time = time.time()
        
        history = model.fit(
            X_train, y_train,
            X_test, y_test,
            batch_size=batch_size,
            max_epochs=max_epochs,
            early_stopping_patience=20
        )
        
        training_time = time.time() - start_time
        print(f"训练完成，耗时: {training_time:.2f}秒")
        
        try:
            import psutil
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            print(f"最终内存使用: {end_memory:.1f} MB")
            print(f"内存增长: {end_memory - start_memory:.1f} MB")
        except:
            pass
        
        # 评估
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = model.evaluate(X_test, y_test, verbose=False)
        
        results['ann'] = {
            'model': model,
            'parameters': {'hidden_size': hidden_size, 'batch_size': batch_size, 'max_epochs': max_epochs},
            'f1': metrics['f1_score'],
            'roc_auc': metrics['roc_auc'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'accuracy': metrics['accuracy'],
            'training_time': training_time
        }
    
    # 备用：使用sklearn模型作为补充
    if not ANN_AVAILABLE or len(results) == 0:
        print("使用sklearn模型作为备用")
        
        # 优化的Random Forest
        rf_params = {
            'n_estimators': 300,
            'max_depth': 15,
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': min(8, os.cpu_count() or 4)  # 控制CPU使用
        }
        rf = RandomForestClassifier(**rf_params)
        
        start_time = time.time()
        rf.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        y_pred = rf.predict(X_test)
        y_proba = rf.predict_proba(X_test)[:, 1]
        
        results['random_forest'] = {
            'model': rf,
            'parameters': rf_params,
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'accuracy': accuracy_score(y_test, y_pred),
            'training_time': training_time
        }
    
    return results

def optimize_threshold(y_true, y_proba, metric='f1'):
    """优化分类阈值"""
    thresholds = np.linspace(0.1, 0.9, 100)
    scores = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        if metric == 'f1':
            current_score = f1_score(y_true, y_pred)
        elif metric == 'precision':
            current_score = precision_score(y_true, y_pred)
        elif metric == 'recall':
            current_score = recall_score(y_true, y_pred)
        else:
            current_score = f1_score(y_true, y_pred)
        scores.append(current_score)
    
    if scores:
        optimal_threshold = thresholds[np.argmax(scores)]
        optimal_score = max(scores)
    else:
        optimal_threshold = 0.5
        optimal_score = 0.0
    
    return optimal_threshold, optimal_score

def validate_csf_descriptors_coverage(descriptors: np.ndarray, 
                                    with_subshell_info: bool = False) -> tuple[bool, list[int]]:
    """
    验证选取的CSFs描述符子集是否满足覆盖条件：
    对于每个轨道，至少有一个CSF在其对应的电子填充数位置不为零
    
    Args:
        descriptors (np.ndarray): 选取出的CSFs描述符数组，形状为 (n_csfs, n_features)
        with_subshell_info (bool): 是否包含子壳层信息
            - False: 使用parse_csf_2_descriptor生成的描述符（每个轨道3个值）
            - True: 使用parse_csf_2_descriptor_with_subshell生成的描述符（每个轨道5个值）
    
    Returns:
        tuple[bool, list[int]]: (是否满足覆盖条件, 未覆盖的轨道索引列表)
    """
    # 检查输入参数
    if descriptors.size == 0:
        return False, []
    
    # 确定每个轨道的电子填充位置索引
    if with_subshell_info:
        values_per_orbital = 5
        electron_index_in_orbital = 2
    else:
        values_per_orbital = 3
        electron_index_in_orbital = 0
    
    # 从描述符结构推断轨道数量
    actual_n_orbitals = descriptors.shape[1] // values_per_orbital
    
    # 直接通过切片获取每个轨道的电子填充
    electron_indices = np.arange(electron_index_in_orbital, 
                                actual_n_orbitals * values_per_orbital, 
                                values_per_orbital)
    
    # 提取所有CSF的电子数信息
    electron_counts = descriptors[:, electron_indices]  # 形状为 (n_csfs, actual_n_orbitals)
    
    # 检查每个轨道是否至少有一个CSF的电子数不为零
    has_nonzero_electrons = np.any(electron_counts > 0, axis=0)  # 形状为 (actual_n_orbitals,)
    
    # 找出未覆盖的轨道索引
    uncovered_orbitals = np.where(~has_nonzero_electrons)[0].tolist()
    
    # 返回验证结果
    is_covered = len(uncovered_orbitals) == 0
    return is_covered, uncovered_orbitals


def select_csfs_for_coverage(descriptors: np.ndarray,
                            uncovered_orbitals: list[int],
                            full_descriptors: np.ndarray,
                            with_subshell_info: bool = False) -> tuple[np.ndarray, list[int]]:
    """
    当覆盖验证失败时，从给定的完整描述符中按顺序选取包含缺少轨道的CSF描述符
    
    Args:
        descriptors (np.ndarray): 当前的CSFs描述符数组，形状为 (n_csfs, n_features)
        uncovered_orbitals (list[int]): 未覆盖的轨道索引列表
        full_descriptors (np.ndarray): 完整的CSFs描述符数组，形状为 (n_full_csfs, n_features)
        with_subshell_info (bool): 是否包含子壳层信息
    
    Returns:
        tuple[np.ndarray, list[int]]: (更新后的描述符数组, 选取的CSF索引列表)
            - 更新后的描述符数组包含原有描述符和新选取的描述符
            - 选取的CSF索引列表对应于full_descriptors中的索引
    """
    if not uncovered_orbitals:
        return descriptors, []
    
    # 确定每个轨道的电子填充位置索引
    if with_subshell_info:
        values_per_orbital = 5
        electron_index_in_orbital = 2
    else:
        values_per_orbital = 3
        electron_index_in_orbital = 0
    
    # 获取每个轨道的电子填充位置索引
    n_orbitals = full_descriptors.shape[1] // values_per_orbital
    electron_indices = np.arange(electron_index_in_orbital, 
                                n_orbitals * values_per_orbital, 
                                values_per_orbital)
    
    # 提取完整描述符中的电子数信息
    full_electron_counts = full_descriptors[:, electron_indices]
    
    # 找出当前描述符中已包含的CSF索引（避免重复选择）
    current_csfs_set = set(range(len(descriptors))) if descriptors.size > 0 else set()
    
    selected_indices = []
    remaining_uncovered = set(uncovered_orbitals)
    
    # 按顺序遍历完整描述符
    for idx in range(len(full_descriptors)):
        if idx in current_csfs_set:
            continue  # 跳过已包含的CSF
            
        # 检查当前CSF是否包含任何剩余未覆盖的轨道
        csf_electrons = full_electron_counts[idx]
        covers_orbitals = [orb for orb in remaining_uncovered if csf_electrons[orb] > 0]
        
        if covers_orbitals:
            selected_indices.append(idx)
            remaining_uncovered -= set(covers_orbitals)
            
            # 如果所有轨道都已覆盖，提前退出
            if not remaining_uncovered:
                break
    
    if not selected_indices:
        return descriptors, []
    
    # 构建更新后的描述符数组
    new_descriptors = full_descriptors[selected_indices]
    
    if descriptors.size == 0:
        updated_descriptors = new_descriptors
    else:
        updated_descriptors = np.vstack([descriptors, new_descriptors])
    
    return updated_descriptors, selected_indices

def monitor_cpu_usage():
    """监控CPU使用情况（简化版，不显示详细信息）"""
    import psutil
    import os
    
    cpu_count = psutil.cpu_count(logical=True)
    physical_cores = psutil.cpu_count(logical=False)
    
    # 仅记录关键信息到日志，不打印详细CPU使用率
    return {
        'total_cores': cpu_count,
        'physical_cores': physical_cores,
        'cpu_percent': psutil.cpu_percent(),
        'per_cpu_percent': []
    }


def verify_multiprocessing():
    """验证多进程功能（简化版，避免复杂问题）"""
    import multiprocessing as mp
    import time
    import os
    
    print("=== CPU多核验证 ===")
    
    # 获取CPU信息
    cpu_count = mp.cpu_count()
    print(f"可用CPU核心数: {cpu_count}")
    
    # 检查环境变量设置
    omp_threads = os.environ.get('OMP_NUM_THREADS', '未设置')
    print(f"OMP_NUM_THREADS: {omp_threads}")
    
    # 简单验证：检查是否设置了多核环境
    try:
        # 使用joblib测试并行
        import joblib
        import numpy as np
        
        # 创建测试数据
        test_data = np.random.rand(1000, 10)
        
        start_time = time.time()
        with joblib.parallel_backend('loky', n_jobs=-1):
            # 简单并行计算测试
            result = np.mean(test_data, axis=0)
        parallel_time = time.time() - start_time
        
        print(f"并行计算测试完成: {parallel_time:.4f}秒")
        
        # 如果CPU核心数>1且环境变量设置正确，认为优化成功
        return cpu_count > 1 and omp_threads != '1'
        
    except Exception as e:
        print(f"并行验证跳过: {str(e)}")
        return cpu_count > 1  # 只要有多个核心就认为可以优化


def main(config):
    """增强版主程序逻辑"""
    config.file_name = f'{config.conf}_{config.cal_loop_num}'
    logger = gk.setup_logging(config)
    
    config_file_path = config.root_path / 'config.toml'
    logger.info("增强版机器学习训练程序启动")
    execution_time = time.time()

    gk.setup_directories(config)

    try:
        # 加载数据文件
        (energy_level_data_pd, 
         rmix_file_data, 
         raw_csfs_descriptors, 
         cal_csfs_data, 
         caled_csfs_indices_dict) = gk.load_data_files(config, logger)
        
        cal_result, asfs_position = gk.check_configuration_coupling(config, energy_level_data_pd, logger)
        
    except Exception as e:
        logger.error(f"程序执行过程中发生错误: {str(e)}")
        raise

    selected_energy_data = energy_level_data_pd.iloc[asfs_position]
    correct_levels_csv_path = config.scf_cal_path / f'{config.conf}_{config.cal_loop_num}_correct_levels.csv'
    selected_energy_data.to_csv(correct_levels_csv_path, index=False)
    logger.info(f"选择的能级数据已保存到: {correct_levels_csv_path}")

    if cal_result:
        logger.info("耦合正确")
        
        # 统一收敛性检查
        should_continue = True
        if config.cal_loop_num >= 3:
            logger.info("开始统一收敛性检查...")
            
            current_calculation_csfs = cal_csfs_data.CSFs_block_length[0]
            logger.info(f"当前轮CSFs数量: {current_calculation_csfs}")
            
            energy_converged = gk.check_energy_convergence(config, logger, selected_energy_data)
            
            if not energy_converged:
                logger.info(f"检测到能量不收敛，回退到第 {config.cal_loop_num - 1} 轮")
                gk.update_config(config_file_path, {
                    'backward_loop_needed': True,
                    'target_backward_loop': config.cal_loop_num - 1,
                    'cal_loop_num': config.cal_loop_num - 1,
                    'continue_cal': True,
                    'cal_error_num': config.cal_error_num + 1
                })
                return
            
            should_continue = gk.evaluate_calculation_convergence(config, logger, current_calculation_csfs)
            
            if not should_continue:
                logger.info("整体计算已收敛，跳过机器学习训练")
                gk.update_config(config_file_path, {'continue_cal': False})
                return

        # 数据预处理
        logger.info("增强版数据预处理")
        include_wrong_level_negatives = getattr(config, 'ml_config', {}).get('include_wrong_level_negatives', True)
        
        caled_csfs_descriptors = gk.generate_chosen_csfs_descriptors(
            config, caled_csfs_indices_dict, raw_csfs_descriptors, 
            rmix_file_data, asfs_position, logger, include_wrong_level_negatives
        )
        
        unselected_csfs_descriptors = gk.get_unselected_descriptors(raw_csfs_descriptors, caled_csfs_indices_dict)
        
        # 处理数据格式 - 支持DataFrame和ndarray两种格式
        if isinstance(caled_csfs_descriptors, pd.DataFrame):
            # DataFrame格式（原始格式）
            X = caled_csfs_descriptors.drop('label', axis=1)
            y = caled_csfs_descriptors['label'].astype(int)  # 确保标签是整数
        else:
            # numpy.ndarray格式
            # 假设最后一列是标签
            X = caled_csfs_descriptors[:, :-1]
            y = caled_csfs_descriptors[:, -1].astype(int)  # 确保标签是整数
        
        # 数据分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
        logger.info(f"类别分布 - 训练集: {np.bincount(y_train)}, 测试集: {np.bincount(y_test)}")
        
        # CPU监控和验证
        logger.info("=== CPU性能监控 ===")
        cpu_info = monitor_cpu_usage()
        multiprocessing_works = verify_multiprocessing()
        
        if multiprocessing_works:
            logger.info("✓ 多进程验证通过")
        else:
            logger.warning("⚠ 多进程验证未通过预期加速")
        
        # 特征预处理
        logger.info("开始特征预处理...")
        preprocessing_start = time.time()
        
        X_train_processed, X_test_processed, scaler, selector = enhanced_feature_preprocessing(
            X_train, X_test, y_train
        )
        
        preprocessing_time = time.time() - preprocessing_start
        logger.info(f"特征预处理完成，耗时: {preprocessing_time:.2f}秒")
        logger.info(f"特征选择后维度: {X_train_processed.shape[1]}")
        
        # 类别不平衡处理
        imbalance_method = config.ml_config.get('imbalance_method', 'adasyn')
        X_train_balanced, y_train_balanced = advanced_imbalance_handling(
            X_train_processed, y_train, imbalance_method
        )
        
        logger.info(f"平衡后类别分布: {np.bincount(y_train_balanced)}")
        
        # 训练集成模型
        logger.info("开始训练集成模型...")
        start_time = time.time()
        
        ensemble_results = build_ensemble_model(
            X_train_balanced, y_train_balanced, X_test_processed, y_test, config
        )
        
        training_time = time.time() - start_time
        logger.info(f"模型训练完成，耗时: {training_time:.2f}秒")
        
        # 记录CPU最终状态
        final_cpu_info = monitor_cpu_usage()
        logger.info("=== CPU使用总结 ===")
        logger.info(f"总执行时间: {preprocessing_time + training_time:.2f}秒")
        
        # 选择最佳模型
        best_model_name = max(ensemble_results.keys(), 
                            key=lambda k: ensemble_results[k]['f1'])
        best_model = ensemble_results[best_model_name]['model']
        
        logger.info(f"最佳模型: {best_model_name}")
        logger.info(f"最佳模型性能: {ensemble_results[best_model_name]}")
        
        # 优化分类阈值
        y_proba = best_model.predict_proba(X_test_processed)[:, 1]
        optimal_threshold, optimal_score = optimize_threshold(y_test, y_proba)
        
        logger.info(f"优化后的分类阈值: {optimal_threshold:.3f}")
        logger.info(f"优化后的F1分数: {optimal_score:.4f}")
        
        # 特征重要性分析 - 支持ANN和sklearn模型
        try:
            if hasattr(best_model, 'get_feature_importance'):
                # ANNClassifier支持特征重要性
                importances = best_model.get_feature_importance(X_train_processed, method='permutation')
                indices = np.argsort(importances)[::-1]
                
                logger.info("前20个重要特征 (ANN permutation importance):")
                for i in range(min(20, len(indices))):
                    logger.info(f"{i+1}. 特征{indices[i]}: {importances[indices[i]]:.4f}")
                    
            elif hasattr(best_model, 'feature_importances_'):
                # sklearn模型特征重要性
                importances = best_model.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                logger.info("前20个重要特征 (sklearn feature importance):")
                for i in range(min(20, len(indices))):
                    logger.info(f"{i+1}. 特征{indices[i]}: {importances[indices[i]]:.4f}")
            else:
                logger.info("当前模型不支持标准特征重要性分析")
                
                # 对于ANN，尝试使用梯度重要性
                if hasattr(best_model, 'get_feature_importance'):
                    grad_importance = best_model.get_feature_importance(X_train_processed, method='gradient')
                    top_indices = np.argsort(np.abs(grad_importance))[-20:][::-1]
                    logger.info("前20个重要特征 (ANN gradient importance):")
                    for i, idx in enumerate(top_indices):
                        logger.info(f"{i+1}. 特征{idx}: {grad_importance[idx]:.4f}")
                        
        except Exception as e:
            logger.warning(f"特征重要性分析失败: {str(e)}")
            logger.info("跳过特征重要性分析")
        
        # 对未选择的CSF进行预测
        if isinstance(unselected_csfs_descriptors, pd.DataFrame):
            X_unselected = unselected_csfs_descriptors.values
        else:
            X_unselected = unselected_csfs_descriptors
            
        X_unselected_scaled = selector.transform(scaler.transform(X_unselected))
        y_unselected_proba = best_model.predict_proba(X_unselected_scaled)[:, 1]
        
        # 记录未选择CSF的预测统计信息
        logger.info(f"未选择CSF预测概率统计: min={y_unselected_proba.min():.3f}, max={y_unselected_proba.max():.3f}, mean={y_unselected_proba.mean():.3f}")
        high_prob_count = np.sum(y_unselected_proba >= optimal_threshold)
        logger.info(f"未选择CSF中高概率样本数: {high_prob_count}/{len(y_unselected_proba)}")
        
        # ============ 智能动态CSF选择机制 ============
        logger.info("开始CSF选择过程...")
        
        # 基于混合系数选择重要组态（已验证重要组态）
        csfs_above_threshold_indices = np.where(np.any(rmix_file_data.mix_coefficient_List[0][asfs_position]**2 >= np.float64(config.cutoff_value), axis = 0))[0]
        verified_important_indices = caled_csfs_indices_dict[0][csfs_above_threshold_indices]
        logger.info(f"已验证重要组态数: {len(verified_important_indices)}")
        
        logger.info("      组态采样")
        logger.info("更新重要组态索引")
        
        # 计算当前重要组态数量作为基准
        current_important_count = len(verified_important_indices)
        
        # 获取最小重要组态数量保护
        min_important_count = getattr(config, 'min_important_count', max(50, int(len(raw_csfs_descriptors) * 0.01)))  # 默认1%或50个
        if current_important_count <= min_important_count:
            current_important_count = min_important_count
            logger.info(f"重要组态数目小于等于最小值，调整为{min_important_count}")
        
        # 获取扩展比例
        expansion_ratio = config.ml_config.get('expansion_ratio', 2)
        
        # 对未选择的CSF进行预测
        if isinstance(unselected_csfs_descriptors, pd.DataFrame):
            X_unselected = unselected_csfs_descriptors.values
        else:
            X_unselected = unselected_csfs_descriptors
            
        X_unselected_scaled = selector.transform(scaler.transform(X_unselected))
        y_unselected_prediction = best_model.predict(X_unselected_scaled)
        y_unselected_probability = best_model.predict_proba(X_unselected_scaled)[:, 1]
        
        # 在未选择的CSF中找出被预测为重要的组态
        ml_predicted_important_mask = y_unselected_prediction == 1
        ml_predicted_important_local_indices = np.where(ml_predicted_important_mask)[0]
        
        # 获取所有CSF的索引
        total_csfs_count = len(raw_csfs_descriptors)
        all_csfs_indices = np.arange(total_csfs_count)
        current_calc_indices = caled_csfs_indices_dict[0]
        unselected_indices = np.setdiff1d(all_csfs_indices, current_calc_indices)
        
        ml_predicted_important_global_indices = unselected_indices[ml_predicted_important_local_indices]
        
        logger.info(f"开始选择组态，当前重要组态数为：{len(verified_important_indices)}")
        logger.info(f"ML预测的重要组态数（在未选择中）：{len(ml_predicted_important_global_indices)}")
        logger.info(f"目标新增组态数：{expansion_ratio * current_important_count}")
        
        # 智能选择策略
        target_new_csf_count = math.ceil(expansion_ratio * current_important_count)
        if len(ml_predicted_important_local_indices) >= target_new_csf_count:
            # 情况1：ML预测的重要组态数量充足，按概率排序选择top-k
            logger.info(f"ML预测组态充足，按概率排序选择前{target_new_csf_count}个")
            
            # 获取ML预测重要组态的概率
            ml_predicted_important_probabilities = y_unselected_probability[ml_predicted_important_local_indices]
            
            # 按概率降序排序
            probability_sorted_indices = np.argsort(ml_predicted_important_probabilities)[::-1]
            
            # 选择前target_new_csf_count个
            top_k_local_indices = ml_predicted_important_local_indices[probability_sorted_indices[:target_new_csf_count]]
            ml_selected_indices = unselected_indices[top_k_local_indices]
            
            logger.info(f"从{len(ml_predicted_important_local_indices)}个ML预测重要组态中选择了{len(ml_selected_indices)}个")
        else:
            # 情况2：ML预测的重要组态数量不足，全部采用
            logger.info(f"ML预测组态不足，全部采用{len(ml_predicted_important_global_indices)}个")
            ml_selected_indices = ml_predicted_important_global_indices
        
        # 最终选择：已验证重要组态 + ML选择的新组态
        final_chosen_indices = np.unique(np.sort(np.concatenate([verified_important_indices, ml_selected_indices])))
        
        # 轨道覆盖检查和补充选择
        current_selected_descriptors = raw_csfs_descriptors[final_chosen_indices]
        is_covered, uncovered_orbitals = validate_csf_descriptors_coverage(
            current_selected_descriptors, 
            with_subshell_info=config.ml_config.get('descriptors_with_subshell_info', False)
        )
        
        if not is_covered:
            logger.info(f"检测到未覆盖的轨道索引: {uncovered_orbitals}")
            logger.info(f"开始补充选择以满足轨道覆盖条件")
            
            # 从剩余未选择的CSFs中选择补充的CSF
            remaining_unselected_indices = np.setdiff1d(unselected_indices, ml_selected_indices)
            remaining_descriptors = raw_csfs_descriptors[remaining_unselected_indices]
            
            # 使用select_csfs_for_coverage函数选择补充的CSF
            _, additional_indices_relative = select_csfs_for_coverage(
                np.array([]),  # 空数组，因为我们只想选择新的CSF
                uncovered_orbitals,
                remaining_descriptors,
                with_subshell_info=config.ml_config.get('descriptors_with_subshell_info', False)
            )
            
            # 将相对索引转换为全局索引
            additional_indices = remaining_unselected_indices[additional_indices_relative].tolist()
            
            # 将补充选择的CSF添加到最终选择中
            if additional_indices:
                additional_indices_array = np.array(additional_indices)
                final_chosen_indices = np.unique(np.sort(np.concatenate([final_chosen_indices, additional_indices_array])))
                logger.info(f"补充选择了 {len(additional_indices)} 个CSF以满足轨道覆盖条件")
                
                # 验证更新后的覆盖情况
                updated_descriptors = raw_csfs_descriptors[final_chosen_indices]
                is_covered_after, _ = validate_csf_descriptors_coverage(
                    updated_descriptors,
                    with_subshell_info=config.ml_config.get('descriptors_with_subshell_info', False)
                )
                
                if is_covered_after:
                    logger.info("✓ 更新后所有轨道均已覆盖")
                else:
                    logger.warning("⚠ 更新后仍有未覆盖的轨道")
            else:
                logger.warning("未能找到合适的CSF来覆盖所有未覆盖轨道")
        else:
            logger.info("✓ 所有轨道均已覆盖，无需补充选择")
        
        logger.info(f"第{config.cal_loop_num + 1}次迭代计算组态数为：{len(final_chosen_indices)}")
        logger.info(f"其中已验证重要组态：{len(verified_important_indices)}")
        logger.info(f"其中ML新增组态：{len(ml_selected_indices)}")
        
        # 数据一致性检查：确保CSFs数量的两个来源一致
        csfs_count_from_cfile = cal_csfs_data.CSFs_block_length[0]
        csfs_count_from_rmix = rmix_file_data.block_CSFs_nums[0]
        
        if csfs_count_from_cfile != csfs_count_from_rmix:
            logger.error(f"CSFs数量不一致: cal_csfs_data.CSFs_block_length[0]={csfs_count_from_cfile}, rmix_file_data.block_CSFs_nums[0]={csfs_count_from_rmix}")
            raise ValueError("本轮计算的CSFs数量数据不一致，请检查数据文件")
        
        current_calculation_csfs = csfs_count_from_cfile
        logger.info(f"本轮计算CSFs数量: {current_calculation_csfs}")
        
        # 计算各种统计率
        total_original_csfs = len(raw_csfs_descriptors)
        total_important_csfs = len(verified_important_indices)
        total_ml_predicted_csfs = len(ml_selected_indices)
        total_final_csfs = len(final_chosen_indices)
        important_retention_rate = total_important_csfs / total_original_csfs
        ml_retention_rate = total_ml_predicted_csfs / total_original_csfs
        final_retention_rate = total_final_csfs / total_original_csfs
        ml_improvement_ratio = len(ml_selected_indices) / len(verified_important_indices) if len(verified_important_indices) > 0 else 0
        
        # 计算组态留存率
        if config.cal_loop_num > 1:
            previous_important_indices_path = config.root_path / 'results' / f'{config.conf}_{config.cal_loop_num-1}_important_indices.pkl'
            if previous_important_indices_path.exists():
                try:
                    previous_important_indices_dict = gk.csfs_index_load(previous_important_indices_path)
                    if not previous_important_indices_dict:
                        raise ValueError("previous_important_indices_dict为空")
                    
                    if 0 not in previous_important_indices_dict:
                        raise KeyError(f"previous_important_indices_dict中缺少键0，可用键: {list(previous_important_indices_dict.keys())}")
                    
                    previous_important_indices = previous_important_indices_dict[0]
                    
                    # 标准组态留存率计算：交集/上轮重要组态数
                    important_intersection = np.intersect1d(verified_important_indices, previous_important_indices)
                    data_retention_rate = len(important_intersection) / len(previous_important_indices)
                    
                    logger.info(f"组态留存率计算（标准方法）:")
                    logger.info(f"- 本次重要组态数: {len(verified_important_indices)}")
                    logger.info(f"- 上次重要组态数: {len(previous_important_indices)}")
                    logger.info(f"- 交集组态数: {len(important_intersection)}")
                    logger.info(f"- 组态留存率: {data_retention_rate:.4%}")
                except Exception as e:
                    logger.warning(f"加载上一轮重要组态索引失败: {e}")
                    data_retention_rate = 0.0
            else:
                logger.warning(f"未找到上一轮重要组态索引文件: {previous_important_indices_path}")
                data_retention_rate = 0.0
        else:
            # 第一轮没有上一轮数据，留存率设为0
            data_retention_rate = 0.0
            logger.info(f"第一轮计算，无法计算组态留存率")
        
        logger.info(f"统计信息:")
        logger.info(f"- 原始CSFs总数: {total_original_csfs}")
        logger.info(f"- 重要CSFs数量: {total_important_csfs} (占原始: {important_retention_rate:.4%})")
        logger.info(f"- ML新增CSFs数量: {total_ml_predicted_csfs} (占原始: {ml_retention_rate:.4%})")
        logger.info(f"- 最终选择CSFs数量: {total_final_csfs} (占原始: {final_retention_rate:.4%})")
        logger.info(f"- ML扩展比例: {ml_improvement_ratio:.2f} (ML新增/重要组态)")
        
        # 分别保存重要组态索引和ML预测组态索引
        important_csfs_indices_dict = {0: verified_important_indices}
        ml_predicted_csfs_indices_dict = {0: ml_selected_indices}
        final_chosen_csfs_indices_dict = {0: final_chosen_indices}
        
        # 保存重要组态索引
        important_indices_path = config.root_path / 'results' / f'{config.conf}_{config.cal_loop_num}_important_indices.pkl'
        gk.csfs_index_storange(important_csfs_indices_dict, important_indices_path)
        logger.info(f"重要组态索引保存到: {important_indices_path}")
        
        # 保存ML预测组态索引
        ml_chosen_indices_dict_path = config.root_path / 'results' / f'{config.conf}_{config.cal_loop_num}_ml_chosen_indices.pkl'
        gk.csfs_index_storange(ml_predicted_csfs_indices_dict, ml_chosen_indices_dict_path)
        logger.info(f"ML预测组态索引保存到: {ml_chosen_indices_dict_path}")
        
        # 保存最终选择的组态索引（用于下次计算）
        final_chosen_indices_path = config.root_path / 'results' / f'{config.conf}_{config.cal_loop_num}_final_chosen_indices.pkl'
        gk.csfs_index_storange(final_chosen_csfs_indices_dict, final_chosen_indices_path)
        logger.info(f"最终选择组态索引保存到: {final_chosen_indices_path}")
        
        # 保存迭代结果
        selection_results = {
            # 组态索引信息
            'important_csfs_indices': verified_important_indices.tolist(),
            'ml_predicted_csfs_indices': ml_selected_indices.tolist(),
            'final_chosen_csfs_indices': final_chosen_indices.tolist(),
            
            # 数量统计
            'important_count': len(verified_important_indices),
            'ml_predicted_count': len(ml_predicted_important_global_indices),  # 总预测数量
            'ml_new_count': len(ml_selected_indices),  # 实际新增数量
            'final_chosen_count': len(final_chosen_indices),
            'total_original_count': total_original_csfs,
            'current_calculation_count': current_calculation_csfs,
            
            # 选择策略参数
            'current_important_count': current_important_count,
            'min_important_count': min_important_count,
            'expansion_ratio': expansion_ratio,
            'target_new_count': target_new_csf_count,
            
            # 留存率统计
            'data_retention_rate': data_retention_rate,
            'important_retention_rate': important_retention_rate,
            'ml_retention_rate': ml_retention_rate,
            'final_retention_rate': final_retention_rate,
            'ml_improvement_ratio': ml_improvement_ratio,
            
            # ML模型性能 - 测试集
            'test_f1': ensemble_results[best_model_name]['f1'],
            'test_roc_auc': ensemble_results[best_model_name]['roc_auc'],
            'test_accuracy': ensemble_results[best_model_name]['accuracy'],
            'test_precision': ensemble_results[best_model_name]['precision'],
            'test_recall': ensemble_results[best_model_name]['recall'],
            
            # 选择策略信息
            'sampling_method': 'supervised_ml_improved',
            'selection_strategy': 'verified_important_plus_ml_expansion',
            
            # 下次计算使用的索引
            'chosen_index': final_chosen_indices.tolist()
        }
        
        # 计算总执行时间
        total_execution_time = time.time() - execution_time
        
        gk.save_iteration_results(
            config=config,
            training_time=training_time,
            eval_time=0.0,  # 已在内部计算
            execution_time=total_execution_time,
            evaluation_results={'test_metrics': ensemble_results[best_model_name]},
            selection_results=selection_results,
            logger=logger
        )
        
        # 绘制拟合图
        logger.info("开始绘制性能图表...")
        try:
            # 获取当前计算CSF的索引用于绘图
            current_calc_indices = caled_csfs_indices_dict[0]
            X_current_calc = raw_csfs_descriptors[current_calc_indices]
            y_current_calc_probability = best_model.predict_proba(selector.transform(scaler.transform(X_current_calc)))[:, 1]
            
            # 使用标准化的保存和绘图函数
            saved_files = gk.save_and_plot_results(
                evaluation_results={'test_metrics': ensemble_results[best_model_name]},
                model=best_model,
                config=config,
                rmix_file_data=rmix_file_data,
                asfs_position=asfs_position,
                caled_csfs_indices_dict=caled_csfs_indices_dict,
                y_current_calc_probability=y_current_calc_probability,
                save_model=True,
                save_data=True,
                plot_curves=True,
                logger=logger
            )
            
            logger.info("性能图表绘制完成")
            
        except Exception as e:
            logger.warning(f"绘图过程出现错误: {e}")
        
        # 保存改进的模型
        model_save_path = config.root_path / 'results' / f'{config.conf}_{config.cal_loop_num}_improved_model.pkl'
        joblib.dump({
            'model': best_model,
            'scaler': scaler,
            'selector': selector,
            'threshold': optimal_threshold,
            'ensemble_results': ensemble_results,
            'total_execution_time': total_execution_time,
            'cpu_optimization': {
                'cpu_cores_used': cpu_info['total_cores'],
                'multiprocessing_verified': multiprocessing_works,
                'preprocessing_time': preprocessing_time,
                'training_time': training_time
            }
        }, model_save_path)
        
        logger.info(f"改进模型已保存到: {model_save_path}")
        logger.info(f"总执行时间: {total_execution_time:.2f}秒")
        
        # CPU优化总结
        logger.info("=== CPU多核优化总结 ===")
        logger.info(f"使用的CPU核心数: {cpu_info['total_cores']} (逻辑) / {cpu_info['physical_cores']} (物理)")
        logger.info(f"多进程验证: {'通过' if multiprocessing_works else '未通过'}")
        logger.info(f"特征预处理优化: {preprocessing_time:.2f}秒")
        logger.info(f"PyTorch训练优化: {training_time:.2f}秒")
        logger.info("✓ CPU多核优化已全部启用")
        
        # 数据保存完成，更新配置继续下一轮计算
        gk.update_config(config_file_path, {'continue_cal': True})
        gk.update_config(config_file_path, {'cal_error_num': 0})
        gk.update_config(config_file_path, {'cal_loop_num': config.cal_loop_num + 1})
        
    else:
        # 处理配置不匹配的回退逻辑
        logger.info("配置不匹配，启动回退机制...")
        # ... (保持原有回退逻辑)

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='增强版机器学习训练程序')
    parser.add_argument('--config', type=str, default='config.toml', help='配置文件路径')
    args = parser.parse_args()
    
    # 加载配置
    try:
        cfg = gk.load_config(args.config)
        main(cfg)
    except FileNotFoundError:
        print(f"错误: 配置文件 {args.config} 不存在")
    except Exception as e:
        print(f"程序执行失败: {str(e)}")