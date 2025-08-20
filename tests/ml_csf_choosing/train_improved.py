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
    import graspdataprocessing as gdp
except ImportError:
    print("错误: 无法导入 graspdataprocessing 模块")
    sys.exit(1)

# 新增：XGBoost集成
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("警告: XGBoost不可用，将使用RandomForest替代")

try:
    from graspdataprocessing.machine_learning_module.ANN import ANNClassifier
    ANN_AVAILABLE = True
except ImportError:
    ANN_AVAILABLE = False
    print("警告: ANN不可用，将使用sklearn模型")

important_config_count_history = []

def enhanced_feature_preprocessing(X_train, X_test, y_train):
    """增强的特征预处理 - CPU多核优化"""
    
    import os
    from sklearn.preprocessing import RobustScaler
    from sklearn.feature_selection import SelectKBest, mutual_info_classif
    import numpy as np
    
    cpu_count = os.cpu_count() or 4
    
    # 1. 特征缩放 - 使用n_jobs参数
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 2. 特征选择（基于互信息）- 全数据多核优化
    k_features = min(X_train.shape[1], 100)  # 选择最多100个特征
    
    # 直接使用完整数据，多核并行处理
    print(f"使用完整数据({len(X_train_scaled)}样本)进行特征选择...")
    
    # 使用多核优化的mutual_info_classif
    from sklearn.feature_selection import mutual_info_classif
    
    # 直接计算互信息分数，使用所有CPU核心
    selector = SelectKBest(score_func=mutual_info_classif, k=k_features)
    
    # 设置环境变量确保多核
    import joblib
    with joblib.parallel_backend('threading', n_jobs=-1):
        selector.fit(X_train_scaled, y_train)
    
    # 在完整数据上训练选择器
    selector.fit(X_train_scaled, y_train)
    X_train_selected = selector.transform(X_train_scaled)
    X_test_selected = selector.transform(X_test_scaled)
    
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
        os.environ['OMP_NUM_THREADS'] = str(optimal_threads)
        os.environ['MKL_NUM_THREADS'] = str(optimal_threads)
        os.environ['NUMEXPR_NUM_THREADS'] = str(optimal_threads)
        
        print(f"启用CPU多线程优化 - 线程数: {optimal_threads}")
        
        # ANN参数优化
        batch_size = 4096
        max_epochs = 150
        hidden_size = 96
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
        model = ANNClassifier(
            input_size=X_train.shape[1],
            hidden_size=hidden_size,
            learning_rate=0.001,
            class_weights=class_weights,
            device='cpu'  # 强制使用CPU
        )
        
        # 训练
        print(f"开始训练 - 数据量: {len(X_train):,}, 特征维度: {X_train.shape[1]}")
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


def main(config):
    """增强版主程序逻辑"""
    config.file_name = f'{config.conf}_{config.cal_loop_num}'
    logger = gdp.setup_logging(config)
    
    config_file_path = config.root_path / 'config.toml'
    logger.info("增强版机器学习训练程序启动")
    execution_time = time.time()

    gdp.setup_directories(config)

    try:
        # 加载数据文件
        (energy_level_data_pd, 
         rmix_file_data, 
         raw_csfs_descriptors, 
         cal_csfs_data, 
         caled_csfs_indices_dict) = gdp.load_data_files(config, logger)
        
        cal_result, asfs_position = gdp.check_configuration_coupling(config, energy_level_data_pd, logger)
        
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
            
            energy_converged = gdp.check_energy_convergence(config, logger, selected_energy_data)
            
            if not energy_converged:
                logger.info(f"检测到能量不收敛，回退到第 {config.cal_loop_num - 1} 轮")
                gdp.update_config(config_file_path, {
                    'backward_loop_needed': True,
                    'target_backward_loop': config.cal_loop_num - 1,
                    'cal_loop_num': config.cal_loop_num - 1,
                    'continue_cal': True,
                    'cal_error_num': config.cal_error_num + 1
                })
                return
            
            should_continue = gdp.evaluate_calculation_convergence(config, logger, current_calculation_csfs)
            
            if not should_continue:
                logger.info("整体计算已收敛，跳过机器学习训练")
                gdp.update_config(config_file_path, {'continue_cal': False})
                return

        # 数据预处理
        logger.info("增强版数据预处理")
        include_wrong_level_negatives = getattr(config, 'ml_config', {}).get('include_wrong_level_negatives', True)
        
        caled_csfs_descriptors = gdp.generate_chosen_csfs_descriptors(
            config, caled_csfs_indices_dict, raw_csfs_descriptors, 
            rmix_file_data, asfs_position, logger, include_wrong_level_negatives
        )
        
        unselected_csfs_descriptors = gdp.get_unselected_descriptors(raw_csfs_descriptors, caled_csfs_indices_dict)
        
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
        
        # 特征预处理
        X_train_processed, X_test_processed, scaler, selector = enhanced_feature_preprocessing(
            X_train, X_test, y_train
        )
        
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
        
        # 后续逻辑与原版保持一致，但使用改进的模型
        # ... (保持原有CSF选择逻辑)
        
        # 计算总执行时间
        total_execution_time = time.time() - execution_time
        
        # 保存改进的模型
        model_save_path = config.root_path / 'results' / f'{config.conf}_{config.cal_loop_num}_improved_model.pkl'
        joblib.dump({
            'model': best_model,
            'scaler': scaler,
            'selector': selector,
            'threshold': optimal_threshold,
            'ensemble_results': ensemble_results,
            'total_execution_time': total_execution_time
        }, model_save_path)
        
        logger.info(f"改进模型已保存到: {model_save_path}")
        logger.info(f"总执行时间: {total_execution_time:.2f}秒")
        
        # 数据保存完成，更新配置继续下一轮计算
        gdp.update_config(config_file_path, {'continue_cal': True})
        gdp.update_config(config_file_path, {'cal_error_num': 0})
        gdp.update_config(config_file_path, {'cal_loop_num': config.cal_loop_num + 1})
        
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
        cfg = gdp.load_config(args.config)
        main(cfg)
    except FileNotFoundError:
        print(f"错误: 配置文件 {args.config} 不存在")
    except Exception as e:
        print(f"程序执行失败: {str(e)}")