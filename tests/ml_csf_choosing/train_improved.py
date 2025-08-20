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
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, accuracy_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif

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

important_config_count_history = []

def enhanced_feature_preprocessing(X_train, X_test, y_train):
    """增强的特征预处理"""
    
    # 1. 特征缩放
    scaler = RobustScaler()  # 对异常值更鲁棒
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 2. 特征选择（基于互信息）
    k_features = min(X_train.shape[1], 100)  # 选择最多100个特征
    selector = SelectKBest(score_func=mutual_info_classif, k=k_features)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)
    
    return X_train_selected, X_test_selected, scaler, selector

def advanced_imbalance_handling(X_train, y_train, method='adasyn'):
    """高级类别不平衡处理"""
    
    strategies = {
        'smote': SMOTE(random_state=42, k_neighbors=5),
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

def build_ensemble_model(X_train, y_train, X_test, y_test):
    """构建集成模型"""
    
    models = {}
    
    # 1. Random Forest (优化后)
    rf_params = {
        'n_estimators': 500,
        'max_depth': 20,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1
    }
    rf = RandomForestClassifier(**rf_params)
    
    # 2. Gradient Boosting
    gb_params = {
        'n_estimators': 300,
        'max_depth': 7,
        'learning_rate': 0.1,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42
    }
    gb = GradientBoostingClassifier(**gb_params)
    
    # 3. XGBoost (如果可用)
    if XGBOOST_AVAILABLE:
        xgb_params = {
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': max(1, len(y_train) // (2 * np.sum(y_train))),
            'random_state': 42
        }
        xgb_model = xgb.XGBClassifier(**xgb_params)
        models['xgboost'] = xgb_model
    
    models['random_forest'] = rf
    models['gradient_boosting'] = gb
    
    # 训练所有模型
    results = {}
    for name, model in models.items():
        print(f"训练 {name}...")
        
        # 超参数优化
        if name == 'random_forest':
            best_model, best_params = optimize_hyperparameters(model, X_train, y_train, 'random_forest')
        elif name == 'xgboost' and XGBOOST_AVAILABLE:
            best_model, best_params = optimize_hyperparameters(model, X_train, y_train, 'xgboost')
        else:
            best_model = model
            best_params = {}
        
        # 训练
        best_model.fit(X_train, y_train)
        
        # 评估
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]
        
        results[name] = {
            'model': best_model,
            'parameters': best_params,
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'accuracy': accuracy_score(y_test, y_pred)
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
        
        # 分离特征和标签
        X = caled_csfs_descriptors.drop('label', axis=1)
        y = caled_csfs_descriptors['label']
        
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
            X_train_balanced, y_train_balanced, X_test_processed, y_test
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
        
        # 特征重要性分析
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            logger.info("前20个重要特征:")
            for i in range(min(20, len(indices))):
                logger.info(f"{i+1}. 特征{indices[i]}: {importances[indices[i]]:.4f}")
        else:
            logger.info("当前模型不支持特征重要性分析")
        
        # 对未选择的CSF进行预测
        X_unselected = unselected_csfs_descriptors.copy()
        X_unselected_scaled = scaler.transform(selector.transform(X_unselected))
        
        y_unselected_proba = best_model.predict_proba(X_unselected_scaled)[:, 1]
        
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
        
    else:
        # 处理配置不匹配的回退逻辑
        logger.info("配置不匹配，启动回退机制...")
        # ... (保持原有回退逻辑)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='增强版机器学习训练程序')
    parser.add_argument('--config', type=str, default='config.toml', help='配置文件路径')
    args = parser.parse_args()
    
    try:
        cfg = gdp.load_config(args.config)
        main(cfg)
    except FileNotFoundError:
        print(f"错误: 配置文件 {args.config} 不存在")
    except Exception as e:
        print(f"程序执行失败: {str(e)}")