#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Id :train.py
@date :2025/05/25 13:53:10
@author :YenochQin (秦毅)
@update :2025/01/22 - 集成ann3_proba.py的ML策略优化
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
        logger.info("************************************************")

    except Exception as e:
            logger.error(f"程序执行过程中发生错误: {str(e)}")
            raise
    # 选择asfs_position索引对应的行
    selected_energy_data = energy_level_data_pd.iloc[asfs_position]
    # 保存正确的能级数据为CSV
    correct_levels_csv_path = config.scf_cal_path / f'{config.conf}_{config.cal_loop_num}_correct_levels.csv'
    selected_energy_data.to_csv(correct_levels_csv_path, index=False)
    logger.info(f"选择的能级数据已保存到: {correct_levels_csv_path}")

    if cal_result:
        # 记录能量信息
        logger.info("能级数据表格：\n%s", 
            tabulate(
                    energy_level_data_pd, 
                    headers='keys', 
                    tablefmt='fancy_grid', 
                    showindex=False, 
                    floatfmt=('d', 'd', 'd', 's', '.7f', '.2f', '.2f', 's')))
        logger.info("耦合正确")
        logger.info("************************************************")

        # 提取特征
        logger.info("数据预处理")
        # 获取是否包含错误能级负样本的配置
        include_wrong_level_negatives = getattr(config, 'ml_config', {}).get('include_wrong_level_negatives', True)
        caled_csfs_descriptors = gdp.generate_chosen_csfs_descriptors(config, caled_csfs_indices_dict, raw_csfs_descriptors, rmix_file_data, asfs_position, logger, include_wrong_level_negatives)
        unselected_csfs_descriptors = gdp.get_unselected_descriptors(raw_csfs_descriptors, caled_csfs_indices_dict)
        X_unselected = unselected_csfs_descriptors.copy()
        logger.info("特征提取完成")

        # 训练模型
        model, X_train, X_test, y_train, y_test, training_time = gdp.train_model(config, caled_csfs_descriptors, rmix_file_data, asfs_position, logger)

        # 评估模型
        evaluation_results = gdp.evaluate_model(
            model, X_train, X_test, y_train, y_test, X_unselected, config, logger
        )

        # 详细的模型评估 - 借鉴ann3_proba.py的评估方式
        test_metrics = evaluation_results['test_metrics']
        train_metrics = evaluation_results['train_metrics']
        
        # 测试集性能
        test_f1, test_roc_auc, test_accuracy, test_precision, test_recall = (
            test_metrics['f1'], test_metrics['roc_auc'], test_metrics['accuracy'],
            test_metrics['precision'], test_metrics['recall']
        )
        
        # 训练集性能
        train_f1, train_roc_auc, train_accuracy, train_precision, train_recall = (
            train_metrics['f1'], train_metrics['roc_auc'], train_metrics['accuracy'],
            train_metrics['precision'], train_metrics['recall']
        )
        
        logger.info("测试集预测结果:")
        logger.info(f"AUC: {test_roc_auc:.4f}, F1: {test_f1:.4f}, Accuracy: {test_accuracy:.4f}")
        logger.info(f"Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")
        
        logger.info("训练集预测结果:")
        logger.info(f"AUC: {train_roc_auc:.4f}, F1: {train_f1:.4f}, Accuracy: {train_accuracy:.4f}")
        logger.info(f"Precision: {train_precision:.4f}, Recall: {train_recall:.4f}")
        
        # 过拟合监控
        overfitting_check = train_f1 - test_f1
        logger.info(f'             过拟合检查差异(训练-测试): {overfitting_check:.4f}')
        if overfitting_check > config.ml_config['overfitting_threshold']:
            logger.warning("检测到可能的过拟合现象")
        elif overfitting_check < config.ml_config['underfitting_threshold']:
            logger.warning("检测到可能的欠拟合现象")
        
        # 模型推理 - 仅对未选择的CSF进行推理（借鉴ann3_proba.py的高效策略）
        logger.info("模型推理")
        start_time = time.time()

        # 验证字典并安全获取当前计算的CSF索引
        if not caled_csfs_indices_dict:
            raise ValueError("caled_csfs_indices_dict为空，无法获取当前计算的CSF索引")
        
        if 0 not in caled_csfs_indices_dict:
            raise KeyError(f"caled_csfs_indices_dict中缺少键0，可用键: {list(caled_csfs_indices_dict.keys())}")
        
        # 获取未选择的CSF索引
        total_csfs_count = len(raw_csfs_descriptors)
        all_csfs_indices = np.arange(total_csfs_count)
        current_calc_indices = caled_csfs_indices_dict[0]
        unselected_indices = np.setdiff1d(all_csfs_indices, current_calc_indices)
        
        # 仅对未选择的CSF进行预测
        X_unselected_for_prediction = raw_csfs_descriptors[unselected_indices]
        y_unselected_prediction = model.predict(X_unselected_for_prediction)
        y_unselected_probability = model.predict_proba(X_unselected_for_prediction)[:, 1]
        
        eval_time = time.time() - start_time
        logger.info(f"模型推理时间: {eval_time:.4f}秒")
        logger.info(f"推理了 {len(y_unselected_probability)} 个未选择CSF组态")
        
        # 为绘图准备当前计算CSF的预测概率
        # 对当前计算的CSF也进行预测（用于绘图和分析）
        X_current_calc = raw_csfs_descriptors[current_calc_indices]
        y_current_calc_probability = model.predict_proba(X_current_calc)[:, 1]
        logger.info(f"当前计算CSF数量: {len(current_calc_indices)}")
        logger.info(f"当前计算CSF预测概率维度: {y_current_calc_probability.shape}")
        
        # 使用标准化的保存和绘图函数
        saved_files = gdp.save_and_plot_results(
            evaluation_results=evaluation_results,
            model=model,
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
        logger.info(f"预测结果与模型保存成功")
        logger.info(f"保存的文件: {list(saved_files.keys())}")

        # 基于混合系数选择重要组态（已验证重要组态）
        csfs_above_threshold_indices = np.where(np.any(rmix_file_data.mix_coefficient_List[0][asfs_position]**2 >= np.float64(config.cutoff_value), axis = 0))[0]
        verified_important_indices = caled_csfs_indices_dict[0][csfs_above_threshold_indices]
        logger.info(f"已验证重要组态数: {len(verified_important_indices)}")
        
        # ============ 智能动态选择机制 ============
        logger.info("      组态采样")
        logger.info("更新重要组态索引")
        
        # 计算当前重要组态数量作为基准
        current_important_count = len(verified_important_indices)
        
        # 获取最小重要组态数量保护
        min_important_count = getattr(config, 'min_important_count', max(50, int(total_csfs_count * 0.01)))  # 默认1%或50个
        if current_important_count <= min_important_count:
            current_important_count = min_important_count
            logger.info(f"重要组态数目小于等于最小值，调整为{min_important_count}")
        
        # 获取扩展比例
        expansion_ratio = config.ml_config.get('expansion_ratio', 2)
        
        # 在未选择的CSF中找出被预测为重要的组态
        ml_predicted_important_mask = y_unselected_prediction == 1
        ml_predicted_important_local_indices = np.where(ml_predicted_important_mask)[0]
        ml_predicted_important_global_indices = unselected_indices[ml_predicted_important_local_indices]
        
        logger.info(f"开始选择组态，当前重要组态数为：{len(verified_important_indices)}")
        logger.info(f"ML预测的重要组态数（在未选择中）：{len(ml_predicted_important_global_indices)}")
        logger.info(f"目标新增组态数：{expansion_ratio * current_important_count}")
        
        # 智能选择策略
        target_new_csf_count = expansion_ratio * current_important_count
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
        
        logger.info(f"第{config.cal_loop_num + 1}次迭代计算组态数为：{len(final_chosen_indices)}")
        logger.info(f"其中已验证重要组态：{len(verified_important_indices)}")
        logger.info(f"其中ML新增组态：{len(ml_selected_indices)}")
        
        # 数据一致性检查：确保CSFs数量的两个来源一致
        csfs_count_from_cal = cal_csfs_data.CSFs_block_length[0]
        csfs_count_from_rmix = rmix_file_data.block_CSFs_nums[0]
        
        if csfs_count_from_cal != csfs_count_from_rmix:
            logger.error(f"CSFs数量不一致: cal_csfs_data.CSFs_block_length[0]={csfs_count_from_cal}, rmix_file_data.block_CSFs_nums[0]={csfs_count_from_rmix}")
            raise ValueError("本轮计算的CSFs数量数据不一致，请检查数据文件")
        
        current_calculation_csfs = csfs_count_from_cal
        logger.info(f"本轮计算CSFs数量: {current_calculation_csfs}")
        
        # 计算数据留存率
        total_original_csfs = len(raw_csfs_descriptors)
        total_important_csfs = len(verified_important_indices)
        total_ml_predicted_csfs = len(ml_selected_indices)
        total_final_csfs = len(final_chosen_indices)
        
        # 修改4：组态留存率使用标准计算方法
        if config.cal_loop_num > 1:
            previous_important_indices_path = config.root_path / 'results' / f'{config.conf}_{config.cal_loop_num-1}_important_indices.pkl'
            if previous_important_indices_path.exists():
                try:
                    previous_important_indices_dict = gdp.csfs_index_load(previous_important_indices_path)
                    # 验证字典结构
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
        
        # 计算各种统计率
        important_retention_rate = total_important_csfs / total_original_csfs
        ml_retention_rate = total_ml_predicted_csfs / total_original_csfs
        final_retention_rate = total_final_csfs / total_original_csfs
        ml_improvement_ratio = len(ml_selected_indices) / len(verified_important_indices) if len(verified_important_indices) > 0 else 0
        
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
        gdp.csfs_index_storange(important_csfs_indices_dict, important_indices_path)
        logger.info(f"重要组态索引保存到: {important_indices_path}")
        
        # 保存ML预测组态索引
        ml_chosen_indices_dict_path = config.root_path / 'results' / f'{config.conf}_{config.cal_loop_num}_ml_chosen_indices.pkl'
        gdp.csfs_index_storange(ml_predicted_csfs_indices_dict, ml_chosen_indices_dict_path)
        logger.info(f"ML预测组态索引保存到: {ml_chosen_indices_dict_path}")
        
        # 保存最终选择的组态索引（用于下次计算）
        final_chosen_indices_path = config.root_path / 'results' / f'{config.conf}_{config.cal_loop_num}_final_chosen_indices.pkl'
        gdp.csfs_index_storange(final_chosen_csfs_indices_dict, final_chosen_indices_path)
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
            'test_f1': test_f1,
            'test_roc_auc': test_roc_auc,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            
            # ML模型性能 - 训练集
            'train_f1': train_f1,
            'train_roc_auc': train_roc_auc,
            'train_accuracy': train_accuracy,
            'train_precision': train_precision,
            'train_recall': train_recall,
            
            # 模型质量评估
            'overfitting_check': overfitting_check,
            
            # 选择策略信息
            'sampling_method': 'supervised_ml_ann3_style',
            'selection_strategy': 'verified_important_plus_ml_expansion',
            
            # 下次计算使用的索引
            'chosen_index': final_chosen_indices.tolist()
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
        
        # 数据保存完成后，检查收敛性
        should_continue = True
        if config.cal_loop_num >= 3:
            logger.info("数据已保存，开始检查收敛性...")
            should_continue = gdp.check_grasp_cal_convergence(config, logger)
            logger.info(f"检查收敛性结果: {'继续计算' if should_continue else '已收敛，停止计算'}")
        
        if should_continue:
            gdp.update_config(config_file_path, {'continue_cal': True})
            gdp.update_config(config_file_path, {'cal_error_num': 0})
            gdp.update_config(config_file_path, {'cal_loop_num': config.cal_loop_num + 1})
        else:
            logger.info("************************************************")
            logger.info("计算收敛，停止计算")
            gdp.update_config(config_file_path, {'continue_cal': False})
        
        # ============ 取消动态选择率修改 ============
        # 注释掉原有的动态选择率计算，保持chosen_ratio不变
        logger.info(f"保持选择率不变: {config.chosen_ratio}")
        logger.info(f"下次计算将使用固定的组态索引，不依赖选择率")
        
        # 不再调用动态选择率计算函数
        # dynamic_ratio = gdp.calculate_dynamic_chosen_ratio(
        #     config, 
        #     caled_csfs_indices_dict[0], 
        #     target_pool_csfs_data, 
        #     y_all_probability, 
        #     evaluation_results, 
        #     energy_level_data_pd, 
        #     logger
        # )
        # config.chosen_ratio = dynamic_ratio
        # gdp.update_config(config_file_path, {'chosen_ratio': dynamic_ratio})

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