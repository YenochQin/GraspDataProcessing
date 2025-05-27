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

sys.path.append('/home/workstation2/AppFiles/GraspDataProcessing/src')
import graspdataprocessing as gdp

def load_config(config_path):
    """加载YAML配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return SimpleNamespace(**config)

def update_config(config_path, updates):
    """更新YAML配置文件
    
    Args:
        config_path: 配置文件路径
        updates: 要更新的键值对字典
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 更新配置值
    config.update(updates)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def setup_logging(config):
    """配置日志系统"""
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/machine_learning_training.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# def level_energy_configuration_compare(config):
    

def traning_model(config):
    """训练模型"""
    logger = setup_logging(config)
    logger.info("开始训练模型")
    

def main(config):
    
    config.file_name = f'{config.conf}_{config.cal_loop_num}'
    """主程序逻辑"""
    logger = setup_logging(config)
    logger.info("机器学习训练程序启动")
    excution_time = time.time()

    # 使用 pathlib 创建目录
    root_path = Path(config.root_path)
    cal_path = root_path.joinpath(f'{config.conf}_{config.cal_loop_num}')
    (cal_path / "models").mkdir(parents=True, exist_ok=True)
    (cal_path / "descripotors").mkdir(parents=True, exist_ok=True)
    (cal_path / "descripotors_stay").mkdir(parents=True, exist_ok=True)
    (cal_path / "test_data").mkdir(parents=True, exist_ok=True)
    (cal_path / "roc_curves").mkdir(parents=True, exist_ok=True)
    (cal_path / "results").mkdir(parents=True, exist_ok=True)

    logger.info(f"初始比例: {config.initial_ratio}")
    logger.info(f"光谱项: {config.spetral_term}")

    # 使用 pathlib 构建路径并创建CSV文件
    result_csv_path = root_path.joinpath('results/results.csv')
    try:
        if not result_csv_path.exists():
            with result_csv_path.open(mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([
                    'training_time', 'eval_time', 'abinitio_time', 'all_time',
                    'f1', 'roc_auc', 'accuracy', 'precision', 'recall',
                    'Es', 'abimport_csfnum', 'MLimport_csfnum', 'MLsampling_ratio', 'next_itr_num',
                    'weight', 'f1_train', 'roc_auc_train', 'accuracy_train', 'precision_train', 'recall_train'
                ])
    except IOError as e:
        logger.error(f"无法创建结果文件 {result_csv_path}: {str(e)}")
        raise
    
    # initial csfs data load
    target_pool_file_path = root_path.joinpath(config.target_pool_file)
    try:
        if not target_pool_file_path.is_file():  # 检查是否是有效文件
            logger.error(f"初始CSFs文件无效或不存在: {target_pool_file_path}")
            raise FileNotFoundError(f"初始CSFs文件无效或不存在: {target_pool_file_path}")
        logger.info(f"成功加载初始CSFs文件: {target_pool_file_path}")
    except PermissionError as e:
        logger.error(f"无权限访问CSFs文件: {target_pool_file_path}")
        raise
    except Exception as e:
        logger.error(f"加载CSFs文件时发生未知错误: {str(e)}")
        raise
    
    
#############################################################################################
## 检查计算结果耦合是否正确
    energy_level_file_path = cal_path.joinpath(f'{config.conf}_{config.cal_loop_num}.level')
    energy_level_file_load = gdp.GraspFileLoad.from_filepath(energy_level_file_path)
    
    energy_level_data_pd = energy_level_file_load.energy_level_2_pd()
    
    cal_configuration_set = set(energy_level_data_pd['configuration'])
    
    if set(config.spetral_term).issubset(cal_configuration_set):
        logger.info(f"cal_loop {config.cal_loop_num} 组态耦合正确")
        cal_result = True
    else:
        logger.error(f"cal_loop {config.cal_loop_num} 组态耦合错误")
        cal_result = False
#############################################################################################

    rmix_file_load = gdp.GraspFileLoad.from_filepath(f'{cal_path}/{config.conf}_{config.cal_loop_num}.m', 'mix')
    rmix_file_data = rmix_file_load.data_file_process()
    ## rmix_file_data.block_num
    ## rmix_file_data.block_index_list
    ## rmix_file_data.block_CSFs_nums
    ## rmix_file_data.block_energy_count_list
    ## rmix_file_data.j_value_location_list
    ## rmix_file_data.parity_list
    ## rmix_file_data.block_levels_index_list
    ## rmix_file_data.block_energy_list
    ## rmix_file_data.block_level_energy_list
    ## rmix_file_data.mix_coefficient_list
    
    raw_csf_file_load = gdp.GraspFileLoad.from_filepath(config.target_pool_file_path)
    raw_csf_data = raw_csf_file_load.data_file_process()
    
    indices_temp = [i for i in range(raw_csf_data.CSFs_block_length[0])] ## 这个可以用enumerate代替
    sum_num_list = []
    retain_csfs_num = math.ceil(raw_csf_data.CSFs_block_length[0] * config.initial_ratio)
    
    ## cal loop_num > 1 --> 用于判断能量有没有下降
    if cal_result:
        for level in rmix_file_data.level_list:
            logger.info(f"             迭代能量：{level}")
        logger.info(f"             耦合正确")
        logger.info("************************************************")
        
        cut_off_csfs_indices_dict = gdp.batch_blocks_mix_square_above_threshold(rmix_file_data, config.cutoff_value)
        ci_temp = cut_off_csfs_indices_dict[0]
        
        unique_indices = gdp.deduplicateanddemerge(ci_temp, config.cutoff_value)
        ci_desc = np.zeros(ci_temp.shape[1], dtype=bool)
        ci_desc[unique_indices] = True
        
        csfs_prim_num, csfs_pool_num = gdp.produce_basis_npy(f'{config.file_name}.npy', f'{config.file_name}.c', 3)
        csfs_prim_num_stay, csfs_pool_num_stay = gdp.produce_basis_npy(f'{config.file_name}_stay.npy', f'{config.file_name}_stay.c', 3)

        descriptors_pool = pd.DataFrame(np.load(f'{config.file_name}.npy'))
        
        new_descriptors_pool = pd.concat([descriptors_pool, pd.DataFrame(ci_desc.T)], axis=1)
        new_descriptors_pool.to_csv(f'descripotors/{config.file_name}_desc.csv', index=False)

        stay_descriptors_pool = pd.DataFrame(np.load(f'{config.file_name}_stay.npy'))
        stay_descriptors_pool.to_csv(f'descripotors/{config.file_name}_stay_desc.csv', index=False)
        
        logger.info("<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>")
        logger.info("             特征提取完成")
        
        total_data = pd.read_csv(f'descripotors/{config.file_name}_desc.csv')
        X = total_data.iloc[:, :-1]
        y = total_data.iloc[:, -1]
        
        stay_data = pd.read_csv(f'descripotors_stay/{config.file_name}_stay_desc.csv')
        X_stay = stay_data.iloc[:, :].values
        X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42)
        
        logger.info("             训练模型")
        
        if config.cal_loop_num == 1:
            model = RandomForestClassifier(class_weight={0:1, 1:3},n_estimators=1000, verbose=True, n_jobs= -1)
        else:
            model = joblib.load(f"models/{config.conf}_{config.cal_loop_num-1}.pkl")

        results = pd.read_csv(f'{config.root_path}/results/iteration_results.csv')

        
        #################### weight ##################
        weight = [1, max(1, 12 - 2*config.cal_loop_num)]
        logger.info("weight:{}".format(weight))
        
        X_resampled, y_resampled = gdp.ANNClassifier.resampling(X_train, y_train, weight)
        start_time = time.time()
        model.fit(X_resampled, y_resampled)
        training_time = time.time() - start_time
        
        # Test set validation and configuration prediction
        logger.info("             预测与评估")
        y_pred = model.predict(X_test)
        start_time = time.time()
        y_pred_other = model.predict(X_stay)
        eval_time = time.time() - start_time
        all_time = excution_time + training_time + eval_time
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred_train = model.predict(X_train)
        y_proba_train = model.predict_proba(X_train)[:, 1]
        
        # Model evaluation
        roc_auc, pr_auc = gdp.ANNClassifier.plot_curve(y_test, y_proba, config.file_name)
        f1, roc_auc, accuracy, precision, recall = gdp.ANNClassifier.model_evaluation(y_test, y_pred, y_proba)
        
        # Overfitting and underfitting monitoring
        f1_train, roc_auc_train, accuracy_train, precision_train, recall_train = gdp.ANNClassifier.model_evaluation(y_train, y_pred_train, y_proba_train)
        
        # Save the result
        result_file = f"test_data/{config.file_name}.csv"
        pd.DataFrame({"y_test": y_test, "y_pred": y_pred, "y_proba": y_proba}).to_csv(result_file, index=False)
        joblib.dump(model, f"models/{config.file_name}.pkl")
        logger.info(f"预测结果与模型保存成功")
        logger.info("<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>")
        
        # Update the minimum number of CI
        sum_num = len(unique_indices)
        sum_num_list.append(sum_num)
        logging.info(f"当前重要组态数目: {sum_num}")
        logging.info(f"迭代重要组态数目: {sum_num_list}")
        if sum_num <= retain_csfs_num:
            sum_num=retain_csfs_num
        
        # Update Important configuration indexes
        indexs_import = unique_indices.tolist()
        indexs_import_temp = [index[i] for i in indexs_import]
        indexs_import_stay = np.where(y_pred_other == 1)[0].tolist()
        indexs_import_stay_temp = [stay_indexs[i] for i in indexs_import_stay]
        np.save(f"results/indexs_import_ab{config.cal_loop_num}.npy",indexs_import_temp)
        np.save(f"results/indexs_import_ml{config.cal_loop_num}.npy",indexs_import_stay_temp)
        
        # Selective configuration
        logger.info(f"开始选择组态，当前重要组态数为：{len(indexs_import_temp)}")
        if config.cal_loop_num != 1:
            logging.info("记录前一步的ML采样率信息")
            results['MLsampling_ratio'][config.cal_loop_num-2] = (len(indexs_import_temp)-len(import_before))/len(ml_add_csfs)

            results.to_csv(f'{config.root_path}/results/iteration_results.csv', index=False)
        
        if  len(indexs_import_stay) >= config.expansion_ratio * sum_num:
            ml_add_csfs= np.random.choice(indexs_import_stay_temp,
                                         size = config.expansion_ratio * sum_num,
                                         replace = False).tolist()
            mc_add_csfs= None
            new_add_csfs= ml_add_csfs
        elif len(indexs_import_stay) <= config.expansion_ratio * sum_num:
            stay_index = gdp.pop_other_ci(indices_temp, 
                                          indexs_import_stay_temp+indexs_import_temp)
            ml_add_csfs= indexs_import_stay_temp
            mc_add_csfs= np.random.choice(stay_index, 
                                         size = config.expansion_ratio * sum_num - len(indexs_import_stay_temp),
                                         replace=False).tolist()
            new_add_csfs= ml_add_csfs+ mc_add_csfs
        MLsampling_ratio = None
        chosen_index = np.sort(np.array(indexs_import_temp + new_add_csfs))
        import_before = indexs_import_temp

        logger.info(f"下一步计算组态数为：{len(chosen_index)}")
        
        ## !TODO: 这里需要修改
        chosen_csfs_data = [csf for i in chosen_index for csf in raw_csf_data.CSFs_block_data[0][i]]
        
        gdp.write_sorted_CSFs_to_cfile(
                                        raw_csf_data.CSFs_file_info,
                                        chosen_csfs_data,
                                        root_path.joinpath(f'{config.conf}_{config.cal_loop_num+1}.c')
        )

        stay_indices= np.array(gdp.pop_other_ci(indices_temp, chosen_index))
        
        unchosen_csfs_data = [csf for i in stay_indices for csf in raw_csf_data.CSFs_block_data[0][i]]
        gdp.write_sorted_CSFs_to_cfile(
                                        raw_csf_data.CSFs_file_info,
                                        unchosen_csfs_data,
                                        root_path.joinpath(f'{config.conf}_{config.cal_loop_num+1}_stay.c')
        )
        # Save results
        with open(f'{config.root_path}/results/iteration_results.csv', mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                training_time, eval_time, excution_time, all_time, f1, roc_auc, accuracy,precision, recall,
                0, len(unique_indices), len(indexs_import_stay), MLsampling_ratio, len(chosen_index),
                weight, f1_train, roc_auc_train, accuracy_train, precision_train, recall_train
            ])
        

        ## 先将能级结果写道一个csv中，然后看csv的长度，长度超过3，再来分析收敛性
        # Perform convergence calculations
        if len(Es_term) >= 3:
            logger.info(f"执行方差计算：{b}")
            E_std = np.std(Es_term[-3:], ddof=1)
            N_rsd = np.std(sum_num_list[-3:], ddof=1)/abs(np.mean(Es_term[-3:]))
            logger.info(f"第{b}次迭代，能量的标准偏差为：{E_std}，重要组态数的相对标准偏差为：{N_rsd}")
            if E_std <= 5e-05 and N_rsd <= 0.005:
                logger.info("达到收敛精度，迭代结束")
                # break
                with open(f'{config.root_path}/run.input', 'w') as file:
                    file.write('False')
                 
        config.cal_loop_num += 1
        update_config(f'{config.root_path}/config.yaml', {'cal_loop_num': config.cal_loop_num})
        
    else:
        cal_error_num = config.cal_error_num + 1
        update_config(f'{config.root_path}/config.yaml', {'cal_error_num': cal_error_num})
        if cal_error_num == 3:
            logger.info("连续三次波函数未改进，迭代收敛，退出筛选程序")
            with open(f'{config.root_path}/run.input', 'w') as file:
                    file.write('False')
            # break
        logger.info('组态选择出现问题：')
        if set(config.spetral_term).issubset(cal_configuration_set):
            logger.info('     能量未降低，波函数未得到改进')
        else:
            logger.info('     耦合出现问题')
        logger.info("正在重选组态")
        indexs_import_temp=np.load(f"results/indexs_import_ab{b-1}.npy")
        stay_indexs = gdp.pop_other_ci(indices_temp, indexs_import_temp)
        ML_sampling_ratio = None
        # mc_add_csfs= np.random.choice(stay_indexs, size=expansion_ratio*sum_num,replace=False).tolist()
        mc_add_csfs= np.random.choice(stay_index, 
                                         size = config.expansion_ratio * sum_num - len(indexs_import_stay_temp),
                                         replace=False).tolist()
        index= np.sort(np.array(list(indexs_import_temp)+mc_add_csfs))
        chosen_csfs_data = [csf for i in chosen_index for csf in raw_csf_data.CSFs_block_data[0][i]]
        
        gdp.write_sorted_CSFs_to_cfile(
                                        raw_csf_data.CSFs_file_info,
                                        chosen_csfs_data,
                                        root_path.joinpath(f'{config.conf}_{config.cal_loop_num}.c')
        )

        stay_indices= np.array(gdp.pop_other_ci(indices_temp, chosen_index))
        
        unchosen_csfs_data = [csf for i in stay_indices for csf in raw_csf_data.CSFs_block_data[0][i]]
        gdp.write_sorted_CSFs_to_cfile(
                                        raw_csf_data.CSFs_file_info,
                                        unchosen_csfs_data,
                                        root_path.joinpath(f'{config.conf}_{config.cal_loop_num}_stay.c')
        )

        

        


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='机器学习训练程序')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    args = parser.parse_args()
    

    # 加载配置
    try:
        cfg = load_config(args.config)
        main(cfg)
    except FileNotFoundError:
        print(f"错误: 配置文件 {args.config} 不存在")
    except yaml.YAMLError as e:
        print(f"错误: 配置文件解析失败 - {str(e)}")