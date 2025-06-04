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
try:
    import graspdataprocessing as gdp
except ImportError:
    print("警告: 无法导入 graspdataprocessing 模块")
    gdp = None

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

def setup_directories(root_path):
    """创建必要的目录结构"""
    root_path = Path(root_path)
    directories = ["models", "descripotors", "descripotors_stay", "test_data", "roc_curves", "results"]
    
    for directory in directories:
        (root_path / directory).mkdir(parents=True, exist_ok=True)
    
    return root_path

def initialize_results_file(result_csv_path, logger):
    """初始化结果CSV文件"""
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

def validate_initial_files(config, root_path, logger):
    """验证初始文件的存在和有效性"""
    # 验证初始CSFs文件
    initial_csfs_path = root_path / config.target_pool_file
    try:
        if not initial_csfs_path.is_file():
            logger.error(f"初始CSFs文件无效或不存在: {initial_csfs_path}")
            raise FileNotFoundError(f"初始CSFs文件无效或不存在: {initial_csfs_path}")
        logger.info(f"成功加载初始CSFs文件: {initial_csfs_path}")
    except PermissionError as e:
        logger.error(f"无权限访问CSFs文件: {initial_csfs_path}")
        raise
    except Exception as e:
        logger.error(f"加载CSFs文件时发生未知错误: {str(e)}")
        raise
    
    return initial_csfs_path

def check_configuration_coupling(config, energy_level_data_pd, logger):
    """检查组态耦合是否正确"""
    cal_configuration_set = set(energy_level_data_pd['configuration'])
    
    if set(config.spetral_term).issubset(cal_configuration_set):
        logger.info(f"cal_loop {config.cal_loop_num} 组态耦合正确")
        return True
    else:
        logger.error(f"cal_loop {config.cal_loop_num} 组态耦合错误")
        return False

def load_data_files(config, root_path, logger):
    """加载数据文件"""
    if gdp is None:
        raise ImportError("graspdataprocessing 模块未正确导入")
    
    cal_path = root_path / f'{config.conf}_{config.cal_loop_num}'
    
    # 加载能级文件
    energy_level_file_path = cal_path / f'{config.conf}_{config.cal_loop_num}.level'
    energy_level_file_load = gdp.GraspFileLoad.from_filepath(str(energy_level_file_path))
    energy_level_data_pd = energy_level_file_load.energy_level_2_pd()
    
    # 加载rmix文件
    rmix_file_path = cal_path / f'{config.conf}_{config.cal_loop_num}.m'
    rmix_file_load = gdp.GraspFileLoad.from_filepath(str(rmix_file_path), 'mix')
    rmix_file_data = rmix_file_load.data_file_process()
    
    # 加载原始CSF文件
    raw_csf_file_load = gdp.GraspFileLoad.from_filepath(str(config.target_pool_file), 'CSFs')
    raw_csf_data = raw_csf_file_load.data_file_process()
    
    return energy_level_data_pd, rmix_file_data, raw_csf_data

def extract_features(config, rmix_file_data, logger):
    """提取特征"""
    if gdp is None:
        raise ImportError("graspdataprocessing 模块未正确导入")
    
    # 获取重要配置索引
    cut_off_csfs_indices_dict = gdp.batch_blocks_mix_square_above_threshold(
        rmix_file_data, float(config.cutoff_value)
    )
    unique_indices = cut_off_csfs_indices_dict[0]
    
    # 创建CI描述符
    ci_desc = np.zeros(rmix_file_data.mix_coefficient_List[0].shape[1], dtype=bool)
    ci_desc[unique_indices] = True
    
    # 生成基础特征文件路径
    npy_file_path = Path(f'{config.file_name}.npy')
    c_file_path = Path(f'{config.file_name}.c')
    stay_npy_file_path = Path(f'{config.file_name}_stay.npy')
    stay_c_file_path = Path(f'{config.file_name}_stay.c')
    
    # 生成基础特征
    csfs_prim_num, csfs_pool_num = gdp.produce_basis_npy(
        str(npy_file_path), str(c_file_path), 3
    )
    csfs_prim_num_stay, csfs_pool_num_stay = gdp.produce_basis_npy(
        str(stay_npy_file_path), str(stay_c_file_path), 3
    )
    
    # 处理描述符
    descriptors_pool = pd.DataFrame(np.load(npy_file_path))
    new_descriptors_pool = pd.concat([descriptors_pool, pd.DataFrame(ci_desc.T)], axis=1)
    
    # 创建描述符目录并保存文件
    desc_dir = Path('descripotors')
    desc_dir.mkdir(exist_ok=True)
    desc_file_path = desc_dir / f'{config.file_name}_desc.csv'
    new_descriptors_pool.to_csv(desc_file_path, index=False)
    
    stay_descriptors_pool = pd.DataFrame(np.load(stay_npy_file_path))
    stay_desc_dir = Path('descripotors_stay')
    stay_desc_dir.mkdir(exist_ok=True)
    stay_desc_file_path = stay_desc_dir / f'{config.file_name}_stay_desc.csv'
    stay_descriptors_pool.to_csv(stay_desc_file_path, index=False)
    
    logger.info("特征提取完成")
    
    return unique_indices

def train_model(config, logger):
    """训练机器学习模型"""
    if gdp is None:
        raise ImportError("graspdataprocessing 模块未正确导入")
    
    # 加载数据
    desc_file_path = Path('descripotors') / f'{config.file_name}_desc.csv'
    total_data = pd.read_csv(desc_file_path)
    X = total_data.iloc[:, :-1]
    y = total_data.iloc[:, -1]
    
    stay_desc_file_path = Path('descripotors_stay') / f'{config.file_name}_stay_desc.csv'
    stay_data = pd.read_csv(stay_desc_file_path)
    X_stay = stay_data.iloc[:, :].values
    
    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=0.2, random_state=42
    )
    
    logger.info("开始训练模型")
    
    # 初始化或加载模型
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    if config.cal_loop_num == 1:
        model = RandomForestClassifier(
            class_weight={0:1, 1:3}, n_estimators=1000, verbose=True, n_jobs=-1
        )
    else:
        model_path = models_dir / f"{config.conf}_{config.cal_loop_num-1}.pkl"
        if model_path.exists():
            model = joblib.load(model_path)
        else:
            model = RandomForestClassifier(
                class_weight={0:1, 1:3}, n_estimators=1000, verbose=True, n_jobs=-1
            )
    
    # 设置权重
    weight = [1, max(1, 12 - 2*config.cal_loop_num)]
    logger.info(f"权重: {weight}")
    
    # 重采样和训练
    X_resampled, y_resampled = gdp.ANNClassifier.resampling(X_train, y_train, weight)
    start_time = time.time()
    model.fit(X_resampled, y_resampled)
    training_time = time.time() - start_time
    
    return model, X_train, X_test, y_train, y_test, X_stay, training_time, weight

def evaluate_model(model, X_train, X_test, y_train, y_test, X_stay, config, logger):
    """评估模型性能"""
    if gdp is None:
        raise ImportError("graspdataprocessing 模块未正确导入")
    
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
    update_config(f'{config.root_path}/config.yaml', {'cal_error_num': cal_error_num})
    
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

def main(config):
    """主程序逻辑"""
    config.file_name = f'{config.conf}_{config.cal_loop_num}'
    logger = setup_logging(config)
    logger.info("机器学习训练程序启动")
    execution_time = time.time()

    # 设置目录结构
    root_path = setup_directories(config.root_path)
    
    # 初始化结果文件
    result_csv_path = root_path / 'results/results.csv'
    initialize_results_file(result_csv_path, logger)
    
    # 验证初始文件
    validate_initial_files(config, root_path, logger)
    
    logger.info(f"初始比例: {config.initial_ratio}")
    logger.info(f"光谱项: {config.spetral_term}")
    
    try:
        # 加载数据文件
        energy_level_data_pd, rmix_file_data, raw_csf_data = load_data_files(config, root_path, logger)
        
        # 检查组态耦合
        cal_result = check_configuration_coupling(config, energy_level_data_pd, logger)
        
        # 初始化索引
        indices_temp = list(range(raw_csf_data.CSFs_block_length[0]))
        sum_num_list = []
        
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
        cfg = load_config(args.config)
        main(cfg)
    except FileNotFoundError:
        print(f"错误: 配置文件 {args.config} 不存在")
    except yaml.YAMLError as e:
        print(f"错误: 配置文件解析失败 - {str(e)}")
    except Exception as e:
        print(f"程序执行失败: {str(e)}")