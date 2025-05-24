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
import time


sys.path.append('/home/workstation2/AppFiles/GraspDataProcessing/src')
import graspdataprocessing as gdp

def load_config(config_path):
    """加载YAML配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return SimpleNamespace(**config)

def setup_logging(config):
    """配置日志系统"""
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/pipeline.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def main(config):
    """主程序逻辑"""
    logger = setup_logging(config)
    logger.info("程序启动")

    # 使用 pathlib 创建目录
    root_path = Path(config.root_path)
    (root_path / "models").mkdir(parents=True, exist_ok=True)
    (root_path / "descripotors").mkdir(parents=True, exist_ok=True)
    (root_path / "descripotors_stay").mkdir(parents=True, exist_ok=True)
    (root_path / "test_data").mkdir(parents=True, exist_ok=True)
    (root_path / "roc_curves").mkdir(parents=True, exist_ok=True)
    (root_path / "results").mkdir(parents=True, exist_ok=True)

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
    initial_csfs_path = root_path.joinpath(config.target_pool_file)
    try:
        if not initial_csfs_path.is_file():  # 检查是否是有效文件
            logger.error(f"初始CSFs文件无效或不存在: {initial_csfs_path}")
            raise FileNotFoundError(f"初始CSFs文件无效或不存在: {initial_csfs_path}")
        logger.info(f"成功加载初始CSFs文件: {initial_csfs_path}")
    except PermissionError as e:
        logger.error(f"无权限访问CSFs文件: {initial_csfs_path}")
        raise
    except Exception as e:
        logger.error(f"加载CSFs文件时发生未知错误: {str(e)}")
        raise

    initial_csfs_data_load = gdp.GraspFileLoad.from_filepath(initial_csfs_path, file_type='CSF')
    initial_csfs_data = initial_csfs_data_load.data_file_process()
    # initial_csfs_data has the following attributes:
    ## initial_csfs_data.subshell_info_raw
    ## initial_csfs_data.CSFs_block_j_value
    ## initial_csfs_data.parity
    ## initial_csfs_data.CSFs_block_data
    ## initial_csfs_data.CSFs_block_length
    ## initial_csfs_data.block_num
    
    chosen_csfs_indices_dict = {}
    chosen_csfs_dict = {}
    unselected_indices_dict = {}
    # 加载索引文件
    if hasattr(config, 'selected_csfs_index_file'):
        selected_csfs_indices_dict = {}
        selected_csfs_indices_dict = gdp.csfs_index_load(config.selected_csfs_index_file)
    
    # 加载CSF文件并建立映射
    if hasattr(config, 'selected_csfs_file'):
        selected_csfs_load = gdp.GraspFileLoad.from_filepath(config.selected_csfs_file, file_type='CSF')  # 修正变量名
        selected_csfs_indices_dict = {}
        selected_csfs_data = selected_csfs_load.data_file_process()
        selected_csfs_indices_dict = gdp.maping_two_csfs_indices(
            selected_csfs_data.CSFs_block_data, 
            initial_csfs_data.CSFs_block_data
        )
    
    for block in range(initial_csfs_data.block_num):
        # 计算每个块的初始采样数量
        chosen_csfs_dict[block], chosen_csfs_indices_dict[block], unselected_indices_dict[block] = gdp.radom_choose_csfs(initial_csfs_data.CSFs_block_data[block], config.initial_ratio, selected_csfs_indices_dict[block])
    
    chosen_csfs_list = [value for key, value in chosen_csfs_dict.items()]
    ## CSFs_file_info: List, sorted_CSFs_data_list: List, output_file: str
    gdp.write_sorted_CSFs_to_cfile(
                                    initial_csfs_data.CSFs_file_info,
                                    chosen_csfs_list,
                                    root_path.joinpath(f'{config.conf}_{config.cal_loop_num}.c')
    )
    logger.info(f"CSFs选择完成，保存到文件{config.conf}_{config.cal_loop_num}.c")
    
    gdp.csfs_index_storange(
                            unselected_indices_dict,
                            root_path.joinpath(f'{config.conf}_{config.cal_loop_num}_unselected_indices.msgpack'
        )
    logger.info(f"未选择CSFs的索引保存到文件{config.conf}_{config.cal_loop_num}_unselected_indices.msgpack")
    
    logger.info("组态筛选完成")


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='机器学习配置交互程序')
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