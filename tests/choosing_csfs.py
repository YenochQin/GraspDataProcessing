#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Id :choosing_csfs.py
@date :2025/06/11 10:45:00
@author :YenochQin (秦毅)
@description: 进行组态选择的流程，支持多轮迭代训练
'''

import argparse
from pathlib import Path
import sys

sys.path.append('/home/workstation3/AppFiles/GraspDataProcessing/src')
try:
    import graspdataprocessing as gdp
except ImportError:
    print("错误: 无法导入 graspdataprocessing 模块")
    sys.exit(1)

def load_target_pool_data(config):
    """
    加载target_pool数据，支持从预处理文件或原始文件加载
    
    Args:
        config: 配置对象
        
    Returns:
        tuple: (target_pool_csfs_data, selected_csfs_indices_dict, logger)
    """
    logger = gdp.setup_logging(config)
    root_path = Path(config.root_path)
    
    # 检查是否存在预处理的数据
    target_pool_binary_path = root_path / f"{config.conf}.pkl"
    selected_indices_path = root_path / f"{config.conf}_selected_indices.pkl"
    
    if target_pool_binary_path.exists():
        # 使用预处理的数据
        logger.info("发现预处理数据，直接加载...")
        target_pool_csfs_data = gdp.load_csfs_binary(target_pool_binary_path)
        logger.info(f"从二进制文件加载CSFs数据: {target_pool_binary_path}")
        
        if selected_indices_path.exists():
            selected_csfs_indices_dict = gdp.csfs_index_load(selected_indices_path)
            logger.info(f"加载初筛CSFs indices: {selected_indices_path}")
        else:
            # 如果没有selected indices，创建空的
            selected_csfs_indices_dict = {block: [] for block in range(target_pool_csfs_data.block_num)}
            logger.info("未找到初筛CSFs indices文件，使用空的indices")
            
    else:
        # 需要重新处理原始数据
        logger.warning("未找到预处理数据，建议先运行 initial_csfs.py 进行数据预处理")
        logger.info("正在从原始文件加载...")
        
        target_pool_file_path = root_path / config.target_pool_file
        target_pool_csfs_load = gdp.GraspFileLoad.from_filepath(target_pool_file_path, file_type='CSF')
        target_pool_csfs_data = target_pool_csfs_load.data_file_process()
        logger.info(f"从原始文件加载CSFs数据: {target_pool_file_path}")
        
        # 创建空的selected indices
        selected_csfs_indices_dict = {block: [] for block in range(target_pool_csfs_data.block_num)}
    
    return target_pool_csfs_data, selected_csfs_indices_dict, logger

def load_previous_chosen_indices(config, logger):
    """
    加载前一轮选择的CSFs索引
    
    Args:
        config: 配置对象
        logger: 日志记录器
        
    Returns:
        dict: 前一轮选择的indices字典
    """
    root_path = Path(config.root_path)
    previous_cal_path = root_path / f'{config.conf}_{config.cal_loop_num-1}'
    previous_indices_file = previous_cal_path / f'{config.conf}_{config.cal_loop_num-1}_chosen_indices'
    
    if previous_indices_file.with_suffix('.pkl').exists():
        selected_csfs_indices_dict = gdp.csfs_index_load(previous_indices_file)
        logger.info(f"加载前一轮chosen indices: {previous_indices_file}")
        return selected_csfs_indices_dict
    else:
        logger.warning(f"未找到前一轮indices文件: {previous_indices_file}")
        return {}

def perform_csfs_selection(config):
    """
    执行组态选择的主要流程
    
    Args:
        config: 配置对象
        
    Returns:
        dict: 包含选择结果的字典
    """
    
    # 步骤1：加载目标池数据
    target_pool_csfs_data, base_selected_indices, logger = load_target_pool_data(config)
    
    logger.info("CSFs选择程序启动")
    logger.info(f'计算循环次数: {config.cal_loop_num}')
    logger.info(f"初始比例: {config.initial_ratio}")
    logger.info(f"光谱项: {config.spetral_term}")
    
    # 步骤2：创建输出目录
    root_path = Path(config.root_path)
    cal_path = root_path / f'{config.conf}_{config.cal_loop_num}'
    cal_path.mkdir(exist_ok=True)
    logger.info(f"创建输出目录: {cal_path}")
    
    # 步骤3：确定已选择的CSFs索引
    if config.cal_loop_num == 1:
        # 第一轮使用base_selected_indices（来自selected_csfs_file或空）
        selected_csfs_indices_dict = base_selected_indices
        logger.info("第一轮选择，使用基础selected indices")
    else:
        # 后续轮次使用前一轮的chosen indices
        selected_csfs_indices_dict = load_previous_chosen_indices(config, logger)
        if not selected_csfs_indices_dict:
            # 如果加载失败，使用空的字典
            selected_csfs_indices_dict = {block: [] for block in range(target_pool_csfs_data.block_num)}
    
    # 步骤4：执行随机选择
    chosen_csfs_indices_dict = {}
    chosen_csfs_dict = {}
    unselected_indices_dict = {}

    for block in range(target_pool_csfs_data.block_num):
        # 计算每个块的初始采样数量
        chosen_csfs_dict[block], chosen_csfs_indices_dict[block], unselected_indices_dict[block] = (
            gdp.radom_choose_csfs(
                target_pool_csfs_data.CSFs_block_data[block], 
                config.initial_ratio, 
                selected_csfs_indices_dict.get(block, [])
            )
        )
    
    logger.info(f"完成CSFs随机选择，共{target_pool_csfs_data.block_num}个块")
    
    # 步骤5：保存选择的CSFs到.c文件
    chosen_csfs_list = [value for key, value in chosen_csfs_dict.items()]
    chosen_csfs_file_path = cal_path / f'{config.conf}_{config.cal_loop_num}.c'
    
    gdp.write_sorted_CSFs_to_cfile(
        target_pool_csfs_data.subshell_info_raw,
        chosen_csfs_list,
        chosen_csfs_file_path
    )
    logger.info(f"CSFs选择完成，保存到文件: {chosen_csfs_file_path}")
    
    # 步骤6：保存chosen indices
    chosen_indices_file = cal_path / f'{config.conf}_{config.cal_loop_num}_chosen_indices'
    gdp.csfs_index_storange(chosen_csfs_indices_dict, chosen_indices_file)
    logger.info(f"已选择CSFs的索引保存到: {chosen_indices_file}.pkl")
    
    # 步骤7：保存unselected indices  
    unselected_indices_file = cal_path / f'{config.conf}_{config.cal_loop_num}_unselected_indices'
    gdp.csfs_index_storange(unselected_indices_dict, unselected_indices_file)
    logger.info(f"未选择CSFs的索引保存到: {unselected_indices_file}.pkl")
    
    logger.info("组态筛选完成")
    total_chosen = sum(len(csfs) for csfs in chosen_csfs_dict.values())
    total_unselected = sum(len(indices) for indices in unselected_indices_dict.values())
    logger.info(f"🔄 计算轮次: {config.cal_loop_num}")
    logger.info(f"📊 选择的CSFs数量: {total_chosen}")
    logger.info(f"📊 未选择的CSFs数量: {total_unselected}")
    
    return {
        'chosen_csfs_dict': chosen_csfs_dict,
        'chosen_csfs_indices_dict': chosen_csfs_indices_dict,
        'unselected_indices_dict': unselected_indices_dict,
        'chosen_csfs_file_path': str(chosen_csfs_file_path),
        'target_pool_csfs_data': target_pool_csfs_data,
        'cal_path': cal_path,
        'logger': logger
    }

def main(config):
    """主程序逻辑"""
    try:
        result = perform_csfs_selection(config)

        print(f"📁 输出.c文件: {result['chosen_csfs_file_path']}")
        print(f"📁 输出目录: {result['cal_path']}")
        
        return result
        
    except Exception as e:
        print(f"❌ 组态选择失败: {str(e)}")
        raise

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='CSFs组态选择程序')
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