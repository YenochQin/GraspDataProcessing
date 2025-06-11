#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Id :initial_csfs.py  
@date :2025/05/25 13:52:19
@author :YenochQin (秦毅)
@description: 处理target_pool_file的数据预处理，包括描述符计算和CSFs数据保存
'''

import argparse
from pathlib import Path
import sys

sys.path.append('D:\\PythonProjects\\GraspDataProcessing\\src')
try:
    import graspdataprocessing as gdp
except ImportError:
    print("错误: 无法导入 graspdataprocessing 模块")
    sys.exit(1)

def process_target_pool_csfs(config):
    """
    处理target_pool CSFs数据：计算描述符、保存二进制文件、生成哈希校验
    
    Args:
        config: 配置对象
        
    Returns:
        dict: 包含selected_csfs_indices_dict的结果字典
    """
    logger = gdp.setup_logging(config)
    logger.info("Target Pool CSFs 数据预处理启动")
    
    root_path = Path(config.root_path)
    target_pool_file_path = root_path.joinpath(config.target_pool_file)
    
    # 验证target_pool_file
    try:
        if not target_pool_file_path.is_file():
            logger.error(f"初始CSFs文件无效或不存在: {target_pool_file_path}")
            raise FileNotFoundError(f"初始CSFs文件无效或不存在: {target_pool_file_path}")
        logger.info(f"成功加载初始CSFs文件: {target_pool_file_path}")
    except PermissionError as e:
        logger.error(f"无权限访问CSFs文件: {target_pool_file_path}")
        raise
    except Exception as e:
        logger.error(f"加载CSFs文件时发生未知错误: {str(e)}")
        raise

    # 步骤1：加载和处理target_pool CSFs
    target_pool_csfs_load = gdp.GraspFileLoad.from_filepath(target_pool_file_path, file_type='CSF')
    target_pool_csfs_data = target_pool_csfs_load.data_file_process()
    logger.info(f"初始CSFs文件{config.target_pool_file} CSFs 读取成功")
    
    # 步骤2：计算描述符
    descriptors_array, labels_array = gdp.batch_process_csfs_with_multi_block(
        target_pool_csfs_data, 
        label_type='sequential'
    )
    logger.info(f"初始CSFs文件{config.target_pool_file} CSFs 描述符计算成功")

    # 步骤3：保存描述符
    target_pool_path = root_path.joinpath(config.conf)
    gdp.save_descriptors_with_multi_block(descriptors_array, labels_array, target_pool_path, 'npy')
    logger.info(f"初始CSFs文件{config.target_pool_file} CSFs 描述符保存成功")

    # 步骤4：保存CSFs二进制文件
    gdp.save_csfs_binary(target_pool_csfs_data, target_pool_path)
    logger.info(f"初始CSFs文件{config.target_pool_file} CSFs 保存成功")
    
    # 步骤5：处理selected_csfs_file（如果存在）
    selected_csfs_indices_dict = {}
    
    if hasattr(config, 'selected_csfs_file') and config.selected_csfs_file:
        # 生成哈希校验文件
        gdp.precompute_large_hash(target_pool_csfs_data.CSFs_block_data, target_pool_path.with_suffix('.pkl'))
        logger.info(f"初始CSFs文件{config.target_pool_file} CSFs 哈希校验文件保存成功")
        
        target_pool_csfs_hash_file = target_pool_path.with_suffix('.pkl')
        
        # 加载selected CSFs
        selected_csfs_file_path = root_path.joinpath(config.selected_csfs_file)
        selected_csfs_load = gdp.GraspFileLoad.from_filepath(selected_csfs_file_path, file_type='CSF')
        selected_csfs_data = selected_csfs_load.data_file_process()
        logger.info(f"已选择CSFs文件{config.selected_csfs_file} CSFs 读取成功")
        
        # 处理混合系数文件（如果存在）
        if hasattr(config, 'selected_csfs_mix_file') and config.selected_csfs_mix_file:
            selected_csfs_mix_coefficient_file = root_path.joinpath(config.selected_csfs_mix_file)
            selected_csfs_mix_coefficient_load = gdp.GraspFileLoad.from_filepath(
                selected_csfs_mix_coefficient_file, 
                file_type='mix'
            )
            
            selected_csfs_mix_coefficient_data = selected_csfs_mix_coefficient_load.data_file_process()
            logger.info(f"已选择CSFs文件{config.selected_csfs_file} CSFs 混合系数文件读取成功")
            
            # 根据阈值筛选
            selected_csfs_mix_coeff_above_threshold_indices = gdp.batch_asfs_mix_square_above_threshold(
                selected_csfs_mix_coefficient_data, 
                threshold=config.cutoff_value
            )
            
            selected_csfs_data.CSFs_block_data = selected_csfs_data.CSFs_block_data[
                selected_csfs_mix_coeff_above_threshold_indices[0]
            ]
            
        # 映射CSFs索引
        selected_csfs_indices_dict = gdp.maping_two_csfs_indices(
            selected_csfs_data.CSFs_block_data, 
            target_pool_csfs_hash_file
        )
        logger.info(f"已选择CSFs文件{config.selected_csfs_file} CSFs 索引映射成功")
        
    else:
        # 初始化空的indices_dict
        selected_csfs_indices_dict = {block: [] for block in range(target_pool_csfs_data.block_num)}
    
    logger.info("Target Pool CSFs 数据预处理完成")
    
    return {
        'selected_csfs_indices_dict': selected_csfs_indices_dict,
        'target_pool_csfs_data': target_pool_csfs_data,
        'logger': logger
    }

def main(config):
    """主程序逻辑"""
    try:
        result = process_target_pool_csfs(config)
        print("✅ Target Pool CSFs 数据预处理成功完成")
        print(f"📊 处理的block数量: {result['target_pool_csfs_data'].block_num}")
        print(f"📁 输出路径: {Path(config.root_path) / config.conf}")
        
        # 保存selected_csfs_indices_dict供后续使用
        selected_indices_file = Path(config.root_path) / f"{config.conf}_selected_indices.pkl"
        gdp.csfs_index_storange(result['selected_csfs_indices_dict'], selected_indices_file)
        print(f"💾 Selected indices已保存: {selected_indices_file}")
        
    except Exception as e:
        print(f"❌ Target Pool CSFs 数据预处理失败: {str(e)}")
        raise

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Target Pool CSFs 数据预处理程序')
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