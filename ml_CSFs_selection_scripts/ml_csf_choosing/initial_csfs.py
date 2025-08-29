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

try:
    import graspkit as gk
except ImportError:
    print("错误: 无法导入 graspkit 模块")
    sys.exit(1)

def process_target_pool_csfs(config):
    """
    处理target_pool CSFs数据：计算描述符、保存二进制文件、生成哈希校验
    
    Args:
        config: 配置对象
        
    Returns:
        dict: 包含selected_csfs_indices_dict和处理状态的结果字典
    """
    logger = gk.setup_logging(config)
    logger.info("Target Pool CSFs 数据预处理启动")
    
    processing_steps = []
    
    root_path = Path(config.root_path)
    target_pool_file_path = root_path.joinpath(config.target_pool_file)
    
    # 步骤0：验证target_pool_file
    try:
        if not target_pool_file_path.is_file():
            error_msg = f"初始CSFs文件无效或不存在: {target_pool_file_path}"
            logger.error(error_msg)
            processing_steps.append({'step': 'file_validation', 'success': False, 'error': error_msg})
            raise FileNotFoundError(error_msg)
        
        logger.info(f"成功加载初始CSFs文件: {target_pool_file_path}")
        processing_steps.append({
            'step': 'file_validation', 
            'success': True, 
            'message': '成功加载初始CSFs文件',
            'file_path': str(target_pool_file_path)
        })
    except PermissionError as e:
        error_msg = f"无权限访问CSFs文件: {target_pool_file_path}"
        logger.error(error_msg)
        processing_steps.append({'step': 'file_validation', 'success': False, 'error': error_msg})
        raise
    except Exception as e:
        error_msg = f"加载CSFs文件时发生未知错误: {str(e)}"
        logger.error(error_msg)
        processing_steps.append({'step': 'file_validation', 'success': False, 'error': error_msg})
        raise

    try:
        # 步骤1：加载和处理target_pool CSFs
        target_pool_csfs_load = gk.GraspFileLoad.from_filepath(target_pool_file_path, file_type='CSF')
        target_pool_csfs_data = target_pool_csfs_load.data_file_process()
        logger.info(f"初始CSFs文件{config.target_pool_file} CSFs 读取成功")
        processing_steps.append({
            'step': 'csfs_loading',
            'success': True,
            'message': f'{config.target_pool_file} CSFs 读取成功',
            'block_count': target_pool_csfs_data.block_num
        })
        
        # 步骤2：计算描述符 (使用C++并行版本)
        target_pool_path = root_path.joinpath(config.conf)
        use_cpp = config.ml_config.get('use_cpp_descriptor_generator', True)
        if use_cpp:
            # 使用C++并行计算
            descriptor_file_path = target_pool_path.with_suffix('.h5')
            descriptor_run = gk.CppDescriptorGenerator(config.ml_config.get('csf_descriptor_executable', None))
            descriptor_run.generate_descriptors(input_file = target_pool_file_path.__str__(),
                            output_file = descriptor_file_path.__str__(),
                            with_subshell_info = config.ml_config.get('descriptors_with_subshell_info', True),
                            cpu_threads = config.cpu_config.get('cpu_threads', 16),
                            quiet = True)

        else:
            # 回退到Python版本
            descriptors_array, labels_array = gk.batch_process_csfs_with_multi_block(
                target_pool_csfs_data, 
                label_type='sequential',
                with_subshell_info=config.ml_config.get('descriptors_with_subshell_info', False)
            )

            gk.save_descriptors_with_multi_block(descriptors_array, labels_array, target_pool_path, 'npy')
            logger.info(f"初始CSFs文件{config.target_pool_file} CSFs 描述符保存成功")
            processing_steps.append({
                'step': 'descriptor_saving',
                'success': True,
                'message': f'{config.target_pool_file} CSFs 描述符保存成功',
                'output_path': str(target_pool_path)
            })
        backend = "C++并行" if use_cpp else "Python"
        logger.info(f"初始CSFs文件{config.target_pool_file} CSFs 描述符计算成功 (使用{backend}版本)")
        

        # 步骤4：保存CSFs二进制文件
        gk.save_csfs_binary(target_pool_csfs_data, target_pool_path)
        logger.info(f"初始CSFs文件{config.target_pool_file} CSFs 保存成功")
        processing_steps.append({
            'step': 'binary_saving',
            'success': True,
            'message': f'{config.target_pool_file} CSFs 保存成功',
            'output_path': str(target_pool_path) + '.pkl'
        })
        
        # 步骤5：处理selected_csfs_file（如果存在）
        selected_csfs_indices_dict = {}
        
        if hasattr(config, 'selected_csfs_file') and config.selected_csfs_file:
            # 生成哈希校验文件
            gk.precompute_large_hash(target_pool_csfs_data.CSFs_block_data, target_pool_path.with_suffix('.pkl'))
            logger.info(f"初始CSFs文件{config.target_pool_file} CSFs 哈希校验文件保存成功")
            processing_steps.append({
                'step': 'hash_generation',
                'success': True,
                'message': f'{config.target_pool_file} CSFs 哈希校验文件保存成功'
            })
            
            target_pool_csfs_hash_file = target_pool_path.with_suffix('.pkl')
            
            # 加载selected CSFs
            selected_csfs_file_path = root_path.joinpath(config.selected_csfs_file)
            selected_csfs_load = gk.GraspFileLoad.from_filepath(selected_csfs_file_path, file_type='CSF')
            selected_csfs_data = selected_csfs_load.data_file_process()
            logger.info(f"已选择CSFs文件{config.selected_csfs_file} CSFs 读取成功")
            processing_steps.append({
                'step': 'selected_csfs_loading',
                'success': True,
                'message': f'{config.selected_csfs_file} CSFs 读取成功',
                'file_path': str(selected_csfs_file_path)
            })
            
            # 处理混合系数文件（如果存在）
            if hasattr(config, 'selected_csfs_mix_file') and config.selected_csfs_mix_file:
                selected_csfs_mix_coefficient_file = root_path.joinpath(config.selected_csfs_mix_file)
                selected_csfs_mix_coefficient_load = gk.GraspFileLoad.from_filepath(
                    selected_csfs_mix_coefficient_file, 
                    file_type='mix'
                )
                
                selected_csfs_mix_coefficient_data = selected_csfs_mix_coefficient_load.data_file_process()
                logger.info(f"已选择CSFs文件{config.selected_csfs_file} CSFs 混合系数文件读取成功")
                processing_steps.append({
                    'step': 'mix_coefficient_loading',
                    'success': True,
                    'message': f'{config.selected_csfs_file} CSFs 混合系数文件读取成功',
                    'file_path': str(selected_csfs_mix_coefficient_file)
                })
                
                # 根据阈值筛选 - 正确处理所有blocks
                selected_csfs_mix_coeff_above_threshold_indices = gk.batch_asfs_mix_square_above_threshold(
                    selected_csfs_mix_coefficient_data, 
                    threshold=config.cutoff_value
                )
                
                threshold_filtering_results = []
                # 对所有blocks应用相应的阈值过滤，确保数据结构一致性
                for block_idx, threshold_indices in selected_csfs_mix_coeff_above_threshold_indices.items():
                    original_count = len(selected_csfs_data.CSFs_block_data[block_idx]) if block_idx < len(selected_csfs_data.CSFs_block_data) else 0
                    if block_idx < len(selected_csfs_data.CSFs_block_data) and len(threshold_indices) > 0:
                        # 使用临时变量避免自引用赋值的bug
                        original_csfs_block = selected_csfs_data.CSFs_block_data[block_idx]
                        selected_csfs_data.CSFs_block_data[block_idx] = [
                            original_csfs_block[i] for i in threshold_indices
                        ]
                        filtered_count = len(selected_csfs_data.CSFs_block_data[block_idx])
                    else:
                        filtered_count = 0
                    
                    threshold_filtering_results.append({
                        'block': block_idx,
                        'original_count': original_count,
                        'filtered_count': filtered_count,
                        'threshold': config.cutoff_value
                    })
                
                processing_steps.append({
                    'step': 'threshold_filtering',
                    'success': True,
                    'message': '根据阈值筛选CSFs',
                    'filtering_results': threshold_filtering_results
                })
            
            # 映射CSFs索引
            selected_csfs_indices_dict = gk.maping_two_csfs_indices(
                selected_csfs_data.CSFs_block_data, 
                target_pool_csfs_hash_file
            )
            logger.info(f"已选择CSFs文件{config.selected_csfs_file} CSFs 索引映射成功")
            processing_steps.append({
                'step': 'index_mapping',
                'success': True,
                'message': f'{config.selected_csfs_file} CSFs 索引映射成功',
                'mapped_indices_count': [len(indices) for indices in selected_csfs_indices_dict.values()]
            })
            
        else:
            # 初始化空的indices_dict
            selected_csfs_indices_dict = {block: [] for block in range(target_pool_csfs_data.block_num)}
            processing_steps.append({
                'step': 'empty_indices_init',
                'success': True,
                'message': '初始化空的indices_dict',
                'block_count': target_pool_csfs_data.block_num
            })
        
        logger.info("Target Pool CSFs 数据预处理完成")
        
        return {
            'selected_csfs_indices_dict': selected_csfs_indices_dict,
            'target_pool_csfs_data': target_pool_csfs_data,
            'processing_steps': processing_steps,
            'success': True
        }
        
    except Exception as e:
        error_msg = f"处理过程中发生错误: {str(e)}"
        logger.error(error_msg)
        processing_steps.append({
            'step': 'processing_error',
            'success': False,
            'error': error_msg
        })
        return {
            'selected_csfs_indices_dict': {},
            'target_pool_csfs_data': None,
            'processing_steps': processing_steps,
            'success': False,
            'error': error_msg
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
        gk.csfs_index_storange(result['selected_csfs_indices_dict'], selected_indices_file)
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
        cfg = gk.load_config(args.config)
        main(cfg)
    except FileNotFoundError:
        print(f"错误: 配置文件 {args.config} 不存在")
    except Exception as e:
        print(f"程序执行失败: {str(e)}")