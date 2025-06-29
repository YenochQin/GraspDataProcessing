#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Id :choosing_csfs.py
@date :2025/06/11 10:45:00
@author :YenochQin (秦毅)
@description: 进行组态选择的流程，支持多轮迭代训练和智能索引选择
'''

import argparse
from pathlib import Path
import sys
import numpy as np

sys.path.append('/home/workstation3/AppFiles/GraspDataProcessing/src')
try:
    import graspdataprocessing as gdp
except ImportError:
    print("错误: 无法导入 graspdataprocessing 模块")
    sys.exit(1)

def load_target_pool_data(config, logger):
    """
    加载target_pool CSFs数据，从预处理文件中加载
    注意：预处理工作已在initial_csfs.py中完成
    
    Args:
        config: 配置对象
        logger: 日志记录器
        
    Returns:
        target_pool_csfs_data: 目标池CSFs数据对象
    """
    root_path = Path(config.root_path)
    target_pool_binary_path = root_path / f"{config.conf}.pkl"
    
    if not target_pool_binary_path.exists():
        logger.error(f"未找到预处理的CSFs数据文件: {target_pool_binary_path}")
        logger.error("请先运行 initial_csfs.py 进行数据预处理")
        raise FileNotFoundError(f"预处理文件不存在: {target_pool_binary_path}")
    
    # 加载CSFs数据
    target_pool_csfs_data = gdp.load_csfs_binary(target_pool_binary_path)
    logger.info(f"从二进制文件加载CSFs数据: {target_pool_binary_path}")
    
    return target_pool_csfs_data

def load_selected_indices(config, logger, target_pool_csfs_data_block_num):
    """
    加载selected indices数据
    
    Args:
        config: 配置对象
        logger: 日志记录器
        target_pool_csfs_data: 目标池CSFs数据对象（用于获取block_num）
        
    Returns:
        dict: selected CSFs indices字典
    """
    root_path = Path(config.root_path)
    selected_indices_path = root_path / f"{config.conf}_selected_indices.pkl"
    
    # 加载selected indices
    if selected_indices_path.exists():
        selected_csfs_indices_dict = gdp.csfs_index_load(selected_indices_path)
        logger.info(f"加载初筛CSFs indices: {selected_indices_path}")
    else:
        # 如果没有selected indices，创建空的
        selected_csfs_indices_dict = {block: [] for block in range(target_pool_csfs_data_block_num)}
        logger.info("未找到初筛CSFs indices文件，使用空的indices")
    
    return selected_csfs_indices_dict


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
    previous_indices_file = root_path / 'results' / f'{config.conf}_{config.cal_loop_num-1}_important_indices'
    
    if previous_indices_file.with_suffix('.pkl').exists():
        final_chosen_indices_dict = gdp.csfs_index_load(previous_indices_file)
        logger.info(f"✅ 加载上一轮计算重要的CSFs indices: {previous_indices_file}.pkl")
        logger.info(f"📊 上一轮计算重要组态数量: {[len(indices) for indices in final_chosen_indices_dict.values()]}")
        return final_chosen_indices_dict
    else:
        logger.info(f"📝 未找到上一轮计算的重要CSFs indices文件: {previous_indices_file}.pkl")
        return None

def load_ml_final_chosen_indices(config, logger):
    """
    加载机器学习训练后生成的最终选择索引（优先级最高）
    
    Args:
        config: 配置对象
        logger: 日志记录器
        
    Returns:
        dict: 机器学习最终选择的indices字典，如果不存在则返回None
    """
    root_path = Path(config.root_path)
    
    # 优先查找final_chosen_indices（train.py生成的最终选择索引）
    final_chosen_path = root_path / 'results' / f'{config.conf}_{config.cal_loop_num-1}_final_chosen_indices'
    
    if final_chosen_path.with_suffix('.pkl').exists():
        final_chosen_indices_dict = gdp.csfs_index_load(final_chosen_path)
        logger.info(f"✅ 加载ML最终选择的CSFs indices: {final_chosen_path}.pkl")
        logger.info(f"📊 ML选择的组态数量: {[len(indices) for indices in final_chosen_indices_dict.values()]}")
        return final_chosen_indices_dict
    else:
        logger.info(f"📝 未找到ML最终选择的indices文件: {final_chosen_path}.pkl")
        return None

def load_previous_ml_chosen_indices(config, logger):
    """
    加载前一轮计算后机器学习选择出的CSFs索引（备用选项）
    
    Args:
        config: 配置对象
        logger: 日志记录器
        
    Returns:
        dict: 前一轮选择的indices字典
    """
    root_path = Path(config.root_path)
    ml_results_path = root_path / 'results' / f'{config.conf}_{config.cal_loop_num-1}_final_chosen_indices'
    
    if ml_results_path.with_suffix('.pkl').exists():
        selected_csfs_indices_dict = gdp.csfs_index_load(ml_results_path)
        logger.info(f"加载机器学习选择的CSFs indices: {ml_results_path}.pkl")
        return selected_csfs_indices_dict
    else:
        logger.warning(f"未找机器学习选择的CSFs indices文件: {ml_results_path}")
        return {}

def build_csfs_from_indices(target_pool_csfs_data, chosen_indices_dict, logger):
    """
    直接从索引构建CSFs数据（类似ann3_proba.py的方式）
    
    Args:
        target_pool_csfs_data: 目标池CSFs数据
        chosen_indices_dict: 选择的索引字典
        logger: 日志记录器
        
    Returns:
        tuple: (chosen_csfs_dict, unselected_indices_dict)
    """
    chosen_csfs_dict = {}
    unselected_indices_dict = {}
    
    for block in range(target_pool_csfs_data.block_num):
        if block in chosen_indices_dict and len(chosen_indices_dict[block]) > 0:
            # 使用指定的索引
            chosen_indices = np.array(chosen_indices_dict[block])
            
            # 验证索引的有效性
            max_index = len(target_pool_csfs_data.CSFs_block_data[block]) - 1
            if np.any(chosen_indices > max_index):
                invalid_indices = chosen_indices[chosen_indices > max_index]
                logger.warning(f"块{block}中发现无效索引: {invalid_indices}，最大有效索引: {max_index}")
                chosen_indices = chosen_indices[chosen_indices <= max_index]
            
            # 构建选择的CSFs
            chosen_csfs_dict[block] = [target_pool_csfs_data.CSFs_block_data[block][i] for i in chosen_indices]
            
            # 计算未选择的索引
            all_indices = np.arange(len(target_pool_csfs_data.CSFs_block_data[block]))
            unselected_indices_dict[block] = np.setdiff1d(all_indices, chosen_indices).tolist()
            
            logger.info(f"块{block}: 通过索引选择了{len(chosen_csfs_dict[block])}个CSFs，剩余{len(unselected_indices_dict[block])}个")
        else:
            # 如果没有指定索引，使用空列表
            chosen_csfs_dict[block] = []
            unselected_indices_dict[block] = list(range(len(target_pool_csfs_data.CSFs_block_data[block])))
            logger.warning(f"块{block}: 没有指定索引，选择0个CSFs")
    
    return chosen_csfs_dict, unselected_indices_dict

def perform_csfs_selection(config):
    """
    执行组态选择的主要流程（支持智能索引选择）
    
    Args:
        config: 配置对象
        
    Returns:
        dict: 包含选择结果的字典
    """
    logger = gdp.setup_logging(config)
    # 步骤1：加载目标池数据
    target_pool_csfs_data = load_target_pool_data(config, logger)
    
    logger.info("CSFs选择程序启动")
    logger.info(f'计算循环次数: {config.cal_loop_num}')
    logger.info(f"光谱项: {config.spetral_term}")
    
    # 步骤2：创建输出目录
    root_path = Path(config.root_path)
    cal_path = root_path / f'{config.conf}_{config.cal_loop_num}'
    cal_path.mkdir(exist_ok=True)
    logger.info(f"创建输出目录: {cal_path}")
    
    # 步骤3：智能确定选择的CSFs索引
    selected_csfs_indices_dict = None
    selection_method = "unknown"
    
    if config.cal_loop_num == 1:
        # 第一轮使用initial_selected_indices（来自selected_csfs_file或空）
        
        selected_csfs_indices_dict = load_selected_indices(config, logger, target_pool_csfs_data.block_num)
        selection_method = "initial_selected"
        logger.info("第一轮选择，使用基础selected indices")
    else:
        # 后续轮次：优先级顺序
        # 1. 优先使用ML最终选择的索引（train.py生成的final_chosen_indices）
        ml_final_indices = load_ml_final_chosen_indices(config, logger)
        if ml_final_indices is not None:
            selected_csfs_indices_dict = ml_final_indices
            selection_method = "ml_final_chosen"
            logger.info("🎯 使用ML最终选择的索引（智能选择）")
        
        # 2. 如果没有ML最终选择，且计算继续，使用ML选择的索引
        elif config.continue_cal:
            selected_csfs_indices_dict = load_previous_ml_chosen_indices(config, logger)
            selection_method = "ml_chosen"
            logger.info("使用机器学习选择的CSFs indices")
        
        # 3. 如果计算不继续（出错），使用前一轮的chosen indices
        else:
            selected_csfs_indices_dict = load_previous_chosen_indices(config, logger)
            selection_method = "previous_chosen"
            logger.info("使用前一轮选择的CSFs indices")
        
        # 4. 如果以上都失败，使用空字典
        if selected_csfs_indices_dict is None:
            selected_csfs_indices_dict = {block: [] for block in range(target_pool_csfs_data.block_num)}
            selection_method = "empty_fallback"
            logger.warning("所有索引加载失败，使用空索引字典")
    
    logger.info(f"📋 选择方法: {selection_method}")
    
    # 步骤4：判断选择方式
    use_direct_indices = (selection_method in ["ml_final_chosen", "ml_chosen"] and 
                        selected_csfs_indices_dict is not None and
                        any(len(indices) > 0 for indices in selected_csfs_indices_dict.values()))
    
    if use_direct_indices:
        # 方式1：直接使用索引构建CSFs
        logger.info("🔧 使用直接索引构建方式")
        chosen_csfs_dict, unselected_indices_dict = build_csfs_from_indices(
            target_pool_csfs_data, selected_csfs_indices_dict, logger
        )
        chosen_csfs_indices_dict = selected_csfs_indices_dict
    else:
        # 方式2：传统的随机选择方式
        logger.info(f"🎲 使用传统随机选择方式，选择率: {config.chosen_ratio}")
        chosen_csfs_indices_dict = {}
        chosen_csfs_dict = {}
        unselected_indices_dict = {}

        for block in range(target_pool_csfs_data.block_num):
            chosen_csfs_dict[block], chosen_csfs_indices_dict[block], unselected_indices_dict[block] = (
                gdp.radom_choose_csfs(
                    target_pool_csfs_data.CSFs_block_data[block], 
                    config.chosen_ratio, 
                    selected_csfs_indices_dict.get(block, [])
                )
            )
    
    logger.info(f"完成CSFs选择，共{target_pool_csfs_data.block_num}个块")
    
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
    
    # 步骤8：统计信息
    total_chosen = sum(len(csfs) for csfs in chosen_csfs_dict.values())
    total_unselected = sum(len(indices) for indices in unselected_indices_dict.values())
    total_csfs = total_chosen + total_unselected
    selection_ratio = total_chosen / total_csfs if total_csfs > 0 else 0
    
    logger.info("=" * 60)
    logger.info("组态筛选完成")
    logger.info(f"🔄 计算轮次: {config.cal_loop_num}")
    logger.info(f"📋 选择方法: {selection_method}")
    logger.info(f"🔧 构建方式: {'直接索引构建' if use_direct_indices else '随机选择构建'}")
    logger.info(f"📊 选择的CSFs数量: {total_chosen}")
    logger.info(f"📊 未选择的CSFs数量: {total_unselected}")
    logger.info(f"📊 总CSFs数量: {total_csfs}")
    logger.info(f"📊 实际选择率: {selection_ratio:.4%}")
    if not use_direct_indices:
        logger.info(f"📊 配置选择率: {config.chosen_ratio:.4%}")
    logger.info("=" * 60)
    
    return {
        'chosen_csfs_dict': chosen_csfs_dict,
        'chosen_csfs_indices_dict': chosen_csfs_indices_dict,
        'unselected_indices_dict': unselected_indices_dict,
        'chosen_csfs_file_path': str(chosen_csfs_file_path),
        'target_pool_csfs_data': target_pool_csfs_data,
        'cal_path': cal_path,
        'logger': logger,
        'selection_method': selection_method,
        'use_direct_indices': use_direct_indices,
        'total_chosen': total_chosen,
        'total_unselected': total_unselected,
        'selection_ratio': selection_ratio
    }

def main(config):
    """主程序逻辑"""
    try:
        result = perform_csfs_selection(config)

        print(f"📁 输出.c文件: {result['chosen_csfs_file_path']}")
        print(f"📁 输出目录: {result['cal_path']}")
        print(f"📋 选择方法: {result['selection_method']}")
        print(f"🔧 构建方式: {'直接索引构建' if result['use_direct_indices'] else '随机选择构建'}")
        print(f"📊 选择数量: {result['total_chosen']}")
        
        return result
        
    except Exception as e:
        print(f"❌ 组态选择失败: {str(e)}")
        raise

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='CSFs组态选择程序（支持智能索引选择）')
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