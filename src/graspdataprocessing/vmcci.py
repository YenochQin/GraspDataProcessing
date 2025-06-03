import numpy as np
import pandas as pd
import re
import os
import logging
from bitarray import bitarray
import random
import subprocess
from collections import defaultdict
from typing import List, Tuple, Dict, Any, Optional, Union

# 预编译正则表达式以提高效率
ENERGY_TOTAL_PATTERN = re.compile(r"Energy\s+Total")
NUMERIC_VALUE_PATTERN = re.compile(r"\d+\.?\d*")
CI_SPLIT_PATTERN = re.compile(r"^\s*(\d+)\s+([+-]?\d*\.\d+)")

class CIConfig:
    """配置管理器"""
    def __init__(self, 
                 cutoff: float = 0.01,
                 min_configs: int = 50,
                 max_iterations: int = 100,
                 chunk_size: int = 1000):
        self.cutoff = cutoff
        self.min_configs = min_configs
        self.max_iterations = max_iterations
        self.chunk_size = chunk_size

def read_ci_blocks(input_file: str) -> Tuple[List[int], List[List[List[str]]], List[str]]:
    """高效读取CI数据文件，解析头部和组态块
    
    Args:
        input_file: 输入CI文件路径
        
    Returns:
        n_ci: 每个块的组态数量列表
        ci_blocks: 组态块的三层列表结构
        headers: 文件头部行
    """
    try:
        with open(input_file, "r", encoding='utf-8') as f:
            data = f.readlines()
    except IOError as e:
        logging.error(f"文件读取失败: {e}")
        raise
    
    headers = data[:5]
    ci_lines = data[5:]
    ci_blocks = []
    current_block = []
    
    # 使用星号作为分块标记
    for line in ci_lines:
        if "*" in line:
            if current_block:
                # 处理完整块
                processed_block = []
                for i in range(0, len(current_block), 3):
                    if i + 2 < len(current_block):
                        processed_block.append(current_block[i:i+3])
                ci_blocks.append(processed_block)
                current_block = []
        else:
            current_block.append(line)
    
    # 处理最后一个块
    if current_block:
        processed_block = []
        for i in range(0, len(current_block), 3):
            if i + 2 < len(current_block):
                processed_block.append(current_block[i:i+3])
        ci_blocks.append(processed_block)
    
    n_ci = [len(block) for block in ci_blocks]
    return n_ci, ci_blocks, headers

def write_selected_ci(output_path: str, n_ci: List[int], ci_blocks: List[List[List[str]]], 
                     headers: List[str], selected_indices: np.ndarray) -> bool:
    """写入选择的CI到文件
    
    Args:
        output_path: 输出文件路径
        n_ci: 每个块的组态数量
        ci_blocks: 组态块数据
        headers: 文件头部行
        selected_indices: 选择的组态索引
    """
    try:
        cumulative_sizes = np.cumsum([0] + n_ci)
        output_lines = []
        
        for block_idx in range(len(n_ci)):
            start_idx = cumulative_sizes[block_idx]
            end_idx = cumulative_sizes[block_idx + 1]
            
            # 获取当前块中选中的索引
            block_indices = selected_indices[(selected_indices >= start_idx) & 
                                            (selected_indices < end_idx)] - start_idx
            
            if block_indices.size > 0:
                if block_idx > 0:
                    output_lines.append(" *\n")
                
                for idx in block_indices:
                    if idx < len(ci_blocks[block_idx]):
                        output_lines.extend(ci_blocks[block_idx][idx])
        
        with open(output_path, "w", encoding='utf-8') as f:
            f.writelines(headers)
            f.writelines(output_lines)
        return True
    except Exception as e:
        logging.error(f"写入文件失败: {e}")
        return False

def get_mix_coefficients(file_path: str) -> List[float]:
    """高效读取混合系数文件
    
    Args:
        file_path: 混合系数文件路径
        
    Returns:
        混合系数列表
    """
    coefficients = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                match = CI_SPLIT_PATTERN.match(line)
                if match:
                    coefficients.append(float(match.group(2)))
        return coefficients
    except IOError as e:
        logging.error(f"读取混合系数失败: {e}")
        return []

def create_contributions_dict(contributions: List[float], 
                             sorted: bool = False) -> Dict[int, float]:
    """创建贡献字典
    
    Args:
        contributions: 贡献值列表
        sorted: 是否按值降序排列
        
    Returns:
        编号到贡献值的映射字典
    """
    contrib_dict = {i+1: val for i, val in enumerate(contributions)}
    return dict(sorted(contrib_dict.items(), key=lambda x: x[1], reverse=True)) if sorted else contrib_dict

def filter_by_threshold(contrib_dict: Dict[int, float], 
                        threshold: float) -> Tuple[List[float], List[int], int]:
    """过滤高于阈值的贡献
    
    Args:
        contrib_dict: 贡献字典
        threshold: 阈值
        
    Returns:
        高于阈值的贡献值列表
        对应的索引列表
        高于阈值的数量
    """
    values_above = []
    indices_above = []
    
    for idx, val in contrib_dict.items():
        if val > threshold:
            values_above.append(val)
            indices_above.append(idx)
    
    return values_above, indices_above, len(values_above)

def calculate_weighted_rank(block: List[List[int]], n: int, sum_num: int) -> List[int]:
    """计算加权排序
    
    Args:
        block: 组态块
        n: 组态数量
        sum_num: 总和数
        
    Returns:
        排序后的循环索引
    """
    positions = []
    for element in block:
        ranks = [element.index(i) + 1 for i in range(sum_num)]
        positions.append(ranks)
    
    avg_ranks = [sum(site[i] for site in positions) / n for i in range(sum_num - 1)]
    
    # 创建索引列表并排序
    cycle_indices = list(range(len(avg_ranks)))
    sorted_indices = [idx for _, idx in sorted(zip(avg_ranks, cycle_indices))]
    
    return sorted_indices

def filter_indices(all_indices: List[int], exclude_indices: List[int]) -> List[int]:
    """使用位图高效过滤索引
    
    Args:
        all_indices: 全部索引列表
        exclude_indices: 要排除的索引
        
    Returns:
        过滤后的索引列表
    """
    if not all_indices:
        return []
    
    max_val = max(all_indices)
    bitmap = bitarray(max_val + 1)
    bitmap.setall(False)
    
    for idx in exclude_indices:
        if idx <= max_val:
            bitmap[idx] = True
    
    return [idx for idx in all_indices if not bitmap[idx]]

def generate_rlevels_filename(conf: str, c: int, 
                             suffix: Optional[str] = None, 
                             spetral_term_index: Optional[int] = None) -> str:
    """生成rlevels文件名
    
    Args:
        conf: 配置前缀
        c: 计数器
        suffix: 后缀(可选)
        spetral_term_index: 光谱项索引(可选)
        
    Returns:
        生成的文件名
    """
    if suffix and spetral_term_index is not None:
        return f"{conf}_{suffix}_{spetral_term_index}_{c}.m"
    return f"{conf}_{c}.m"

def read_rlevels_output(conf: str, b: int, 
                       suffix: Optional[str] = None, 
                       spetral_term: Optional[str] = None,
                       spetral_term_index: Optional[int] = None) -> Tuple[List[str], Optional[List[float]]]:
    """读取rlevels输出
    
    Args:
        conf: 配置前缀
        b: 计数器
        suffix: 后缀(可选)
        spetral_term: 光谱项(可选)
        spetral_term_index: 光谱项索引(可选)
        
    Returns:
        输出配置列表
        能量值列表(如果有光谱项)
    """
    filename = generate_rlevels_filename(conf, b, suffix, spetral_term_index)
    temp_file = "rlevels_temp_jin.txt"
    
    try:
        subprocess.run(f"rlevels {filename}", shell=True, check=True, stdout=open(temp_file, 'w'))
        
        with open(temp_file, 'r') as f:
            lines = f.readlines()
        
        os.remove(temp_file)
        
        output_confs = []
        energies = []
        energy_found = False
        
        for line in lines:
            if "Energy Total" in line:
                energy_found = True
            elif spetral_term and spetral_term in line:
                values = NUMERIC_VALUE_PATTERN.findall(line)
                if len(values) > 4:
                    energies.append(float(values[4]))
        
        # 提取配置
        for line in lines:
            if not line.strip() or line.startswith('-'):
                continue
            parts = line.split()
            if len(parts) > 7:
                output_confs.append(' '.join(parts[7:]))
        
        return (output_confs, energies) if spetral_term else (output_confs, None)
    
    except (subprocess.CalledProcessError, IOError) as e:
        logging.error(f"执行rlevels失败: {e}")
        return [], None

def initialize_configuration(n_ci: List[int], config: CIConfig) -> Tuple[np.ndarray, np.ndarray]:
    """初始化配置
    
    Args:
        n_ci: 每个块的组态数量
        config: 配置参数
        
    Returns:
        初始索引
        所有可能索引
    """
    total_configs = sum(n_ci)
    all_indices = np.arange(total_configs)
    
    # 确保基本配置数量不超过总配置数
    min_configs = min(config.min_configs, total_configs)
    
    # 初始配置为前min_configs个
    initial_indices = all_indices[:min_configs]
    
    # 添加随机配置
    if min_configs < total_configs:
        additional = np.random.choice(
            np.arange(min_configs, total_configs),
            size=min(min_configs, total_configs - min_configs),  # 确保不超过总数
            replace=False
        )
        initial_indices = np.concatenate([initial_indices, additional])
    
    return initial_indices, all_indices

def process_large_ci(ci_data: List[Any], config: CIConfig) -> List[Any]:
    """分批处理大型CI数据
    
    Args:
        ci_data: CI数据
        config: 配置参数
        
    Returns:
        处理结果列表
    """
    results = []
    for i in range(0, len(ci_data), config.chunk_size):
        chunk = ci_data[i:i+config.chunk_size]
        # 处理当前分块 (示例)
        results.extend([item for item in chunk])
    return results

def extract_rlevels_data(filepath: str) -> List[Dict[str, Any]]:
    """提取rlevels数据到字典列表
    
    Args:
        filepath: 输入文件路径
        
    Returns:
        解析后的数据字典列表
    """
    data_records = []
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('--') or 'Configuration' in line:
                    continue
                
                if re.match(r'^\d', line):
                    fields = re.split(r'\s+', line, maxsplit=6)
                    if len(fields) < 7:
                        continue
                    
                    record = {
                        'No': int(fields[0]),
                        'Pos': int(fields[1]),
                        'J': fields[2],
                        'Parity': fields[3],
                        'Energy': float(fields[4]),
                        'Total': float(fields[5]),
                        'Splitting': float(fields[6]),
                        'Configuration': fields[7] if len(fields) > 7 else ''
                    }
                    data_records.append(record)
        return data_records
    except IOError as e:
        logging.error(f"读取rlevels数据失败: {e}")
        return []

def save_energy_configuration(data: List[Dict[str, Any]], output_file: str) -> None:
    """保存能量和配置数据
    
    Args:
        data: 数据字典列表
        output_file: 输出文件路径
    """
    try:
        with open(output_file, 'w') as f:
            f.write("Energy Total (cm^-1)\tConfiguration\n")
            for record in data:
                energy = record.get('Energy', '')
                config = record.get('Configuration', '')
                f.write(f"{energy}\t{config}\n")
    except IOError as e:
        logging.error(f"保存数据失败: {e}")

def process_ci_matrix(index: np.ndarray, ci_matrix: np.ndarray) -> pd.DataFrame:
    """处理CI矩阵
    
    Args:
        index: 索引数组
        ci_matrix: CI系数矩阵
        
    Returns:
        包含索引和权重的结果DataFrame
    """
    if ci_matrix.size == 0:
        return pd.DataFrame(columns=['index', 'weight'])
    
    max_col_idx = np.argmax(ci_matrix, axis=0)
    min_col_idx = np.argmin(ci_matrix, axis=0)
    
    max_vals = ci_matrix[max_col_idx, np.arange(ci_matrix.shape[1])]
    min_vals = ci_matrix[min_col_idx, np.arange(ci_matrix.shape[1])]
    
    # 比较绝对值选择更大的值
    abs_max = np.abs(max_vals)
    abs_min = np.abs(min_vals)
    mask = abs_max >= abs_min
    
    result_indices = np.where(mask, max_col_idx, min_col_idx)
    result_weights = np.where(mask, max_vals, min_vals)
    
    result_df = pd.DataFrame({
        'index': index,
        'weight': result_weights
    })
    
    # 按权重降序排列
    return result_df.sort_values('weight', ascending=False)

def aggregate_datasets(root_path: str, step: int, config: CIConfig) -> Tuple[np.ndarray, np.ndarray]:
    """聚合历史数据集
    
    Args:
        root_path: 根目录路径
        step: 当前步骤
        config: 配置参数
        
    Returns:
        索引数组
        CI系数数组
    """
    dataset_dir = os.path.join(root_path, 'dataset')
    if not os.path.exists(dataset_dir):
        return np.array([]), np.array([])
    
    # 获取所有数据集文件
    try:
        files = sorted([
            os.path.join(dataset_dir, f) 
            for f in os.listdir(dataset_dir)
            if f.endswith('.csv')
        ])
        
        if not files:
            return np.array([]), np.array([])
        
        # 读取最近的文件
        start_index = max(0, len(files) - config.max_iterations)
        datasets = []
        
        for file_path in files[start_index:]:
            try:
                df = pd.read_csv(file_path)
                datasets.append(df)
            except Exception as e:
                logging.warning(f"跳过无法读取的文件 {file_path}: {e}")
        
        if not datasets:
            return np.array([]), np.array([])
        
        # 合并数据集
        combined = pd.concat(datasets, ignore_index=True)
        
        # 按索引分组取最大权重
        if not combined.empty:
            aggregated = combined.groupby('index', as_index=False).agg({'weight': 'max'})
            return aggregated['index'].values, aggregated['weight'].values
    
    except Exception as e:
        logging.error(f"聚合数据集失败: {e}")
    
    return np.array([]), np.array([])

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

if __name__ == "__main__":
    # 示例用法
    config = CIConfig(cutoff=0.05, min_configs=100)
    
    # 读取CI数据
    n_ci, ci_blocks, headers = read_ci_blocks("input.ci")
    
    # 初始化配置
    init_indices, all_indices = initialize_configuration(n_ci, config)
    
    # 写入选择的CI
    write_selected_ci("output.ci", n_ci, ci_blocks, headers, init_indices)
    
    # 读取混合系数
    mix_coeffs = get_mix_coefficients("mixcoefficient.txt")
    
    # 处理大型数据集
    large_data = list(range(10000))
    processed = process_large_ci(large_data, config)
    
    logging.info("CI处理流程完成")
