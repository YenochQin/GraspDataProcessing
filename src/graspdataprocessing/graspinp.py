import numpy as np
import re
import pandas as pd
from typing import List, Tuple, Any, Optional

def chunk_string(s: str, n: int) -> list[str]:
    """将字符串分割成固定长度的块"""
    return [s[i:i+n] for i in range(0, len(s), n)]

def get_max_electrons(speclab: str) -> int:
    """获取轨道最大电子容量"""
    specsyms = 'spdfghiklmnoqrtuv'
    l = specsyms.index(speclab[0])
    # 处理轨道是否有 '+' 标记 (如 s+)
    plus_els = 2 if speclab[1].strip() == '+' else 0
    return 2 * l + plus_els

def parse_j2_value(lab_str: str, defj2: int) -> int:
    """解析J2值（格式: 'J' 或 'J/2'）"""
    if not lab_str.strip():
        return defj2
    value, _, denom = lab_str.partition('/')
    j2 = int(value)
    return j2 * 2 if not denom else j2

def parse_csf_lines(orbs: List[str], csf_lines: List[str], J2tot: int) -> List[int]:
    """解析CSF的三行数据为电子配置数组"""
    # 预处理三行数据
    num_line = csf_lines[0].rstrip()
    J2nu_line = csf_lines[1].rstrip().ljust(len(num_line))
    J2cpl_line = csf_lines[2].rstrip()[5:-4].ljust(len(num_line))
    
    # 分块处理
    num_chunks = chunk_string(num_line, 9)
    J2nu_chunks = chunk_string(J2nu_line, 9)
    J2cpl_chunks = chunk_string(J2cpl_line, 9)
    
    cells = [0] * (3 * len(orbs))
    prev_j2cpl = 0
    
    for i, (num_ch, J2nu_ch) in enumerate(zip(num_chunks, J2nu_chunks)):
        orb_label = num_ch[:5].strip()
        if not orb_label:  # 跳过空轨道
            continue
        try:
            orb_idx = orbs.index(orb_label)
        except ValueError:
            continue
            
        # 解析电子数
        num_elec = int(num_ch[6:8])
        
        # 解析J2值
        J2nu_parts = J2nu_ch.split(';')
        J2 = parse_j2_value(J2nu_parts[-1].strip(), 0)
        
        # 解析耦合J2值
        if i < len(J2cpl_chunks) - 1:
            J2cpl = parse_j2_value(J2cpl_chunks[i].strip(), prev_j2cpl)
        else:
            J2cpl = J2tot
        
        prev_j2cpl = J2cpl if J2cpl != 0 else prev_j2cpl
        start_idx = orb_idx * 3
        cells[start_idx : start_idx+3] = [num_elec, J2, J2cpl]
    
    # 填充未指定的耦合值
    for i in range(1, len(orbs)):
        current_start = i * 3
        if cells[current_start] == 0:  # 空轨道
            prev_j2cpl = cells[current_start - 1]  # 前一个轨道的耦合值
            if prev_j2cpl != 0:
                cells[current_start + 2] = prev_j2cpl
    return cells

def parse_orbitals_line(line: str) -> List[str]:
    """从行中提取轨道标签"""
    return [orb.strip().ljust(5) for orb in chunk_string(line.rstrip(), 5)]

def extract_orbitals(filename: str) -> List[str]:
    """从文件头中提取轨道列表"""
    with open(filename, "r") as f:
        for _ in range(3):  # 跳过前三行
            f.readline()
        return parse_orbitals_line(f.readline())

def count_csfs(filename: str) -> Tuple[int, int]:
    """统计主组态和池组态数量"""
    prim_count, pool_count = 0, 0
    with open(filename, "r") as f:
        # 跳过前5行
        for _ in range(5):
            f.readline()
        
        # 统计主组态
        while f.readline().strip():
            f.readline()  # 跳过中间行
            f.readline()  # 跳过结束行
            prim_count += 1
        
        # 重置文件指针
        f.seek(0)
        for _ in range(5 + 3 * prim_count):
            f.readline()
        
        # 统计池组态
        while f.readline().strip():
            f.readline()
            f.readline()
            pool_count += 1
            
    return prim_count, pool_count

def generate_basis_npy(output_file: str, input_file: str, J2tot: int) -> Tuple[int, int]:
    """生成CSF数据的NumPy文件"""
    prim_count, pool_count = count_csfs(input_file)
    orbitals = extract_orbitals(input_file)
    
    csfs_data = np.zeros((pool_count, 3 * len(orbitals)), dtype=np.float32)
    
    with open(input_file, "r") as f:
        # 跳过前5行和主组态
        for _ in range(5 + 3 * prim_count):
            f.readline()
        
        # 处理每个池组态
        for i in range(pool_count):
            lines = [f.readline().rstrip() for _ in range(3)]
            csfs_data[i, :] = parse_csf_lines(orbitals, lines, J2tot)
    
    np.save(output_file, csfs_data)
    return prim_count, pool_count

def trim_file_first_char(input_path: str, output_path: str):
    """删除文件每行的首字符"""
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        outfile.writelines(line[1:] for line in infile)

def extract_configurations(filename: str, cmin: float) -> Tuple[List[Tuple[int, float, str]], np.ndarray]:
    """从输出文件中提取组态信息"""
    pattern = re.compile(r"^\s*(\d+)\s+([+-]?\d*\.?\d+)\s*$")
    configs = []
    current_block = []
    current_ci = 0.0
    current_idx = 0
    
    with open(filename, 'r') as f:
        for line in f:
            if match := pattern.match(line):
                # 保存上一个组态（如果有数据）
                if current_block and current_ci != 0:
                    configs.append((current_idx, current_ci, "".join(current_block)))
                
                # 开始新组态
                current_idx = int(match.group(1))
                current_ci = float(match.group(2))
                current_block = []
            elif line.strip():  # 组态内容行
                current_block.append(line)
    
    # 处理最后一个组态
    if current_block and current_ci != 0:
        configs.append((current_idx, current_ci, "".join(current_block)))
    
    # 筛选CI系数平方大于阈值的索引
    indices = np.array([idx for idx, ci, _ in configs if ci**2 >= cmin])
    return configs, indices

def compute_class_weights(pos_ratio: float, target_ratio: float) -> Tuple[float, float]:
    """计算动态类别权重"""
    target_pos = target_ratio / (1 + target_ratio)
    target_neg = 1 - target_pos
    
    weight_pos = target_pos / pos_ratio
    weight_neg = target_neg / (1 - pos_ratio)
    
    return weight_pos, weight_neg
