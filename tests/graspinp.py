import numpy as np
import pandas as pd
import re

def chunkit(str, n):
    return [str[i:i+n] for i in range(0, len(str), n)]


def get_max_num(speclab):
    specsyms = 'spdfghiklmnoqrtuv'
    l = specsyms.index(speclab[0])
    plus_els = 0 if speclab[1].strip() else 2
    return 2 * l + plus_els


def readj2(lab_str, defj2):
    if lab_str.strip()=='':
        return defj2
    strspl = lab_str.split('/')
    j2 = int(strspl[0])
    if len(strspl)==1:
        j2 *= 2
    return j2


def lines_to_csf(orbs, csf_3lines, J2tot):

    cells = [0] * (3*len(orbs))
    
    num_line = csf_3lines[0].rstrip()
    J2nu_line = csf_3lines[1].rstrip().ljust(len(num_line))
    J2cpl_line = csf_3lines[2].rstrip()[5:-4].ljust(len(num_line))

    num_chunk = chunkit(num_line, 9)
    J2nu_chunk = chunkit(J2nu_line, 9)
    J2cpl_chunk = chunkit(J2cpl_line, 9)

    lenchunks = len(num_chunk)
    J2cpl_lst = 0
    for ich in range(0, lenchunks):
        orb = num_chunk[ich][:5]
        num = int(num_chunk[ich][6:8])
        # maxnum = get_max_num(orb[3:5])
        J2nu = J2nu_chunk[ich].split(';')
        J2 = readj2(J2nu[-1], 0)
        nu = int(J2nu[0]) if len(J2nu)==2 else np.nan
        if J2cpl_lst == 0:
            J2cpl_lst = J2
            J2cpl = J2
        elif J2 == 0:
            J2cpl = J2cpl_lst
        else:
            J2cpl = readj2(J2cpl_chunk[ich], J2cpl_lst) if ich < lenchunks-1 else J2tot
            J2cpl_lst = J2cpl

        iorb = 3 * orbs.index(orb)
        cells[iorb : iorb+3] = num, J2, J2cpl

    for ii, num in enumerate(cells[::3]):
        if (num == 0) and ii != 0:
            cellstart_this = ii * 3
            cellstart_last = (ii - 1) * 3
            Jcpl_lst = cells[cellstart_last + 2]
            if Jcpl_lst != 0:
                cells[cellstart_this + 2] = Jcpl_lst
                
    return cells


def line_to_orbs(line):
    orb_line = line.rstrip()
    orbs = chunkit(orb_line, 5)
    orbs[-1] = orbs[-1].ljust(5)
    return orbs


def extract_orbs(flnm_head):
    with open(flnm_head, "r") as f_head:
        for _ in range(0,3):
            f_head.readline()
        orbsline = f_head.readline()
        orbs = line_to_orbs(orbsline)
    return orbs
def count_prim_pool(flnm_full, flnm_head):
    with open(flnm_full, "r") as f_full:
        with open(flnm_head, "r") as f_head:
            for _ in range(0, 5):
                f_head.readline()
                f_full.readline()

            csfs_prim_num = 0
            while True:
                ln = f_head.readline()
                if not ln:
                    break
                else:
                    f_head.readline()
                    f_head.readline()
                    # 这里不再跳过 flnm_full 中的对应行
                    csfs_prim_num += 1

        csfs_pool_num = 0
        while True:
            ln = f_full.readline()
            if not ln:
                break
            else:
                f_full.readline()
                f_full.readline()
                csfs_pool_num += 1

    return csfs_prim_num, csfs_pool_num

def produce_basis_npy(flnm_npy, flnm_full, J2tot):

    # 统计CSF文件中原始组态和池组态的数量
    csfs_prim_num, csfs_pool_num = count_prim_pool(flnm_full, flnm_full)
    
    # 提取轨道信息
    orbs = extract_orbs(flnm_full)
    
    # 初始化NumPy数组 (池组态数量 × 3×轨道数)
    csfs_np = np.zeros( ( csfs_pool_num, 3*len(orbs) ), dtype=np.float32)
    
    # 读取CSF文件并处理每3行为一个CSF
    with open(flnm_full, "r") as f_full:
        # 跳过前5行头信息
        for _ in range(5):
            f_full.readline()
    
        # 逐CSF处理
        for csf_ii in range(csfs_pool_num):
            # 读取3行CSF数据
            ln1 = f_full.readline()
            ln2 = f_full.readline()
            ln3 = f_full.readline()
            # 转换为数值并存入数组
            csfs_np[csf_ii, :] = lines_to_csf(orbs, [ln1, ln2, ln3], J2tot)
    
        # 保存NumPy数组到文件
        with open(flnm_npy, "wb") as f_npy:
            np.save(f_npy, csfs_np)
    
        # 返回原始组态和池组态数量
        return csfs_prim_num, csfs_pool_num

def remove_first_char_from_file(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            # 删除每行的第一个字符，并写入到新文件
            outfile.write(line[1:])

def extract_confinfo_part_ind(filename,cmin):
    """
    从指定文件中逐行提取编号、CI系数和组态。
    返回一个包含编号、CI系数和组态的列表。
    """
    pattern = r"^\s*(\d+)\s+([+-]?\d*\.?\d+)\s*$"
    curr_index_ci_conf = []
    current_block = []
    current_ci = 0
    current_index = 0

    with open(filename, 'r') as file:
        for line in file:
            match = re.match(pattern, line)

            if match:
                # 如果在一个组态块中，保存符合条件的块
                if current_block and current_ci != 0:
                    curr_index_ci_conf.append((current_index, current_ci, "".join(current_block)))
                
                # 开始新的组态块（不包括匹配行），将 current_index 转换为 int
                current_index = int(match.group(1))
                current_ci = float(match.group(2))
                current_block = []  # 清空组态块
            elif current_block or line.strip():  # 处理组态行
                current_block.append(line)  # 保留原始行内容
    
    # 处理最后一个组态块
    if current_block and current_ci != 0:
        curr_index_ci_conf.append((current_index, current_ci, "".join(current_block)))
        
    # 根据CI系数的平方进行筛选
    part_ind = [int(item[0]) for item in curr_index_ci_conf if item[1] ** 2 >= cmin]
    part_ind = np.array(part_ind)  # 转为numpy数组

    return curr_index_ci_conf, part_ind

def calculate_dynamic_weights(current_ratio,target_ratio):
    
    # 计算目标比例对应的总样本占比
    target_pos_ratio = target_ratio / (1 + target_ratio)
    target_neg_ratio = 1 - target_pos_ratio

    # 计算当前比例（加入平滑项）
    current_pos_ratio = current_ratio
    current_neg_ratio = 1-current_ratio

    # 动态计算权重
    weight_pos = target_pos_ratio / current_pos_ratio
    weight_neg = target_neg_ratio / current_neg_ratio

    return float(weight_pos),float(weight_neg)

# def extract_orbs(flnm_head):
#     with open(flnm_head, "r") as f_head:
#         for _ in range(0,3):
#             f_head.readline()
#         orbsline = f_head.readline()
#         orbs = line_to_orbs(orbsline)
#         orbs = [item.strip() for item in orbs]
#     return orbs