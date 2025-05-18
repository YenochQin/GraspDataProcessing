import numpy as np
import pandas as pd
import re

def chunkit(str, n):
    """
    将字符串分割成长度为n的块
    :param str: 输入字符串
    :param n: 每块长度
    :return: 分割后的字符串列表
    """
    return [str[i:i+n] for i in range(0, len(str), n)]

def get_max_num(speclab):
    """
    获取轨道最大电子数
    :param speclab: 轨道标签(如'sp', 'd'等)
    :return: 最大电子数
    """
    specsyms = 'spdfghiklmnoqrtuv'  # 轨道符号序列
    l = specsyms.index(speclab[0])  # 获取轨道角量子数
    plus_els = 0 if speclab[1].strip() else 2  # 判断是否有额外电子
    return 2 * l + plus_els  # 计算最大电子数

def readj2(lab_str, defj2):
    """
    读取J2值
    :param lab_str: 标签字符串
    :param defj2: 默认J2值
    :return: 解析后的J2值
    """
    if lab_str.strip() == '':  # 空字符串返回默认值
        return defj2
    strspl = lab_str.split('/')  # 分割字符串
    j2 = int(strspl[0])  # 获取J2值
    if len(strspl) == 1:  # 判断是否需要乘以2
        j2 *= 2
    return j2

def lines_to_csf(orbs, csf_3lines, J2tot):
    """
    将三行CSF数据转换为数值数组
    :param orbs: 轨道列表
    :param csf_3lines: 三行CSF数据
    :param J2tot: 总J2值
    :return: 数值数组
    """
    cells = [0] * (3 * len(orbs))  # 初始化结果数组
    
    # 处理三行数据
    num_line = csf_3lines[0].rstrip()  # 第一行(数字行)
    J2nu_line = csf_3lines[1].rstrip().ljust(len(num_line))  # 第二行(J2nu行)
    J2cpl_line = csf_3lines[2].rstrip()[5:-4].ljust(len(num_line))  # 第三行(J2cpl行)

    # 将每行分割成块
    num_chunk = chunkit(num_line, 9)
    J2nu_chunk = chunkit(J2nu_line, 9)
    J2cpl_chunk = chunkit(J2cpl_line, 9)

    lenchunks = len(num_chunk)
    J2cpl_lst = 0
    for ich in range(0, lenchunks):
        # 解析轨道和电子数
        orb = num_chunk[ich][:5]
        num = int(num_chunk[ich][6:8])
        maxnum = get_max_num(orb[3:5])
        
        # 解析J2nu值
        J2nu = J2nu_chunk[ich].split(';')
        J2 = readj2(J2nu[-1], 0)
        nu = int(J2nu[0]) if len(J2nu) == 2 else np.nan
        
        # 解析J2cpl值
        if J2cpl_lst == 0:
            J2cpl_lst = J2
            J2cpl = J2
        elif J2 == 0:
            J2cpl = J2cpl_lst
        else:
            J2cpl = readj2(J2cpl_chunk[ich], J2cpl_lst) if ich < lenchunks-1 else J2tot
            J2cpl_lst = J2cpl

        # 填充结果数组
        iorb = 3 * orbs.index(orb)
        cells[iorb : iorb+3] = num, J2, J2cpl

    # 处理耦合值
    for ii, num in enumerate(cells[::3]):
        if (num == 0) and ii != 0:
            cellstart_this = ii * 3
            cellstart_last = (ii - 1) * 3
            Jcpl_lst = cells[cellstart_last + 2]
            if Jcpl_lst != 0:
                cells[cellstart_this + 2] = Jcpl_lst
                
    return cells

def line_to_orbs(line):
    """
    将轨道行转换为轨道列表
    :param line: 输入行
    :return: 轨道列表
    """
    orb_line = line.rstrip()
    orbs = chunkit(orb_line, 5)  # 每5个字符分割
    orbs[-1] = orbs[-1].ljust(5)  # 最后一个轨道补齐长度
    return orbs

def extract_orbs(flnm_head):
    """
    从头文件中提取轨道信息
    :param flnm_head: 头文件路径
    :return: 轨道列表
    """
    with open(flnm_head, "r") as f_head:
        for _ in range(0,3):
            f_head.readline()  # 跳过前3行
        orbsline = f_head.readline()  # 读取轨道行
        orbs = line_to_orbs(orbsline)
    return orbs

def count_prim_pool(flnm_full, flnm_head):
    """
    计算主组态和池组态数量
    :param flnm_full: 完整文件路径
    :param flnm_head: 头文件路径
    :return: (主组态数, 池组态数)
    """
    with open(flnm_full, "r") as f_full:
        with open(flnm_head, "r") as f_head:
            # 跳过前5行
            for _ in range(0, 5):
                f_head.readline()
                f_full.readline()

            # 计算主组态数量
            csfs_prim_num = 0
            while True:
                ln = f_head.readline()
                if not ln:
                    break
                else:
                    f_head.readline()
                    f_head.readline()
                    csfs_prim_num += 1

        # 计算池组态数量
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
    """
    生成基础npy文件
    :param flnm_npy: 输出npy文件路径
    :param flnm_full: 输入完整文件路径
    :param J2tot: 总J2值
    :return: (主组态数, 池组态数)
    """
    # 获取组态数量和轨道信息
    csfs_prim_num, csfs_pool_num = count_prim_pool(flnm_full, flnm_full)
    orbs = extract_orbs(flnm_full)
    
    # 初始化numpy数组
    csfs_np = np.zeros((csfs_pool_num, 3*len(orbs)), dtype=np.float32)
    
    # 读取并处理文件
    with open(flnm_full, "r") as f_full:
        # 跳过前5行
        for _ in range(5):
            f_full.readline()

        # 处理每个组态
        for csf_ii in range(csfs_pool_num):
            ln1 = f_full.readline()  # 第一行
            ln2 = f_full.readline()  # 第二行
            ln3 = f_full.readline()  # 第三行
            csfs_np[csf_ii, :] = lines_to_csf(orbs, [ln1, ln2, ln3], J2tot)

    # 保存npy文件
    with open(flnm_npy, "wb") as f_npy:
        np.save(f_npy, csfs_np)
    return csfs_prim_num, csfs_pool_num

def remove_first_char_from_file(input_file, output_file):
    """
    删除文件中每行的第一个字符
    :param input_file: 输入文件
    :param output_file: 输出文件
    """
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            outfile.write(line[1:])  # 写入去掉第一个字符的行

def extract_confinfo_part_ind(filename, cmin):
    """
    从文件中提取组态信息
    :param filename: 输入文件
    :param cmin: CI系数最小平方值
    :return: (组态信息列表, 符合条件的索引数组)
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
                # 保存当前块
                if current_block and current_ci != 0:
                    curr_index_ci_conf.append((current_index, current_ci, "".join(current_block)))
                
                # 开始新块
                current_index = int(match.group(1))
                current_ci = float(match.group(2))
                current_block = []
            elif current_block or line.strip():
                current_block.append(line)
    
    # 处理最后一个块
    if current_block and current_ci != 0:
        curr_index_ci_conf.append((current_index, current_ci, "".join(current_block)))
        
    # 筛选符合条件的索引
    part_ind = [int(item[0]) for item in curr_index_ci_conf if item[1] ** 2 >= cmin]
    part_ind = np.array(part_ind)

    return curr_index_ci_conf, part_ind

def calculate_dynamic_weights(current_ratio, target_ratio):
    """
    计算动态权重
    :param current_ratio: 当前比例
    :param target_ratio: 目标比例
    :return: (正样本权重, 负样本权重)
    """
    # 计算目标比例
    target_pos_ratio = target_ratio / (1 + target_ratio)
    target_neg_ratio = 1 - target_pos_ratio

    # 计算当前比例
    current_pos_ratio = current_ratio
    current_neg_ratio = 1 - current_ratio

    # 计算权重
    weight_pos = target_pos_ratio / current_pos_ratio
    weight_neg = target_neg_ratio / current_neg_ratio

    return float(weight_pos), float(weight_neg)