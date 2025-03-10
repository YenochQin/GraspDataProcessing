#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Id :CSFs_choosing.py
@date :2024/08/02 20:38:24
@author :YenochQin (秦毅)
'''


def mix_data_abs_above_threshold(mix_data_array: np.ndarray, threshold=0.1):
    # 1. 生成布尔掩码：绝对值是否超过阈值（0.1）
    abs_above_threshold_mask = np.abs(mix_data_array) > threshold

    # 2. 提取满足条件的数值：绝对值超过阈值的元素值
    values_above_threshold = mix_data_array[abs_above_threshold_mask]

    # 3. 提取满足条件的索引：满足条件的多维坐标
    indices_where_abs_above_threshold = np.argwhere(abs_above_threshold_mask)

    # 组合结果（可选）
    result_pairs = list(zip(values_above_threshold, indices_where_abs_above_threshold))
    
    sorted_result = sorted(result_pairs, key=lambda x: abs(x[0]), reverse=True)
    
    return sorted_result

def csf_mix_above_threshold_coupling_info(mix_data_above_threshold_list: List, csf_data_list: List):
    
    csf_coupling_info = []
    
    for i in mix_data_above_threshold_list:



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

# 读取文件并删除每行的第一个字符
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

def generate_onoff(basis_size, csfs_prim_num, part_ind):
    onoff = np.zeros(basis_size, dtype=bool)
    onoff[part_ind-1] = True
    onoff[csfs_prim_num:] = True
    mark_train = onoff.copy()
    mark_apply = ~onoff
    true_count = np.count_nonzero(onoff)
    return onoff, mark_train, mark_apply, true_count

def generate_import_onoff(csfs_prim_num, part_ind):
    onoff = np.zeros(csfs_prim_num, dtype=bool)
    onoff[part_ind-1] = True
    mark_train = onoff.copy()
    mark_apply = ~onoff
    return onoff, mark_train, mark_apply

def write_atcomp_input(curr_grasp_inp, full_grasp_inp, basis_size, onoff):
    with open(curr_grasp_inp, "w") as f_curr:
        with open(full_grasp_inp, "r") as f_full:
            # 先写入full_grasp_inp文件的前5行
            for _ in range(5):
                ln = f_full.readline()
                f_curr.write(ln)
            
            # 继续按照onoff[csfs_ind]的信息写入
            for csfs_ind in range(basis_size):
                ln1 = f_full.readline()
                ln2 = f_full.readline()
                ln3 = f_full.readline()

                if onoff[csfs_ind]:
                    f_curr.write(ln1)
                    f_curr.write(ln2)
                    f_curr.write(ln3)
                    
    return None