# 导入必要的库
import numpy as np  # 数值计算
import pandas as pd  # 数据处理
import re           # 正则表达式
import os           # 操作系统接口

def get_total_ci(input_file):
    """
    从输入文件中获取总组态信息
    :param input_file: 输入文件路径
    :return: 
        N_ci - 每个块的组态数量列表
        cis_ts - 组态数据列表
        head - 文件头信息
        indexss - 组态索引列表
    """
    datas = []
    with open(input_file, "r", encoding='utf-8') as f:
        for line in f:
            datas.append(line)

    head = datas[0:5]  # 前5行为文件头
    cis = datas[5:]    # 剩余为组态数据
    cis_t = []
    count = []

    # 查找分隔符(*)位置
    for i in range(len(cis)):
        if "*" in cis[i]:
            count.append(i)

    cis_ts = []   # 存储所有组态块
    indexss = []  # 存储索引

    # 分割组态数据块
    start = 0
    for end in count:
        ci = cis[start:end]  # 获取一个完整块
        cis_t = []
        # 每3行组成一个组态
        for j in range(int(len(ci) / 3)):
            cis_t.append([ci[j * 3 + 0], ci[j * 3 + 1], ci[j * 3 + 2]])
            indexss.append([j * 3 + 0, j * 3 + 1, j * 3 + 2])
        cis_ts.append(cis_t)
        start = end + 1

    # 处理最后一部分数据
    ci = cis[start:]
    cis_t = []
    for j in range(int(len(ci) / 3)):
        cis_t.append([ci[j * 3 + 0], ci[j * 3 + 1], ci[j * 3 + 2]])
        indexss.append([j * 3 + 0, j * 3 + 1, j * 3 + 2])
    cis_ts.append(cis_t)

    # 计算每个块的组态数量
    N_ci = []
    for ci in cis_ts:
        N_ci.append(len(ci))
    return N_ci, cis_ts, head, indexss

def save_ci(index, output_path, N_ci, cis_ts, head):
    """
    保存组态到文件
    :param index: 要保存的组态索引
    :param output_path: 输出文件路径
    :param N_ci: 每个块的组态数量
    :param cis_ts: 组态数据
    :param head: 文件头信息
    :return: True
    """
    id_split = [0]+np.cumsum(N_ci).tolist()  # 计算分割点
    out_data = []
    
    # 按块处理组态
    for i in range(len(id_split)-1):
        if i+1==len(id_split)-1:
            ids = [int(j-id_split[i]) for j in index[(index>=id_split[i])&(index<=id_split[i+1])]]
        else:
            ids = [int(j-id_split[i]) for j in index[(index>=id_split[i])&(index<id_split[i+1])]]
        
        # 获取当前块的组态
        out = [cis_ts[i][id] for id in ids]
        outs = []
        for j in out:
            outs += j
        
        # 添加分隔符
        if i==0:
            out_data += outs
        else:
            out_data += [' *\n']
            out_data += outs
            
    # 写入文件
    with open(output_path, "w", encoding='utf-8') as f:
        for line in head+out_data:
            f.write(line)
    return True

def get_mixcoefficient(path):
    """
    从文件中获取混合系数
    :param path: 文件路径
    :return: 系数列表
    """
    coefficients = []
    with open(path, 'r') as file:
        for line in file:
            # 使用正则匹配数字和浮点数
            match = re.search(r'^\s*(\d+)\s+([+-]?\d*\.\d+)', line)
            if match:
                coefficient = float(match.group(2))
                coefficients.append(coefficient)
    return coefficients

def create_contributions_dict(contributions_list):
    """
    创建贡献字典
    :param contributions_list: 贡献值列表
    :return: 字典{索引:贡献值}
    """
    contributions_dict = {}
    for index, value in enumerate(contributions_list):
        contributions_dict[index + 1] = value  # 索引从1开始
    return contributions_dict

def create_sorted_contributions_dict(contributions_list):
    """
    创建排序后的贡献字典
    :param contributions_list: 贡献值列表
    :return: 按贡献值降序排列的字典
    """
    contributions_dict = {}
    for index, value in enumerate(contributions_list):
        contributions_dict[index + 1] = value
    # 按值降序排序
    sorted_contributions = {k: v for k, v in sorted(contributions_dict.items(), key=lambda item: item[1], reverse=True)}
    return sorted_contributions

def extract_values_greater_than(d, threshold):
    """
    提取大于阈值的值
    :param d: 字典
    :param threshold: 阈值
    :return: 值列表,键列表,计数
    """
    values_greater_than_threshold = [value for value in d.values() if value > threshold]
    indices_greater_than_threshold = [key for key, value in d.items() if value > threshold]
    count_greater_than_threshold = len(values_greater_than_threshold)
    return values_greater_than_threshold, indices_greater_than_threshold, count_greater_than_threshold

def weight_sorted(Block, n, cycle_index, sum_num):
    """
    权重排序
    :param Block: 块数据
    :param n: 循环次数
    :param cycle_index: 循环索引
    :param sum_num: 总数
    :return: 排序后的循环索引
    """
    l = []
    sorted_average = []
    for element in Block:
        k = []
        for i in range(0, sum_num):
            j = element.index(i) + 1
            k.append(j)
        l.append(k)
    
    locate = []
    average = []    
    for i in range(0, sum_num-1):
        locate = []
        for site in l:
            locate.append(site[i])
        average.append(sum(locate)/n)

    # 排序处理
    zipped = zip(average, cycle_index)
    sorted_zipped = sorted(zipped)
    sorted_average, sorted_cycle_index = zip(*sorted_zipped)
    return sorted_cycle_index

def pop_other_ci(indexs, indexs_import):
    """
    获取不在导入索引中的索引
    :param indexs: 所有索引
    :param indexs_import: 导入的索引
    :return: 剩余的索引
    """
    stay_indexs = []
    for i in indexs:
        if i not in indexs_import:
            stay_indexs.append(i)
    return stay_indexs

def trans_ci(index, indexs_import):
    """
    转换组态索引
    :param index: 原始索引
    :param indexs_import: 导入的索引
    :return: 转换后的索引
    """
    return index[indexs_import]

def rlevels(conf, c, suffix, spetral_term_index):
    """
    生成能级文件名
    :param conf: 组态名
    :param c: 计数器
    :param suffix: 后缀
    :param spetral_term_index: 光谱项索引
    :return: 生成的文件名
    """
    if suffix is None and spetral_term_index is None:
        return conf + '_' + str(c) + '.m'
    else:
        return conf + '_' + suffix + '_' + str(spetral_term_index) + '_' + str(c) + '.m'

def readconf(conf, b, suffix, spetral_term, spetral_term_index):
    """
    读取组态信息
    :param conf: 组态名
    :param b: 计数器
    :param suffix: 后缀
    :param spetral_term: 光谱项
    :param spetral_term_index: 光谱项索引
    :return: 组态信息,能量值(如果有光谱项)
    """
    param = " " + rlevels(conf, b, suffix, spetral_term_index)
    os.system("rlevels" + param + " >> rlevels_temp_jin.txt")
    with open("rlevels_temp_jin.txt", 'r') as f:
        E = f.readlines()
    Es = []
    for i, e in zip(range(len(E)), E):
        if "Energy Total" in e:
            index = i
        if spetral_term is not None:
            for spetral_term_element in [spetral_term]:
                if spetral_term_element in e:
                    term_index = i
                    Es += [float(re.findall(r"\d+\.?\d*", e)[4]) for e in E[term_index:term_index+1]]
    line = E[index+3:-1]
    outconf = []
    for d in range(1, len(line)+1):
        outconf.append(' '.join(line[d-1].split()[7:]))
    os.system("rm rlevels_temp_jin.txt")
    if spetral_term is not None:
        return outconf, Es
    else:
        return outconf

def initial_configuration(N_ci, indexs, sum_num_min, n):
    """
    初始化组态配置
    :param N_ci: 组态数量列表
    :param indexs: 索引
    :param sum_num_min: 最小组态数
    :param n: 扩展系数
    :return: 初始索引,临时索引
    """
    indexs = np.arange(sum(N_ci)-1)
    indexs_temp = indexs + 1
    index = indexs_temp[:sum_num_min-1]
    indexs_select = [i for i in indexs_temp if i not in index]
    indexs_select = index.tolist() + np.random.choice(indexs_select, size=n * sum_num_min, replace=False).tolist()
    index = np.array([0] + indexs_select)
    index.sort()
    return index, indexs_temp

def iteration_index(indexs_temp, sum_num, spetral_term_inoutconf, Block, index, spetral_term_index, indexs_import_temp_term):
    """
    迭代索引处理
    :param indexs_temp: 临时索引
    :param sum_num: 总数
    :param spetral_term_inoutconf: 光谱项输入输出配置
    :param Block: 块数据
    :param index: 索引
    :param spetral_term_index: 光谱项索引
    :param indexs_import_temp_term: 导入的临时索引项
    :return: 排序后的索引
    """
    reference_list = list(range(sum_num))
    element = Block[spetral_term_inoutconf]
    missing_number = []
    if all(i in element for i in reference_list):
        print("All elements are present in the reference_list.")
    else:
        print("Some elements are missing from the reference_list.")
        for i in reference_list:
            if i not in element:
                element.append(i)
                missing_number.append(i)
        print("missing_number: ", missing_number)
    indexs_import = element[:sum_num]
    print('indexs_import from jj2lsj', indexs_import)

    indexs_import_temp = [index[i] for i in indexs_import]
    np.save("indexs_import_temp{}.npy".format(spetral_term_index), indexs_import_temp)
    indexs_import_temp_term.append(indexs_import_temp)
    pd.DataFrame(indexs_import_temp_term).to_csv("indexs_import_temp_term{}.csv".format(spetral_term_index))
    print('index from jj2lsj to index_temp', max(indexs_import_temp), indexs_import_temp)
    stay_indexs = pop_other_ci(indexs_temp, indexs_import_temp)
    return np.sort(np.array(indexs_import_temp + np.random.choice(stay_indexs, size=sum_num, replace=False).tolist()))

def initialize_import_index(cis_ts, cis_ts_import):
    """
    初始化导入索引
    :param cis_ts: 组态数据
    :param cis_ts_import: 要导入的组态数据
    :return: 导入的索引列表
    """
    element_index_dict = {tuple(element): index for index, element in enumerate(cis_ts[0])}
    indices_import = []
    for sub_list_import in cis_ts_import[0]:
        index = element_index_dict.get(tuple(sub_list_import))
        indices_import.append(index)
    return indices_import

def divide_poolwithorb(cis_ts, orb_groups):
    """
    按轨道分组
    :param cis_ts: 组态数据
    :param orb_groups: 轨道组
    :return: 索引池,CSF池
    """
    index_pool = {}
    csfs_pool = {}
    for orb_name, orb_str in orb_groups.items():
        orb_list = orb_str.split()
        index_pool[orb_name] = []
        for i, sub_list in enumerate(cis_ts[0]):
            if any(any(orb in line for line in sub_list) for orb in orb_list):
                index_pool[orb_name].append(i)
        csfs_pool[orb_name] = [cis_ts[0][i] for i in index_pool[orb_name]]
    return index_pool, csfs_pool

def deduplicateanddemerge(ci_temp, cutoff_value):
    """
    去重和分离
    :param ci_temp: 临时CI数据
    :param cutoff_value: 截断值
    :return: 唯一索引
    """
    valid_indices_list = []
    for i in range(ci_temp.shape[0]):
        squared_coeffs = np.square(ci_temp[i])
        valid_indices = np.where(squared_coeffs > cutoff_value)[0]
        valid_indices_list.append(valid_indices)
        all_valid_indices = np.concatenate(valid_indices_list)
        unique_indices = np.unique(all_valid_indices)
        return unique_indices

def extract_rlevels_to_dict(filepath):
    """
    从能级文件提取数据到字典
    :param filepath: 文件路径
    :return: 数据字典列表
    """
    data_dict_list = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('--') or 'Configuration' in line:
                continue
            if re.match(r'^\d', line):
                fields = re.split(r'\s+', line)
                if len(fields) < 7:
                    continue  
                row_dict = {
                    'No': int(fields[0]),
                    'Pos': int(fields[1]),
                    'J': fields[2],
                    'Parity': fields[3],
                    'Energy': float(fields[4]),
                    'Total': float(fields[5]),
                    'Splitting': float(fields[6]),
                    'Configuration': ' '.join(fields[7:])
                }
                data_dict_list.append(row_dict)
    return data_dict_list

def save_data(data, output_file):
    """
    保存数据到文件
    :param data: 数据列表
    :param output_file: 输出文件路径
    """
    with open(output_file, 'w') as f:
        f.write("Energy Total (cm^-1)\tConfiguration\n")
        for item in data:
            energy = item.get('Energy', '')
            config = item.get('Configuration', '')
            f.write(f"{energy}\t{config}\n")