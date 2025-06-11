import numpy as np
import pandas as pd
import re
import os
from bitarray import bitarray
import random

def get_total_ci(input_path):
    datas = []
    with open(input_path, "r", encoding='utf-8') as f:
        for line in f:
            datas.append(line)

    head = datas[0:5]
    cis = datas[5:]
    cis_t = []
    count = [-1]

    # 标记每个数据块的边界
    for i in range(len(cis)):
        if "*" in cis[i]:
            count.append(i)
    count.append(len(cis))  # 修改：使用len(cis)确保包含所有数据

    cis_ts = []
    indexss = []
    
    # 处理每个数据块
    for i in range(len(count)-1):
        ci = cis[count[i]+1:count[i+1]]
        cis_t = []
        
        # 修改：确保处理所有数据，即使不是3的倍数
        for j in range(0, len(ci), 3):
            # 确保不会越界
            if j + 2 < len(ci):
                cis_t.append([ci[j], ci[j+1], ci[j+2]])
                indexss.append([j, j+1, j+2])
            else:
                # 处理不足3行的情况
                remaining = ci[j:]
                if remaining:
                    cis_t.append(remaining)
                    indexss.append(list(range(j, j+len(remaining))))
        
        cis_ts.append(cis_t)

    N_ci = [len(ci) for ci in cis_ts]
    return N_ci, cis_ts, head, indexss

def save_ci(index, output_path, N_ci, cis_ts, head):
    id_split = [0]+np.cumsum(N_ci).tolist()
    out_data = []
    for i in range(len(id_split)-1):
        if i+1==len(id_split)-1:
            ids = [int(j-id_split[i]) for j in index[(index>=id_split[i])&(index<=id_split[i+1])]]
        else:
            ids = [int(j-id_split[i]) for j in index[(index>=id_split[i])&(index<id_split[i+1])]]
        out = [cis_ts[i][id] for id in ids]
        outs = []

        for j in out:
            outs += j
        if i==0:
            out_data += outs
        else:
            out_data += [' *\n']
            out_data += outs
            
    with open(output_path, "w", encoding='utf-8') as f:
        for line in head+out_data:
            f.write(line)
    return True

def get_mixcoefficient(path):
    # Initialize an empty list to store the coefficients
    coefficients = []

    # Read the file line by line
    with open(path, 'r') as file:
        for line in file:
            # Use regular expression to find numeric identifier followed by a floating point number
            match = re.search(r'^\s*(\d+)\s+([+-]?\d*\.\d+)', line)
            if match:
                coefficient = float(match.group(2))
                coefficients.append(coefficient)

    return coefficients

#将编号和贡献对应起来，组成dict，索引即为编号,contributions_dict[编号]即为贡献值
def create_contributions_dict(contributions_list):
    contributions_dict = {}
    for index, value in enumerate(contributions_list):
        # Add 1 to the index to match the element's numeric identifier
        contributions_dict[index + 1] = value
    return contributions_dict

#将编号和贡献对应起来的dict，按照贡献降序排列，索引仍为编号，得到的是对应编号的组态贡献值
def create_sorted_contributions_dict(contributions_list):
    contributions_dict = {}
    for index, value in enumerate(contributions_list):
        # Add 1 to the index to match the element's numeric identifier
        contributions_dict[index + 1] = value

    # Sort the contributions_dict in descending order by values
    sorted_contributions = {k: v for k, v in sorted(contributions_dict.items(), key=lambda item: item[1], reverse=True)}

    return sorted_contributions

#获取高于截断的贡献、编号形成一个新的list，并获取其数量
def extract_values_greater_than(d, threshold):
    # Create a list of values greater than the threshold
    values_greater_than_threshold = [value for value in d.values() if value > threshold]

    # Create a list of corresponding keys (indices)
    indices_greater_than_threshold = [key for key, value in d.items() if value > threshold]

    # Get the count of values greater than the threshold
    count_greater_than_threshold = len(values_greater_than_threshold)

    return values_greater_than_threshold, indices_greater_than_threshold, count_greater_than_threshold

def weight_sorted(Block,n,cycle_index,sum_num):
    l=[]
    sorted_average=[]
    for element in Block:
        k=[]
        for i in range(0,sum_num):
            j=element.index(i)+1
            k.append(j)
        # print('k',k)
        l.append(k)
        print('l',l)
    locate=[]
    average=[]    
    for i in range(0,sum_num-1):
        locate=[]
        for site in l:
            # print(element[i])
            locate.append(site[i])
        # print(locate)
        average.append(sum(locate)/n)
    print(average)

    # 使用zip()函数将两个list打包成一个元组列表
    zipped = zip(average, cycle_index)

    # 使用sorted()函数对元组列表进行排序
    sorted_zipped = sorted(zipped)

    # 使用zip()函数将排序后的元组列表拆分成两个list
    sorted_average, sorted_cycle_index = zip(*sorted_zipped)
    return sorted_cycle_index

def pop_other_ci(indexs, indexs_import):
    stay_indexs = []
    for i in indexs:
        if i not in indexs_import:
            stay_indexs.append(i)
    return stay_indexs

def pop_other_ci_bitmap(indexs, indexs_import):
    # 确定位图的大小（假设 indexs 中的元素都是非负整数）
    max_val = max(indexs) if indexs else 0
    bitmap = bitarray(max_val + 1)
    bitmap.setall(False)
    
    # 将 indexs_import 中的元素在位图中标记为 True
    for i in indexs_import:
        if i <= max_val:  # 确保不超出位图范围
            bitmap[i] = True
    
    # 过滤 indexs
    return [i for i in indexs if not bitmap[i]]


def trans_ci(index, indexs_import):
    return index[indexs_import]

def rlevels(conf,c,suffix,spetral_term_index):
    if suffix is None and spetral_term_index is None:
        return  conf + '_' + str(c) + '.m'
    else:
        return  conf + '_'+ suffix + '_' + str(spetral_term_index) + '_' + str(c) + '.m'

def readconf(conf,b,suffix,spetral_term,spetral_term_index):
    param = " "+rlevels(conf,b,suffix,spetral_term_index)
    os.system("rlevels" +param+ " >> rlevels_temp_jin.txt")
    with open("rlevels_temp_jin.txt",'r') as f:
        E = f.readlines()
    Es = []
    for i, e in zip(range(len(E)), E):
        if "Energy Total" in e:
            index = i
        if spetral_term is not None:
            for spetral_term_element in [spetral_term]:
                if spetral_term_element in e:
                    term_index = i
                    Es+=[float(re.findall(r"\d+\.?\d*",e)[4]) for e in E[term_index:term_index+1]]
    line = E[index+3:-1]
    outconf = []
    for d in range(1, len(line)+1):
        outconf.append(' '.join(line[d-1].split()[7:]))
    os.system("rm rlevels_temp_jin.txt")
    if spetral_term is not None:
        return outconf, Es
    else:
        return outconf

def designation_initial_configuration(N_ci, sum_num_min, n , index):
    indexs_temp = np.arange(sum(N_ci))
    num_sample = n * sum_num_min
    indexs_select = np.random.randint(sum_num_min,len(indexs_temp), num_sample*2)
    indexs_select = np.unique(indexs_select)
    indexs_select = indexs_temp[indexs_select[:num_sample]]
    index = np.concatenate((index,indexs_select))
    return index,indexs_temp

def fixedratio_initial_configuration(N_ci, sum_num_min):
    indexs_temp = np.arange(sum(N_ci))
    index = indexs_temp[:sum_num_min]
    return index,indexs_temp

def random_initial_configuration(N_ci, sum_num_min, n):
    indexs_temp = np.arange(sum(N_ci))
    index = indexs_temp[:sum_num_min]
    num_sample = n * sum_num_min
    indexs_select = np.random.randint(sum_num_min,len(indexs_temp), num_sample*2)
    indexs_select = np.unique(indexs_select)
    indexs_select = indexs_temp[indexs_select[:num_sample]]
    index = np.concatenate((index,indexs_select))
    return index,indexs_temp

def iteration_index(indexs_temp,sum_num,spetral_term_inoutconf,Block,index,spetral_term_index,indexs_import_temp_term):
    reference_list=list(range(sum_num))#是range(sum_num)而不是range(1,sum_num+1)是因为Block= get_import_ci,Block中最后读出来的是[0,1,2,.....,sum_add-1]
    element=Block[spetral_term_inoutconf]
    missing_number=[]
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
    print('indexs_import from jj2lsj',indexs_import)

    indexs_import_temp = [index[i] for i in indexs_import]
    # indexs_import_temp = indexs_import_temp[:initial_importnum]
    np.save("indexs_import_temp{}.npy".format(spetral_term_index),indexs_import_temp)
    indexs_import_temp_term.append(indexs_import_temp)
    pd.DataFrame(indexs_import_temp_term).to_csv("indexs_import_temp_term{}.csv".format(spetral_term_index))
    print('index from jj2lsj to index_temp',max(indexs_import_temp),indexs_import_temp)
    # indexs_import = index[indexs_import].tolist()
    stay_indexs = pop_other_ci(indexs_temp, indexs_import_temp)
    return np.sort(np.array(indexs_import_temp+np.random.choice(stay_indexs,size=sum_num,replace=False).tolist()))

def initialize_import_index(cis_ts, cis_ts_import):
    element_index_dict = {tuple(element): index for index, element in enumerate(cis_ts[0])}
    indices_import = []
    for sub_list_import in cis_ts_import[0]:
        index = element_index_dict.get(tuple(sub_list_import))
        indices_import.append(index)
    return indices_import

def divide_poolwithorb(cis_ts,orb_groups):
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
    squared_coeffs = np.square(ci_temp)
    row_indices, col_indices = np.where(squared_coeffs > cutoff_value)
    unique_indices = np.unique(col_indices)
    
    return unique_indices

def extract_rlevels_to_dict(filepath):
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
    with open(output_file, 'w') as f:
        f.write("Energy Total (cm^-1)\tConfiguration\n")
        for item in data:
            energy = item.get('Energy', '')
            config = item.get('Configuration', '')
            f.write(f"{energy}\t{config}\n")

def random_choice(indexs, Num):
    index_rand = np.random.randint(0, len(indexs), size=Num*3)
    index_rand = np.unique(index_rand)
    index_rand = index_rand[np.random.permutation(len(index_rand))]
    return indexs[index_rand[:Num]]

# def process_ci_temp(index, ci_temp):
#     max_idx = np.argmax(ci_temp, axis=0)
#     min_idx = np.argmin(ci_temp, axis=0)
    
#     # 计算每列的最大值和最小值
#     max_vals = ci_temp[max_idx, np.arange(ci_temp.shape[1])]
#     min_vals = ci_temp[min_idx, np.arange(ci_temp.shape[1])]
    
#     # 比较最大值和最小值的绝对值，选择较大的
#     abs_max = np.abs(max_vals)
#     abs_min = np.abs(min_vals)
#     mask = abs_max >= abs_min
    
#     # 创建结果数组
#     result_indices = np.where(mask, max_idx, min_idx)
#     result_weights = np.where(mask, max_vals, min_vals)
    
#     # 构建DataFrame
#     result_df = pd.DataFrame({
#         'index': index[np.arange(ci_temp.shape[1])],
#         'weight': result_weights
#     })
    
#     return result_df

def process_ci_temp(index, ci_temp):
    max_idx = np.argmax(ci_temp, axis=0)
    min_idx = np.argmin(ci_temp, axis=0)
    
    # 计算每列的最大值和最小值
    max_vals = ci_temp[max_idx, np.arange(ci_temp.shape[1])]
    min_vals = ci_temp[min_idx, np.arange(ci_temp.shape[1])]
    
    # 比较最大值和最小值的绝对值，选择较大的
    abs_max = np.abs(max_vals)
    abs_min = np.abs(min_vals)
    mask = abs_max >= abs_min
    
    # 创建结果数组
    result_indices = np.where(mask, max_idx, min_idx)
    result_weights = np.where(mask, max_vals, min_vals)
    
    # 确保索引不超出范围
    valid_indices = np.arange(min(len(index), ci_temp.shape[1]))
    
    # 构建DataFrame
    result_df = pd.DataFrame({
        'index': index[valid_indices],
        'weight': result_weights[valid_indices]
    })
    
    return result_df

def read_dataset(root_path, step=1):
    # start_step = max(0, step - 6)  # 读取最近5次的数据（包含当前step）
    # if step <= cfg.classifier.dataset['init_step']:
    #     start_step = 0
    # else:
    #     start_step = min(step-cfg.classifier.dataset['max_step'], 2*(step - cfg.classifier.dataset['init_step']))
    start_step = step
    # 读取dadaset/下的csv文件
    files = os.listdir(root_path+'/dataset/')
    files = sorted([root_path+'/dataset/'+file for file in files])
    dataset = pd.read_csv(files[0])
    # 初始化dataset
    # dataset = dataset.iloc[:2]
    for i in range(start_step, len(files)):
        dataset_path = files[i]
        if os.path.exists(dataset_path):
            dataset_temp = pd.read_csv(dataset_path)
            dataset_temp.columns = dataset_temp.columns.astype(str)
            # 垂直拼接数据集，保留历史数据
            dataset = pd.concat([dataset, dataset_temp], ignore_index=True, axis=0)

    # 去重并保留最大权重
    if not dataset.empty:
        identifier_col = 'index'
        dataset = dataset.groupby(identifier_col, as_index=False).agg({
            **{col: 'first' for col in dataset.columns[:-1]},
            'weight': 'max'
        }).reset_index(drop=True)

    ci = dataset['weight'].values
    index = dataset['index'].values

    return index, ci