import numpy as np
import pandas as pd
import re
import os

def get_total_ci(input_file):
    datas = []
    with open(input_file, "r", encoding='utf-8') as f:
        for line in f:
            datas.append(line)

    head = datas[0:5]
    cis = datas[5:]
    cis_t = []
    count = []

    for i in range(len(cis)):
        if "*" in cis[i]:
            count.append(i)

    cis_ts = []
    indexss = []

    start = 0
    for end in count:
        ci = cis[start:end]
        cis_t = []
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

    N_ci = []
    for ci in cis_ts:
        N_ci.append(len(ci))
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

def initial_configuration(N_ci, indexs, sum_num_min, n):
    # N_ci, cis_ts, head, indexs = get_total_ci(path+ conf +"_10.c")
    indexs = np.arange(sum(N_ci)-1)
    indexs_temp = indexs+1
    index = indexs_temp[:sum_num_min-1]#np.random.choice(indexs_temp,size=1000,replace=False)
    indexs_select = [i for i in indexs_temp if i not in index]
    indexs_select= index.tolist()+np.random.choice(indexs_select,size=n * sum_num_min,replace=False).tolist()
    # indexs_select= np.sort(np.array(index).tolist())
    index = np.array([0]+indexs_select)
    index.sort()
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
def deduplicateanddemerge(ci_temp,cutoff_value):
    valid_indices_list = []
    for i in range(ci_temp.shape[0]):
        squared_coeffs = np.square(ci_temp[i])
        valid_indices = np.where(squared_coeffs > cutoff_value)[0]
        valid_indices_list.append(valid_indices)
        all_valid_indices = np.concatenate(valid_indices_list)
        unique_indices = np.unique(all_valid_indices)
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