from ANN import ANNClassifier
import vmcci as vmc
import graspinp as ginp
import graspdataprocessing as gdp
import subprocess
import time
import numpy as np
import os
import sys
import math
import random
import pandas as pd
import logging
import time
import csv
import joblib
import json
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def main(cfg):
    block = cfg.block
    difference = cfg.difference
    cutoff_value = cfg.cutoff_value
    initial_method = cfg.initial_method
    initial_ratio = cfg.initial_ratio
    expansion_ratio = cfg.expansion_ratio
    conf = cfg.conf
    target_pool_file = cfg.target_pool_file
    original_path = cfg.root_path
    spetral_term = cfg.spetral_term
    path = original_path + '/'
    csfs_pool_name = path+ target_pool_file

    # 日志配置
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        filename="logs/pipeline.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger()

    # 确保目录存在
    os.makedirs("models", exist_ok=True)
    os.makedirs("descripotors", exist_ok=True)
    os.makedirs("descripotors_stay", exist_ok=True)
    os.makedirs("test_data", exist_ok=True)
    os.makedirs("roc_curves", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # 初始化结果 CSV 文件
    results_csv = "results/iteration_results.csv"
    if not os.path.exists(results_csv):
        with open(results_csv, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([ 
                'Es', 'abimport_csfnum', 'MLimport_csfnum', 'sampling_method', 'sampling_ratio', 'current_itr_num',
                'training_time', 'eval_time','abinitio_time', 'all_time', 'f1', 'roc_auc', 'accuracy', 'precision', 'recall',
                # 'weight', 
                'f1_train', 'roc_auc_train','accuracy_train','precision_train','recall_train'
            ])
    logger.info("初始化结果 CSV 文件完成。")

    # 定义总的组态空间
    N_ci, cis_ts, head, indexs = vmc.get_total_ci(csfs_pool_name)
    sum_num_min = round(math.ceil(N_ci[0]*initial_ratio))
    # 描述符处理
    csfs_prim_num, csfs_pool_num = ginp.produce_basis_npy("descriptior.npy", csfs_pool_name, 3)
    desc_pool = pd.DataFrame(np.load("descriptior.npy"))

    # 读取历史数据
    step = 1
    mc_step = 0
    Es_term=[]
    sum_num_list=[]
    # while os.path.exists(path+ conf + "_" + str(block) + "_" + str(step) + ".c"):
    #     step

    # 迭代过程
    while step > 0:
        file_name = conf + "_" + str(block) + "_"
        # 组态采样
        if mc_step == 0:
            if step == 1:
                logger.info("************************************************")
                logger.info(f"第{step}次迭代开始")
                # 初始化采样
                logger.info("      初始化采样")
                if initial_method == "designation":
                    if not os.path.exists(f"{path}{file_name}{step}.npy"):
                        logger.error("指定初始化文件不存在，请检查文件路径")
                        sys.exit(1)
                    logger.info("      指定初始化")
                    sampling_method = "designation"
                    index,indexs_temp = vmc.designation_initial_configuration(N_ci, sum_num_min, expansion_ratio , np.load(path + file_name + str(step) + ".npy"))
                elif initial_method == "fixedratio":
                    logger.info("      固定比例初始化")
                    sampling_method = "fixedratio"
                    index,indexs_temp = vmc.fixedratio_initial_configuration(N_ci, sum_num_min)
                elif initial_method == "random":
                    logger.info("      随机初始化")
                    sampling_method = "random"
                    index,indexs_temp = vmc.random_initial_configuration(N_ci, sum_num_min, expansion_ratio)#index指的是初始化的组态索引，indexs_temp指的是所有的组态索引
                vmc.save_ci(index, path + file_name + str(step) + ".c", N_ci, cis_ts, head)
                stay_indexs= np.setdiff1d(indexs_temp, index)#stay_indexs指的是剩余的组态索引
                # vmc.save_ci(stay_indexs, path + file_name + str(step) + "_stay.c", N_ci, cis_ts, head)#将剩余组态保存以备机器预测与筛选
                # index,indexs_temp 分别表示选取的组态索引和所有组态索引
            else:
                logger.info("************************************************")
                logger.info(f"第{step}次迭代开始")
                logger.info("<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>")
                # 监督学习采样
                logger.info("      监督学习采样")
                sampling_method = "supervised"
                # Feature engineering
                logger.info("             开始特征提取")
                index_accumulate, ci = vmc.read_dataset(original_path, step=1)
                # index_accumulate 表示组态在总组态池中的索引

                ci_desc = np.zeros(index_accumulate.shape, dtype=bool)
                ci_desc[ci**2 >= cutoff_value] = True
                new_desc = desc_pool.iloc[index_accumulate].copy()
                new_desc['label'] = ci_desc
                stay_desc = desc_pool.iloc[indexs_temp]
                new_desc.to_csv("descripotors/" + file_name + str(step) + "_desc.csv", index=False)
                # stay_desc.to_csv("descripotors_stay/" + file_name + str(step) + "_stay_desc.csv", index=False)
                
                logger.info("             特征提取完成")
                # Data preprocessing
                logger.info("             数据预处理")
                total_data = new_desc
                X = total_data.iloc[:, :-1]
                y = total_data.iloc[:, -1]
                X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42)
                
                # Processing unbalanced data, data resampling
                # weight = [1, max(1, 12 - 2*step)]
                # logger.info("weight:{}".format(weight))
                # X_resampled, y_resampled = ANNClassifier.resampling(X_train, y_train, weight)
                X_resampled, y_resampled = X_train, y_train
                
                # Model initialization
                model = ANNClassifier(input_size=X_train.shape[1], hidden_size=128)
                
                # Model training
                logger.info("             训练模型")
                start_time = time.time()
                model.fit(X_resampled, y_resampled)
                training_time = time.time() - start_time
                
                # Model evaluation
                logger.info("             预测与评估")
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]
                y_pred_train = model.predict(X_train)
                y_proba_train = model.predict_proba(X_train)[:, 1]
                y_proba_all = model.predict_proba(X.values)
                print(y_proba_all[:, 1].shape)
                roc_auc, pr_auc = ANNClassifier.plot_curve(ci, y_proba_all, y_test, y_proba, file_name+str(step-1))
                f1, roc_auc, accuracy, precision, recall = ANNClassifier.model_evaluation(y_test, y_pred, y_proba)
                logger.info ("测试集预测结果:")
                logger.info (f"AUC:{roc_auc}, pr_auc:{pr_auc}, f1:{f1}, accuracy:{accuracy}, precision:{precision}, recall:{recall}")
                # Overfitting and underfitting monitoring
                f1_train, roc_auc_train, accuracy_train, precision_train, recall_train = ANNClassifier.model_evaluation(y_train, y_pred_train, y_proba_train)
                logger.info (f"训练集预测结果:")
                logger.info (f"AUC:{roc_auc_train}, f1:{f1_train}, accuracy:{accuracy_train}, precision:{precision_train}, recall:{recall_train}")
                
                # Model reasoning
                logger.info("             模型推理")
                start_time = time.time()
                X_stay = stay_desc.iloc[:, :].values
                y_stay_pred = model.predict(X_stay)
                y_stay_proba = model.predict_proba(X_stay)[:, 1]
                eval_time = time.time() - start_time
                logger.info(f"             模型推理时间:{eval_time}")
                
                # Save the result
                result_file = f"test_data/{file_name}{step-1}.csv"
                pd.DataFrame({"y_test": y_test, "y_pred": y_pred, "y_proba": y_proba}).to_csv(result_file, index=False)
                joblib.dump(model, f"models/{file_name}{step-1}.pkl")
                logger.info(f"             预测结果与模型保存成功")
                
                # Configuration sampling
                logger.info("      组态采样")
                
                # Update Important configuration indexes
                logger.info("             更新重要组态索引")
                indexs_import_temp = index[unique_indices]
                indexs_import_stay = np.where(y_stay_pred == 1)[0]
                indexs_import_stay_temp = indexs_temp[indexs_import_stay]
                np.save("results/indexs_import_ab{}_{}.npy".format(block,step-1),indexs_import_temp)
                np.save("results/indexs_import_ml{}_{}.npy".format(block,step-1),indexs_import_stay_temp)
                
                # Selective configuration
                logger.info(f"             开始选择组态，当前重要组态数为：{len(indexs_import_temp)}")
                if  len(indexs_import_stay)>=expansion_ratio*sum_num:
                    sorted_indices_within_important = np.argsort(y_stay_proba[indexs_import_stay])[::-1]
                    top_k_indices_in_y_stay = indexs_import_stay[sorted_indices_within_important[:expansion_ratio * sum_num]]
                    ml_addcsf = indexs_temp[top_k_indices_in_y_stay]
                    # ml_addcsf = vmc.random_choice(indexs_import_stay_temp, expansion_ratio*sum_num)
                    # sorted_indices = np.argsort(y_stay_proba)[::-1]
                    # ml_addcsf = stay_indexs[sorted_indices[:expansion_ratio * sum_num]]
                else:
                    ml_addcsf = indexs_import_stay_temp
                index = np.unique(np.sort(np.concatenate([indexs_import_temp,ml_addcsf])))

                logger.info(f"             第{step}次迭代计算组态数为：{len(index)}")
                vmc.save_ci(index, path+ file_name + str(step) + ".c", N_ci, cis_ts, head)
                stay_indexs= np.setdiff1d(indexs_temp, index)
                # vmc.save_ci(stay_indexs, path+ file_name + str(step) + "_stay.c", N_ci, cis_ts, head)
                logger.info("<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>")
        else:
            logger.info("************************************************")
            logger.info(f"             第{step}次迭代开始")
            if spetral_term[0] in outconf:
                logger.info('     能量未降低，波函数未得到改进')
            else:
                logger.info('     耦合出现问题')
            logger.info("     正在重选组态")
            sampling_method = "random"
            indexs_import_temp=np.load("results/indexs_import_ab{}_{}.npy".format(block,step-1))
            stay_indexs = np.setdiff1d(indexs_temp, indexs_import_temp)
            mc_addcsf = vmc.random_choice(stay_indexs,expansion_ratio*sum_num)
            index= np.sort(np.concatenate([indexs_import_temp,mc_addcsf]))
            vmc.save_ci(index, path+ file_name + str(step) + ".c", N_ci, cis_ts, head)
            stay_indexs= np.setdiff1d(indexs_temp, index)
            # vmc.save_ci(stay_indexs, path+ file_name + str(step) + "_stay.c", N_ci, cis_ts, head)
            
        # Ab initio calculation
        logger.info(f"     开始从头算")
        os.system("cp " + file_name + str(step) + ".c rcsf.inp")
        start_time = time.time()
        os.system("echo y | mpirun --use-hwthread-cpus -np 64 rangular_mpi")
        logger.info("             rangular done")
        logger.info(f"             rangular运行时间:{time.time() - start_time}")
        # if step == 1:
        #     os.system("rwfnestimate < rwfnestimate.input1")
        #     logger.info("             rwfnestimate done")
        #     scfstart_time = time.time()
        #     os.system("mpirun --use-hwthread-cpus -np 64 rmcdhf_mpi < rmcdhf_mpi.input1")
        # else:
        #     os.system("rwfnestimate < rwfnestimate.input2")
        #     scfstart_time = time.time()
        #     os.system("mpirun --use-hwthread-cpus -np 64 rmcdhf_mpi < rmcdhf_mpi.input2")
        os.system("rwfnestimate < rwfnestimate.input1")
        logger.info("             rwfnestimate done")
        scfstart_time = time.time()
        os.system("mpirun --use-hwthread-cpus -np 64 rmcdhf_mpi < rmcdhf_mpi.input1")        
        logger.info("             rmcdhf done")
        logger.info(f"             rmcdhf运行时间:{time.time() - scfstart_time}")
        excution_time = time.time() - start_time
        os.system("rsave " + file_name + str(step))
        subprocess.run(["jj2lsj"], input=(file_name + str(step)).encode() + b"\nn\ny\ny\n", stdout=open("jj2lsj{}.log".format(block), "w"), stderr=open("jj2lsj{}.log".format(block), "w"), shell=True)
        logger.info("             从头算计算完成")

        # Read the information
        os.system("rlevels " + file_name + str(step) + ".m >>" + file_name + str(step) +".lev")
        energylist = vmc.extract_rlevels_to_dict(file_name + str(step) + ".lev")
        os.system("rm " + file_name + str(step) + ".lev")
        energy = [energylist[i]['Energy'] for i in range(len(energylist))]
        outconf = [energylist[i]['Configuration'] for i in range(len(energylist))]
        logger.info(f"             本次迭代结果：{energy}")
        logger.info(f"             耦合组态：{outconf}")
        # if set(spetral_term) == set(outconf):
        #     if step == 1:
        #         result = True
        #     else:
        #         energy_diff = energy[0] - Es_term[step-2]
        #         result = energy_diff <= difference
        # else:
        #     result = False
        if step == 1:
                result = True
        else:
            energy_diff = energy[0] - Es_term[step-2]
            result = energy_diff <= difference
        
        if result == True:
            Es_term.append(energy[0])
            logger.info(f"             迭代能量：{Es_term}")
            logger.info(f"             能量降低，耦合正确")

            # Get mix coefficient
            plottest = {
                "atom": "NdGelike", 
                "file_dir": original_path, 
                "file_name": file_name + str(step) + ".m", 
                "level_parameter": "cv3pCI",
                "file_type": "mix"
                }
            # ci_file = gdp.GraspFileLoad(plottest).data_file_process()# gdp_dev_1.3
            # ci_temp = ci_file.mix_coefficient_list[block-1] # gdp_dev_1.3
            ci_temp = gdp.GraspFileLoad(plottest).data_file_process()[-1][0]# gdp_dev_1.2.5
            index_ci = vmc.process_ci_temp(index, ci_temp)
            os.makedirs(path+"/dataset", exist_ok=True)
            file_name = path+"/dataset/index_ci"+str(step)+".csv"
            pd.DataFrame(index_ci).to_csv(file_name)
            
            # Update important configuration indexes
            unique_indices = vmc.deduplicateanddemerge(ci_temp,cutoff_value)
            sum_num = len(unique_indices)
            sum_num_list.append(sum_num)
            logging.info(f"当前重要组态数目: {sum_num}")
            logging.info(f"迭代重要组态数目: {sum_num_list}")
            if sum_num <= sum_num_min:
                sum_num=sum_num_min
                logger.info(f"重要组态数目小于等于最小值，调整为{sum_num_min}")
                
            # Update sampling ratio
            if step == 1:
                sampling_ratio = len(unique_indices)/len(index)
            else:
                sampling_ratio = (len(unique_indices)-len(import_before))/len(index)
            import_before = unique_indices
            
            # Save results
            if step == 1:
                with open(results_csv, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        energy, len(unique_indices), None, sampling_method, sampling_ratio, len(index),
                        None, None, excution_time, excution_time, None, None, None,None, None,
                        None, None, None, None, None, None
                    ])
            else:
                with open(results_csv, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        energy, len(unique_indices), len(indexs_import_stay), sampling_method, sampling_ratio, len(index),
                        training_time, eval_time, excution_time, excution_time + training_time + eval_time, f1, roc_auc, accuracy,precision, recall,
                        # weight,
                        f1_train, roc_auc_train, accuracy_train, precision_train, recall_train
                    ])
        
            # Perform convergence calculations
            if len(Es_term) >= 3:
                logger.info(f"执行方差计算：{step}")
                E_std = np.std(Es_term[-3:], ddof=1)
                N_rsd = np.std(sum_num_list[-3:], ddof=1)/abs(np.mean(Es_term[-3:]))
                logger.info(f"第{step}次迭代，能量的标准偏差为：{E_std}，重要组态数的相对标准偏差为：{N_rsd}")
                if E_std <= 1e-05 and N_rsd <= 0.001:
                    logger.info("达到收敛精度，迭代结束")
                    # calculate the import configuration wave function and run the rci calculation
                    vmc.save_ci(np.array(indexs_import_temp), path + conf + "import.c", N_ci, cis_ts, head)
                    logger.info("开始后处理")
                    start_time = time.time()
                    os.system("cp " + path + conf + "import.c rcsf.inp")
                    os.system("echo y | mpirun --use-hwthread-cpus -np 64 rangular_mpi")
                    os.system("rwfnestimate < rwfnestimate.input1")
                    os.system("mpirun --use-hwthread-cpus -np 64 rmcdhf_mpi < rmcdhf_mpi.input1")
                    os.system("rsave " + path + conf + "import")
                    os.system("mpirun --use-hwthread-cpus -np 64 rci_mpi < rci_mpi.input")
                    subprocess.run(["jj2lsj"], input=(file_name + str(step)).encode() + b"\ny\ny\ny\n", stdout=open("jj2lsj_import.log", "w"), stderr=open("jj2lsj_import.log", "w"), shell=True)
                    logger.info("             import ci calculation done")
                    logger.info(f"             运行时间:{time.time() - start_time}")
                    logger.info("后处理结束,mlsci_grasp程序结束")
                    break
            logger.info("************************************************")
            step += 1
            mc_step = 0
        else:
            logger.info('组态选择出现问题：')
            mc_step += 1
            if mc_step == 3:
                logger.info("连续三次波函数未改进，迭代收敛，退出筛选程序")
                break
            logger.info(f"             耦合错误,进行MC采样")
            

if __name__ == "__main__":

    cfg = {
        "atom": "NdGelike",#原子体系
        "conf": "5d6s6p",
        "block": 1,
        "difference": 0,#是否严格要求波函数改进
        "cutoff_value": 1e-8,#截断值
        "initial_method":"random",#指定初始化：designation；随机初始化：random；固定比例初始化：fixedratio
        "initial_ratio": 0.03,#初始化比例
        "expansion_ratio": 2,#拓展比例，下一次实际计算的组态数N=expansion_ratio*重要组态
        "target_pool_file": "pool.c",#目标组态空间文件
        "root_path": "/home/workstation6/4thdd",
        "spetral_term": ['4d(10)1S0.4f(14)1S1_1S.5s(2).5p(6).6s(2).6p_2P','4d(10)1S0.4f(14)1S1_1S.5s(2).5p(6).5d_2D.6s_3D.6p_4D','4d(10)1S0.4f(14)1S1_1S.5s(2).5p(6).5d_2D.6s_3D.6p_4P','4d(10)1S0.4f(14)1S1_1S.5s(2).5p(6).5d_2D.6s_1D.6p_2P','4d(10)1S0.4f(14)1S1_1S.5s(2).5p(6).5d_2D.6s_3D.6p_2P']
        }
    # cfg['root_path'] = f"/home/workstation1/8thdd/recode_mlsci/mlsic_recode_v2/NdGelike_block1-3/{cfg['atom']}_{cfg['conf']}_{cfg['block']}"
    cfg['root_path'] = "/home/workstation6/ssd4t/LuI/5d6s6p_13_odd1_repeat/5d6s6p_13_odd1"
    cfg['target_pool_file'] = f'13_inter_odd{cfg['block']}.c'
    
    # 使用 types.SimpleNamespace 将 dict 转为 class
    from types import SimpleNamespace
    cfg = SimpleNamespace(**cfg)
    # 将参数写入日志文件
    with open(cfg.root_path+'/config.txt', 'w') as f:
        for key, value in cfg.__dict__.items():
            f.write(f"{key}: {value}\n")

    main(cfg)
