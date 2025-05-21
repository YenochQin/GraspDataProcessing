import graspdataprocessing as gdp
import subprocess
import time
import numpy as np
import os
import math
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
    initial_ratio = cfg.initial_ratio
    expansion_ratio = cfg.expansion_ratio
    conf = cfg.conf
    target_pool_file = cfg.target_pool_file
    original_path = cfg.root_path
    spetral_term = cfg.spetral_term
    path = original_path + '/'
    csfs_pool_name = path+ target_pool_file
    smote = SMOTE(random_state=42)
    rus = RandomUnderSampler(random_state=42)

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
                'training_time', 'eval_time','abinitio_time', 'all_time', 'f1', 'roc_auc', 'accuracy', 'precision', 'recall', 
                'Es', 'abimport_csfnum', 'MLimport_csfnum', 'MLsampling_ratio', 'next_itr_num',
                'weight', 'f1_train', 'roc_auc_train','accuracy_train','precision_train','recall_train'
            ])
    logger.info("初始化结果 CSV 文件完成。")

    # 定义总的组态空间
    N_ci, cis_ts, head, indexs = gdp.get_total_ci(csfs_pool_name)
    sum_num_min = round(math.ceil(N_ci[0]*initial_ratio))

    # 初始化迭代空间
    index,indexs_temp = gdp.initial_configuration(N_ci, indexs, sum_num_min, expansion_ratio)#index指的是初始化的组态索引，indexs_temp指的是所有的组态索引
    gdp.save_ci(index, path+ conf + "_" + str(block) +"_1.c", N_ci, cis_ts, head)#保存文件操作
    stay_indexs= np.array(gdp.pop_other_ci(indexs_temp, index))#stay_indexs指的是剩余的组态索引
    gdp.save_ci(stay_indexs, path+ conf + "_" + str(block) +"_1_stay.c", N_ci, cis_ts, head)#将剩余组态保存以备机器预测与筛选

    # 组态筛选
    if 1:
        Es_term=[]
        sum_num_list=[]
        spetral_term_index=block
        b = 1
        c = 0
        while b > 0:
            file_name = conf + "_" + str(spetral_term_index) + "_" + str(b)
            # Ab initio calculation
            logger.info("************************************************")
            logger.info(f"             第{b}次迭代开始")
            os.system("cp " + file_name + ".c rcsf.inp")
            start_time = time.time()
            os.system("mpirun -hostfile hostfile -np 12 -mca btl ^openib rangular_mpi < rangular.input")
            logger.info("             rangular done")
            logger.info(f"             rangular运行时间:{time.time() - start_time}")
            if b == 1:
                os.system("rwfnestimate < rwfnestimate.input")
                logger.info("             rwfnestimate done")
                scfstart_time = time.time()
                os.system("mpirun -hostfile hostfile -np 12 -mca btl ^openib rmcdhf_mem_mpi < rmcdhf_mem_mpi.input")
                os.system("rsave " + file_name)
                logger.info("             rmcdhf done")
                logger.info(f"             rmcdhf运行时间:{time.time() - scfstart_time}")
                os.system("mpirun -hostfile hostfile -np 12 -mca btl ^openib rci_mpi < rci_mpi.input")
                logger.info("             rci done")
                logger.info(f"             rci运行时间:{time.time() - scfstart_time}")
            else:
                # os.system("rwfnestimate < rwfnestimate.input2")
                os.system("cp "+ conf + "_" + str(spetral_term_index) + "_" + str(b-1) + ".w " + file_name+ ".w")
                scfstart_time = time.time()
                os.system("mpirun -hostfile hostfile -np 12 -mca btl ^openib rci_mpi < rci_mpi.input")
                logger.info("             rci done")
                logger.info(f"             rci运行时间:{time.time() - scfstart_time}")
            
            excution_time = time.time() - start_time
            subprocess.run(["jj2lsj"], input=(conf + '_{}_{}'.format(spetral_term_index,b)).encode() + b"\ny\ny\ny\n", stdout=open("jj2lsj{}.log".format(spetral_term_index), "w"), stderr=open("jj2lsj{}.log".format(spetral_term_index), "w"), shell=True)
            logger.info("             从头算计算完成")

            # Read the information
            os.system("rlevels " + file_name + ".cm >>" + file_name +".lev")
            energylist = gdp.extract_rlevels_to_dict(file_name + ".lev")
            energy = [energylist[i]['Energy'] for i in range(len(energylist))]
            outconf = [energylist[i]['Configuration'] for i in range(len(energylist))]
            logger.info(f"             本次迭代结果：{energy}")
            logger.info(f"             耦合组态：{outconf}")
            if spetral_term in outconf:
                if b == 1:
                    result = True
                else:
                    energy_diff = abs(energy[0] - Es_term[b-2])
                    result = energy_diff >= difference
            else:
                result = False
            
            if result == True:
                Es_term.append(energy[0])
                logger.info(f"             迭代能量：{Es_term}")
                logger.info(f"             能量降低，耦合正确")
                logger.info("************************************************")
                # Get mix coefficient
                plottest = {
                    "atom": "NdGelike", 
                    "file_dir": original_path, 
                    "file_name": file_name+ ".cm", 
                    "level_parameter": "cv3pCI",
                    "file_type": "mix"
                    }
                # ci_file = gdp.GraspFileLoad(plottest).data_file_process()
                # ci_temp = ci_file.mix_coefficient_list[spetral_term_index-1]
                ci_temp = gdp.GraspFileLoad(plottest).data_file_process()[-1][0]# ci_temp type is np.array
                unique_indices = gdp.deduplicateanddemerge(ci_temp,cutoff_value)
                ci_desc = np.zeros(ci_temp.shape[1], dtype=bool)
                ci_desc[unique_indices] = True
                
                # Feature engineering, building data sets
                csfs_prim_num, csfs_pool_num = gdp.produce_basis_npy(file_name + ".npy", file_name + ".c", 3)
                csfs_prim_num_stay, csfs_pool_num_stay = gdp.produce_basis_npy(file_name + "_stay.npy", file_name + "_stay.c", 3)

                desc_pool = pd.DataFrame(np.load(file_name + ".npy"))
                new_desc = pd.concat([desc_pool,pd.DataFrame(ci_desc.T)], axis=1)
                new_desc.to_csv("descripotors/" + file_name + "_desc.csv", index=False)

                stay_desc = pd.DataFrame(np.load(file_name + "_stay.npy"))
                stay_desc.to_csv("descripotors_stay/" + file_name + "_stay_desc.csv", index=False)
                logger.info("<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>")
                logger.info("             特征提取完成")

                # Read the features and train the model
                total_data = pd.read_csv("descripotors/" + file_name + "_desc.csv")
                X = total_data.iloc[:, :-1]
                y = total_data.iloc[:, -1]
                
                stay_data = pd.read_csv("descripotors_stay/" + file_name + "_stay_desc.csv")
                X_stay = stay_data.iloc[:, :].values
                
                X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42)
                
                logger.info("             训练模型")
                if b == 1:
                    model = RandomForestClassifier(class_weight={0:1, 1:3},n_estimators=1000, verbose=True, n_jobs= -1)
                else:
                    model = joblib.load(f"models/{conf}_{spetral_term_index}_{b-1}.pkl")
                
                # Processing unbalanced data, data resampling
                results = pd.read_csv(original_path+'/results/iteration_results.csv')
                weight = [1, max(1, 12 - 2*b)]
                logger.info("weight:{}".format(weight))
                
                X_resampled, y_resampled = gdp.ANNClassifier.resampling(X_train, y_train, weight)
                start_time = time.time()
                model.fit(X_resampled, y_resampled)
                training_time = time.time() - start_time
                
                # Test set validation and configuration prediction
                logger.info("             预测与评估")
                y_pred = model.predict(X_test)
                start_time = time.time()
                y_pred_other = model.predict(X_stay)
                eval_time = time.time() - start_time
                all_time = excution_time + training_time + eval_time
                y_proba = model.predict_proba(X_test)[:, 1]
                y_pred_train = model.predict(X_train)
                y_proba_train = model.predict_proba(X_train)[:, 1]
                
                # Model evaluation
                roc_auc, pr_auc = gdp.ANNClassifier.plot_curve(y_test, y_proba, file_name)
                f1, roc_auc, accuracy, precision, recall = gdp.ANNClassifier.model_evaluation(y_test, y_pred, y_proba)
                
                # Overfitting and underfitting monitoring
                f1_train, roc_auc_train, accuracy_train, precision_train, recall_train = gdp.ANNClassifier.model_evaluation(y_train, y_pred_train, y_proba_train)
                
                # Save the result
                result_file = f"test_data/{file_name}.csv"
                pd.DataFrame({"y_test": y_test, "y_pred": y_pred, "y_proba": y_proba}).to_csv(result_file, index=False)
                joblib.dump(model, f"models/{file_name}.pkl")
                logger.info(f"预测结果与模型保存成功")
                logger.info("<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>")
                
                # Update the minimum number of CI
                sum_num = len(unique_indices)
                sum_num_list.append(sum_num)
                logging.info(f"当前重要组态数目: {sum_num}")
                logging.info(f"迭代重要组态数目: {sum_num_list}")
                if sum_num <= sum_num_min:
                    sum_num=sum_num_min
                
                # Update Important configuration indexes
                indexs_import = unique_indices.tolist()
                indexs_import_temp = [index[i] for i in indexs_import]
                indexs_import_stay = np.where(y_pred_other == 1)[0].tolist()
                indexs_import_stay_temp = [stay_indexs[i] for i in indexs_import_stay]
                np.save("results/indexs_import_ab{}_{}.npy".format(spetral_term_index,b),indexs_import_temp)
                np.save("results/indexs_import_ml{}_{}.npy".format(spetral_term_index,b),indexs_import_stay_temp)
                
                # Selective configuration
                logger.info(f"开始选择组态，当前重要组态数为：{len(indexs_import_temp)}")
                if c == 0 and b != 1:
                    logging.info("记录前一步的ML采样率信息")
                    results['MLsampling_ratio'][b-2] = (len(indexs_import_temp)-len(import_before))/len(ml_addcsf)
                    results.to_csv(original_path+'/results/iteration_results.csv', index=False)
                
                if  len(indexs_import_stay)>=expansion_ratio*sum_num:
                    ml_addcsf = np.random.choice(indexs_import_stay_temp,size=expansion_ratio*sum_num,replace=False).tolist()
                    mc_addcsf = None
                    new_addcsf = ml_addcsf
                elif len(indexs_import_stay)<=expansion_ratio*sum_num:
                    stay_index = gdp.pop_other_ci(indexs_temp, indexs_import_stay_temp+indexs_import_temp)
                    ml_addcsf = indexs_import_stay_temp
                    mc_addcsf = np.random.choice(stay_index,size=expansion_ratio*sum_num-len(indexs_import_stay_temp),replace=False).tolist()
                    new_addcsf = ml_addcsf + mc_addcsf
                MLsampling_ratio = None
                index = np.sort(np.array(indexs_import_temp + new_addcsf))
                import_before = indexs_import_temp

                logger.info(f"下一步计算组态数为：{len(index)}")
                gdp.save_ci(index, path+ conf + "_" + str(spetral_term_index) + "_" + str(b+1) + ".c", N_ci, cis_ts, head)
                stay_indexs= np.array(gdp.pop_other_ci(indexs_temp, index))
                gdp.save_ci(stay_indexs, path+ conf + "_" + str(spetral_term_index) + "_" + str(b+1) + "_stay.c", N_ci, cis_ts, head)

                # Save results
                with open(results_csv, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        training_time, eval_time, excution_time, all_time, f1, roc_auc, accuracy,precision, recall,
                        energy, len(unique_indices), len(indexs_import_stay), MLsampling_ratio, len(index),
                        weight, f1_train, roc_auc_train, accuracy_train, precision_train, recall_train
                    ])
                
                # Perform convergence calculations
                if len(Es_term) >= 3:
                    logger.info(f"执行方差计算：{b}")
                    E_std = np.std(Es_term[-3:], ddof=1)
                    N_rsd = np.std(sum_num_list[-3:], ddof=1)/abs(np.mean(Es_term[-3:]))
                    logger.info(f"第{b}次迭代，能量的标准偏差为：{E_std}，重要组态数的相对标准偏差为：{N_rsd}")
                    if E_std <= 5e-05 and N_rsd <= 0.005:
                        logger.info("达到收敛精度，迭代结束")
                        break
                b += 1
                c = 0
            else:
                c += 1
                if c == 3:
                    logger.info("连续三次波函数未改进，迭代收敛，退出筛选程序")
                    break
                logger.info('组态选择出现问题：')
                if term in outconf:
                    logger.info('     能量未降低，波函数未得到改进')
                else:
                    logger.info('     耦合出现问题')
                logger.info("正在重选组态")
                indexs_import_temp=np.load("results/indexs_import_ab{}_{}.npy".format(spetral_term_index,b-1))
                stay_indexs = gdp.pop_other_ci(indexs_temp, indexs_import_temp)
                ML_sampling_ratio = None
                mc_addcsf = np.random.choice(stay_indexs,size=expansion_ratio*sum_num,replace=False).tolist()
                index= np.sort(np.array(list(indexs_import_temp)+mc_addcsf))
                gdp.save_ci(index, path+ file_name + ".c", N_ci, cis_ts, head)
                stay_indexs= np.array(gdp.pop_other_ci(indexs_temp, index))
                gdp.save_ci(stay_indexs, path+ file_name + "_stay.c", N_ci, cis_ts, head)
                pass


if __name__ == "__main__":

    # models = models[4:]
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logger = logging.getLogger(__name__)
    term = [["5s(2).5p(6).4f(14)1S1_1S.6s(2).5d_2D"],
            ["5s(2).5p(6).4f(14)1S1_1S.6s(2).5d_2D"],
            ["5s(2).5p(6).4f(14)1S1_1S.6s(2).5d_2D"],
            ["5s(2).5p(6).4f(14)1S1_1S.6s(2).5d_2D"],
            ["5s(2).5p(6).4f(14)1S1_1S.6s(2).5d_2D"]]
    for block in range(1,5):
        cfg = {
            "atom": "NdGelike",#原子体系
            "conf": "5d6s2",
            "block": block,
            "difference": 0,#是否严格要求波函数改进
            "cutoff_value": 1e-7,#截断值
            "initial_ratio": 0.06,#初始化比例
            "expansion_ratio": 2,#拓展比例，下一次实际计算的组态数N=expansion_ratio*重要组态
            "target_pool_file": "5d6s2_10.c",#目标组态空间文件
            "root_path": "/home/workstation1/8thdd/recode_mlsci/mlsciscript_rerun",
            "spetral_term": term[block-1],
        }
        cfg['root_path'] = f"/home/workstation1/8thdd/recode_mlsci/mlsciscript_rerun/{cfg['atom']}_{cfg['conf']}_{cfg['block']}"
        # 使用 types.SimpleNamespace 将 dict 转为 class
        from types import SimpleNamespace
        cfg = SimpleNamespace(**cfg)
        if not os.path.exists(cfg.root_path):
            os.makedirs(cfg.root_path)
        # 将参数写入日志文件
        with open(cfg.root_path+'/config.txt', 'w') as f:
            for key, value in cfg.__dict__.items():
                f.write(f"{key}: {value}\n")
        # ====================== 日志配置 ======================
        # 为每个模型创建独立的文件处理器
        # 日志配置
        os.makedirs("logs", exist_ok=True)
        logging.basicConfig(
            filename="logs/pipeline.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        logger = logging.getLogger()
        
        file_handler = logging.FileHandler(cfg.root_path+'/calculation.log')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

        logger.info("开始计算")

        main(cfg)
