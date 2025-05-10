from ANN import ANNClassifier
import vmcci as vmc
import graspinp as ginp
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
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score,roc_curve,f1_score,accuracy_score,precision_score,recall_score
from sklearn.model_selection import train_test_split

# 初始化参数
block = 1
mr_num=['1']
difference=0
cutoff_value=1e-08
initial_ratio = 0.02
expansion_ratio = 2
conf = "4f145d6s2"
original_path = '/home/workstation6/ssd2t/LuI/4f145d6s2_J3_MLSCI'
spetral_term=['5s(2).5p(6).4f(14)1S1_1S.6s(2).5d_2D']
smote = SMOTE(random_state=42)
rus = RandomUnderSampler(random_state=42)
path = original_path + '/'

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
            'training_time', 'eval_time','abinitio_time', 'all_time', 'f1', 'roc_auc', 'accuracy', 'precision', 'recall', 
            'Es', 'abimport_csfnum', 'MLimport_csfnum', 'MLsampling_ratio', 'next_itr_num',
            'weight', 'f1_train', 'roc_auc_train','accuracy_train','precision_train','recall_train'
        ])
logger.info("初始化结果 CSV 文件完成。")

# 定义总的组态空间
N_ci, cis_ts, head, indexs = vmc.get_total_ci(path+ "4f145d6s2_10inter_even1.c")
sum_num_min = round(math.ceil(N_ci[0]*initial_ratio))
index,indexs_temp = vmc.initial_configuration(N_ci, indexs, sum_num_min, expansion_ratio)

# 初始化迭代空间
vmc.save_ci(index, path+ conf + "_temp_"+ str(block) +"_1.c", N_ci, cis_ts, head)
stay_indexs= np.array(vmc.pop_other_ci(indexs_temp, index))
vmc.save_ci(stay_indexs, path+ conf + "_temp_"+ str(block) +"_1_stay.c", N_ci, cis_ts, head)

# 组态筛选
for term in spetral_term:
    Es_term=[]
    indexs_select_list=[]
    sum_num_list=[]
    spetral_term_index=block
    b = 1
    c = 0
    while b > 0:
        # Ab initio calculation
        logger.info("************************************************")
        logger.info(f"             第{b}次迭代开始")
        os.system("cp " + conf + "_temp_" + str(spetral_term_index) + "_" + str(b) + ".c rcsf.inp")
        start_time = time.time()
        subprocess.run("mpirun --use-hwthread-cpus -np 64 rangular_mpi", input=b"y\n", stdout=open("rangular_temp{}.log".format(spetral_term_index), "w"), shell=True)
        logger.info("             rangular done")
        logger.info(f"             rangular运行时间:{time.time() - start_time}")
        cmd_input = f"y\n1\n4f145d6s2_10_single.w\n*\n2\n*"
        subprocess.run("rwfnestimate", input=cmd_input.encode(), stdout=open("rwfnestimate{}_np.log".format(block), "w"), stderr=open("rwfnestimate{}_np.log".format(block), "w"), shell=True)
        logger.info("             rwfnestimate done")
        scfstart_time = time.time()
        subprocess.run("mpirun --use-hwthread-cpus -np 64 rmcdhf_mem_mpi", input=b"y\n"+ mr_num[block -1].encode() +b"\n\n\n100", stdout=open("rmcdhf_temp{}_np.log".format(spetral_term_index), "w"), stderr=open("rmcdhf_temp{}_np.log".format(spetral_term_index), "w"), shell=True)
        logger.info("             rmcdhf done")
        logger.info(f"             rmcdhf运行时间:{time.time() - scfstart_time}")
        os.system("rsave " + conf + "_temp_" + str(spetral_term_index) + "_" + str(b))
        excution_time = time.time() - start_time
        subprocess.run(["jj2lsj"], input=(conf + '_temp_{}_{}'.format(spetral_term_index,b)).encode() + b"\nn\ny\ny\n", stdout=open("jj2lsj{}.log".format(spetral_term_index), "w"), stderr=open("jj2lsj{}.log".format(spetral_term_index), "w"), shell=True)
        logger.info("             从头算计算完成")

        # Read the information
        os.system("rlevels " + conf + "_temp_" + str(spetral_term_index) + "_" + str(b) + ".m >> rlevels_temp.txt")
        energylist = vmc.extract_rlevels_to_dict('rlevels_temp.txt')
        energy = [energylist[i]['Energy'] for i in range(len(energylist))]
        Es_term.append(energy[0])
        outconf = [energylist[i]['Configuration'] for i in range(len(energylist))]
        logger.info(f"             本次迭代结果：{energy}")
        logger.info(f"             耦合组态：{outconf}")
        logger.info("************************************************")
        
        # Get mix coefficient
        plottest = {
            "atom": "NdGelike", 
            "file_dir": original_path, 
            "file_name": conf + "_temp_" + str(spetral_term_index) + "_" + str(b)+ ".m", 
            "level_parameter": "cv3pCI",
            "file_type": "mix"
            }
        ci_temp = gdp.GraspFileLoad(plottest).data_file_process()[-1][0]# ci_temp type is np.array
        unique_indices = vmc.deduplicateanddemerge(ci_temp,cutoff_value)
        ci_desc = np.zeros(ci_temp.shape[1], dtype=bool)
        ci_desc[unique_indices] = True
        
        # Feature engineering, building data sets
        csfs_prim_num, csfs_pool_num = ginp.produce_basis_npy(conf + "_temp_" + str(spetral_term_index) + "_" + str(b) + ".npy", conf + "_temp_" + str(spetral_term_index) + "_" + str(b) + ".c", 3)
        csfs_prim_num_stay, csfs_pool_num_stay = ginp.produce_basis_npy(conf + "_temp_" + str(spetral_term_index) + "_" + str(b) + "_stay.npy", conf + "_temp_" + str(spetral_term_index) + "_" + str(b) + "_stay.c", 3)

        desc_pool = pd.DataFrame(np.load(conf + "_temp_" + str(spetral_term_index) + "_" + str(b) + ".npy"))
        new_desc = pd.concat([desc_pool,pd.DataFrame(ci_desc.T)], axis=1)
        new_desc.to_csv("descripotors/" + conf + "_temp_" + str(spetral_term_index) + "_" + str(b) + "_desc.csv", index=False)

        stay_desc = pd.DataFrame(np.load(conf + "_temp_" + str(spetral_term_index) + "_" + str(b) + "_stay.npy"))
        stay_desc.to_csv("descripotors_stay/" + conf + "_temp_" + str(spetral_term_index) + "_" + str(b) + "_stay_desc.csv", index=False)
        logger.info("<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>")
        logger.info("             特征提取完成")

        # Read the features and train the model
        total_data = pd.read_csv("descripotors/" + conf + "_temp_" + str(spetral_term_index) + "_" + str(b) + "_desc.csv")
        X = total_data.iloc[:, :-1]
        y = total_data.iloc[:, -1]
        
        stay_data = pd.read_csv("descripotors_stay/" + conf + "_temp_" + str(spetral_term_index) + "_" + str(b) + "_stay_desc.csv")
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
        
        X_resampled, y_resampled = ANNClassifier.resampling(X_train, y_train, weight)
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
        
        # Model evaluation
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        # Overfitting and underfitting monitoring
        y_pred_train = model.predict(X_train)
        y_proba_train = model.predict_proba(X_train)[:, 1]
        f1_train = f1_score(y_train, y_pred_train)
        roc_auc_train = roc_auc_score(y_train, y_proba_train)
        accuracy_train = accuracy_score(y_train, y_pred_train)
        precision_train = precision_score(y_train, y_pred_train)
        recall_train = recall_score(y_train, y_pred_train)
        
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
        plt.title(f"ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        roc_curve_file = f"roc_curves/{conf}_{spetral_term_index}_{b}.png"
        plt.savefig(roc_curve_file)
        plt.close()
        
        # Save the result
        result_file = f"test_data/{conf}_{spetral_term_index}_{b}.csv"
        pd.DataFrame({"y_test": y_test, "y_pred": y_pred, "y_proba": y_proba}).to_csv(result_file, index=False)
        joblib.dump(model, f"models/{conf}_{spetral_term_index}_{b}.pkl")
        logger.info(f"预测结果与模型保存成功")
        logger.info("<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>")
        
        # Update the minimum number of CI
        sum_num = len(unique_indices)
        sum_num_list.append(sum_num)
        logging.info(f"重要组态数目: {sum_num}")
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
            stay_index = vmc.pop_other_ci(indexs_temp, indexs_import_stay_temp+indexs_import_temp)
            ml_addcsf = indexs_import_stay_temp
            mc_addcsf = np.random.choice(stay_index,size=expansion_ratio*sum_num-len(indexs_import_stay_temp),replace=False).tolist()
            new_addcsf = ml_addcsf + mc_addcsf
        MLsampling_ratio = None
        index = np.sort(np.array(indexs_import_temp + new_addcsf))
        import_before = indexs_import_temp

        logger.info(f"下一步计算组态数为：{len(index)}")
        vmc.save_ci(index, path+ conf + "_temp_" + str(spetral_term_index) + "_" + str(b+1) + ".c", N_ci, cis_ts, head)
        stay_indexs= np.array(vmc.pop_other_ci(indexs_temp, index))
        vmc.save_ci(stay_indexs, path+ conf + "_temp_" + str(spetral_term_index) + "_" + str(b+1) + "_stay.c", N_ci, cis_ts, head)

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
            standard = np.std(Es_term[-3:], ddof=1)
            std_sum_num = np.std(sum_num_list[-3:], ddof=1)/3
            if standard <= 1e-04 and std_sum_num <= 0.1:
                logger.info("达到收敛精度，迭代结束")
                break
        b += 1
        c = 0

        # else:
        #     c += 1
        #     if c == 3:
        #         logger.info("连续三次波函数未改进，迭代收敛，退出筛选程序")
        #         break
        #     logger.info('组态选择出现问题：')
        #     if term in outconf:
        #         logger.info('     能量未降低，波函数未得到改进')
        #     else:
        #         logger.info('     耦合出现问题')
        #     logger.info("正在重选组态")
        #     indexs_import_temp=np.load("results/indexs_import_ab{}_{}.npy".format(spetral_term_index,b-1))
        #     stay_indexs = vmc.pop_other_ci(indexs_temp, indexs_import_temp)
        #     ML_sampling_ratio = None
        #     mc_addcsf = np.random.choice(stay_indexs,size=expansion_ratio*sum_num,replace=False).tolist()
        #     index= np.sort(np.array(list(indexs_import_temp)+mc_addcsf))
        #     vmc.save_ci(index, path+ conf + "_temp_" + str(spetral_term_index) + "_" + str(b) + ".c", N_ci, cis_ts, head)
        #     stay_indexs= np.array(vmc.pop_other_ci(indexs_temp, index))
        #     vmc.save_ci(stay_indexs, path+ conf + "_temp_" + str(spetral_term_index) + "_" + str(b) + "_stay.c", N_ci, cis_ts, head)
        #     pass
