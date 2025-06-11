"""
重构的机器学习训练模块
主要改进：
1. 分离配置、日志、目录初始化
2. 将大函数拆分为小函数
3. 改进错误处理和日志记录
4. 增加类型提示和文档
5. 优化代码结构和可读性
"""

import os
import sys
import time
import math
import csv
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass

from ANN import ANNClassifier
import vmcci as vmc
import graspinp as ginp
import graspdataprocessing as gdp
from sklearn.model_selection import train_test_split


@dataclass
class IterationConfig:
    """迭代配置数据类"""
    block: int
    difference: int
    cutoff_value: float
    initial_method: str
    initial_ratio: float
    expansion_ratio: float
    conf: str
    target_pool_file: str
    root_path: str
    spetral_term: List[str]


class MLPipeline:
    """机器学习管道类"""
    
    def __init__(self, config: IterationConfig):
        self.config = config
        self.path = config.root_path + '/'
        self.csfs_pool_name = self.path + config.target_pool_file
        self.logger = self._setup_logging()
        self._setup_directories()
        self._initialize_results_csv()
        
    def _setup_logging(self) -> logging.Logger:
        """设置日志配置"""
        os.makedirs("logs", exist_ok=True)
        logging.basicConfig(
            filename="logs/pipeline.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            encoding='utf-8'  # 解决中文编码问题
        )
        return logging.getLogger()
    
    def _setup_directories(self) -> None:
        """创建必要的目录"""
        directories = [
            "models", "descripotors", "descripotors_stay", 
            "test_data", "roc_curves", "results"
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def _initialize_results_csv(self) -> None:
        """初始化结果CSV文件"""
        results_csv = "results/iteration_results.csv"
        if not os.path.exists(results_csv):
            headers = [
                'Es', 'abimport_csfnum', 'MLimport_csfnum', 'sampling_method', 
                'sampling_ratio', 'current_itr_num', 'training_time', 'eval_time',
                'abinitio_time', 'all_time', 'f1', 'roc_auc', 'accuracy', 
                'precision', 'recall', 'f1_train', 'roc_auc_train',
                'accuracy_train', 'precision_train', 'recall_train'
            ]
            with open(results_csv, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(headers)
        self.logger.info("初始化结果 CSV 文件完成。")
    
    def initialize_configuration_space(self) -> Tuple[List[int], Any, Any, np.ndarray]:
        """初始化组态空间"""
        self.logger.info("初始化组态空间...")
        N_ci, cis_ts, head, indexs = vmc.get_total_ci(self.csfs_pool_name)
        
        # 描述符处理
        csfs_prim_num, csfs_pool_num = ginp.produce_basis_npy(
            "descriptior.npy", self.csfs_pool_name, 3
        )
        self.desc_pool = pd.DataFrame(np.load("descriptior.npy"))
        
        self.sum_num_min = round(math.ceil(N_ci[0] * self.config.initial_ratio))
        self.logger.info(f"组态空间初始化完成，最小组态数: {self.sum_num_min}")
        
        return N_ci, cis_ts, head, indexs
    
    def initial_sampling(self, N_ci: List[int], cis_ts: Any, head: Any, 
                        step: int) -> Tuple[np.ndarray, np.ndarray, str]:
        """初始采样"""
        file_name = f"{self.config.conf}_{self.config.block}_"
        
        if self.config.initial_method == "designation":
            if not os.path.exists(f"{self.path}{file_name}{step}.npy"):
                self.logger.error("指定初始化文件不存在，请检查文件路径")
                sys.exit(1)
            self.logger.info("指定初始化")
            sampling_method = "designation"
            index, indexs_temp = vmc.designation_initial_configuration(
                N_ci, self.sum_num_min, self.config.expansion_ratio,
                np.load(self.path + file_name + str(step) + ".npy")
            )
        elif self.config.initial_method == "fixedratio":
            self.logger.info("固定比例初始化")
            sampling_method = "fixedratio"
            index, indexs_temp = vmc.fixedratio_initial_configuration(N_ci, self.sum_num_min)
        elif self.config.initial_method == "random":
            self.logger.info("随机初始化")
            sampling_method = "random"
            index, indexs_temp = vmc.random_initial_configuration(
                N_ci, self.sum_num_min, self.config.expansion_ratio
            )
        else:
            raise ValueError(f"不支持的初始化方法: {self.config.initial_method}")
        
        # 保存组态
        vmc.save_ci(index, self.path + file_name + str(step) + ".c", N_ci, cis_ts, head)
        self.logger.info(f"初始采样完成，选择了 {len(index)} 个组态")
        
        return index, indexs_temp, sampling_method
    
    def feature_extraction(self, step: int, indexs_temp: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """特征提取"""
        self.logger.info("开始特征提取")
        file_name = f"{self.config.conf}_{self.config.block}_"
        
        try:
            index_accumulate, ci = vmc.read_dataset(self.config.root_path, step=1)
            
            # 创建布尔描述符
            ci_desc = np.zeros(index_accumulate.shape, dtype=bool)
            ci_desc[ci**2 >= self.config.cutoff_value] = True
            
            # 构建带标签的描述符
            new_desc = self.desc_pool.iloc[index_accumulate].copy()
            new_desc['label'] = ci_desc
            
            stay_desc = self.desc_pool.iloc[indexs_temp]
            
            # 保存描述符
            new_desc.to_csv(f"descripotors/{file_name}{step}_desc.csv", index=False)
            
            self.logger.info("特征提取完成")
            return new_desc, stay_desc
            
        except Exception as e:
            self.logger.error(f"特征提取失败: {e}")
            raise
    
    def train_model(self, total_data: pd.DataFrame, step: int) -> Tuple[ANNClassifier, Dict[str, float], float]:
        """训练模型"""
        self.logger.info("数据预处理")
        
        X = total_data.iloc[:, :-1]
        y = total_data.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(
            X.values, y.values, test_size=0.2, random_state=42
        )
        
        # 模型初始化
        model = ANNClassifier(input_size=X_train.shape[1], hidden_size=128)
        
        # 模型训练
        self.logger.info("训练模型")
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # 模型评估
        self.logger.info("预测与评估")
        metrics = self._evaluate_model(model, X_train, X_test, y_train, y_test, X.values, step)
        
        return model, metrics, training_time
    
    def _evaluate_model(self, model: ANNClassifier, X_train: np.ndarray, X_test: np.ndarray, 
                       y_train: np.ndarray, y_test: np.ndarray, X_all: np.ndarray, 
                       step: int) -> Dict[str, float]:
        """评估模型性能"""
        file_name = f"{self.config.conf}_{self.config.block}_"
        
        # 预测
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred_train = model.predict(X_train)
        y_proba_train = model.predict_proba(X_train)[:, 1]
        y_proba_all = model.predict_proba(X_all)
        
        # 混合系数数据（如果可用）
        try:
            index_accumulate, ci = vmc.read_dataset(self.config.root_path, step=1)
            roc_auc, pr_auc = ANNClassifier.plot_curve(
                ci, y_proba_all, y_test, y_proba, file_name + str(step-1)
            )
        except Exception as e:
            self.logger.warning(f"绘制曲线失败: {e}")
            roc_auc = pr_auc = 0.0
        
        # 计算指标
        f1, roc_auc, accuracy, precision, recall = ANNClassifier.model_evaluation(
            y_test, y_pred, y_proba
        )
        f1_train, roc_auc_train, accuracy_train, precision_train, recall_train = ANNClassifier.model_evaluation(
            y_train, y_pred_train, y_proba_train
        )
        
        # 日志记录
        self.logger.info(f"测试集预测结果: AUC:{roc_auc:.4f}, f1:{f1:.4f}, accuracy:{accuracy:.4f}")
        self.logger.info(f"训练集预测结果: AUC:{roc_auc_train:.4f}, f1:{f1_train:.4f}, accuracy:{accuracy_train:.4f}")
        
        # 保存结果
        result_file = f"test_data/{file_name}{step-1}.csv"
        pd.DataFrame({
            "y_test": y_test, "y_pred": y_pred, "y_proba": y_proba
        }).to_csv(result_file, index=False)
        
        joblib.dump(model, f"models/{file_name}{step-1}.pkl")
        self.logger.info("预测结果与模型保存成功")
        
        return {
            'f1': f1, 'roc_auc': roc_auc, 'accuracy': accuracy,
            'precision': precision, 'recall': recall,
            'f1_train': f1_train, 'roc_auc_train': roc_auc_train,
            'accuracy_train': accuracy_train, 'precision_train': precision_train,
            'recall_train': recall_train, 'pr_auc': pr_auc
        }
    
    def model_inference(self, model: ANNClassifier, stay_desc: pd.DataFrame) -> Tuple[np.ndarray, float]:
        """模型推理"""
        self.logger.info("模型推理")
        start_time = time.time()
        
        X_stay = stay_desc.iloc[:, :].values
        y_stay_pred = model.predict(X_stay)
        
        eval_time = time.time() - start_time
        self.logger.info(f"模型推理时间: {eval_time:.2f}秒")
        
        return y_stay_pred, eval_time
    
    def supervised_sampling(self, step: int, indexs_temp: np.ndarray) -> Tuple[str, Dict[str, float], float]:
        """监督学习采样"""
        self.logger.info("监督学习采样")
        
        # 特征提取
        total_data, stay_desc = self.feature_extraction(step, indexs_temp)
        
        # 模型训练
        model, metrics, training_time = self.train_model(total_data, step)
        
        # 模型推理
        y_stay_pred, eval_time = self.model_inference(model, stay_desc)
        
        return "supervised", metrics, training_time
    
    def run_iteration(self, step: int) -> bool:
        """运行单次迭代"""
        try:
            self.logger.info("*" * 48)
            self.logger.info(f"第{step}次迭代开始")
            
            if step == 1:
                # 初始化组态空间
                N_ci, cis_ts, head, indexs = self.initialize_configuration_space()
                
                # 初始采样
                index, indexs_temp, sampling_method = self.initial_sampling(N_ci, cis_ts, head, step)
                
            else:
                # 监督学习采样
                sampling_method, metrics, training_time = self.supervised_sampling(step, indexs_temp)
                
                # 这里需要添加组态选择逻辑
                # ... (configuration selection logic)
            
            self.logger.info(f"第{step}次迭代完成")
            return True
            
        except Exception as e:
            self.logger.error(f"第{step}次迭代失败: {e}")
            return False
    
    def run(self) -> None:
        """运行主循环"""
        step = 1
        
        while step > 0:
            success = self.run_iteration(step)
            if not success:
                break
            
            # 检查收敛条件或其他终止条件
            # ... (convergence check logic)
            
            step += 1
            
            # 添加最大迭代次数限制
            if step > 100:  # 防止无限循环
                self.logger.warning("达到最大迭代次数，停止执行")
                break


def main(cfg):
    """主函数 - 重构版本"""
    try:
        # 转换配置为数据类
        config = IterationConfig(
            block=cfg.block,
            difference=cfg.difference,
            cutoff_value=cfg.cutoff_value,
            initial_method=cfg.initial_method,
            initial_ratio=cfg.initial_ratio,
            expansion_ratio=cfg.expansion_ratio,
            conf=cfg.conf,
            target_pool_file=cfg.target_pool_file,
            root_path=cfg.root_path,
            spetral_term=cfg.spetral_term
        )
        
        # 创建管道并运行
        pipeline = MLPipeline(config)
        pipeline.run()
        
    except Exception as e:
        logging.error(f"程序执行失败: {e}")
        raise


if __name__ == "__main__":
    # 示例配置
    class Config:
        block = 1
        difference = 10
        cutoff_value = 0.01
        initial_method = "random"
        initial_ratio = 0.1
        expansion_ratio = 2.0
        conf = "test"
        target_pool_file = "pool.c"
        root_path = "/path/to/data"
        spetral_term = ["1S", "3P"]
    
    main(Config()) 