# 导入必要的库
import torch
import torch.nn as nn  # PyTorch神经网络模块
import torch.optim as optim  # 优化器模块
from sklearn.model_selection import train_test_split  # 数据集分割
from sklearn.metrics import accuracy_score  # 准确率计算
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score  # 评估指标
import matplotlib.pyplot as plt  # 绘图
import numpy as np  # 数值计算
import pandas as pd  # 数据处理
import joblib  # 模型保存/加载
import logging  # 日志记录

############################
#####  ANN Classifier  #####
############################
class ANNClassifier:
    def __init__(self, input_size, hidden_size=150, output_size=2, learning_rate=0.001):
        """
        初始化ANN分类器
        :param input_size: 输入特征维度
        :param hidden_size: 隐藏层神经元数量，默认150
        :param output_size: 输出类别数，默认2(二分类)
        :param learning_rate: 学习率，默认0.001
        """
        # 自动检测并使用GPU或CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 定义神经网络结构
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),  # 输入层到隐藏层
            nn.ReLU(),  # ReLU激活函数
            nn.Linear(hidden_size, output_size)  # 隐藏层到输出层
        ).to(self.device)
        
        # 使用Adam优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # 交叉熵损失函数，设置类别权重[9,1]处理不平衡数据
        self.criterion = nn.CrossEntropyLoss(weight=torch.Tensor([9,1]).to(self.device))

    def fit(self, X_train, y_train, X_val=None, y_val=None, batch_size=2048, max_epochs=1000, weight = [1,1]):
        """
        训练ANN模型
        :param X_train: 训练数据(numpy数组)
        :param y_train: 训练标签(numpy数组)
        :param X_val: 验证数据(可选)
        :param y_val: 验证标签(可选)
        :param batch_size: 批大小，默认2048
        :param max_epochs: 最大训练轮数，默认1000
        :param weight: 类别权重，默认[1,1]
        """
        # 将数据转换为PyTorch张量并移至相应设备
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(self.device)

        # 如果有验证数据，同样处理
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)
            y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(self.device)

        # 训练循环
        self.model.train()
        for epoch in range(max_epochs):
            permutation = torch.randperm(X_train_tensor.size(0))  # 随机打乱数据顺序
            epoch_loss = 0.0  # 记录当前epoch的总损失

            # 分批训练
            for i in range(0, X_train_tensor.size(0), batch_size):
                indices = permutation[i:i + batch_size]
                batch_X, batch_y = X_train_tensor[indices], y_train_tensor[indices]

                # 梯度清零
                self.optimizer.zero_grad()
                
                # 前向传播
                outputs = self.model(batch_X)
                
                # 计算损失
                loss = self.criterion(outputs, batch_y)
                
                # 反向传播
                loss.backward()
                
                # 参数更新
                self.optimizer.step()

                epoch_loss += loss.item()

            # 每50个epoch打印一次训练信息
            if (epoch + 1) % 50 == 0:
                logging.getLogger().info(f"Epoch [{epoch + 1}/{max_epochs}], Loss: {epoch_loss:.8f}")

                # 如果有验证数据，进行评估
                if X_val is not None and y_val is not None:
                    self.evaluate(X_val, y_val, verbose=True)

    def predict(self, X):
        """
        预测输入数据的类别
        :param X: 输入数据(numpy数组)
        :return: 预测标签(numpy数组)
        """
        # 获取预测概率
        predictions = self.predict_proba(X)
        # 使用0.5作为阈值进行二分类
        predictions = predictions[:, 1]>0.5
        return predictions
    
    def predict_proba(self, X):
        """
        预测输入数据的类别概率
        :param X: 输入数据(numpy数组)
        :return: 预测概率(numpy数组)
        """
        self.model.eval()  # 设置为评估模式
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        # 禁用梯度计算以加速预测
        with torch.no_grad():
            # 使用softmax获取概率
            outputs = torch.softmax(self.model(X_tensor), dim=1)

        return outputs.cpu().numpy()  # 返回numpy数组

    def evaluate(self, X, y, verbose=True):
        """
        评估模型性能
        :param X: 输入数据(numpy数组)
        :param y: 真实标签(numpy数组)
        :param verbose: 是否打印评估指标
        :return: 包含评估指标的字典
        """
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            y_pred = torch.argmax(outputs, dim=1).cpu().numpy()  # 预测类别
            y_proba = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # 正类概率

        # 计算各种评估指标
        metrics = {
            "accuracy": accuracy_score(y, y_pred),  # 准确率
            "f1_score": f1_score(y, y_pred),  # F1分数
            "precision": precision_score(y, y_pred),  # 精确率
            "recall": recall_score(y, y_pred),  # 召回率
            "roc_auc": roc_auc_score(y, y_proba)  # ROC AUC
        }

        # 打印评估结果
        if verbose:
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"F1 Score: {metrics['f1_score']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"ROC AUC: {metrics['roc_auc']:.4f}")

        return metrics

    def plot_roc_curve(self, X, y):
        """
        绘制ROC曲线
        :param X: 输入数据(numpy数组)
        :param y: 真实标签(numpy数组)
        """
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            y_proba = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()

        # 计算ROC曲线
        fpr, tpr, _ = roc_curve(y, y_proba)
        
        # 绘制ROC曲线
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc_score(y, y_proba):.4f})")
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.show()

    def save_model(self, path):
        """
        保存模型参数
        :param path: 保存路径
        """
        torch.save(self.model.state_dict(), path)  # 保存PyTorch模型参数
        joblib.dump(self, path + '_ann.pkl')  # 保存整个对象

    def load_model(self, path):
        """
        加载模型参数
        :param path: 加载路径
        :return: 加载的模型
        """
        self.model.load_state_dict(torch.load(path))  # 加载PyTorch模型参数
        self.model.to(self.device)
        return joblib.load(path + '_ann.pkl')  # 返回加载的模型
    
    def resampling(X_train, y_train, weight):
        """
        上采样方法处理不平衡数据
        :param X_train: 训练数据
        :param y_train: 训练标签
        :param weight: 类别权重
        :return: 重采样后的数据和标签
        """
        # 分离正负样本
        X_positive = X_train[y_train==1]
        X_negtive = X_train[y_train==0]
        Y_positive = y_train[y_train==1]
        Y_negtive = y_train[y_train==0]

        # 确定采样数量
        Num_min = max(X_positive.shape[0], X_negtive.shape[0])
        N_sample_positive, N_sample_negtive = int(weight[0]*Num_min), int(weight[1]*Num_min)
        
        # 随机采样
        id_positive = np.random.randint(0, X_positive.shape[0], N_sample_positive)
        id_negtive = np.random.randint(0, X_negtive.shape[0], N_sample_negtive)

        # 合并采样结果
        X_resampled = np.concatenate((X_positive[id_positive], X_negtive[id_negtive]), axis=0)
        y_resampled = np.concatenate((Y_positive[id_positive], Y_negtive[id_negtive]), axis=0)

        # 打乱顺序
        shuffle_index = np.random.permutation(X_resampled.shape[0])
        X_resampled = X_resampled[shuffle_index]
        y_resampled = y_resampled[shuffle_index]
        return X_resampled, y_resampled

    def downsampling(X_train, y_train, weight):
        """
        下采样方法处理不平衡数据
        :param X_train: 训练数据
        :param y_train: 训练标签
        :param weight: 类别权重
        :return: 重采样后的数据和标签
        """
        # 分离正负样本
        X_positive = X_train[y_train==1]
        X_negative = X_train[y_train==0]
        Y_positive = y_train[y_train==1]
        Y_negative = y_train[y_train==0]

        # 确定采样数量
        Num_max = min(X_positive.shape[0], X_negative.shape[0])
        N_sample_positive, N_sample_negative = int(weight[0]*Num_max), int(weight[1]*Num_max)
        
        # 随机采样(不重复)
        id_positive = np.random.choice(X_positive.shape[0], N_sample_positive, replace=False)
        id_negative = np.random.choice(X_negative.shape[0], N_sample_negative, replace=False)

        # 合并采样结果
        X_resampled = np.concatenate((X_positive[id_positive], X_negative[id_negative]), axis=0)
        y_resampled = np.concatenate((Y_positive[id_positive], Y_negative[id_negative]), axis=0)

        # 打乱顺序
        shuffle_index = np.random.permutation(X_resampled.shape[0])
        X_resampled = X_resampled[shuffle_index]
        y_resampled = y_resampled[shuffle_index]
        return X_resampled, y_resampled
        
# # 读取特征并训练模型
# total_data = pd.read_csv("/home/workstation1/ssd2t/ANNSCI/Lu_4f145d6s2_J3_1e-09_1to3/descripotors/5d6s2_temp_1_1_desc.csv")
# X = total_data.iloc[:, :-1]
# stay_data = pd.read_csv("/home/workstation1/ssd2t/ANNSCI/Lu_4f145d6s2_J3_1e-09_1to3/descripotors_stay/5d6s2_temp_1_1_stay_desc.csv")
# X_stay = stay_data.iloc[:, :].values
# y = (total_data.iloc[:, -1] ** 2 > 1e-9).astype(int)
# X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42)

# ann_model = ANNClassifier(input_size=X_train.shape[1], hidden_size=150)
# X_resampled, y_resampled = X_train, y_train
# ann_model.fit(X_resampled, y_resampled)

# y_pred = ann_model.predict(X_test)
# y_pred_other = ann_model.predict(X_stay)
# y_proba = ann_model.predict_proba(X_test)[:, 1]