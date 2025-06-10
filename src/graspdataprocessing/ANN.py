from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score
from sklearn.metrics import precision_recall_curve, confusion_matrix
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import joblib
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Union, List


class TensorNet(nn.Module):
    def __init__(self, input_shape, hidden_dim=128, num_classes=2):
        super().__init__()
        seq_length, in_channels = input_shape
        
        # 参数初始化
        self.w1 = nn.Parameter(torch.empty(seq_length, in_channels, hidden_dim))
        self.b1 = nn.Parameter(torch.empty(hidden_dim, in_channels))
        self.w2 = nn.Parameter(torch.empty(in_channels))
        self.b2 = nn.Parameter(torch.empty(1))
        self.w3 = nn.Parameter(torch.empty(hidden_dim, num_classes))
        self.b3 = nn.Parameter(torch.empty(num_classes))
        
        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w3)
        nn.init.zeros_(self.b1)
        nn.init.zeros_(self.b2)
        nn.init.zeros_(self.b3)
        nn.init.normal_(self.w2, mean=0.0, std=0.01)
        
        self.input_shape = input_shape
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, *self.input_shape)
        
        a1 = torch.einsum('bjk,jkl->blk', x, self.w1) + self.b1
        a2 = torch.einsum('blk,k->bl', self.relu(a1), self.w2) + self.b2
        a3 = torch.matmul(self.relu(a2), self.w3) + self.b3
        return a3

class ANNClassifier:
    """
    优化的人工神经网络分类器
    
    主要改进：
    1. 添加类型提示和完整文档
    2. 改进错误处理和验证
    3. 支持早停和学习率调度
    4. 优化内存使用和性能
    5. 增强日志记录
    6. 改进代码结构和可读性
    """
    
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int = 150, 
                 output_size: int = 2, 
                 learning_rate: float = 0.001,
                 class_weights: Optional[List[float]] = None,
                 device: Optional[str] = None):
        """
        初始化ANN分类器
        
        Args:
            input_size: 输入特征数量
            hidden_size: 隐藏层神经元数量
            output_size: 输出类别数量（默认2用于二分类）
            learning_rate: 学习率
            class_weights: 类别权重用于处理不平衡数据
            device: 计算设备，如果为None则自动选择
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # 设备配置
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        # 类别权重处理
        if class_weights is None:
            class_weights = [9.0, 1.0]  # 默认权重
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
        
        # 构建模型
        self.model = self._build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        
        # 训练历史记录
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        self.logger = logging.getLogger(__name__)

    def _build_model(self) -> nn.Module:
        """构建神经网络模型"""
        model = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.BatchNorm1d(self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_size // 2, self.output_size)
        ).to(self.device)
        
        # 初始化权重
        self._initialize_weights(model)
        return model

    def _initialize_weights(self, model: nn.Module):
        """初始化模型权重"""
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def fit(self, 
            X_train: np.ndarray, 
            y_train: np.ndarray, 
            X_val: Optional[np.ndarray] = None, 
            y_val: Optional[np.ndarray] = None, 
            batch_size: int = 2048, 
            max_epochs: int = 150,
            early_stopping_patience: int = 20,
            min_delta: float = 1e-4,
            weight: Optional[List[float]] = None) -> Dict[str, List[float]]:
        """
        训练ANN模型
        
        Args:
            X_train: 训练数据
            y_train: 训练标签
            X_val: 验证数据（可选）
            y_val: 验证标签（可选）
            batch_size: 批次大小
            max_epochs: 最大训练轮数
            early_stopping_patience: 早停耐心值
            min_delta: 最小改进阈值
            weight: 重采样权重（如果提供）
            
        Returns:
            训练历史记录字典
        """
        # 数据验证
        self._validate_input_data(X_train, y_train)
        
        # 重采样处理
        if weight is not None:
            X_train, y_train = self.resampling(X_train, y_train, weight)
            
        # 数据转换
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(self.device)

        # 验证数据处理
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)
            y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(self.device)
        else:
            X_val_tensor = y_val_tensor = None

        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        # 早停机制
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        # 训练循环
        self.model.train()
        val_loss = val_accuracy = 0.0  # 初始化变量
        
        for epoch in range(max_epochs):
            epoch_loss = self._train_epoch(X_train_tensor, y_train_tensor, batch_size)
            self.training_history['train_loss'].append(epoch_loss)

            # 验证评估
            if X_val_tensor is not None:
                val_loss, val_accuracy = self._validate_epoch(X_val_tensor, y_val_tensor)
                self.training_history['val_loss'].append(val_loss)
                self.training_history['val_accuracy'].append(val_accuracy)
                
                # 学习率调度
                scheduler.step(val_loss)
                
                # 早停检查
                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                    
                if patience_counter >= early_stopping_patience:
                    self.logger.info(f"早停触发，在第 {epoch + 1} 轮停止训练")
                    if best_model_state is not None:
                        self.model.load_state_dict(best_model_state)
                    break

            # 日志记录
            if (epoch + 1) % 50 == 0:
                log_msg = f"Epoch [{epoch + 1}/{max_epochs}], Loss: {epoch_loss:.6f}"
                if X_val_tensor is not None:
                    log_msg += f", Val Loss: {val_loss:.6f}, Val Acc: {val_accuracy:.4f}"
                self.logger.info(log_msg)

        return self.training_history

    def _train_epoch(self, X_train: torch.Tensor, y_train: torch.Tensor, batch_size: int) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        permutation = torch.randperm(X_train.size(0))
        
        for i in range(0, X_train.size(0), batch_size):
            indices = permutation[i:i + batch_size]
            batch_X, batch_y = X_train[indices], y_train[indices]

            self.optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def _validate_epoch(self, X_val: torch.Tensor, y_val: torch.Tensor) -> Tuple[float, float]:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            outputs = self.model(X_val)
            loss = self.criterion(outputs, y_val)
            
            _, predicted = torch.max(outputs.data, 1)
            total = y_val.size(0)
            correct = (predicted == y_val).sum().item()
            
            total_loss = loss.item()
            accuracy = correct / total

        return total_loss, accuracy

    def _validate_input_data(self, X: np.ndarray, y: np.ndarray):
        """验证输入数据的有效性"""
        if X.shape[0] != y.shape[0]:
            raise ValueError("X和y的样本数量不匹配")
        
        if X.shape[1] != self.input_size:
            raise ValueError(f"输入特征维度 {X.shape[1]} 与模型期望的 {self.input_size} 不匹配")
        
        if len(np.unique(y)) > self.output_size:
            raise ValueError(f"标签类别数 {len(np.unique(y))} 超过模型输出维度 {self.output_size}")

    def predict(self, X):
        """
        Predict the labels for the given input data.
        :param X: Input data (numpy array).
        :return: Predicted labels (numpy array).
        """
        predictions = self.predict_proba(X)
        predictions = predictions[:, 1]>0.5

        return predictions
    
    def predict_proba(self, X):
        """
        Predict the labels for the given input data.
        :param X: Input data (numpy array).
        :return: Predicted labels (numpy array).
        """
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outputs = torch.softmax(self.model(X_tensor), dim=1)

        return outputs.cpu().numpy()

    def evaluate(self, X, y, verbose=True):
        """
        Evaluate the model on the given data and labels.
        :param X: Input data (numpy array).
        :param y: True labels (numpy array).
        :param verbose: If True, print the evaluation metrics.
        :return: A dictionary containing evaluation metrics.
        """
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            y_pred = torch.argmax(outputs, dim=1).cpu().numpy()
            y_proba = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()

        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "f1_score": f1_score(y, y_pred),
            "precision": precision_score(y, y_pred),
            "recall": recall_score(y, y_pred),
            "roc_auc": roc_auc_score(y, y_proba)
        }

        if verbose:
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"F1 Score: {metrics['f1_score']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"ROC AUC: {metrics['roc_auc']:.4f}")

        return metrics

    def plot_roc_curve(self, X, y):
        """
        Plot the ROC curve for the model.
        :param X: Input data (numpy array).
        :param y: True labels (numpy array).
        """
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            y_proba = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()

        fpr, tpr, _ = roc_curve(y, y_proba)
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc_score(y, y_proba):.4f})")
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.show()

    def save_model(self, path):
        """
        Save the model parameters to a file.
        :param path: File path to save the model.
        """
        torch.save(self.model.state_dict(), path)
        joblib.dump(self, path + '_ann.pkl')

    def load_model(self, path):
        """
        Load the model parameters from a file.
        :param path: File path to load the model.
        """
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
        return joblib.load(path + '_ann.pkl')
    
    def resampling(self, X_train, y_train, weight):
        X_positive = X_train[y_train==1]
        X_negtive = X_train[y_train==0]
        Y_positive = y_train[y_train==1]
        Y_negtive = y_train[y_train==0]

        Num_min = max(X_positive.shape[0], X_negtive.shape[0])

        N_sample_positive, N_sample_negtive = int(weight[0]*Num_min), int(weight[1]*Num_min)
        id_positive = np.random.randint(0, X_positive.shape[0], N_sample_positive)
        id_negtive = np.random.randint(0, X_negtive.shape[0], N_sample_negtive)

        X_resampled = np.concatenate((X_positive[id_positive], X_negtive[id_negtive]), axis=0)
        y_resampled = np.concatenate((Y_positive[id_positive], Y_negtive[id_negtive]), axis=0)

        # 打乱顺序
        shuffle_index = np.random.permutation(X_resampled.shape[0])
        X_resampled = X_resampled[shuffle_index]
        y_resampled = y_resampled[shuffle_index]
        return X_resampled, y_resampled

    def downsampling(self, X_train, y_train, weight):
        X_positive = X_train[y_train==1]
        X_negative = X_train[y_train==0]
        Y_positive = y_train[y_train==1]
        Y_negative = y_train[y_train==0]

        Num_max = min(X_positive.shape[0], X_negative.shape[0])

        N_sample_positive, N_sample_negative = int(weight[0]*Num_max), int(weight[1]*Num_max)
        id_positive = np.random.choice(X_positive.shape[0], N_sample_positive, replace=False)
        id_negative = np.random.choice(X_negative.shape[0], N_sample_negative, replace=False)

        X_resampled = np.concatenate((X_positive[id_positive], X_negative[id_negative]), axis=0)
        y_resampled = np.concatenate((Y_positive[id_positive], Y_negative[id_negative]), axis=0)

        # 打乱顺序
        shuffle_index = np.random.permutation(X_resampled.shape[0])
        X_resampled = X_resampled[shuffle_index]
        y_resampled = y_resampled[shuffle_index]
        return X_resampled, y_resampled
    
    @staticmethod
    def plot_curve(cal_mix_coeff_List, y_proba_all, y_test, y_proba, filename):
        # 绘制 ROC 曲线
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)
        plt.figure(figsize=(10, 8))
        plt.subplot(2, 2, 1)
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")

        # 绘制 PR 曲线
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(recall, precision)
        plt.subplot(2, 2, 2)
        plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.4f}")
        plt.title("PR Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend(loc="lower left")

        # 绘制混淆矩阵
        y_pred = np.where(y_proba > 0.5, 1, 0)
        cm = confusion_matrix(y_test, y_pred)
        plt.subplot(2, 2, 3)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')

        # 绘制可解释性曲线
        plt.subplot(2, 2, 4)
        # 确保两个数组大小一致
        min_len = min(len(y_proba_all[:, 1]), len(cal_mix_coeff_List))
        if min_len > 0:
            plt.scatter(y_proba_all[:min_len, 1], np.log(abs(cal_mix_coeff_List[:min_len])))
            plt.xlabel('Predicted Probability')
            plt.ylabel('Log|Ci Values|')
            plt.title('Ci Values vs Predicted Probability')
        else:
            plt.text(0.5, 0.5, 'No data to plot', ha='center', va='center', transform=plt.gca().transAxes)

        plt.tight_layout()
        
        # filename is a Path object representing the full desired path.
        # Ensure the parent directory exists before saving.
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filename)
        plt.close()

        return roc_auc, pr_auc
    
    @staticmethod
    def model_evaluation(y_test: np.ndarray, 
                        y_pred: np.ndarray, 
                        y_proba: np.ndarray) -> Tuple[float, float, float, float, float]:
        """
        模型评估
        
        Args:
            y_test: 测试集真实标签
            y_pred: 预测标签
            y_proba: 预测概率
            
        Returns:
            f1, roc_auc, accuracy, precision, recall
        """
        try:
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_proba)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
        except Exception as e:
            logging.getLogger(__name__).warning(f"评估指标计算出错: {e}")
            # 返回默认值或可计算的指标
            f1 = roc_auc = accuracy = precision = recall = 0.0
            if len(y_test) > 0:
                accuracy = accuracy_score(y_test, y_pred)
        
        return f1, roc_auc, accuracy, precision, recall

    def get_feature_importance(self, X: np.ndarray, method: str = 'permutation') -> np.ndarray:
        """
        计算特征重要性
        
        Args:
            X: 输入数据
            method: 计算方法 ('permutation' 或 'gradient')
            
        Returns:
            特征重要性数组
        """
        if method == 'permutation':
            return self._permutation_importance(X)
        elif method == 'gradient':
            return self._gradient_importance(X)
        else:
            raise ValueError("方法必须是 'permutation' 或 'gradient'")
    
    def _permutation_importance(self, X: np.ndarray, n_repeats: int = 10) -> np.ndarray:
        """置换重要性"""
        baseline_score = self.evaluate(X, self.predict(X), verbose=False)['accuracy']
        importance_scores = []
        
        for feature_idx in range(X.shape[1]):
            scores = []
            for _ in range(n_repeats):
                X_permuted = X.copy()
                X_permuted[:, feature_idx] = np.random.permutation(X_permuted[:, feature_idx])
                permuted_score = self.evaluate(X_permuted, self.predict(X_permuted), verbose=False)['accuracy']
                scores.append(baseline_score - permuted_score)
            importance_scores.append(np.mean(scores))
        
        return np.array(importance_scores)
    
    def _gradient_importance(self, X: np.ndarray) -> np.ndarray:
        """梯度重要性"""
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32, requires_grad=True).to(self.device)
        
        outputs = self.model(X_tensor)
        # 对于二分类，使用正类的输出
        if self.output_size == 2:
            target_output = outputs[:, 1].sum()
        else:
            target_output = outputs.sum()
            
        target_output.backward()
        
        # 计算梯度的绝对值作为重要性
        gradients = X_tensor.grad.abs().mean(dim=0)
        
        return gradients.cpu().numpy()
