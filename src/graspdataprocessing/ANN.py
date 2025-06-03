# ANN.py
from sklearn.model_selection import train_test_split
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
    def __init__(self, input_size, hidden_size=128, output_size=2, learning_rate=0.001):
        assert input_size % 3 == 0, "输入特征数必须能被3整除"
        torch.set_num_threads(min(12, torch.get_num_threads()))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TensorNet(
            input_shape=(input_size // 3, 3),
            hidden_dim=hidden_size,
            num_classes=output_size
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = None

    def fit(self, X_train, y_train, X_val=None, y_val=None, batch_size=2048, max_epochs=150):
        # 转换数据格式
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long, device=self.device)
        
        # 自动计算类别权重
        class_counts = torch.bincount(y_train_tensor)
        weights = (1.0 / class_counts.float()).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=weights)
        
        # 训练循环
        self.model.train()
        for epoch in range(max_epochs):
            permutation = torch.randperm(X_train_tensor.size(0))
            epoch_loss = 0.0
            
            for i in range(0, X_train_tensor.size(0), batch_size):
                indices = permutation[i:i + batch_size]
                batch_X, batch_y = X_train_tensor[indices], y_train_tensor[indices]
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
            # 每50个epoch打印日志
            if (epoch + 1) % 50 == 0 or epoch == max_epochs - 1:
                logging.info(f"Epoch [{epoch + 1}/{max_epochs}], Loss: {epoch_loss:.6f}")
                if X_val is not None and y_val is not None:
                    self.evaluate(X_val, y_val, verbose=True)

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
            outputs = self.model(X_tensor)
            return torch.softmax(outputs, dim=1).cpu().numpy()
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return proba[:, 1] > 0.5

    def evaluate(self, X, y, verbose=True):
        proba = self.predict_proba(X)
        y_pred = proba[:, 1] > 0.5
        
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "f1_score": f1_score(y, y_pred, zero_division=0),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y, proba[:, 1]) if len(np.unique(y)) > 1 else 0.5
        }
        
        if verbose:
            for metric, value in metrics.items():
                print(f"{metric.capitalize()}: {value:.4f}")
                
        return metrics

    @staticmethod
    def resample(X, y, ratios=(1, 1)):
        """高效重采样实现"""
        classes, counts = np.unique(y, return_counts=True)
        min_count = min(counts) if min(counts) > 0 else 1
        
        resampled = []
        for cls, (target_ratio, current_count) in zip(classes, zip(ratios, counts)):
            n_samples = int(target_ratio * min_count)
            indices = np.where(y == cls)[0]
            selected = np.random.choice(indices, size=n_samples, replace=current_count < n_samples)
            resampled.append((X[selected], y[selected]))
        
        return np.vstack([r[0] for r in resampled]), np.hstack([r[1] for r in resampled])

    def save_model(self, path):
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict()
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
