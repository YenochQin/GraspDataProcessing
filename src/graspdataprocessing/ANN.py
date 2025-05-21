import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score, precision_recall_curve, auc, confusion_matrix
from sklearn.calibration import calibration_curve
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import joblib
import logging
############################
#####  ANN Classifier  #####
############################
class ANNClassifier:
    def __init__(self, input_size, hidden_size=150, output_size=2, learning_rate=0.001):
        """
        Initialize the ANN classifier.
        :param input_size: Number of input features.
        :param hidden_size: Number of neurons in the hidden layer.
        :param output_size: Number of output classes (default: 2 for binary classification).
        :param learning_rate: Learning rate for the optimizer.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss(weight=torch.Tensor([9,1]).to(self.device))

    def fit(self, X_train, y_train, X_val=None, y_val=None, batch_size=2048, max_epochs=1000, weight = [1,1]):
        """
        Train the ANN model.
        :param X_train: Training data (numpy array).
        :param y_train: Training labels (numpy array).
        :param X_val: Validation data (optional, numpy array).
        :param y_val: Validation labels (optional, numpy array).
        :param batch_size: Size of each training batch.
        :param max_epochs: Maximum number of epochs.
        """

        # Convert data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(self.device)

        if X_val is not None and y_val is not None:
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)
            y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(self.device)

        # Training loop
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

            if (epoch + 1) % 50 == 0:
                logging.getLogger().info(f"Epoch [{epoch + 1}/{max_epochs}], Loss: {epoch_loss:.8f}")

                # Optionally evaluate on validation set
                if X_val is not None and y_val is not None:
                    self.evaluate(X_val, y_val, verbose=True)

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
    
    def resampling(X_train, y_train, weight):
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

    def downsampling(X_train, y_train, weight):
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
    
    def plot_curve(y_test, y_proba, filename):
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

        # 绘制校准曲线
        prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)
        plt.subplot(2, 2, 4)
        plt.plot(prob_pred, prob_true, 's-')
        plt.plot([0, 1], [0, 1], '--', color='gray')
        plt.title('Calibration Curve')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')

        plt.tight_layout()
        
        result_file = f"roc_curves/{filename}.png"
        plt.savefig(result_file)
        plt.close()

        return roc_auc, pr_auc
    
    def model_evaluation(y_test, y_pred, y_proba):

        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        return f1, roc_auc, accuracy, precision, recall
    
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