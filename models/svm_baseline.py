import numpy as np
import os
import joblib
import time
import settings

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

class PCASVMClassifier:
    def __init__(self, pca_components=128, svm_c=1.0, kernel="rbf"):
        self.pca_components = pca_components
        self.svm_c = svm_c
        self.kernel = kernel

        self.scaler = None
        self.pca = None
        self.svm = None

    def _flatten_loader(self, loader):
        X = []
        y = []
        for images, labels in loader:
            arr = images.numpy().astype(np.float32)
            arr = arr.reshape(arr.shape[0], -1)
            X.append(arr)
            y.append(labels.numpy())
        return np.concatenate(X, axis=0), np.concatenate(y, axis=0)

    def fit(self, train_loader, val_loader=None):
        t_start = time.time()

        X_train, y_train = self._flatten_loader(train_loader)

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)

        self.pca = PCA(n_components=self.pca_components)
        X_train_pca = self.pca.fit_transform(X_train_scaled)

        self.svm = SVC(C=self.svm_c, kernel=self.kernel)
        self.svm.fit(X_train_pca, y_train)

        train_preds = self.svm.predict(X_train_pca)
        train_acc = accuracy_score(y_train, train_preds)
        train_conf = confusion_matrix(y_train, train_preds)

        val_acc = None
        val_conf = None

        if val_loader is not None:
            X_val, y_val = self._flatten_loader(val_loader)
            X_val_scaled = self.scaler.transform(X_val)
            X_val_pca = self.pca.transform(X_val_scaled)

            val_preds = self.svm.predict(X_val_pca)
            val_acc = accuracy_score(y_val, val_preds)
            val_conf = confusion_matrix(y_val, val_preds)

        history = {
            "seed": settings.SEED,
            "train_acc": [float(train_acc)],
            "train_conf_matrix": train_conf.tolist(),
            "val_acc": [float(val_acc)] if val_acc is not None else [],
            "val_conf_matrix": val_conf.tolist() if val_conf is not None else [],
            "total_time": time.time() - t_start,
        }

        return history

    def evaluate(self, loader):
        X, y = self._flatten_loader(loader)
        X_scaled = self.scaler.transform(X) # pyright: ignore
        X_pca = self.pca.transform(X_scaled) # pyright: ignore
        preds = self.svm.predict(X_pca) # pyright: ignore
        return accuracy_score(y, preds)

    def predict(self, images):
        arr = images.numpy().astype(np.float32)
        arr = arr.reshape(arr.shape[0], -1)
        arr_scaled = self.scaler.transform(arr) # pyright: ignore
        arr_pca = self.pca.transform(arr_scaled) # pyright: ignore
        return self.svm.predict(arr_pca) # pyright: ignore

    def save(self, prefix):
        os.makedirs(os.path.dirname(prefix), exist_ok=True)

        scaler_path = prefix + "_scaler.joblib"
        pca_path    = prefix + "_pca.joblib"
        svm_path    = prefix + "_svm.joblib"

        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.pca, pca_path)
        joblib.dump(self.svm, svm_path)

        return {
            "scaler": scaler_path,
            "pca": pca_path,
            "svm": svm_path,
        }

    @staticmethod
    def load(prefix):
        model = PCASVMClassifier()
        model.scaler = joblib.load(prefix + "_scaler.joblib")
        model.pca = joblib.load(prefix + "_pca.joblib")
        model.svm = joblib.load(prefix + "_svm.joblib")
        return model
