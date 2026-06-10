import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class MLPNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)


class MLPModel:
    def __init__(self, input_size, lr=1e-3, epochs=30, batch_size=512):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model = MLPNet(input_size).to(self.device)

    def _make_loader(self, X, y, shuffle=True):
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        return DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=shuffle)

    def fit(self, X_train, y_train, X_val, y_val):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        train_loader = self._make_loader(X_train, y_train)

        print(f"Training MLP on {self.device}...")
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                out = self.model(xb)
                loss = criterion(out, yb)
                if torch.isnan(loss):
                    print(f"[WARN] NaN loss at epoch {epoch}, reducing lr")
                    for g in optimizer.param_groups:
                        g['lr'] *= 0.1
                    continue
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if epoch % 5 == 0 or epoch == 1:
                val_acc = self._val_accuracy(X_val, y_val)
                print(f"  Epoch {epoch:3d} | loss: {total_loss/len(train_loader):.4f} | val_acc: {val_acc:.4f}")

        print("MLP training complete.")

    def _val_accuracy(self, X, y):
        self.model.eval()
        with torch.no_grad():
            xb = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self.model(xb)
            preds = logits.argmax(dim=1).cpu().numpy()
        return accuracy_score(y, preds)

    def predict_proba(self, X):
        self.model.eval()
        results = []
        loader = DataLoader(
            TensorDataset(torch.tensor(X, dtype=torch.float32)),
            batch_size=self.batch_size, shuffle=False
        )
        with torch.no_grad():
            for (xb,) in loader:
                logits = self.model(xb.to(self.device))
                proba = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                results.append(proba)
        return np.concatenate(results)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def evaluate(self, X, y, scenario=''):
        proba = self.predict_proba(X)
        preds = (proba >= 0.5).astype(int)
        metrics = {
            'Accuracy': accuracy_score(y, preds),
            'Precision': precision_score(y, preds, zero_division=0),
            'Recall': recall_score(y, preds, zero_division=0),
            'F1': f1_score(y, preds, zero_division=0),
            'ROC_AUC': roc_auc_score(y, proba) if len(np.unique(y)) > 1 else 0.0,
        }
        if scenario:
            print(f"[MLP {scenario}] {metrics}")
        return metrics, proba

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path, input_size):
        self.model = MLPNet(input_size).to(self.device)
        self.model.load_state_dict(torch.load(path, map_location=self.device))
