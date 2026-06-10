import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        # x: (batch, seq_len=1, features)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last time step
        return self.fc(out)


class LSTMModel:
    def __init__(self, input_size, lr=1e-3, epochs=30, batch_size=512):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.input_size = input_size
        self.model = LSTMNet(input_size).to(self.device)

    def _make_loader(self, X, y, shuffle=True):
        # Add seq_len=1 dimension
        X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        y_t = torch.tensor(y, dtype=torch.long)
        return DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=shuffle)

    def fit(self, X_train, y_train, X_val, y_val):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        train_loader = self._make_loader(X_train, y_train)

        print(f"Training LSTM on {self.device}...")
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

        print("LSTM training complete.")

    def _val_accuracy(self, X, y):
        self.model.eval()
        with torch.no_grad():
            xb = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(self.device)
            logits = self.model(xb)
            preds = logits.argmax(dim=1).cpu().numpy()
        return accuracy_score(y, preds)

    def predict_proba(self, X):
        self.model.eval()
        results = []
        loader = DataLoader(
            TensorDataset(torch.tensor(X, dtype=torch.float32).unsqueeze(1)),
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
            print(f"[LSTM {scenario}] {metrics}")
        return metrics, proba

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
