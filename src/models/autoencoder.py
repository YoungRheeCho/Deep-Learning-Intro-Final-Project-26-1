import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class AENet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, input_size),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class AutoencoderModel:
    def __init__(self, input_size, lr=1e-3, epochs=50, batch_size=512):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.input_size = input_size
        self.model = AENet(input_size).to(self.device)
        self.threshold = None

    def fit(self, X_train, y_train, X_val, y_val):
        # Train on BENIGN only
        X_benign = X_train[y_train == 0]
        print(f"Training Autoencoder on {len(X_benign)} BENIGN samples, device={self.device}...")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        loader = DataLoader(
            TensorDataset(torch.tensor(X_benign, dtype=torch.float32)),
            batch_size=self.batch_size, shuffle=True
        )

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss = 0
            for (xb,) in loader:
                xb = xb.to(self.device)
                optimizer.zero_grad()
                recon = self.model(xb)
                loss = criterion(recon, xb)
                if torch.isnan(loss):
                    print(f"[WARN] NaN loss at epoch {epoch}, reducing lr")
                    for g in optimizer.param_groups:
                        g['lr'] *= 0.1
                    continue
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if epoch % 10 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d} | recon_loss: {total_loss/len(loader):.6f}")

        # Compute threshold on BENIGN validation samples
        X_val_benign = X_val[y_val == 0]
        errors = self._reconstruction_errors(X_val_benign)
        self.threshold = float(np.percentile(errors, 95))
        print(f"Autoencoder threshold (95th percentile on val BENIGN): {self.threshold:.6f}")
        print("Autoencoder training complete.")

    def _reconstruction_errors(self, X):
        self.model.eval()
        errors = []
        loader = DataLoader(
            TensorDataset(torch.tensor(X, dtype=torch.float32)),
            batch_size=self.batch_size, shuffle=False
        )
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device)
                recon = self.model(xb)
                err = ((xb - recon) ** 2).mean(dim=1).cpu().numpy()
                errors.append(err)
        return np.concatenate(errors)

    def predict_proba(self, X):
        return self._reconstruction_errors(X)

    def predict(self, X):
        if self.threshold is None:
            raise ValueError("Threshold not set. Run fit() first.")
        errors = self._reconstruction_errors(X)
        return (errors > self.threshold).astype(int)

    def evaluate(self, X, y, scenario=''):
        errors = self._reconstruction_errors(X)
        preds = (errors > self.threshold).astype(int)
        metrics = {
            'Accuracy': accuracy_score(y, preds),
            'Precision': precision_score(y, preds, zero_division=0),
            'Recall': recall_score(y, preds, zero_division=0),
            'F1': f1_score(y, preds, zero_division=0),
            'ROC_AUC': roc_auc_score(y, errors) if len(np.unique(y)) > 1 else 0.0,
        }
        if scenario:
            print(f"[AE {scenario}] {metrics}")
        return metrics, errors

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model = AENet(self.input_size).to(self.device)
        self.model.load_state_dict(torch.load(path, map_location=self.device))
