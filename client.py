# client.py
import os
import flwr as fl
import torch
import joblib
from typing import Dict, List, Tuple
from model import SimpleRegressionNet
from data_loader import load_client_data
from config import DEVICE, EPOCHS, LEARNING_RATE

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id: int):
        self.client_id = client_id
        
        # Load data and optionally get preprocessors (only for client 0)
        if client_id == 0:
            self.train_loader, self.val_loader, input_dim, self.scaler, self.imputer = \
                load_client_data(client_id, return_preprocessors=True)
            # Save artifacts
            os.makedirs("artifacts", exist_ok=True)
            joblib.dump(self.scaler, "artifacts/scaler.pkl")
            joblib.dump(self.imputer, "artifacts/imputer.pkl")
            print("✅ Scaler and imputer saved from Client 0")
        else:
            self.train_loader, self.val_loader, input_dim = \
                load_client_data(client_id, return_preprocessors=False)

        self.model = SimpleRegressionNet(input_dim).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.criterion = torch.nn.MSELoss()

    def get_parameters(self, config) -> List:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: List, config: Dict) -> Tuple[List, int, Dict]:
        self.set_parameters(parameters)
        self.model.train()
        for _ in range(EPOCHS):
            for X, y in self.train_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(X), y)
                loss.backward()
                self.optimizer.step()
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters: List, config: Dict) -> Tuple[float, int, Dict]:
        self.set_parameters(parameters)
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for X, y in self.val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                total_loss += self.criterion(self.model(X), y).item()
        mse = total_loss / len(self.val_loader)
        return float(mse), len(self.val_loader.dataset), {"mse": mse}