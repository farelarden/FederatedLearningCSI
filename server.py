# server.py
import flwr as fl
import torch
from model import SimpleRegressionNet
from config import NUM_CLIENTS, NUM_ROUNDS

def main():
    def save_model(server_round, parameters, config):
        if server_round == NUM_ROUNDS:
            # Infer input_dim from parameters (first weight matrix)
            input_dim = parameters[0].shape[1]  # fc1.weight: [64, input_dim]
            model = SimpleRegressionNet(input_dim)
            params_dict = zip(model.state_dict().keys(), parameters)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            model.load_state_dict(state_dict, strict=True)
            torch.save(model.state_dict(), "artifacts/final_model.pth")
            print(f"✅ Final model saved (input_dim={input_dim})!")
        return None

    strategy = fl.server.strategy.FedAvg(
        min_available_clients=NUM_CLIENTS,
        evaluate_fn=save_model,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()