import sys
import subprocess

def main():
    if len(sys.argv) < 2:
        print("Usage: python run.py [server|client <id>]")
        return

    if sys.argv[1] == "server":
        from server import main as start_server
        start_server()
    elif sys.argv[1] == "client":
        client_id = int(sys.argv[2])
        from client import FlowerClient
        import flwr as fl
        fl.client.start_numpy_client(server_address="localhost:8080", client=FlowerClient(client_id))
    else:
        print("Invalid command")

if __name__ == "__main__":
    main()