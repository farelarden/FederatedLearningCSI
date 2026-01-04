# In simulate_federated_simple.py - UPDATED
import flwr as fl
import sys
import os
from typing import Dict, Optional, Tuple, List
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
import pickle
import pandas as pd
import lightgbm as lgb
import json
import time
import multiprocessing
from simple_model_averager import SimpleModelAverager
import traceback
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from client_federated import LightGBMClient

# Map client IDs to their data files (keep existing)
CLIENT_DATA_MAP = {
    1: "data/group0_combined.csv",
    2: "data/group1_combined.csv", 
    3: "data/group2_combined.csv",
    4: "data/group3_combined.csv",
    5: "data/group4_combined.csv",
    6: "data/group5_combined.csv",
    7: "data/group6_combined.csv",
    8: "data/group7_combined.csv",
    9: "data/group8_combined.csv"
}

class AdaptiveFedAvgStrategy(fl.server.strategy.FedAvg):
    """Working adaptive strategy for Flower 1.25.0"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client_performances = {}
        self.round_learning_rates = {}
        self.performance_history = []
        self.performance_weighted = True
        self.dynamic_lr = True
        self.min_lr = 0.01
        self.max_lr = 0.3
        self.best_client_performances = {}  # NEW: Store best MAE per client
        print("✅ AdaptiveFedAvgStrategy initialized")
    
    # In AdaptiveFedAvgStrategy class
    def aggregate_fit(self, server_round, results, failures):
        """Aggregate training results with client weighting"""
        print(f"\n🔄 ROUND {server_round} - AGGREGATING {len(results)} CLIENTS")
        print("=" * 60)
        
        # Track client performances
        client_weights = {}
        has_performance_data = False  # Track if we got any MAE values
        
        for client_proxy, fit_res in results:
            # Extract client ID properly
            client_id = None
            
            # Method 1: From the results themselves (most reliable)
            if fit_res.metrics and 'client_id' in fit_res.metrics:
                client_id = fit_res.metrics['client_id']
            # Method 2: Try to extract from client proxy
            elif hasattr(client_proxy, 'cid'):
                client_id = client_proxy.cid
            else:
                # Generate a stable ID from the proxy string
                import hashlib
                client_str = str(client_proxy)
                client_id = str(int(hashlib.md5(client_str.encode()).hexdigest()[:8], 16))
            
            print(f"DEBUG: Processing client {client_id}")
            print(f"DEBUG: Metrics keys: {list(fit_res.metrics.keys()) if fit_res.metrics else 'None'}")
            
            num_examples = fit_res.num_examples
            metrics = fit_res.metrics
            
            # DEBUG: Print all available metrics
            print(f"  Client {client_id}: metrics = {metrics}")
            
            # Extract MAE - try different keys
            mae_value = None
            if metrics:
                for key in ['test_mae', 'mae', 'train_mae']:
                    if key in metrics:
                        mae_value = metrics[key]
                        break
            
            if mae_value is not None:
                print(f"  ✅ Client {client_id}: Found MAE={mae_value:.4f}")
                has_performance_data = True
                
                if client_id not in self.client_performances:
                    self.client_performances[client_id] = []
                
                self.client_performances[client_id].append(mae_value)
                
                # Update best performance
                if (client_id not in self.best_client_performances or 
                    mae_value < self.best_client_performances[client_id]):
                    self.best_client_performances[client_id] = mae_value
                    print(f"  🏆 New best MAE for client {client_id}: {mae_value:.4f}")
                
                # Calculate weight based on performance (lower MAE = higher weight)
                weight = 1.0 / max(mae_value, 0.001)  # Avoid division by zero
            else:
                print(f"  ⚠️ Client {client_id}: No MAE found in metrics")
                weight = 1.0  # Default weight
            
            # Normalize by number of examples
            weight *= num_examples
            client_weights[client_id] = weight
        
        # Only save if we have performance data
        if has_performance_data:
            self._save_client_performances_to_file(server_round)
        else:
            print(f"  ⚠️ No performance data collected in round {server_round}")
        
        # Normalize weights
        total_weight = sum(client_weights.values())
        if total_weight > 0:
            for client_id in client_weights:
                client_weights[client_id] /= total_weight
            print(f"\n📊 Client weights: {client_weights}")
        else:
            print(f"\n⚠️ No valid weights calculated")
        
        # Call parent aggregate_fit
        return super().aggregate_fit(server_round, results, failures)

    def _save_client_performances_to_file(self, server_round=None):
        """Save client performances to JSON file"""
        try:
            # Ensure we have data to save
            if not self.best_client_performances and not self.client_performances:
                print(f"⚠️ No performance data to save (round {server_round})")
                return
            
            performance_data = {
                'best_client_performances': self.best_client_performances,
                'all_client_performances': self.client_performances,
                'server_round': server_round if server_round else 'unknown',
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'total_clients': len(self.best_client_performances)
            }
            
            with open('client_performances.json', 'w') as f:
                json.dump(performance_data, f, indent=2)
            
            print(f"📊 Saved client performances to client_performances.json")
            print(f"   Best performances: {self.best_client_performances}")
            
            # Also print to debug file
            with open('client_performances_debug.txt', 'a') as f:
                f.write(f"\n=== Round {server_round} ===\n")
                f.write(json.dumps(performance_data, indent=2) + "\n")
                
        except Exception as e:
            print(f"⚠️ Error saving performances: {e}")
            # Simple error message without traceback
            import sys
            print(f"Error type: {type(e).__name__}")

    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate evaluation results"""
        if not results:
            return None, {}
        
        print(f"\n📈 ROUND {server_round} - EVALUATION RESULTS")
        
        total_loss = 0.0
        total_examples = 0
        
        for client_proxy, eval_res in results:
            total_loss += eval_res.loss * eval_res.num_examples
            total_examples += eval_res.num_examples
        
        avg_loss = total_loss / total_examples if total_examples > 0 else 1.0
        self.performance_history.append(avg_loss)
        
        print(f"  Average loss: {avg_loss:.4f}")
        print(f"  History (last 5): {[f'{x:.4f}' for x in self.performance_history[-5:]]}")
        
        return avg_loss, {"loss": avg_loss}
    
    def _save_performance_history(self):
        """Save history to file"""
        import pickle
        history_data = {
            'performance_history': self.performance_history,
            'client_performances': self.client_performances,
            'round_learning_rates': self.round_learning_rates
        }
        
        with open('adaptive_strategy_history.pkl', 'wb') as f:
            pickle.dump(history_data, f)

def client_fn(cid: str):
    """Create a client for simulation"""
    client_id = int(cid) + 1
    
    # Get the correct data file for this client
    data_file = CLIENT_DATA_MAP.get(client_id)
    
    if data_file is None or not os.path.exists(data_file):
        print(f"⚠️ [Factory] Warning: No data file for Client {client_id}")
        data_file = "data/group0_combined.csv"
    
    print(f"[Factory] Creating Client {client_id} with data: {data_file}")
    
    # Create client WITH OPTUNA enabled
    real_client = LightGBMClient(
        csv_path=data_file,
        client_id=client_id,
        use_optuna=True,           # ← ENABLE OPTUNA
        optuna_trials=10,          # ← 10 trials per client
        use_global_registry=False   # ← Disable for now
    )
    
    return real_client.to_client()
def main():
    print("🚀 Starting Adaptive Federated Learning with LightGBM")
    print("=" * 60)
    print("Configuration:")
    print(f"  Clients: 9 (one per data group)")
    print(f"  Rounds: 10")
    print(f"  Strategy: AdaptiveFedAvg (Performance-weighted)")
    print("=" * 60)
    
    # Check if all data files exist
    missing_files = []
    for client_id, data_file in CLIENT_DATA_MAP.items():
        if not os.path.exists(data_file):
            missing_files.append(data_file)
    
    if missing_files:
        print(f"\n⚠️  Warning: {len(missing_files)} data files not found:")
        for f in missing_files:
            print(f"    - {f}")
        print("  Some clients will use fallback data.")
    
    print("\n📁 Client data mapping:")
    for client_id, data_file in CLIENT_DATA_MAP.items():
        exists = "✓" if os.path.exists(data_file) else "✗"
        print(f"  Client {client_id}: {exists} {data_file}")
    
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("client_artifacts", exist_ok=True)
    os.makedirs("global_ensemble", exist_ok=True)
    
    # Configure ADAPTIVE strategy
    strategy = AdaptiveFedAvgStrategy(
        fraction_fit=1.0,      # 100% of clients train each round
        fraction_evaluate=1.0, # 100% of clients evaluate each round
        min_fit_clients=2,     # Minimum 2 clients to train
        min_evaluate_clients=2,# Minimum 2 clients to evaluate
        min_available_clients=2, # Minimum 2 clients available
        evaluate_fn=None,
    )
    
    # Start simulation
    print("\n" + "=" * 60)
    print("🏃 Starting adaptive simulation...")
    print("=" * 60)
    
    try:
        from flwr.server import ServerConfig
    


        # Get available CPU cores (leave 2 for OS/system)
        # Get available CPU cores
        available_cores = multiprocessing.cpu_count()
        print(f"System has {available_cores} CPU cores")

        # Calculate realistic client resources
        clients_to_run = 9  # You want 9 clients
        cpus_per_client = max(0.25, available_cores / clients_to_run / 2)  # Leave room for system

        print(f"Running {clients_to_run} clients with {cpus_per_client:.2f} CPUs each")

        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=clients_to_run,  # 9 clients
            config=ServerConfig(num_rounds=15),
            strategy=strategy,
            client_resources={"num_cpus": cpus_per_client},  # REALISTIC CPU allocation
            ray_init_args={
                "ignore_reinit_error": True,
                "include_dashboard": False,
                "num_cpus": available_cores,
                "object_store_memory": 500 * 1024 * 1024,  # 500MB (NOT 4GB!)
            }
        )    
        print("\n" + "=" * 60)
        print("✅ Adaptive simulation completed successfully!")
        print("=" * 60)
        
        # Create final ensemble
        create_final_ensemble()
        
    except Exception as e:
        print(f"\n❌ Simulation error: {e}")
        import traceback
        traceback.print_exc()
        
        # Try with simpler settings
        try:
            print("\n🔄 Trying with 3 clients, 5 rounds...")
            history = fl.simulation.start_simulation(
                client_fn=client_fn,
                num_clients=3,
                config=fl.server.ServerConfig(num_rounds=5),
                strategy=strategy,
                client_resources={"num_cpus": 1},
            )
            print("✅ Success with reduced settings!")
            create_final_ensemble()
        except Exception as e2:
            print(f"❌ Still failing: {e2}")
    
    print("\n" + "=" * 60)
    print("📁 Generated files:")
    print("  - models/client_X/             : Individual client models")
    print("  - global_ensemble/             : Ensemble model configurations")
    print("  - global_best_model.txt        : Latest global model")
    print("  - adaptive_strategy_history.pkl: Performance history")
    print("\n🎯 Next steps:")
    print("  1. Run: python select_best_model.py (with ensemble option)")
    print("  2. Run: python predict_submission_federated.py")
    print("=" * 60)
def create_advanced_final_ensemble():
    """Create advanced final ensemble"""
    print("\n🎯 Creating ADVANCED final ensemble...")
    
    # Initialize ensemble builder
    ensemble_builder = AdvancedEnsembleBuilder(ensemble_type='stacking')
    
    # Load models
    ensemble_builder.load_models_from_directory("models")
    
    if len(ensemble_builder.base_models) < 2:
        print("⚠️ Not enough models for advanced ensemble")
        return None
    
    # Create validation data from client data
    print("\n📊 Creating validation set for ensemble...")
    X_val, y_val = create_validation_data(CLIENT_DATA_MAP, validation_ratio=0.1)
    
    if X_val is None or y_val is None:
        print("⚠️ Could not create validation set, using equal weights")
        # Fallback to weighted ensemble using performance history
        if os.path.exists('adaptive_strategy_history.pkl'):
            with open('adaptive_strategy_history.pkl', 'rb') as f:
                history = pickle.load(f)
            
            # Get latest performances
            performances = []
            for client_id in range(1, 10):
                client_perf = history.get('client_performances', {}).get(str(client_id), [1.0])
                performances.append(client_perf[-1] if client_perf else 1.0)
            
            # Create weighted ensemble
            weighted_predict, weights = ensemble_builder.create_weighted_ensemble_by_performance(
                performances[:len(ensemble_builder.base_models)]
            )
            
            # Save ensemble
            ensemble_builder.weights = weights
            ensemble_builder.save_ensemble("global_ensemble_advanced")
            
            print(f"✅ Created weighted ensemble with {len(ensemble_builder.base_models)} models")
            return weighted_predict
        else:
            print("❌ No validation data and no performance history")
            return None
    
    # Create stacked ensemble with validation data
    print("\n🔧 Training stacked ensemble...")
    ensemble = ensemble_builder.create_stacked_ensemble(
        X_val,  # Using validation data for training the meta-learner
        y_val,
        X_val,   # Also using same data for validation (for demonstration)
        y_val
    )
    
    # Save the ensemble
    ensemble_builder.save_ensemble("global_ensemble_advanced")
    
    # Test ensemble on validation data
    y_pred = ensemble.predict(X_val)
    final_mae = mean_absolute_error(y_val, y_pred)
    final_r2 = r2_score(y_val, y_pred)
    
    print(f"\n✅ ADVANCED ensemble created!")
    print(f"   Final validation MAE: {final_mae:.4f}")
    print(f"   Final validation R²: {final_r2:.4f}")
    print(f"   Saved to: global_ensemble_advanced/")
    
    return ensemble

def create_final_ensemble():
    """Create final ensemble from best models only - ROBUST VERSION"""
    print(f"\n🎯 Creating final ensemble...")
    
    # Create global_ensemble directory if it doesn't exist
    os.makedirs("global_ensemble", exist_ok=True)
    print(f"✅ Created/Verified global_ensemble directory")
    
    # Check if models directory exists
    if not os.path.exists("models"):
        print(f"❌ models directory doesn't exist!")
        return None
    
    print(f"📁 Checking models directory...")
    
    # List all client directories
    client_dirs = []
    for item in os.listdir("models"):
        if item.startswith("client_") and os.path.isdir(os.path.join("models", item)):
            client_dirs.append(item)
    
    print(f"   Found {len(client_dirs)} client directories: {client_dirs}")
    """Create final ensemble from best models only - ROBUST VERSION"""
    print(f"\n🎯 Creating final ensemble...")
    
    top_k = 5  # Select top 5 models
    
    # 1. Load client performances with error handling
    perf_file = 'client_performances.json'
    if not os.path.exists(perf_file):
        print(f"⚠️ {perf_file} not found.")
        print("   Creating fallback ensemble instead...")
        return create_fallback_ensemble()
    
    try:
        with open(perf_file, 'r') as f:
            performance_data = json.load(f)
        print(f"✅ Loaded performance data from {perf_file}")
    except Exception as e:
        print(f"❌ Error loading {perf_file}: {e}")
        return create_fallback_ensemble()
    
    # 2. Get MAE data from any available source
    mae_dict = {}
    
    # Try best_client_performances first
    if 'best_client_performances' in performance_data:
        mae_dict = performance_data['best_client_performances']
        print(f"📊 Using best_client_performances: {len(mae_dict)} clients")
    
    # If empty, try all_client_performances
    elif 'all_client_performances' in performance_data:
        print("📊 Using all_client_performances (extracting best MAE)...")
        all_perf = performance_data['all_client_performances']
        for client_id_str, maes in all_perf.items():
            if maes and isinstance(maes, list):
                mae_dict[client_id_str] = min(maes)
        print(f"   Extracted best MAE for {len(mae_dict)} clients")
    
    else:
        print("❌ No performance data found in the file")
        print("   Keys available:", list(performance_data.keys()))
        return create_fallback_ensemble()
    
    if not mae_dict:
        print("⚠️ MAE dictionary is empty")
        return create_fallback_ensemble()
    
    print(f"\n📈 Performance data found for clients: {list(mae_dict.keys())}")
    
    # 3. Collect models with their MAE
    model_info = []
    
    for client_id_str, mae in mae_dict.items():
        try:
            client_id = int(client_id_str)
            client_dir = f"models/client_{client_id}"
            best_model_path = os.path.join(client_dir, "best_model.txt")
            
            if os.path.exists(best_model_path):
                model = lgb.Booster(model_file=best_model_path)
                model_info.append({
                    'client_id': client_id,
                    'model': model,
                    'path': best_model_path,
                    'mae': float(mae)
                })
                print(f"  ✓ Client {client_id}: MAE={mae:.4f}, Model loaded")
            else:
                print(f"  ✗ Client {client_id}: No model file at {best_model_path}")
        except Exception as e:
            print(f"  ✗ Client {client_id_str} error: {e}")
            continue
    
    if len(model_info) < 2:
        print(f"⚠️ Only {len(model_info)} models with MAE data")
        print("   Falling back to simple ensemble...")
        return create_fallback_ensemble()
    
    print(f"\n✅ Found {len(model_info)} models with MAE data")
    
    # 4. Sort by MAE (lower is better)
    model_info.sort(key=lambda x: x['mae'])
    
    # 5. Select top K
    selected_models = model_info[:min(top_k, len(model_info))]
    
    # 6. Calculate weights based on MAE
    print(f"\n📊 Calculating MAE-based weights...")
    
    # Get MAE values
    mae_values = [m['mae'] for m in selected_models]
    print(f"   MAE values: {[f'{m:.4f}' for m in mae_values]}")
    
    # Method: Inverse MAE squared (gives more weight to significantly better models)
    weights = []
    for m in selected_models:
        # Weight = 1 / (mae^2 + epsilon)
        weight = 1.0 / (m['mae'] ** 2 + 0.01)
        weights.append(weight)
    
    # Normalize weights
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]
    
    # Assign weights back to models
    for i, m in enumerate(selected_models):
        m['weight'] = normalized_weights[i]
    
    # 7. Create and save ensemble
    all_models = [m['model'] for m in selected_models]
    all_paths = [m['path'] for m in selected_models]
    all_weights = [m['weight'] for m in selected_models]
    
    print(f"\n📈 Selected models for ensemble:")
    for i, m in enumerate(selected_models):
        print(f"  {i+1}. Client {m['client_id']}: MAE={m['mae']:.4f}, Weight={m['weight']:.4f}")
    
    # 8. Save ensemble
    ensemble_config = SimpleModelAverager.save_model_collection(
        all_models, all_paths, all_weights, "final_ensemble"
    )
    
    print(f"\n✅ Created weighted ensemble with {len(selected_models)} models")
    print(f"   Weight sum: {sum(all_weights):.4f}")
    print(f"   Average MAE: {np.mean(mae_values):.4f}")
    
    # Also save ensemble info for debugging
    ensemble_info = {
        'client_ids': [m['client_id'] for m in selected_models],
        'mae_values': mae_values,
        'weights': all_weights,
        'average_mae': float(np.mean(mae_values))
    }
    
    with open('global_ensemble/ensemble_info.json', 'w') as f:
        json.dump(ensemble_info, f, indent=2)
    
    return ensemble_config


def create_fallback_ensemble():
    """Create fallback ensemble when performance data is not available"""
    print("🔄 Creating fallback ensemble with equal weights...")
    
    # Simple equal-weight ensemble (DIFFERENT from your weighted code!)
    all_models = []
    all_paths = []
    
    for client_id in range(1, 10):
        client_dir = f"models/client_{client_id}"
        best_model_path = os.path.join(client_dir, "best_model.txt")
        
        if os.path.exists(best_model_path):
            try:
                model = lgb.Booster(model_file=best_model_path)
                all_models.append(model)
                all_paths.append(best_model_path)
                print(f"  ✓ Added model from Client {client_id}")
            except:
                continue
    
    if len(all_models) >= 2:
        # Equal weights
        all_weights = [1.0/len(all_models)] * len(all_models)
        
        print(f"\n📊 Fallback ensemble: {len(all_models)} models with equal weights")
        
        ensemble_config = SimpleModelAverager.save_model_collection(
            all_models, all_paths, all_weights, "final_ensemble_fallback"
        )
        
        print(f"✅ Created fallback ensemble with {len(all_models)} models")
        return ensemble_config
    
    print("⚠️ Not enough models for even a fallback ensemble")
    return None

def get_client_performance(self, client_id):
    """Get the best performance for a client from history"""
    # Try to load from performance history
    if os.path.exists('adaptive_strategy_history.pkl'):
        try:
            with open('adaptive_strategy_history.pkl', 'rb') as f:
                history = pickle.load(f)
            
            # Get client's performance history
            client_key = str(client_id)
            if client_key in history.get('client_performances', {}):
                performances = history['client_performances'][client_key]
                if performances:
                    return min(performances)  # Best (lowest) MAE
        except:
            pass
    
    # Fallback: Use default MAE
    return 10.0  # Default if no history found

if __name__ == "__main__":
    main()