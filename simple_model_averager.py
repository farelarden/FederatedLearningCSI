# simple_model_averager.py - Practical LightGBM Model Averaging
import lightgbm as lgb
import numpy as np
from typing import List, Dict
import pickle
import os

class SimpleModelAverager:
    """
    Simple but effective approach to aggregate LightGBM models
    Uses prediction averaging and model stacking
    """
    
    @staticmethod
    def create_ensemble_from_predictions(models: List[lgb.Booster], 
                                         X_reference: np.ndarray,
                                         weights: List[float] = None) -> lgb.Booster:
        """
        Create ensemble by training a meta-model on predictions
        
        Args:
            models: List of trained LightGBM models
            X_reference: Reference data for training meta-model
            weights: Optional weights for each model
            
        Returns:
            Single LightGBM model that approximates the ensemble
        """
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        
        weights = np.array(weights) / np.sum(weights)
        
        print(f"Creating ensemble from {len(models)} models")
        print(f"Model weights: {weights}")
        
        # Step 1: Get predictions from all models
        all_predictions = []
        for model in models:
            preds = model.predict(X_reference)
            all_predictions.append(preds)
        
        # Step 2: Create meta-features (predictions from each model)
        meta_X = np.column_stack(all_predictions)
        
        # Step 3: Create target as weighted average
        target = np.zeros_like(all_predictions[0])
        for pred, weight in zip(all_predictions, weights):
            target += pred * weight
        
        # Step 4: Train a simple meta-model
        meta_model = lgb.LGBMRegressor(
            n_estimators=50,
            learning_rate=0.1,
            num_leaves=31,
            verbosity=-1,
            n_jobs=-1,
            random_state=42
        )
        
        meta_model.fit(meta_X, target)
        
        return meta_model.booster_
    
    @staticmethod
    def average_parameters(models: List[lgb.Booster], weights: List[float] = None):
        """
        Average model parameters and create new model
        Simpler but less accurate than prediction averaging
        """
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        
        # Extract parameters from all models
        all_params = []
        for model in models:
            params = model.params.copy()
            # Remove non-serializable items
            if 'callbacks' in params:
                del params['callbacks']
            all_params.append(params)
        
        # Average numeric parameters
        avg_params = {}
        for key in all_params[0].keys():
            if isinstance(all_params[0][key], (int, float)):
                values = [params.get(key, all_params[0][key]) for params in all_params]
                # Weighted average
                weighted_sum = sum(v * w for v, w in zip(values, weights))
                avg_params[key] = weighted_sum
            else:
                # Keep non-numeric params from first model
                avg_params[key] = all_params[0][key]
        
        # Create new model with averaged parameters
        # This is a simplified approach - real averaging requires tree merging
        return avg_params
    
    @staticmethod
    def save_model_collection(models: List[lgb.Booster], 
                             paths: List[str], 
                             weights: List[float],
                             output_dir: str = "ensemble_models"):
        """
        Save multiple models with their weights for later ensemble prediction
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save models
        model_files = []
        for i, (model, path) in enumerate(zip(models, paths)):
            model_file = os.path.join(output_dir, f"model_{i:03d}.txt")
            model.save_model(model_file)
            model_files.append(model_file)
        
        # Save ensemble configuration
        ensemble_config = {
            'model_files': model_files,
            'weights': weights,
            'num_models': len(models),
            'feature_names': models[0].feature_name() if hasattr(models[0], 'feature_name') else []
        }
        
        config_path = os.path.join(output_dir, "ensemble_config.pkl")
        with open(config_path, 'wb') as f:
            pickle.dump(ensemble_config, f)
        
        print(f"Saved ensemble configuration to {config_path}")
        print(f"Models: {len(model_files)}, Weights: {weights}")
        
        return ensemble_config