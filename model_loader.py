# model_loader.py - CORRECT VERSION
import lightgbm as lgb
import json
import pickle
import os

def save_model_with_features(model, feature_names, path, client_id=None):
    """
    Save LightGBM model with feature names properly embedded
    """
    client_info = f"[Client {client_id}] " if client_id is not None else ""
    print(f"{client_info}Saving model with {len(feature_names)} features to {path}")
    
    # 1. First save the LightGBM model
    model.save_model(path)
    print(f"{client_info}Model saved to {path}")
    
    # 2. Save feature names to a JSON file
    base_name = os.path.splitext(path)[0]  # Remove .txt extension
    features_json_path = f"{base_name}_features.json"
    
    with open(features_json_path, 'w') as f:
        json.dump(feature_names, f, indent=2)
    
    print(f"{client_info}Feature names saved to {features_json_path}")
    
    # 3. Also save as pickle
    features_pkl_path = f"{base_name}_features.pkl"
    with open(features_pkl_path, 'wb') as f:
        pickle.dump(feature_names, f)
    
    # 4. Update model's internal feature names
    # This is CRITICAL for LightGBM to remember feature names
    if hasattr(model, '_Booster'):
        # For sklearn API
        model._Booster.feature_name = feature_names
    elif hasattr(model, 'feature_name'):
        # For native LightGBM
        model.feature_name = feature_names
    else:
        # Create a custom attribute
        model.feature_name_ = feature_names
    
    # 5. Re-save the model with feature names embedded
    model.save_model(path)
    print(f"{client_info}Model re-saved with embedded feature names")
    
    return True

def load_model_with_features(path):
    """
    Load LightGBM model and restore feature names
    """
    print(f"Loading model from {path}")
    
    # Load the model
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    
    model = lgb.Booster(model_file=path)
    
    # Try to load feature names from JSON first
    base_name = os.path.splitext(path)[0]
    features_json_path = f"{base_name}_features.json"
    features_pkl_path = f"{base_name}_features.pkl"
    
    feature_names = None
    
    # Try JSON first
    if os.path.exists(features_json_path):
        try:
            with open(features_json_path, 'r') as f:
                feature_names = json.load(f)
            print(f"✅ Loaded {len(feature_names)} feature names from {features_json_path}")
        except Exception as e:
            print(f"⚠️ Error loading JSON features: {e}")
    
    # Try pickle if JSON failed
    if feature_names is None and os.path.exists(features_pkl_path):
        try:
            with open(features_pkl_path, 'rb') as f:
                feature_names = pickle.load(f)
            print(f"✅ Loaded {len(feature_names)} feature names from pickle")
        except Exception as e:
            print(f"⚠️ Error loading pickle features: {e}")
    
    # Check model's internal feature names
    if hasattr(model, 'feature_name'):
        internal_names = model.feature_name()
        print(f"Model has {len(internal_names)} internal feature names")
        
        # Are they generic?
        generic_count = sum(1 for f in internal_names if f.startswith('Column_'))
        if generic_count == len(internal_names):
            print("⚠️ Model has only generic Column_X names internally")
        else:
            print("✅ Model has real feature names internally")
            feature_names = internal_names
    
    # Set the feature names attribute
    if feature_names:
        model.feature_name_ = feature_names
        # Also update the booster
        if hasattr(model, 'set_feature_name'):
            model.set_feature_name(feature_names)
    else:
        print("❌ Could not load feature names. Using generic names.")
        model.feature_name_ = [f'Column_{i}' for i in range(model.num_feature())]
    
    return model

def get_model_features(model):
    """Get feature names from a model"""
    if hasattr(model, 'feature_name_') and model.feature_name_:
        return model.feature_name_
    elif hasattr(model, 'feature_name'):
        return model.feature_name()
    else:
        return [f'Column_{i}' for i in range(model.num_feature())]