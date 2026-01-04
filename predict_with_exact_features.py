# predict_with_exact_features_FIXED.py
import pandas as pd
import numpy as np
import lightgbm as lgb
import json
import os
import sys
from ast import literal_eval
import traceback

# Import preprocessing function from client_federated
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from client_federated import LightGBMClient
except ImportError as e:
    print(f"❌ Failed to import LightGBMClient: {e}")
    print("Make sure client_federated.py is in the same directory")
    sys.exit(1)

def load_test_data(test_csv_path):
    """Load test data with proper separator"""
    print(f"📊 Loading: {test_csv_path}")
    
    try:
        # Try semicolon first
        df = pd.read_csv(test_csv_path, sep=';')
        print(f"✅ Loaded with semicolon separator")
    except:
        try:
            df = pd.read_csv(test_csv_path)
            print(f"✅ Loaded with comma separator")
        except Exception as e:
            print(f"❌ Failed to load: {e}")
            return None
    
    print(f"  Shape: {df.shape}")
    print(f"  First 10 columns: {list(df.columns)[:10]}")
    return df

def extract_meaningful_time_series_features(series_column, prefix):
    """Extract physiologically meaningful features from time series"""
    print(f"  Extracting {prefix} time series features...")
    
    features = {}
    feature_names = [
        f'{prefix}_mean', f'{prefix}_std', f'{prefix}_trend',
        f'{prefix}_rmssd', f'{prefix}_recovery_rate'
    ]
    
    for name in feature_names:
        features[name] = []
    
    for series in series_column:
        try:
            # Convert to list if string
            if isinstance(series, str) and series.startswith('['):
                series = literal_eval(series)
            
            if isinstance(series, list) and len(series) > 30:  # Need enough data
                clean_series = [float(x) for x in series if x is not None and not pd.isna(x)]
                
                if len(clean_series) > 30:
                    arr = np.array(clean_series)
                    
                    # Basic stats
                    features[f'{prefix}_mean'].append(np.mean(arr))
                    features[f'{prefix}_std'].append(np.std(arr))
                    
                    # Trend (first half vs second half)
                    split = len(arr) // 2
                    first_half_mean = np.mean(arr[:split])
                    second_half_mean = np.mean(arr[split:])
                    features[f'{prefix}_trend'].append(second_half_mean - first_half_mean)
                    
                    # RMSSD for HRV-like measurements
                    diffs = np.diff(arr)
                    if len(diffs) > 0:
                        rmssd = np.sqrt(np.mean(diffs ** 2))
                        features[f'{prefix}_rmssd'].append(rmssd)
                    else:
                        features[f'{prefix}_rmssd'].append(0)
                    
                    # Recovery rate (last 20% vs first 20%)
                    recovery_start = int(len(arr) * 0.2)
                    recovery_end = int(len(arr) * 0.8)
                    if recovery_end > recovery_start:
                        early_mean = np.mean(arr[:recovery_start])
                        late_mean = np.mean(arr[recovery_end:])
                        features[f'{prefix}_recovery_rate'].append(early_mean - late_mean)
                    else:
                        features[f'{prefix}_recovery_rate'].append(0)
                else:
                    # Not enough data
                    for name in feature_names:
                        features[name].append(0)
            else:
                # No data
                for name in feature_names:
                    features[name].append(0)
        except:
            # Error case
            for name in feature_names:
                features[name].append(0)
    
    return features

def calculate_stress_recovery(stress_series):
    """Calculate how much stress decreases during the night"""
    recovery_rates = []
    
    for series in stress_series:
        try:
            if isinstance(series, str) and series.startswith('['):
                series = literal_eval(series)
            
            if isinstance(series, list) and len(series) > 20:
                clean_series = [float(x) for x in series if x is not None and not pd.isna(x)]
                
                if len(clean_series) > 20:
                    # First quarter average
                    first_quarter = int(len(clean_series) * 0.25)
                    start_stress = np.mean(clean_series[:first_quarter])
                    
                    # Last quarter average
                    last_quarter_start = int(len(clean_series) * 0.75)
                    end_stress = np.mean(clean_series[last_quarter_start:])
                    
                    # Recovery = stress reduction
                    recovery = start_stress - end_stress
                    recovery_rates.append(recovery)
                else:
                    recovery_rates.append(0)
            else:
                recovery_rates.append(0)
        except:
            recovery_rates.append(0)
    
    return recovery_rates

def preprocess_data_with_time_series(df, client_id=999):
    """EXACT SAME preprocessing as in client_federated.py"""
    print(f"🔧 Preprocessing with time series extraction (Client {client_id})...")
    
    processed = df.copy()
    
    # Clean column names first
    rename_dict = {}
    for col in processed.columns:
        if col != 'label':
            cleaned = ''.join(c if c.isalnum() or c in ['_', '.'] else '_' for c in str(col))
            # Remove multiple underscores
            while '__' in cleaned:
                cleaned = cleaned.replace('__', '_')
            # Remove leading/trailing underscores
            cleaned = cleaned.strip('_')
            if '.' in cleaned:
                cleaned = cleaned.replace('.', '_dot_')
            if cleaned and cleaned[0].isdigit():
                cleaned = f'f_{cleaned}'
            if cleaned != col:
                rename_dict[col] = cleaned
    
    if rename_dict:
        processed = processed.rename(columns=rename_dict)
        print(f"  Cleaned {len(rename_dict)} column names")
    
    # Extract time series features
    if 'hr_time_series' in processed.columns:
        print("  Extracting HR time series features...")
        hr_features = extract_meaningful_time_series_features(
            processed['hr_time_series'],
            prefix='hr'
        )
        for feature_name, values in hr_features.items():
            processed[feature_name] = values
    
    if 'resp_time_series' in processed.columns:
        print("  Extracting respiration time series features...")
        resp_features = extract_meaningful_time_series_features(
            processed['resp_time_series'],
            prefix='resp'
        )
        for feature_name, values in resp_features.items():
            processed[feature_name] = values
    
    if 'stress_time_series' in processed.columns:
        print("  Extracting stress time series features...")
        stress_features = extract_meaningful_time_series_features(
            processed['stress_time_series'],
            prefix='stress'
        )
        for feature_name, values in stress_features.items():
            processed[feature_name] = values
        
        # Also calculate stress recovery
        stress_recovery = calculate_stress_recovery(processed['stress_time_series'])
        processed['stress_recovery_rate'] = stress_recovery
    
    # Remove raw time series columns
    time_series_cols = ['hr_time_series', 'resp_time_series', 'stress_time_series']
    for col in time_series_cols:
        if col in processed.columns:
            processed = processed.drop(columns=[col])
    
    # Create composite features (SAME AS IN client_federated.py)
    print("  Creating composite features...")
    
    # stress_per_sleep
    if 'stress_mean' in processed.columns and 'sleep_sleepTimeSeconds' in processed.columns:
        sleep_hours = processed['sleep_sleepTimeSeconds'] / 3600
        sleep_hours = sleep_hours.replace(0, 1)
        processed['stress_per_sleep'] = processed['stress_mean'] / sleep_hours
    elif 'stress_mean' in processed.columns:
        processed['stress_per_sleep'] = 0.0
    
    # hrv_stress_interaction
    if 'hr_rmssd' in processed.columns and 'stress_mean' in processed.columns:
        processed['hrv_stress_interaction'] = processed['hr_rmssd'] * processed['stress_mean']
    elif 'hr_rmssd' in processed.columns:
        processed['hrv_stress_interaction'] = 0.0
    
    # hrv_stress_squared
    if 'hr_rmssd' in processed.columns and 'stress_mean' in processed.columns:
        processed['hrv_stress_squared'] = processed['hr_rmssd'] * (processed['stress_mean'] ** 2)
    elif 'hr_rmssd' in processed.columns:
        processed['hrv_stress_squared'] = 0.0
    
    # optimal_stress_zone
    if 'stress_mean' in processed.columns:
        processed['optimal_stress_zone'] = ((processed['stress_mean'] > 20) & 
                                           (processed['stress_mean'] < 50)).astype(int)
    else:
        processed['optimal_stress_zone'] = 0
    
    # calories_per_recovery
    if 'act_totalCalories' in processed.columns and 'hr_recovery_rate' in processed.columns:
        processed['calories_per_recovery'] = processed['act_totalCalories'] / (processed['hr_recovery_rate'].abs() + 1)
    elif 'act_totalCalories' in processed.columns:
        processed['calories_per_recovery'] = 0.0
    
    # recovery_activity_interaction
    if 'hr_recovery_rate' in processed.columns and 'act_totalCalories' in processed.columns:
        processed['recovery_activity_interaction'] = processed['hr_recovery_rate'] * processed['act_totalCalories']
    elif 'hr_recovery_rate' in processed.columns:
        processed['recovery_activity_interaction'] = 0.0
    
    # sympathetic_tone
    if 'stress_mean' in processed.columns and 'hr_std' in processed.columns:
        processed['sympathetic_tone'] = processed['stress_mean'] * processed['hr_std']
    elif 'stress_mean' in processed.columns:
        processed['sympathetic_tone'] = 0.0
    
    # optimal_hr_zone
    if 'hr_mean' in processed.columns:
        processed['optimal_hr_zone'] = ((processed['hr_mean'] > 55) & 
                                       (processed['hr_mean'] < 70)).astype(int)
    else:
        processed['optimal_hr_zone'] = 0
    
    # Fill NaN values
    for col in processed.columns:
        if processed[col].isnull().any():
            if processed[col].dtype in ['float64', 'int64']:
                median_val = processed[col].median()
                if pd.isna(median_val):
                    median_val = 0.0
                processed[col] = processed[col].fillna(median_val)
            else:
                processed[col] = processed[col].fillna(0.0)
    
    # Remove columns with all zeros or very low variance
    columns_to_keep = []
    for col in processed.columns:
        if col == 'label' or col == 'id':
            columns_to_keep.append(col)
        elif processed[col].std() > 0.01:
            columns_to_keep.append(col)
        else:
            print(f"  Dropping low-variance column: {col}")
    
    processed = processed[columns_to_keep]
    
    print(f"✅ Preprocessing complete: {len(processed.columns)} features")
    return processed

def get_best_model_features():
    """Load the exact 26 features from the best model"""
    # Try to find best model
    model_paths = [
        'final_ensemble/model_000.txt',
        'models/client_1/best_model.txt',
        'models/client_2/best_model.txt',
        'models/client_3/best_model.txt',
        'global_ensemble/final_ensemble/model_000.txt'
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            print(f"📂 Found model: {model_path}")
            
            # Load model
            try:
                model = lgb.Booster(model_file=model_path)
                model_features = model.feature_name()
                print(f"✅ Model has {len(model_features)} features")
                
                # Also try to load feature info from JSON if available
                feature_info_path = model_path.replace('.txt', '_features.json')
                if os.path.exists(feature_info_path):
                    with open(feature_info_path, 'r') as f:
                        feature_info = json.load(f)
                    print(f"✅ Loaded feature info from JSON")
                    return feature_info.get('feature_names', model_features)
                
                return model_features
            except Exception as e:
                print(f"❌ Error loading model: {e}")
    
    # Fallback: Use the 26 features you listed
    print("⚠️ Using hardcoded feature list as fallback")
    return [
        'act_totalCalories', 'str_avgStressLevel', 'stress_per_sleep',
        'hrv_stress_interaction', 'hrv_stress_squared', 'optimal_stress_zone',
        'stress_rmssd', 'calories_per_recovery', 'stress_std',
        'stress_recovery_rate', 'stress_mean', 'sleep_sleepTimeSeconds',
        'stress_trend', 'act_activeKilocalories', 'recovery_activity_interaction',
        'resp_avgTomorrowSleepRespirationValue', 'sleep_lightSleepSeconds',
        'sympathetic_tone', 'hr_rmssd', 'sleep_awakeCount', 'user_id',
        'optimal_hr_zone', 'day', 'Unnamed_0', 'hr_mean', 'hr_recovery_rate'
    ]

def align_to_model_features(df, model_features):
    """Align dataframe to match exact model features"""
    print(f"\n🔍 Aligning to model's {len(model_features)} features...")
    
    aligned_df = pd.DataFrame(index=df.index)
    
    # First, try exact matches
    for feature in model_features:
        if feature in df.columns:
            aligned_df[feature] = df[feature].copy()
            print(f"  ✓ {feature}")
        else:
            # Try to find similar column (case-insensitive, with underscores)
            found = False
            df_lower = {col.lower().replace('_', ''): col for col in df.columns}
            feature_lower = feature.lower().replace('_', '')
            
            if feature_lower in df_lower:
                actual_col = df_lower[feature_lower]
                aligned_df[feature] = df[actual_col].copy()
                print(f"  🔄 {feature} (mapped from {actual_col})")
                found = True
            
            if not found:
                # Provide reasonable defaults
                if 'stress' in feature.lower():
                    if 'zone' in feature.lower() or 'flag' in feature.lower():
                        aligned_df[feature] = 0  # Binary flag
                    else:
                        aligned_df[feature] = 30.0  # Mid-range stress
                elif 'hr' in feature.lower():
                    if 'zone' in feature.lower() or 'flag' in feature.lower():
                        aligned_df[feature] = 0
                    elif 'mean' in feature.lower():
                        aligned_df[feature] = 65.0  # Average HR
                    elif 'rmssd' in feature.lower():
                        aligned_df[feature] = 40.0  # Average HRV
                    else:
                        aligned_df[feature] = 0.0
                elif 'calories' in feature.lower():
                    aligned_df[feature] = 2000.0
                elif 'sleep' in feature.lower():
                    if 'Seconds' in feature:
                        aligned_df[feature] = 28800  # 8 hours
                    else:
                        aligned_df[feature] = 0.0
                else:
                    aligned_df[feature] = 0.0
                print(f"  ⚠️ {feature} (default)")
    
    # Ensure correct order
    aligned_df = aligned_df[model_features]
    
    # Final NaN check
    for col in aligned_df.columns:
        if aligned_df[col].isnull().any():
            median_val = aligned_df[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            aligned_df[col] = aligned_df[col].fillna(median_val)
    
    return aligned_df

def main():
    print("="*60)
    print("PREDICTION WITH EXACT CLIENT PREPROCESSING")
    print("="*60)
    
    # Find test data
    test_csv_path = 'data/x_test.csv'
    if not os.path.exists(test_csv_path):
        test_csv_path = 'x_test.csv'
    
    if not os.path.exists(test_csv_path):
        print("❌ No test data found!")
        return
    
    # Load test data
    df = load_test_data(test_csv_path)
    if df is None:
        return
    
    # Get the exact 26 features from best model
    model_features = get_best_model_features()
    print(f"\n🎯 Model expects {len(model_features)} features:")
    for i, feat in enumerate(model_features[:10]):
        print(f"  {i+1}. {feat}")
    if len(model_features) > 10:
        print(f"  ... and {len(model_features)-10} more")
    
    # Preprocess EXACTLY like client_federated.py
    df_preprocessed = preprocess_data_with_time_series(df, client_id=999)
    
    # Show what we created
    print(f"\n📊 Preprocessed features created: {len(df_preprocessed.columns)}")
    print("First 20 features:")
    print(df_preprocessed.columns[:20].tolist())
    
    # Align to model features
    X_aligned = align_to_model_features(df_preprocessed, model_features)
    
    print(f"\n✅ Final aligned features: {X_aligned.shape}")
    
    # Find and load model
    model_path = 'final_ensemble/model_000.txt'
    if not os.path.exists(model_path):
        # Try to find any best model
        import glob
        best_models = glob.glob('models/client_*/best_model.txt') + glob.glob('global_ensemble/*/model_*.txt')
        if best_models:
            model_path = best_models[0]
            print(f"🔍 Found alternative model: {model_path}")
    
    if not os.path.exists(model_path):
        print("❌ No model found!")
        return
    
    print(f"\n🎯 Loading model: {model_path}")
    model = lgb.Booster(model_file=model_path)
    
    # Verify feature alignment
    model_actual_features = model.feature_name()
    print(f"Model actual features: {len(model_actual_features)}")
    
    if len(model_actual_features) != len(model_features):
        print(f"⚠️ Warning: Feature count mismatch!")
        print(f"  Model has: {len(model_actual_features)}")
        print(f"  We prepared: {len(model_features)}")
        
        # Try to align based on model's actual features
        X_aligned = align_to_model_features(df_preprocessed, model_actual_features)
    
    # Check if features match exactly
    missing_features = set(model_actual_features) - set(X_aligned.columns)
    extra_features = set(X_aligned.columns) - set(model_actual_features)
    
    if missing_features:
        print(f"❌ Missing features: {len(missing_features)}")
        for feat in list(missing_features)[:5]:
            print(f"  - {feat}")
    if extra_features:
        print(f"⚠️ Extra features: {len(extra_features)}")
    
    if not missing_features:
        print("✅ All features match!")
    
    # Convert to numpy array
    X_array = X_aligned.values.astype(np.float32)
    
    # Make predictions
    predictions = model.predict(X_array)
    
    print(f"\n✅ Predictions made: {len(predictions)} samples")
    print(f"  Range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"  Mean: {predictions.mean():.4f}")
    print(f"  Std: {predictions.std():.4f}")
    
    # Check if predictions need clipping (but NOT scaling!)
    print(f"\n📊 Checking prediction validity...")
    
    # Clip to reasonable range if needed, but don't scale
    if predictions.min() < 0 or predictions.max() > 100:
        print(f"  Clipping predictions to [0, 100] range...")
        predictions = np.clip(predictions, 0, 100)
        print(f"  After clipping: [{predictions.min():.4f}, {predictions.max():.4f}]")
    else:
        print(f"  Predictions are already in reasonable range [0, 100]")
    
    # Create submission
    print(f"\n📄 Creating submission file...")
    
    # Use IDs if available
    if 'id' in df.columns:
        test_ids = df['id'].values
    elif 'ID' in df.columns:
        test_ids = df['ID'].values
    else:
        test_ids = np.arange(len(predictions))
    
    submission = pd.DataFrame({
        'id': test_ids,
        'label': predictions
    })
    
    # Save submission
    submission_path = 'submission_final_exact.csv'
    submission.to_csv(submission_path, index=False)
    
    print(f"\n✅ Submission saved to: {submission_path}")
    print(f"\n📈 Submission statistics:")
    print(f"  Samples: {len(submission)}")
    print(f"  Label range: [{submission['label'].min():.6f}, {submission['label'].max():.6f}]")
    print(f"  Label mean: {submission['label'].mean():.6f}")
    print(f"  Label std: {submission['label'].std():.6f}")
    
    print(f"\n👀 First 10 predictions:")
    print(submission.head(10).to_string(index=False))
    
    # Save feature info for debugging
    X_aligned.to_csv('features_aligned_debug.csv', index=False)
    print(f"\n📁 Features saved to: features_aligned_debug.csv")
    
    # Also save preprocessing info
    df_preprocessed.to_csv('features_preprocessed_debug.csv', index=False)
    print(f"📁 Preprocessed features saved to: features_preprocessed_debug.csv")

if __name__ == "__main__":
    main()