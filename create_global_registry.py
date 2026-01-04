# create_global_registry.py - FIXED VERSION
import pandas as pd
import numpy as np
import pickle
import os
import json
import ast


def calculate_client_statistics(df):
    """
    Calculate comprehensive statistics for a client's data
    Returns: Dict with feature statistics
    """
    from dataset import (
        simple_preprocess_heart_rate,
        extract_hr_features,
        resample_and_smooth_series,
        detect_hr_outliers,
        classify_sleep_stages,
        extract_advanced_features,
        preprocess_respiration_data,
        extract_respiration_features,
        combined_sleep_stage_classification,
        preprocess_stress_data,
        extract_stress_features,
        triple_combined_sleep_classification
    )
    
    # Apply preprocessing pipeline (same as in client)
    processed = df.copy()
    processed = simple_preprocess_heart_rate(processed)
    processed = extract_hr_features(processed)
    processed = resample_and_smooth_series(processed, target_length=360)
    processed = detect_hr_outliers(processed)
    processed = classify_sleep_stages(processed)
    processed = extract_advanced_features(processed)
    
    df_resp_processed = preprocess_respiration_data(processed)
    df_with_resp_features = extract_respiration_features(df_resp_processed)
    processed = combined_sleep_stage_classification(df_with_resp_features)
    
    df_stress_processed = preprocess_stress_data(processed)
    df_with_stress_features = extract_stress_features(df_stress_processed)
    processed = triple_combined_sleep_classification(df_with_stress_features)
    
    # Select only numeric columns
    processed = processed.select_dtypes(include=['int', 'float'])
    
    # Calculate statistics
    client_stats = {}
    numeric_cols = processed.select_dtypes(include=['float64', 'int64']).columns
    
    for col in numeric_cols:
        try:
            col_data = processed[col].dropna()
            if len(col_data) == 0:
                continue
                
            client_stats[col] = {
                'mean': float(col_data.mean()),
                'std': float(col_data.std()),
                'median': float(col_data.median()),
                'min': float(col_data.min()),
                'max': float(col_data.max()),
                'q01': float(col_data.quantile(0.01)),
                'q25': float(col_data.quantile(0.25)),
                'q75': float(col_data.quantile(0.75)),
                'q99': float(col_data.quantile(0.99)),
                'iqr': float(col_data.quantile(0.75) - col_data.quantile(0.25)),
                'count': len(col_data),
                'missing_pct': float((processed[col].isnull().sum() / len(processed)) * 100)
            }
        except Exception as e:
            print(f"  ⚠️ Error calculating stats for {col}: {e}")
            continue
    
    return client_stats

def create_global_feature_registry(data_files, registry_path="global_feature_registry.pkl"):
    """Create a global feature registry with WEIGHTED statistics"""
    
    print("=" * 60)
    print("Creating IMPROVED Global Feature Registry (Weighted)")
    print("=" * 60)
    
    all_feature_stats = {}  # {feature: [(weight, stats), ...]}
    client_weights = {}
    feature_frequencies = {}
    
    # Step 1: Collect weighted statistics from all data files
    print("📊 Collecting weighted statistics from all clients...")
    
    for client_id, data_path in enumerate(data_files):
        if not os.path.exists(data_path):
            print(f"⚠️ File not found: {data_path}")
            continue
            
        print(f"  Processing Client {client_id}: {data_path}")
        
        try:
            # Load data
            df = pd.read_csv(data_path, sep=";")
            
            # Weight by dataset size
            client_weight = len(df)
            client_weights[client_id] = client_weight
            print(f"    Samples: {client_weight}, Weight: {client_weight}")
            
            # Calculate client statistics
            client_stats = calculate_client_statistics(df)
            
            # Store with weights
            for feature, stats in client_stats.items():
                if feature not in all_feature_stats:
                    all_feature_stats[feature] = []
                    feature_frequencies[feature] = 0
                
                all_feature_stats[feature].append((client_weight, stats))
                feature_frequencies[feature] += 1
                
        except Exception as e:
            print(f"❌ Error processing {data_path}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n📈 Collected statistics for {len(all_feature_stats)} features")
    print(f"   Processed {len([w for w in client_weights.values() if w > 0])} clients")
    
    # Step 2: Calculate WEIGHTED global statistics
    print("\n📊 Calculating weighted global statistics...")
    
    global_stats = {}
    global_transformations = {}
    selected_features = []
    
    for feature, weighted_stats in all_feature_stats.items():
        if len(weighted_stats) == 0:
            continue
            
        # Calculate frequency percentage
        freq_pct = feature_frequencies[feature] / len(data_files)
        
        # Only include features present in >50% of clients
        if freq_pct < 0.5:
            print(f"  ⚠️ Excluding {feature}: only in {freq_pct:.1%} of clients")
            continue
        
        # Calculate weighted statistics
        total_weight = sum(w for w, _ in weighted_stats)
        
        # Weighted mean
        weighted_mean = sum(w * s['mean'] for w, s in weighted_stats) / total_weight
        
        # Weighted std (pooled variance formula)
        weighted_variance = sum(w * (s['std']**2 + (s['mean'] - weighted_mean)**2) 
                              for w, s in weighted_stats) / total_weight
        weighted_std = np.sqrt(weighted_variance)
        
        # Weighted percentiles (more complex - use median of medians)
        weighted_median = sum(w * s['median'] for w, s in weighted_stats) / total_weight
        
        # For percentiles, collect all values and weight them
        all_q25 = []
        all_q75 = []
        all_q01 = []
        all_q99 = []
        
        for weight, stats in weighted_stats:
            # Repeat values according to weight (approximation)
            all_q25.extend([stats['q25']] * int(weight/100))
            all_q75.extend([stats['q75']] * int(weight/100))
            all_q01.extend([stats['q01']] * int(weight/100))
            all_q99.extend([stats['q99']] * int(weight/100))
        
        weighted_q25 = np.median(all_q25) if all_q25 else weighted_median * 0.75
        weighted_q75 = np.median(all_q75) if all_q75 else weighted_median * 1.25
        weighted_q01 = np.median(all_q01) if all_q01 else weighted_mean - 2*weighted_std
        weighted_q99 = np.median(all_q99) if all_q99 else weighted_mean + 2*weighted_std
        
        # Store weighted statistics
        global_stats[f'{feature}_mean'] = weighted_mean
        global_stats[f'{feature}_std'] = weighted_std
        global_stats[f'{feature}_median'] = weighted_median
        global_stats[f'{feature}_min'] = min(s['min'] for _, s in weighted_stats)
        global_stats[f'{feature}_max'] = max(s['max'] for _, s in weighted_stats)
        global_stats[f'{feature}_q01'] = weighted_q01
        global_stats[f'{feature}_q25'] = weighted_q25
        global_stats[f'{feature}_q75'] = weighted_q75
        global_stats[f'{feature}_q99'] = weighted_q99
        global_stats[f'{feature}_iqr'] = weighted_q75 - weighted_q25
        global_stats[f'{feature}_upper_bound'] = weighted_q75 + 1.5 * (weighted_q75 - weighted_q25)
        
        # Add to selected features
        selected_features.append(feature)
        
        print(f"  ✓ {feature}: mean={weighted_mean:.2f}, weight={total_weight:.0f}, freq={freq_pct:.1%}")
    
    print(f"\n✅ Calculated weighted statistics for {len(global_stats)//12} features")
    
    # Step 3: Determine transformations using the LARGEST client's data
    print("\n📈 Analyzing transformations (using largest client's data)...")
    
    # Find largest client
    largest_client_id = max(client_weights.items(), key=lambda x: x[1])[0]
    largest_client_file = data_files[largest_client_id]
    
    if os.path.exists(largest_client_file):
        print(f"  Using largest client: Client {largest_client_id} ({client_weights[largest_client_id]} samples)")
        
        try:
            sample_df = pd.read_csv(largest_client_file, sep=";")
            
            # Apply preprocessing
            from dataset import (
                simple_preprocess_heart_rate,
                extract_hr_features,
                resample_and_smooth_series,
                detect_hr_outliers,
                classify_sleep_stages,
                extract_advanced_features,
                preprocess_respiration_data,
                extract_respiration_features,
                combined_sleep_stage_classification,
                preprocess_stress_data,
                extract_stress_features,
                triple_combined_sleep_classification
            )
            
            # Apply preprocessing
            processed = sample_df.copy()
            processed = simple_preprocess_heart_rate(processed)
            processed = extract_hr_features(processed)
            processed = resample_and_smooth_series(processed, target_length=360)
            processed = detect_hr_outliers(processed)
            processed = classify_sleep_stages(processed)
            processed = extract_advanced_features(processed)
            
            df_resp_processed = preprocess_respiration_data(processed)
            df_with_resp_features = extract_respiration_features(df_resp_processed)
            processed = combined_sleep_stage_classification(df_with_resp_features)
            
            df_stress_processed = preprocess_stress_data(processed)
            df_with_stress_features = extract_stress_features(df_stress_processed)
            processed = triple_combined_sleep_classification(df_with_stress_features)
            
            # Select only numeric columns
            processed = processed.select_dtypes(include=['int', 'float'])
            
            # Define features to transform
            features_to_transform = [
                'hr_maxHeartRate', 'hr_minHeartRate',
                'hr_restingHeartRate', 'hr_lastSevenDaysAvgRestingHeartRate',
                'resp_lowestRespirationValue',
                'resp_highestRespirationValue', 'resp_avgWakingRespirationValue',
                'resp_avgSleepRespirationValue',
                'resp_avgTomorrowSleepRespirationValue',
                'str_maxStressLevel', 'str_avgStressLevel',
                'sleep_sleepTimeSeconds', 'sleep_napTimeSeconds',
                'sleep_unmeasurableSleepSeconds', 'sleep_deepSleepSeconds',
                'sleep_lightSleepSeconds', 'sleep_remSleepSeconds',
                'sleep_awakeSleepSeconds', 'act_totalCalories',
                'act_activeKilocalories', 'act_distance',
                'sleep_averageRespirationValue', 'sleep_lowestRespirationValue',
                'sleep_highestRespirationValue', 'sleep_awakeCount',
                'sleep_avgSleepStress', 'sleep_avgHeartRate'
            ]
            
            # For each feature, determine best transformation
            for feature in features_to_transform:
                if feature in processed.columns:
                    try:
                        from dataset import test_all_transformations
                        
                        results, transforms = test_all_transformations(
                            processed,
                            feature,
                            'label'
                        )
                        
                        # Get top 3 transformations
                        top3_names = results.iloc[:3]['transform'].tolist()
                        
                        # Store COMPLETE metadata with WEIGHTED statistics
                        global_transformations[feature] = {
                            'selected_transforms': top3_names,
                            # Use WEIGHTED statistics from global_stats
                            'mean': global_stats.get(f'{feature}_mean', processed[feature].mean()),
                            'std': global_stats.get(f'{feature}_std', processed[feature].std()),
                            'median': global_stats.get(f'{feature}_median', processed[feature].median()),
                            'q01': global_stats.get(f'{feature}_q01', processed[feature].quantile(0.01)),
                            'q25': global_stats.get(f'{feature}_q25', processed[feature].quantile(0.25)),
                            'q75': global_stats.get(f'{feature}_q75', processed[feature].quantile(0.75)),
                            'q99': global_stats.get(f'{feature}_q99', processed[feature].quantile(0.99)),
                            'iqr': global_stats.get(f'{feature}_iqr', 
                                                   processed[feature].quantile(0.75) - processed[feature].quantile(0.25)),
                            'upper_bound': global_stats.get(f'{feature}_upper_bound',
                                                          processed[feature].quantile(0.75) + 1.5 * 
                                                          (processed[feature].quantile(0.75) - processed[feature].quantile(0.25))),
                            'min': global_stats.get(f'{feature}_min', processed[feature].min()),
                            'max': global_stats.get(f'{feature}_max', processed[feature].max()),
                            'bins_5': np.linspace(processed[feature].min(), processed[feature].max(), 6).tolist(),
                            'bins_10': np.linspace(processed[feature].min(), processed[feature].max(), 11).tolist()
                        }
                        
                        # Create feature names for transformed versions
                        for rank, trans_name in enumerate(top3_names, 1):
                            feature_name = f"{feature}_{rank}"
                            if feature_name not in selected_features:
                                selected_features.append(feature_name)
                        
                        print(f"  ✓ {feature}: {top3_names}")
                        
                    except Exception as e:
                        print(f"  ⚠️ Error transforming {feature}: {e}")
                        continue
        
        except Exception as e:
            print(f"❌ Error analyzing transformations: {e}")
            import traceback
            traceback.print_exc()
    
    # Step 4: Add original features (not transformed)
    original_features = [f for f in selected_features 
                        if not any(ft in f for ft in features_to_transform)]
    selected_features.extend(original_features)
    
    # Add label
    if 'label' not in selected_features:
        selected_features.append('label')
    
    # Remove duplicates
    selected_features = list(set(selected_features))
    
    # Step 5: Save the IMPROVED registry
    registry_data = {
        'global_stats': global_stats,
        'global_transformations': global_transformations,
        'selected_features': selected_features,
        'client_weights': client_weights,
        'feature_frequencies': feature_frequencies,
        'registry_type': 'weighted_v2',
        'creation_date': pd.Timestamp.now().isoformat()
    }
    
    with open(registry_path, 'wb') as f:
        pickle.dump(registry_data, f)
    
    print(f"\n✅ IMPROVED Global registry created!")
    print(f"   Statistics: {len(global_stats)} weighted metrics")
    print(f"   Transformations: {len(global_transformations)} features")
    print(f"   Selected features: {len(selected_features)}")
    print(f"   Client weights used: {len(client_weights)} clients")
    print(f"   Saved to: {registry_path}")
    
    # Save feature list as text
    with open("global_feature_list_weighted.txt", "w") as f:
        f.write("IMPROVED WEIGHTED FEATURE REGISTRY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Created: {pd.Timestamp.now().isoformat()}\n")
        f.write(f"Total features: {len(selected_features)}\n")
        f.write(f"Client weights used: {sum(client_weights.values())} total samples\n")
        f.write("\n" + "=" * 60 + "\n\n")
        
        for i, feature in enumerate(sorted(selected_features)):
            freq = feature_frequencies.get(feature.replace('_1', '').replace('_2', '').replace('_3', ''), 0)
            f.write(f"{i:4d}: {feature:<50} (in {freq}/{len(data_files)} clients)\n")
    
    print(f"   Feature list saved to: global_feature_list_weighted.txt")
    
    return registry_data

def load_and_verify_registry(registry_path="global_feature_registry.pkl"):
    """Load and verify the global registry"""
    if not os.path.exists(registry_path):
        print(f"❌ Registry not found: {registry_path}")
        return None
    
    with open(registry_path, 'rb') as f:
        registry = pickle.load(f)
    
    # Verify all required keys are present
    required_keys = ['mean', 'std', 'median', 'q01', 'q25', 'q75', 'q99', 'iqr', 'upper_bound', 'min', 'max', 'bins_5', 'bins_10']
    
    print(f"\n🔍 Verifying registry...")
    print(f"   Global stats: {len(registry['global_stats'])} entries")
    print(f"   Transformations: {len(registry['global_transformations'])} features")
    print(f"   Selected features: {len(registry['selected_features'])}")
    
    # Check a few features
    for i, (feature, meta) in enumerate(list(registry['global_transformations'].items())[:5]):
        missing_keys = [key for key in required_keys if key not in meta]
        if missing_keys:
            print(f"  ⚠️ {feature}: Missing keys: {missing_keys}")
        else:
            print(f"  ✓ {feature}: All keys present")
    
    return registry

if __name__ == "__main__":
    # List all client data files
    data_files = [
        "data/group0_combined.csv",
        "data/group1_combined.csv",
        "data/group2_combined.csv",
        "data/group3_combined.csv",
        "data/group4_combined.csv",
        "data/group5_combined.csv",
        "data/group6_combined.csv",
        "data/group7_combined.csv",
        "data/group8_combined.csv"
    ]
    
    # Create the registry
    registry = create_global_feature_registry(data_files)
    
    # Verify it was created correctly
    load_and_verify_registry()