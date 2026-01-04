# client_federated.py - FIXED VERSION with model initialization
import flwr as fl
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import mean_absolute_percentage_error, explained_variance_score
import pickle
import warnings
import sys
import optuna
from sklearn.model_selection import KFold
import pandas as pd
import os
import json  
import traceback
from simple_model_averager import SimpleModelAverager
warnings.filterwarnings('ignore')
import time

class GlobalFeatureRegistry:
    """Enhanced global feature registry with NeuroKit features"""
    
    def __init__(self, registry_path="global_feature_registry.pkl"):
        self.registry_path = registry_path
        self.global_stats = {}
        self.global_transformations = {}
        self.selected_features = []
        self.client_weights = {}
        self.feature_frequencies = {}
        self.neurokit_features = {}  # Store NeuroKit feature templates
        self.load_or_create()         

    def load_or_create(self):
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, 'rb') as f:
                    data = pickle.load(f)
                    self.global_stats = data.get('global_stats', {})
                    self.global_transformations = data.get('global_transformations', {})
                    self.selected_features = data.get('selected_features', [])
                    self.client_weights = data.get('client_weights', {})
                    self.feature_frequencies = data.get('feature_frequencies', {})
                    
                registry_type = data.get('registry_type', 'unknown')
                print(f"✅ Loaded {registry_type} registry with {len(self.selected_features)} features")
                print(f"   Based on {len(self.client_weights)} clients, {sum(self.client_weights.values())} total samples")
                
                self._validate_and_fix_transformations()
                
            except Exception as e:
                print(f"⚠️ Error loading registry: {e}")
                self._create_empty_registry()
        else:
            print(f"⚠️ No global registry found at {self.registry_path}")
            print("   Run: python create_global_registry.py")
            self._create_empty_registry()
    
    def _create_empty_registry(self):
        self.global_stats = {}
        self.global_transformations = {}
        self.selected_features = []
        self.client_weights = {}
        self.feature_frequencies = {}
        print("⚠️ Created empty local registry (no global registry found)")
    
    def _validate_and_fix_transformations(self):
        """Ensure all transformation metadata has ALL required keys"""
        required_keys = ['mean', 'std', 'median', 'q01', 'q25', 'q75', 'q99', 
                        'iqr', 'upper_bound', 'min', 'max', 'bins_5', 'bins_10']
        
        fixed_count = 0
        for feature, meta in self.global_transformations.items():
            for key in required_keys:
                if key not in meta:
                    stat_key = f'{feature}_{key}'
                    if stat_key in self.global_stats:
                        meta[key] = self.global_stats[stat_key]
                    else:
                        if key == 'mean':
                            meta[key] = meta.get('median', 0.0)
                        elif key == 'std':
                            meta[key] = 1.0
                        elif key == 'median':
                            meta[key] = meta.get('mean', 0.0)
                        elif key in ['q01', 'q25']:
                            meta[key] = meta.get('min', 0.0)
                        elif key in ['q75', 'q99', 'max']:
                            meta[key] = meta.get('max', 1.0)
                        elif key == 'iqr':
                            meta[key] = meta.get('q75', 0.75) - meta.get('q25', 0.25)
                        elif key == 'upper_bound':
                            q75 = meta.get('q75', 0.75)
                            iqr = meta.get('iqr', 0.5)
                            meta[key] = q75 + 1.5 * iqr
                        elif key == 'bins_5':
                            meta[key] = [meta.get('min', 0.0), meta.get('max', 1.0)]
                        elif key == 'bins_10':
                            meta[key] = [meta.get('min', 0.0), meta.get('max', 1.0)]
                    
                    fixed_count += 1
        
        if fixed_count > 0:
            print(f"⚠️ Fixed {fixed_count} missing metadata keys in transformations")

class LightGBMClient(fl.client.NumPyClient):
    def __init__(self, csv_path, client_id=0, use_optuna=True, optuna_trials=20,
                use_global_registry=False, registry_path="global_feature_registry.pkl"):
        
        print(f"[Client {client_id}] Loading with GLOBAL REGISTRY feature alignment")
        self.client_id = client_id
        self.use_optuna = use_optuna
        self.optuna_trials = optuna_trials
        self.best_test_mae = float('inf')
        self.best_model_path = None
        self.use_global_registry = use_global_registry
        self.round_history = []
        self.model = None
        if use_global_registry:
            self.registry = GlobalFeatureRegistry(registry_path)
            self.clean_registry_feature_names()
        else:
            self.registry = None
        try:
            # 1. Load data
            df = pd.read_csv(csv_path, sep=";")
            print(f"[Client {client_id}] Raw data shape: {df.shape}")
            
            # 2. SIMPLE cleaning - just drop columns with too many NaN
            df_clean = df.copy()
            
            # Drop columns with >50% NaN
            nan_threshold = len(df_clean) * 0.5
            cols_to_keep = []
            for col in df_clean.columns:
                if col == 'label':
                    cols_to_keep.append(col)
                    continue
                    
                nan_count = df_clean[col].isnull().sum()
                if nan_count < nan_threshold:
                    cols_to_keep.append(col)
                else:
                    print(f"  Dropping {col} ({nan_count}/{len(df_clean)} NaN)")
            
            df_clean = df_clean[cols_to_keep]
            
            # 3. Fill remaining NaN
            for col in df_clean.columns:
                if col != 'label' and df_clean[col].isnull().any():
                    if df_clean[col].dtype in ['float64', 'int64']:
                        median_val = df_clean[col].median()
                        if pd.isna(median_val):
                            median_val = 0.0
                        df_clean[col] = df_clean[col].fillna(median_val)
                    else:
                        df_clean[col] = df_clean[col].fillna(0.0)
            
            # 4. Split data
            from sklearn.model_selection import train_test_split
            self.train_df, self.test_df = train_test_split(
                df_clean, test_size=0.2, random_state=42 + client_id
            )
            
            print(f"[Client {client_id}] Train: {len(self.train_df)}, Test: {len(self.test_df)}")
            
            # 5. Create features
            self.train_processed = self.preprocess_data_with_time_series(self.train_df)
            self.test_processed = self.preprocess_data_with_time_series(self.test_df)

            if self.use_global_registry and self.registry:
                # Get common features from registry or align with other clients
                common_features = self.select_features_cross_client(top_k=30)
                
                self.train_processed = self.train_processed[common_features]
                self.test_processed = self.test_processed[common_features]

            # Feature selection
            self.train_processed, self.test_processed = self.select_features_by_correlation(
                min_correlation=0.15
            )

            # Get feature names
            self.feature_names = [c for c in self.train_processed.columns if c != 'label']
            
            # VALIDATE and clean feature names
            self.feature_names = self.validate_feature_names_for_lightgbm(self.feature_names)
            
            # Also clean dataframe column names to match
            rename_dict = {}
            for i, col in enumerate(self.train_processed.columns):
                if col != 'label':
                    if i < len(self.feature_names):
                        new_name = self.feature_names[i]
                        if new_name != col:
                            rename_dict[col] = new_name
                    else:
                        # Clean the name
                        cleaned = self.clean_feature_names([col])[0]
                        if cleaned != col:
                            rename_dict[col] = cleaned
            
            if rename_dict:
                print(f"  Renaming dataframe columns to match cleaned feature names...")
                self.train_processed = self.train_processed.rename(columns=rename_dict)
                self.test_processed = self.test_processed.rename(columns=rename_dict)
                # Re-extract feature names
                self.feature_names = [c for c in self.train_processed.columns if c != 'label']
            
            # Prepare arrays - ensure we use the cleaned column names
            try:
                self.X_train = self.train_processed[self.feature_names].values
                self.y_train = self.train_processed['label'].values
                self.X_test = self.test_processed[self.feature_names].values  
                self.y_test = self.test_processed['label'].values
            except KeyError as e:
                print(f"  ❌ ERROR: Missing column: {e}")
                print(f"  Available columns: {list(self.train_processed.columns)}")
                print(f"  Looking for: {self.feature_names}")
                raise

            # Check if we should even bother training
            if self.y_train.std() < 0.5:
                print(f"\n❌ CRITICAL: Target has no variance (std={self.y_train.std():.4f})")
                print("Model training is pointless - fix your target variable!")
                sys.exit(1)

            print(f"\n[Client {self.client_id}] Label column analysis:")
            print(f"  Unique values in 'label': {np.unique(self.y_train)}")
            print(f"  Value counts:")
            unique, counts = np.unique(self.y_train, return_counts=True)
            for val, count in zip(unique[:10], counts[:10]):
                print(f"    {val}: {count} times ({count/len(self.y_train)*100:.1f}%)")
            # ========== ADD THIS NEW CODE ==========
            # Select best features using cross-client or local selection
            if self.use_global_registry and self.registry:
                # Use cross-client feature selection
                selected_features = self.select_best_features_cross_client(top_k=30)
            else:
                # Use local feature selection
                selected_features = self.select_features_locally(top_k=30)
            # Ensure label is included
            selected_features = [f for f in selected_features if f != 'label']
            selected_features = selected_features + ['label'] if 'label' not in selected_features else selected_features

            # Filter dataframes to only selected features
            available_features = [f for f in selected_features if f in self.train_processed.columns]
            self.train_processed = self.train_processed[available_features]
            self.test_processed = self.test_processed[available_features]

            # Update feature names (excluding label)
            self.feature_names = [f for f in available_features if f != 'label']
                        
            print(f"\n[Client {self.client_id}] Using {len(self.feature_names)} simple features")
   
            # 8. Final NaN check
            if np.isnan(self.X_train).any() or np.isnan(self.y_train).any():
                self.X_train = np.where(np.isnan(self.X_train), 0.0, self.X_train)
                self.y_train = np.where(np.isnan(self.y_train), 0.0, self.y_train)
                self.X_test = np.where(np.isnan(self.X_test), 0.0, self.X_test)
                self.y_test = np.where(np.isnan(self.y_test), 0.0, self.y_test)
            
            print(f"\n[Client {self.client_id}] Data shapes:")
            print(f"  X_train: {self.X_train.shape}, y_train: {self.y_train.shape}")
            
            # 9. Quick baseline check
            baseline_pred = np.full_like(self.y_test, self.y_train.mean())
            baseline_mae = mean_absolute_error(self.y_test, baseline_pred)
            print(f"  Baseline MAE (predicting mean): {baseline_mae:.2f}")
            
            # 10. Create initial model
            self.model = self.create_initial_model()
            
            # After creating initial model in __init__
            if self.model is not None:
                y_pred = self.model.predict(self.X_test)
                test_mae = mean_absolute_error(self.y_test, y_pred)
                test_r2 = r2_score(self.y_test, y_pred)
                
                baseline_pred = np.full_like(self.y_test, self.y_train.mean())
                baseline_mae = mean_absolute_error(self.y_test, baseline_pred)
                baseline_r2 = r2_score(self.y_test, baseline_pred)
                
                print(f"\n[Client {client_id}] Initial evaluation:")
                print(f"  Test MAE: {test_mae:.2f} (Baseline: {baseline_mae:.2f})")
                print(f"  Test R²:  {test_r2:.4f} (Baseline R²: {baseline_r2:.4f})")
                print(f"  Improvement: {(baseline_mae - test_mae)/baseline_mae*100:.1f}%")
                
            print(f"[Client {client_id}] Initialization complete")
            
        except Exception as e:
            print(f"[Client {client_id}] Error: {e}")
            import traceback
            traceback.print_exc()
            raise
 
    def extract_meaningful_time_series_features(self, series_column, prefix, sampling_rate=1/60):
        """Extract physiologically meaningful features from time series"""
        features = {}
        

        feature_names = [
            f'{prefix}_mean', f'{prefix}_std', f'{prefix}_trend',
            f'{prefix}_rmssd', f'{prefix}_recovery_rate'
        ]
        
        for name in feature_names:
            features[name] = []
        
        for series in series_column:
            try:
                if isinstance(series, str) and series.startswith('['):
                    import ast
                    series = ast.literal_eval(series)
                
                if isinstance(series, list) and len(series) > 30:  
                    clean_series = [float(x) for x in series if x is not None and not pd.isna(x)]
                    
                    if len(clean_series) > 30:
                        arr = np.array(clean_series)
                        
                        features[f'{prefix}_mean'].append(np.mean(arr))
                        features[f'{prefix}_std'].append(np.std(arr))
                        
                        split = len(arr) // 2
                        first_half_mean = np.mean(arr[:split])
                        second_half_mean = np.mean(arr[split:])
                        features[f'{prefix}_trend'].append(second_half_mean - first_half_mean)
                        
                        diffs = np.diff(arr)
                        if len(diffs) > 0:
                            rmssd = np.sqrt(np.mean(diffs ** 2))
                            features[f'{prefix}_rmssd'].append(rmssd)
                        else:
                            features[f'{prefix}_rmssd'].append(0)
                        
                        recovery_start = int(len(arr) * 0.2)
                        recovery_end = int(len(arr) * 0.8)
                        if recovery_end > recovery_start:
                            early_mean = np.mean(arr[:recovery_start])
                            late_mean = np.mean(arr[recovery_end:])
                            features[f'{prefix}_recovery_rate'].append(early_mean - late_mean)
                        else:
                            features[f'{prefix}_recovery_rate'].append(0)
                    else:
                        for name in feature_names:
                            features[name].append(0)
                else:
                    for name in feature_names:
                        features[name].append(0)
            except:

                for name in feature_names:
                    features[name].append(0)
        
        return features
   
    def clean_feature_names(self, feature_names):
        """Clean feature names to remove ALL special characters"""
        cleaned_names = []
        
        for i, name in enumerate(feature_names):
            if not isinstance(name, str):
                name = str(name)
            
            # First, convert to string and strip
            name = str(name).strip()
            
            # Remove ALL non-alphanumeric characters except underscore and dot
            cleaned = ''.join(c if c.isalnum() or c in ['_', '.'] else '_' for c in name)
            
            # Ensure it doesn't start with a number
            if cleaned and cleaned[0].isdigit():
                cleaned = f'f_{cleaned}'
            
            # Ensure it's not empty
            if not cleaned:
                cleaned = f'feature_{i}'
            
            # Remove multiple underscores
            while '__' in cleaned:
                cleaned = cleaned.replace('__', '_')
            
            # Remove leading/trailing underscores
            cleaned = cleaned.strip('_')
            
            # Make sure it's valid for LightGBM
            # LightGBM has issues with certain patterns
            if '.' in cleaned:
                cleaned = cleaned.replace('.', '_dot_')
            
            # Check for reserved keywords
            reserved_words = {'true', 'false', 'null', 'nan', 'inf', 'none'}
            if cleaned.lower() in reserved_words:
                cleaned = f'_{cleaned}_'
            
            cleaned_names.append(cleaned)
        
        # FIX: ADD THIS RETURN STATEMENT
        return cleaned_names
    def validate_feature_names_for_lightgbm(self, feature_names):
        """Validate feature names are safe for LightGBM"""
        print(f"[Client {self.client_id}] Validating feature names for LightGBM...")
        
        problematic = []
        validated_names = []
        
        for i, name in enumerate(feature_names):
            name = str(name)
            
            # Check for problematic characters
            problematic_chars = ['[', ']', '{', '}', ':', ',', ';', '(', ')', '"', "'", 
                            '\\', '/', '|', '?', '!', '@', '#', '$', '%', '^', '&', 
                            '*', '+', '=', '<', '>', '~', '`', ' ', '\t', '\n', '\r']
            
            has_problem = any(char in name for char in problematic_chars)
            
            if has_problem:
                # Clean the name
                cleaned = self.clean_feature_names([name])[0]
                problematic.append((i, name, cleaned))
                validated_names.append(cleaned)
            else:
                validated_names.append(name)
        
        if problematic:
            print(f"  ⚠️ Found {len(problematic)} problematic feature names:")
            for i, original, cleaned in problematic[:5]:  # Show first 5
                print(f"    {i}: '{original}' -> '{cleaned}'")
            if len(problematic) > 5:
                print(f"    ... and {len(problematic) - 5} more")
        
        # Ensure uniqueness
        unique_names = []
        seen = set()
        for name in validated_names:
            if name in seen:
                # Make it unique by adding index
                idx = 1
                new_name = f"{name}_{idx}"
                while new_name in seen:
                    idx += 1
                    new_name = f"{name}_{idx}"
                unique_names.append(new_name)
                seen.add(new_name)
            else:
                unique_names.append(name)
                seen.add(name)
        
        print(f"  ✅ Validated {len(unique_names)} feature names")
        return unique_names

    def create_features(self, df):
        """Create features from global registry with domain-specific composite features"""
        print(f"[Client {self.client_id}] Creating registry-based features...")
        
        processed = df.copy()
        
        # ========== STEP 1: Apply global transformations from registry ==========
        if self.registry and self.registry.global_transformations:
            print(f"  Applying global transformations from registry...")
            for feature, transforms in self.registry.global_transformations.items():
                if feature in processed.columns:
                    if 'selected_transforms' in transforms and transforms['selected_transforms']:
                        best_transform = transforms['selected_transforms'][0]
                        
                        # Apply the transformation based on global stats
                        feature_data = processed[feature]
                        meta = transforms
                        
                        if best_transform == 'log':
                            # Use global min to avoid log(0)
                            global_min = meta.get('min', 0.0)
                            cleaned_feature_name = self.clean_feature_names([f'{feature}_log'])[0]
                            processed[cleaned_feature_name] = np.log1p(feature_data - global_min + 1e-10)
                        elif best_transform == 'sqrt':
                            cleaned_feature_name = self.clean_feature_names([f'{feature}_sqrt'])[0]
                            processed[cleaned_feature_name] = np.sqrt(np.abs(feature_data))
                        elif best_transform == 'normalize':
                            global_mean = meta.get('mean', 0.0)
                            global_std = meta.get('std', 1.0)
                            cleaned_feature_name = self.clean_feature_names([f'{feature}_norm'])[0]
                            processed[cleaned_feature_name] = (feature_data - global_mean) / global_std
                        elif best_transform == 'robust_scale':
                            global_median = meta.get('median', 0.0)
                            global_iqr = meta.get('iqr', 1.0)
                            cleaned_feature_name = self.clean_feature_names([f'{feature}_robust'])[0]
                            processed[cleaned_feature_name] = (feature_data - global_median) / global_iqr
        
        # ========== STEP 2: Create domain-specific composite features ==========
        print(f"  Creating domain-specific composite features...")
        
        # Helper function to safely create features with fallback
        def safe_feature(func, feature_name, *args, default=0.0):
            try:
                result = func(*args)
                cleaned_name = self.clean_feature_names([feature_name])[0]
                return cleaned_name, result
            except:
                cleaned_name = self.clean_feature_names([feature_name])[0]
                return cleaned_name, default
        
        # Dictionary to store new features
        new_features = {}
        
        # Recovery-Stress Balance (Critical!)
        if all(col in processed.columns for col in ['hr_recovery_rate', 'stress_recovery_rate']):
            feat_name, values = safe_feature(
                lambda: processed['hr_recovery_rate'] / (processed['stress_recovery_rate'] + 1),
                'recovery_stress_balance'
            )
            new_features[feat_name] = values
        
        if all(col in processed.columns for col in ['deep_sleep_ratio', 'high_stress_pct']):
            feat_name, values = safe_feature(
                lambda: processed['deep_sleep_ratio'] * (1 - processed['high_stress_pct']),
                'sleep_stress_efficiency'
            )
            new_features[feat_name] = values
        
        if all(col in processed.columns for col in ['hr_rmssd', 'stress_mean']):
            feat_name, values = safe_feature(
                lambda: processed['hr_rmssd'] / (processed['stress_mean'] + 1),
                'hrv_stress_ratio'
            )
            new_features[feat_name] = values
        
        # Autonomic Nervous System Balance
        if all(col in processed.columns for col in ['hr_rmssd', 'resp_cv']):
            feat_name, values = safe_feature(
                lambda: processed['hr_rmssd'] / (processed['resp_cv'] + 1e-10),
                'ans_balance'
            )
            new_features[feat_name] = values
        
        if all(col in processed.columns for col in ['stress_mean', 'hr_std']):
            feat_name, values = safe_feature(
                lambda: processed['stress_mean'] * processed['hr_std'],
                'sympathetic_tone'
            )
            new_features[feat_name] = values
        
        # Sleep Quality Adjusted Metrics
        if all(col in processed.columns for col in ['sleep_avgSleepStress', 'deep_sleep_ratio']):
            feat_name, values = safe_feature(
                lambda: processed['sleep_avgSleepStress'] * (1 - processed['deep_sleep_ratio']),
                'stress_penetrated_sleep'
            )
            new_features[feat_name] = values
        
        # U-shaped relationships (optimal zones)
        if 'hr_mean' in processed.columns:
            # Use global stats for optimal zones if available
            hr_q25 = self.registry.global_stats.get('hr_mean_q25', 55) if self.registry else 55
            hr_q75 = self.registry.global_stats.get('hr_mean_q75', 70) if self.registry else 70
            feat_name = self.clean_feature_names(['optimal_hr_zone'])[0]
            new_features[feat_name] = np.where(
                (processed['hr_mean'] > hr_q25) & (processed['hr_mean'] < hr_q75), 1, 0
            )
        
        if 'stress_mean' in processed.columns:
            stress_q25 = self.registry.global_stats.get('stress_mean_q25', 20) if self.registry else 20
            stress_q75 = self.registry.global_stats.get('stress_mean_q75', 50) if self.registry else 50
            feat_name = self.clean_feature_names(['optimal_stress_zone'])[0]
            new_features[feat_name] = np.where(
                (processed['stress_mean'] > stress_q25) & (processed['stress_mean'] < stress_q75), 1, 0
            )
        
        # Sleep Architecture Quality
        if all(col in processed.columns for col in ['rem_sleep_ratio', 'deep_sleep_ratio']):
            feat_name, values = safe_feature(
                lambda: processed['rem_sleep_ratio'] / (processed['deep_sleep_ratio'] + 0.001),
                'rem_deep_ratio'
            )
            new_features[feat_name] = values
        
        # Recovery Efficiency
        if all(col in processed.columns for col in ['act_totalCalories', 'hr_recovery_rate']):
            feat_name, values = safe_feature(
                lambda: processed['act_totalCalories'] / (processed['hr_recovery_rate'] + 1),
                'calories_per_recovery'
            )
            new_features[feat_name] = values
        
        if all(col in processed.columns for col in ['stress_mean', 'sleep_sleepTimeSeconds']):
            feat_name, values = safe_feature(
                lambda: processed['stress_mean'] / (processed['sleep_sleepTimeSeconds'] / 3600 + 1),
                'stress_per_sleep'
            )
            new_features[feat_name] = values
        
        # State transition metrics
        if all(col in processed.columns for col in ['hr_awake_pct', 'hr_deep_sleep_pct']):
            feat_name, values = safe_feature(
                lambda: processed['hr_awake_pct'] - processed['hr_deep_sleep_pct'],
                'awake_to_sleep_transition'
            )
            new_features[feat_name] = values
        
        if all(col in processed.columns for col in ['stress_first_third', 'stress_last_third']):
            feat_name, values = safe_feature(
                lambda: processed['stress_first_third'] - processed['stress_last_third'],
                'stress_to_recovery'
            )
            new_features[feat_name] = values
        
        # Variability in transitions
        if all(col in processed.columns for col in ['hr_change_point_density', 'stress_volatility']):
            feat_name, values = safe_feature(
                lambda: processed['hr_change_point_density'] * processed['stress_volatility'],
                'transition_variability'
            )
            new_features[feat_name] = values
        
        # Heart-Brain coherence (simplified)
        if all(col in processed.columns for col in ['hr_resp_correlation', 'rsa_strength']):
            feat_name, values = safe_feature(
                lambda: processed['hr_resp_correlation'] * processed['rsa_strength'],
                'hr_resp_coherence'
            )
            new_features[feat_name] = values
        
        # System-wide coherence
        if all(col in processed.columns for col in ['hr_resp_correlation', 'stress_hr_correlation', 'stress_resp_correlation']):
            feat_name, values = safe_feature(
                lambda: (processed['hr_resp_correlation'] + 
                        processed['stress_hr_correlation'] + 
                        processed['stress_resp_correlation']) / 3,
                'physiological_coherence'
            )
            new_features[feat_name] = values
        
        # Overall Resilience Score using global max/min if available
        if all(col in processed.columns for col in ['hr_rmssd', 'high_stress_pct', 'deep_sleep_ratio', 'hr_recovery_rate']):
            # Get normalization factors from global registry or local data
            hrv_max = self.registry.global_stats.get('hr_rmssd_max', processed['hr_rmssd'].max()) if self.registry else processed['hr_rmssd'].max()
            recovery_max = self.registry.global_stats.get('hr_recovery_rate_max', processed['hr_recovery_rate'].max()) if self.registry else processed['hr_recovery_rate'].max()
            
            feat_name, values = safe_feature(
                lambda: (0.25 * processed['hr_rmssd'] / max(hrv_max, 1) +
                        0.25 * (1 - processed['high_stress_pct']) +
                        0.25 * processed['deep_sleep_ratio'] +
                        0.25 * processed['hr_recovery_rate'] / max(recovery_max, 1)),
                'resilience_score'
            )
            new_features[feat_name] = values
        
        # Clinical threshold flags using global percentiles
        if 'hr_rmssd' in processed.columns:
            hrv_q25 = self.registry.global_stats.get('hr_rmssd_q25', 
                                                    processed['hr_rmssd'].quantile(0.25)) if self.registry else processed['hr_rmssd'].quantile(0.25)
            feat_name = self.clean_feature_names(['low_hrv_flag'])[0]
            new_features[feat_name] = (processed['hr_rmssd'] < hrv_q25).astype(int)
        
        if 'high_stress_pct' in processed.columns:
            stress_q75 = self.registry.global_stats.get('high_stress_pct_q75',
                                                    processed['high_stress_pct'].quantile(0.75)) if self.registry else processed['high_stress_pct'].quantile(0.75)
            feat_name = self.clean_feature_names(['high_stress_flag'])[0]
            new_features[feat_name] = (processed['high_stress_pct'] > stress_q75).astype(int)
        
        if 'deep_sleep_ratio' in processed.columns:
            sleep_q25 = self.registry.global_stats.get('deep_sleep_ratio_q25',
                                                    processed['deep_sleep_ratio'].quantile(0.25)) if self.registry else processed['deep_sleep_ratio'].quantile(0.25)
            feat_name = self.clean_feature_names(['poor_sleep_flag'])[0]
            new_features[feat_name] = (processed['deep_sleep_ratio'] < sleep_q25).astype(int)
        
        # Recovery failure flags
        if all(col in processed.columns for col in ['low_hrv_flag', 'high_stress_flag', 'poor_sleep_flag']):
            low_hrv_flag_name = self.clean_feature_names(['low_hrv_flag'])[0]
            high_stress_flag_name = self.clean_feature_names(['high_stress_flag'])[0]
            poor_sleep_flag_name = self.clean_feature_names(['poor_sleep_flag'])[0]
            
            if low_hrv_flag_name in new_features and high_stress_flag_name in new_features and poor_sleep_flag_name in new_features:
                feat_name = self.clean_feature_names(['recovery_failure'])[0]
                new_features[feat_name] = (
                    (new_features[low_hrv_flag_name] == 1) |
                    (new_features[high_stress_flag_name] == 1) |
                    (new_features[poor_sleep_flag_name] == 1)
                ).astype(int)
        
        # Critical interactions for tree-based models
        if all(col in processed.columns for col in ['hr_rmssd', 'stress_mean']):
            feat_name, values = safe_feature(
                lambda: processed['hr_rmssd'] * processed['stress_mean'],
                'hrv_stress_interaction'
            )
            new_features[feat_name] = values
        
        if all(col in processed.columns for col in ['deep_sleep_ratio', 'stress_mean']):
            feat_name, values = safe_feature(
                lambda: processed['deep_sleep_ratio'] * processed['stress_mean'],
                'sleep_stress_interaction'
            )
            new_features[feat_name] = values
        
        if all(col in processed.columns for col in ['hr_recovery_rate', 'act_totalCalories']):
            feat_name, values = safe_feature(
                lambda: processed['hr_recovery_rate'] * processed['act_totalCalories'],
                'recovery_activity_interaction'
            )
            new_features[feat_name] = values
        
        # Quadratic interactions
        if all(col in processed.columns for col in ['hr_rmssd', 'stress_mean']):
            feat_name, values = safe_feature(
                lambda: processed['hr_rmssd'] * (processed['stress_mean'] ** 2),
                'hrv_stress_squared'
            )
            new_features[feat_name] = values
        
        # Add new features to processed dataframe
        for feat_name, values in new_features.items():
            processed[feat_name] = values
        
        # ========== STEP 3: Clean existing feature names ==========
        print(f"  Cleaning existing feature names...")
        rename_dict = {}
        for col in processed.columns:
            if col != 'label':
                cleaned_name = self.clean_feature_names([col])[0]
                if cleaned_name != col:
                    rename_dict[col] = cleaned_name
        
        if rename_dict:
            processed = processed.rename(columns=rename_dict)
            print(f"  Renamed {len(rename_dict)} features")
        
        # ========== STEP 4: Align features with global registry ==========
        if self.registry and self.registry.selected_features:
            print(f"  Aligning with global registry features...")
            
            # Clean registry feature names too
            registry_features_cleaned = self.clean_feature_names(self.registry.selected_features)
            registry_features = set(registry_features_cleaned)
            current_features = set(processed.columns)
            
            # Find features that need to be added
            features_to_add = registry_features - current_features
            
            for feature in features_to_add:
                if feature != 'label':
                    # Add missing features with appropriate defaults
                    if feature.endswith('_flag'):
                        processed[feature] = 0
                    elif any(term in feature for term in ['_log', '_sqrt', '_norm', '_robust', '_ratio', '_score']):
                        processed[feature] = 0.0
                    else:
                        processed[feature] = 0.0
            
            # Keep only registry features + label
            features_to_keep = list(registry_features)
            if 'label' not in features_to_keep:
                features_to_keep.append('label')
            
            # Ensure all features exist
            available_features = [f for f in features_to_keep if f in processed.columns]
            processed = processed[available_features]
        
        # ========== STEP 5: Handle NaN values ==========
        print(f"  Handling NaN values...")
        for col in processed.columns:
            if col != 'label' and processed[col].isnull().any():
                # Use global median if available, otherwise local median
                if self.registry and f'{col}_median' in self.registry.global_stats:
                    fill_value = self.registry.global_stats[f'{col}_median']
                else:
                    fill_value = processed[col].median()
                
                if pd.isna(fill_value):
                    fill_value = 0.0
                
                processed[col] = processed[col].fillna(fill_value)
        
        print(f"  Created {len(processed.columns)} features (aligned with registry)")
        return processed

    def select_features_cross_client(self, top_k=30):
        """Select features based on cross-client consensus"""
        print(f"[Client {self.client_id}] Selecting top {top_k} features via cross-client consensus...")
        
        if not self.registry or not self.registry.feature_frequencies:
            print("  No registry found, using local feature selection")
            return self.select_features_locally(top_k)
        
        # Step 1: Get feature frequency across clients
        feature_freq = self.registry.feature_frequencies
        total_clients = len(self.registry.client_weights)
        
        # Step 2: Calculate reliability score (frequency + weight)
        feature_scores = {}
        for feature, freq in feature_freq.items():
            # Frequency score (0-1)
            freq_score = freq / total_clients
            
            # Calculate weighted importance if available
            weight_score = 1.0
            if feature in self.registry.global_stats:
                # Features with larger variance across clients are more important
                if f'{feature}_std' in self.registry.global_stats:
                    std_val = self.registry.global_stats[f'{feature}_std']
                    weight_score = min(std_val / 10.0, 1.0)  # Normalize
            
            # Combined reliability score
            reliability = 0.7 * freq_score + 0.3 * weight_score
            feature_scores[feature] = reliability
        
        # Step 3: Sort by reliability
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Step 4: Select features present in this client's data
        available_features = []
        for feature, score in sorted_features:
            if feature in self.train_processed.columns:
                available_features.append(feature)
            if len(available_features) >= top_k:
                break
        
        # Step 5: Add fallback if not enough features
        if len(available_features) < 10:
            print(f"  ⚠️ Only {len(available_features)} registry features available")
            print(f"  Adding local features...")
            
            # Get local feature importance
            from sklearn.feature_selection import mutual_info_regression
            local_scores = []
            
            for i, feature_name in enumerate(self.feature_names):
                if feature_name not in available_features:
                    # Calculate local MI score
                    try:
                        mi = mutual_info_regression(
                            self.X_train[:, i:i+1],
                            self.y_train,
                            random_state=42
                        )[0]
                        local_scores.append((feature_name, mi))
                    except:
                        continue
            
            # Add top local features
            local_scores.sort(key=lambda x: x[1], reverse=True)
            additional_features = [f for f, _ in local_scores[:top_k - len(available_features)]]
            available_features.extend(additional_features)
        
        print(f"  Selected {len(available_features)} features via cross-client consensus")
    
        # Make sure label is included
        if 'label' not in available_features:
            available_features.append('label')
        
        return available_features
    
    def preprocess_data_with_time_series(self, df):
        """Preprocess data including time series feature extraction"""
        processed = df.copy()
        
        # Extract time series features
        if 'hr_time_series' in processed.columns:
            print("  Extracting HR time series features...")
            hr_features = self.extract_meaningful_time_series_features(
                processed['hr_time_series'],
                prefix='hr',
                sampling_rate=1/60
            )
            for feature_name, values in hr_features.items():
                processed[feature_name] = values
        
        if 'resp_time_series' in processed.columns:
            print("  Extracting respiration time series features...")
            resp_features = self.extract_meaningful_time_series_features(
                processed['resp_time_series'],
                prefix='resp',
                sampling_rate=1/60
            )
            for feature_name, values in resp_features.items():
                processed[feature_name] = values
        
        if 'stress_time_series' in processed.columns:
            print("  Extracting stress time series features...")
            stress_features = self.extract_meaningful_time_series_features(
                processed['stress_time_series'],
                prefix='stress',
                sampling_rate=1/60
            )
            for feature_name, values in stress_features.items():
                processed[feature_name] = values
        
        # Remove raw time series columns
        time_series_cols = ['hr_time_series', 'resp_time_series', 'stress_time_series']
        for col in time_series_cols:
            if col in processed.columns:
                processed = processed.drop(columns=[col])
        
        # Now apply the registry-based feature creation
        processed = self.create_features(processed)
        
        return processed
    
    def calculate_stress_recovery(self, stress_series):
        """Calculate how much stress decreases during the night"""
        recovery_rates = []
        
        for series in stress_series:
            try:
                if isinstance(series, str) and series.startswith('['):
                    import ast
                    series = ast.literal_eval(series)
                
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
 
    def select_best_features_cross_client(self, top_k=20):
        """Select features based on cross-client performance"""
        print(f"[Client {self.client_id}] Selecting best {top_k} features...")
        
        # If using global registry, get feature importance across clients
        if self.registry and self.registry.feature_frequencies:
            # Sort features by frequency across clients
            sorted_features = sorted(
                self.registry.feature_frequencies.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Take top features that appear in at least 2 clients
            reliable_features = [f for f, count in sorted_features if count >= 2][:top_k]
            
            print(f"  Selected {len(reliable_features)} reliable features from registry")
            return reliable_features
        
        # Fallback: local feature selection
        return self.select_features_locally(top_k)

    def select_features_locally(self, top_k=20):
        """Select features based on local data"""
        from sklearn.feature_selection import RFE
        from sklearn.linear_model import LinearRegression
        
        # Use Recursive Feature Elimination
        estimator = LinearRegression()
        selector = RFE(estimator, n_features_to_select=min(top_k, self.X_train.shape[1]))
        
        selector.fit(self.X_train, self.y_train)
        selected_indices = selector.support_
        
        selected_features = [self.feature_names[i] for i, selected in enumerate(selected_indices) if selected]
        
        print(f"  Selected {len(selected_features)} features locally")
        return selected_features

    def create_initial_model(self):
        """Create initial model with optional Optuna tuning"""
        print(f"[Client {self.client_id}] Creating initial model...")
        
        if self.use_optuna:
            # Use Optuna optimization
            best_params, best_mae = self.optimize_with_optuna(n_trials=self.optuna_trials)
            
            # Save best params
            self.best_params = best_params
            print(f"[Client {self.client_id}] Best params from Optuna: {best_params}")
        else:
            # Use simple training
            train_data = lgb.Dataset(
                self.X_train,
                label=self.y_train,
                feature_name=self.feature_names,
                params={'verbose': -1}
            )
            
            self.model = lgb.train(
                self.get_default_params(),
                train_data,
                num_boost_round=100
            )
        
        # Save model
        model_dir = f"models/client_{self.client_id}"
        os.makedirs(model_dir, exist_ok=True)
        initial_path = os.path.join(model_dir, "initial_model.txt")
        self.save_model_with_features(self.model, initial_path)
        
        print(f"[Client {self.client_id}] Initial model created")
        return self.model
    
    def save_feature_info_to_csv(self, train_processed, test_processed, client_id):
        """Save feature information to CSV files for debugging"""
        
        feature_info_dir = f"feature_info/client_{client_id}"
        os.makedirs(feature_info_dir, exist_ok=True)
        
        train_features = train_processed.drop(columns=['label', 'Unnamed: 0', 'day', 'act_activeTime'], errors='ignore')
        train_features_info = []
        
        for i, col in enumerate(train_features.columns):
            train_features_info.append({
                'index': i,
                'feature_name': str(col),
                'original_name': str(col),
                'cleaned_name': self.clean_feature_names([str(col)])[0],
                'dtype': str(train_features[col].dtype),
                'missing_count': train_features[col].isnull().sum(),
                'missing_pct': (train_features[col].isnull().sum() / len(train_features)) * 100,
                'unique_values': train_features[col].nunique(),
                'mean': train_features[col].mean() if train_features[col].dtype in ['float64', 'int64'] else None,
                'std': train_features[col].std() if train_features[col].dtype in ['float64', 'int64'] else None,
                'min': train_features[col].min() if train_features[col].dtype in ['float64', 'int64'] else None,
                'max': train_features[col].max() if train_features[col].dtype in ['float64', 'int64'] else None,
                'has_special_chars': any(c in str(col) for c in ['[', ']', '{', '}', '<', '>', ':', '"', "'", ',', ';', '(', ')']),
                'special_chars': ''.join([c for c in str(col) if c in ['[', ']', '{', '}', '<', '>', ':', '"', "'", ',', ';', '(', ')']]) or None
            })
        
        train_features_df = pd.DataFrame(train_features_info)
        train_csv_path = os.path.join(feature_info_dir, "train_features_info.csv")
        train_features_df.to_csv(train_csv_path, index=False)
        
        test_features = test_processed.drop(columns=['label', 'Unnamed: 0', 'day', 'act_activeTime'], errors='ignore')
        test_features_info = []
        
        for i, col in enumerate(test_features.columns):
            test_features_info.append({
                'index': i,
                'feature_name': str(col),
                'original_name': str(col),
                'cleaned_name': self.clean_feature_names([str(col)])[0],
                'dtype': str(test_features[col].dtype),
                'missing_count': test_features[col].isnull().sum(),
                'missing_pct': (test_features[col].isnull().sum() / len(test_features)) * 100,
                'unique_values': test_features[col].nunique(),
                'has_special_chars': any(c in str(col) for c in ['[', ']', '{', '}', '<', '>', ':', '"', "'", ',', ';', '(', ')']),
                'special_chars': ''.join([c for c in str(col) if c in ['[', ']', '{', '}', '<', '>', ':', '"', "'", ',', ';', '(', ')']]) or None
            })
        
        test_features_df = pd.DataFrame(test_features_info)
        test_csv_path = os.path.join(feature_info_dir, "test_features_info.csv")
        test_features_df.to_csv(test_csv_path, index=False)
        
        train_set = set(train_features.columns)
        test_set = set(test_features.columns)
        
        comparison_info = {
            'train_features_count': len(train_set),
            'test_features_count': len(test_set),
            'common_features_count': len(train_set & test_set),
            'train_only_features_count': len(train_set - test_set),
            'test_only_features_count': len(test_set - train_set),
            'train_only_features': list(train_set - test_set),
            'test_only_features': list(test_set - train_set),
            'common_features': list(train_set & test_set)
        }
        
        comparison_json_path = os.path.join(feature_info_dir, "feature_comparison.json")
        with open(comparison_json_path, 'w') as f:
            json.dump(comparison_info, f, indent=2)
        
        cleaned_names = self.clean_feature_names(list(train_features.columns))
        cleaned_df = pd.DataFrame({
            'original_name': list(train_features.columns),
            'cleaned_name': cleaned_names,
            'index': range(len(cleaned_names))
        })
        
        cleaned_csv_path = os.path.join(feature_info_dir, "cleaned_feature_names.csv")
        cleaned_df.to_csv(cleaned_csv_path, index=False)
        
        summary_info = {
            'client_id': client_id,
            'train_shape': train_processed.shape,
            'test_shape': test_processed.shape,
            'train_features_count': len(train_features.columns),
            'test_features_count': len(test_features.columns),
            'cleaned_names_count': len(cleaned_names),
            'has_mismatch': len(cleaned_names) != len(train_features.columns),
            'mismatch_amount': abs(len(cleaned_names) - len(train_features.columns)),
            'problematic_features': [str(col) for col in train_features.columns 
                                    if any(c in str(col) for c in ['[', ']', '{', '}', '<', '>', ':', '"', "'", ',', ';', '(', ')'])],
            'files_generated': [
                train_csv_path,
                test_csv_path,
                comparison_json_path,
                cleaned_csv_path
            ]
        }
        
        summary_json_path = os.path.join(feature_info_dir, "summary.json")
        with open(summary_json_path, 'w') as f:
            json.dump(summary_info, f, indent=2)
        
        print(f"\n📊 [Client {client_id}] Feature information saved:")
        print(f"   Train features: {len(train_features.columns)}")
        print(f"   Test features: {len(test_features.columns)}")
        print(f"   Cleaned names: {len(cleaned_names)}")
        print(f"   Mismatch: {'YES' if summary_info['has_mismatch'] else 'NO'}")
        
        if summary_info['problematic_features']:
            print(f"   ⚠️ Problematic features with special characters: {len(summary_info['problematic_features'])}")
        
        print(f"   📁 Files saved to: {feature_info_dir}")
        
        return summary_info

    def apply_local_transformations(self, df):
        """Apply transformations using local metadata"""
        if not hasattr(self, 'transformation_metadata') or not self.transformation_metadata:
            return df
            
        from dataset import apply_transformations_to_test
        return apply_transformations_to_test(df, self.transformation_metadata)
    
    def preprocess_data(self, df, is_training=True):
        """Preprocess data using either global or local statistics"""
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
        
        processed = processed.select_dtypes(include=['int', 'float'])
        
        if self.registry and self.registry.global_stats:
            processed = self.impute_with_global_stats(processed)
            processed = self.create_features_with_global_stats(processed)
            
            if self.registry.global_transformations:
                try:
                    from dataset import apply_transformations_to_test
                    processed = apply_transformations_to_test(processed, self.registry.global_transformations)
                except Exception as e:
                    print(f"⚠️ Error applying global transformations: {e}")
                    print("Falling back to local transformations")
                    if is_training:
                        self.transformation_metadata = self.analyze_transformations(processed)
                        processed = self.apply_local_transformations(processed)
        else:
            if is_training:
                self.local_stats = self.extract_local_statistics(processed)
                self.transformation_metadata = self.analyze_transformations(processed)
            
            processed = self.impute_with_local_stats(processed)
            processed = self.create_features_with_local_stats(processed)
            
            if is_training:
                processed = self.apply_local_transformations(processed)
            elif hasattr(self, 'transformation_metadata'):
                processed = self.apply_local_transformations(processed)
        
        return processed
    
    def extract_local_statistics(self, df_train):
        """Extract statistics from this client's training data only"""
        stats = {}
        
        numeric_cols = df_train.select_dtypes(include=['float64', 'int64']).columns
        
        for col in numeric_cols:
            try:
                stats[f'{col}_mean'] = float(df_train[col].mean())
                stats[f'{col}_median'] = float(df_train[col].median())
                stats[f'{col}_std'] = float(df_train[col].std())
                stats[f'{col}_min'] = float(df_train[col].min())
                stats[f'{col}_max'] = float(df_train[col].max())
                stats[f'{col}_q01'] = float(df_train[col].quantile(0.01))
                stats[f'{col}_q25'] = float(df_train[col].quantile(0.25))
                stats[f'{col}_q75'] = float(df_train[col].quantile(0.75))
                stats[f'{col}_q99'] = float(df_train[col].quantile(0.99))
            except:
                continue
        
        print(f"[Client {self.client_id}] Extracted {len(stats)} local statistics")
        return stats
    
    def analyze_transformations(self, df_train):
        """Analyze which transformations work best for each feature"""
        from dataset import test_all_transformations
        
        transformation_metadata = {}
        
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
        
        existing_features = [f for f in features_to_transform if f in df_train.columns]
        print(f"[Client {self.client_id}] Analyzing transformations for {len(existing_features)} features...")
        
        for feature in existing_features:
            try:
                results, transforms = test_all_transformations(
                    df_train,
                    feature,
                    'label'
                )
                
                top3_names = results.iloc[:3]['transform'].tolist()
                
                feature_stats = {
                    'selected_transforms': top3_names,
                    'mean': df_train[feature].mean(),
                    'std': df_train[feature].std(),
                    'median': df_train[feature].median(),
                    'q01': df_train[feature].quantile(0.01),
                    'q99': df_train[feature].quantile(0.99),
                    'q25': df_train[feature].quantile(0.25),
                    'q75': df_train[feature].quantile(0.75),
                }
                transformation_metadata[feature] = feature_stats
                
            except Exception as e:
                print(f"[Client {self.client_id}] Error analyzing {feature}: {e}")
                continue
        
        return transformation_metadata
    
    def impute_with_local_stats(self, df):
        """Impute missing values using this client's local statistics"""
        columns_to_impute = [
            'resp_avgSleepRespirationValue',
            'sleep_sleepTimeSeconds',
            'sleep_napTimeSeconds',
            'sleep_unmeasurableSleepSeconds',
            'sleep_deepSleepSeconds',
            'sleep_lightSleepSeconds',
            'sleep_remSleepSeconds',
            'sleep_awakeSleepSeconds',
            'sleep_averageRespirationValue',
            'sleep_lowestRespirationValue',
            'sleep_highestRespirationValue',
            'sleep_awakeCount',
            'sleep_avgSleepStress',
            'sleep_avgHeartRate'
        ]
        
        for col in columns_to_impute:
            if col in df.columns:
                stat_key = f'{col}_median'
                if hasattr(self, 'local_stats') and stat_key in self.local_stats:
                    df[col] = df[col].fillna(self.local_stats[stat_key])
                else:
                    df[col] = df[col].fillna(df[col].median())
        
        return df
    
    def impute_with_global_stats(self, df):
        """Impute missing values using global statistics"""
        columns_to_impute = [
            'resp_avgSleepRespirationValue',
            'sleep_sleepTimeSeconds',
            'sleep_napTimeSeconds',
            'sleep_unmeasurableSleepSeconds',
            'sleep_deepSleepSeconds',
            'sleep_lightSleepSeconds',
            'sleep_remSleepSeconds',
            'sleep_awakeSleepSeconds',
            'sleep_averageRespirationValue',
            'sleep_lowestRespirationValue',
            'sleep_highestRespirationValue',
            'sleep_awakeCount',
            'sleep_avgSleepStress',
            'sleep_avgHeartRate'
        ]
        
        for col in columns_to_impute:
            if col in df.columns:
                stat_key = f'{col}_median'
                if stat_key in self.registry.global_stats:
                    df[col] = df[col].fillna(self.registry.global_stats[stat_key])
                else:
                    df[col] = df[col].fillna(df[col].median())
        
        return df
    
    def create_features_with_local_stats(self, df):
        """Create engineered features using this client's local statistics"""
        if not hasattr(self, 'local_stats'):
            return df
            
        print(f"[Client {self.client_id}] Creating features with local stats...")
        
        def get_local_stat(stat_name, default=None):
            if stat_name in self.local_stats:
                return self.local_stats[stat_name]
            elif default is not None:
                return default
            else:
                if 'mean' in stat_name:
                    return 0.0
                elif 'std' in stat_name or 'max' in stat_name or 'q' in stat_name:
                    return 1.0
                else:
                    return 1.0
        
        hr_rmssd_max = get_local_stat('hr_rmssd_max', 1.0)
        hr_recovery_rate_max = get_local_stat('hr_recovery_rate_max', 1.0)
        hr_rmssd_min = get_local_stat('hr_rmssd_min', df['hr_rmssd'].min() if 'hr_rmssd' in df.columns else 0.0)
        hr_recovery_rate_min = get_local_stat('hr_recovery_rate_min', df['hr_recovery_rate'].min() if 'hr_recovery_rate' in df.columns else 0.0)
        hr_rmssd_q25 = get_local_stat('hr_rmssd_q25', 0.25)
        high_stress_pct_q75 = get_local_stat('high_stress_pct_q75', 0.75)
        deep_sleep_ratio_q25 = get_local_stat('deep_sleep_ratio_q25', 0.25)
        
        if 'hr_recovery_rate' in df.columns and 'stress_recovery_rate' in df.columns:
            df['recovery_stress_balance'] = df['hr_recovery_rate'] / (df['stress_recovery_rate'] + 1)
        
        if 'deep_sleep_ratio' in df.columns and 'high_stress_pct' in df.columns:
            df['sleep_stress_efficiency'] = df['deep_sleep_ratio'] * (1 - df['high_stress_pct'])
        
        if 'hr_rmssd' in df.columns and 'stress_mean' in df.columns:
            df['hrv_stress_ratio'] = df['hr_rmssd'] / (df['stress_mean'] + 1)
        
        if 'hr_rmssd' in df.columns and 'resp_cv' in df.columns:
            df['ans_balance'] = df['hr_rmssd'] / (df['resp_cv'] + 1e-10)
        
        if 'stress_mean' in df.columns and 'hr_std' in df.columns:
            df['sympathetic_tone'] = df['stress_mean'] * df['hr_std']
        
        # if 'hr_mean' in df.columns:
        #     df['hr_mean_squared'] = df['hr_mean'] ** 2
        
        # if 'stress_mean' in df.columns:
        #     df['stress_mean_squared'] = df['stress_mean'] ** 2
        
        # if 'sleep_sleepTimeSeconds' in df.columns:
        #     df['sleep_squared'] = df['sleep_sleepTimeSeconds'] ** 2
        
        if 'act_totalCalories' in df.columns and 'hr_recovery_rate' in df.columns:
            df['calories_per_recovery'] = df['act_totalCalories'] / (df['hr_recovery_rate'] + 1)
        
        if 'stress_mean' in df.columns and 'sleep_sleepTimeSeconds' in df.columns:
            df['stress_per_sleep'] = df['stress_mean'] / (df['sleep_sleepTimeSeconds'] / 3600 + 1)
        
        if 'hr_awake_pct' in df.columns and 'hr_deep_sleep_pct' in df.columns:
            df['awake_to_sleep_transition'] = df['hr_awake_pct'] - df['hr_deep_sleep_pct']
        
        if 'stress_first_third' in df.columns and 'stress_last_third' in df.columns:
            df['stress_to_recovery'] = df['stress_first_third'] - df['stress_last_third']
        
        if 'hr_change_point_density' in df.columns and 'stress_volatility' in df.columns:
            df['transition_variability'] = df['hr_change_point_density'] * df['stress_volatility']
        
        if 'hr_rmssd' in df.columns and 'stress_mean' in df.columns:
            df['hrv_stress_interaction'] = df['hr_rmssd'] * df['stress_mean']
        
        if 'deep_sleep_ratio' in df.columns and 'stress_mean' in df.columns:
            df['sleep_stress_interaction'] = df['deep_sleep_ratio'] * df['stress_mean']
        
        if 'hr_recovery_rate' in df.columns and 'act_totalCalories' in df.columns:
            df['recovery_activity_interaction'] = df['hr_recovery_rate'] * df['act_totalCalories']
        
        # if 'hr_rmssd' in df.columns and 'stress_mean' in df.columns:
        #     df['hrv_stress_squared'] = df['hr_rmssd'] * (df['stress_mean'] ** 2)
        
        if 'hr_rmssd' in df.columns:
            hr_rmssd_range = hr_rmssd_max - hr_rmssd_min + 1e-10
            df['hr_rmssd_normalized'] = (df['hr_rmssd'] - hr_rmssd_min) / hr_rmssd_range
        
        if 'hr_recovery_rate' in df.columns:
            hr_recovery_rate_range = hr_recovery_rate_max - hr_recovery_rate_min + 1e-10
            df['hr_recovery_rate_normalized'] = (df['hr_recovery_rate'] - hr_recovery_rate_min) / hr_recovery_rate_range
        
        if all(col in df.columns for col in ['hr_rmssd_normalized', 'high_stress_pct', 'deep_sleep_ratio', 'hr_recovery_rate_normalized']):
            df['resilience_score'] = (
                0.25 * df['hr_rmssd_normalized'] +
                0.25 * (1 - df['high_stress_pct']) +
                0.25 * df['deep_sleep_ratio'] +
                0.25 * df['hr_recovery_rate_normalized']
            )
        
        if 'hr_rmssd' in df.columns:
            df['low_hrv_flag'] = (df['hr_rmssd'] < hr_rmssd_q25).astype(int)
        
        if 'high_stress_pct' in df.columns:
            df['high_stress_flag'] = (df['high_stress_pct'] > high_stress_pct_q75).astype(int)
        
        if 'deep_sleep_ratio' in df.columns:
            df['poor_sleep_flag'] = (df['deep_sleep_ratio'] < deep_sleep_ratio_q25).astype(int)
        
        print(f"[Client {self.client_id}] Created {len([c for c in df.columns if '_flag' in c])} flag features")
        
        return df
    def clean_all_nan_values(self, df):
        """Clean ALL NaN and None values from dataframe"""
        print(f"[Client {self.client_id}] Cleaning all NaN values...")
        
        df_clean = df.copy()
        
        # 1. First convert string lists to actual lists
        time_series_cols = ['hr_time_series', 'resp_time_series', 'stress_time_series']
        for col in time_series_cols:
            if col in df_clean.columns:
                if isinstance(df_clean[col].iloc[0], str) and df_clean[col].iloc[0].startswith('['):
                    import ast
                    df_clean[col] = df_clean[col].apply(
                        lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else []
                    )
        
        # 2. Clean lists (remove None and NaN)
        for col in df_clean.columns:
            if df_clean[col].dtype == object:
                # Check if it contains lists
                sample = df_clean[col].iloc[0] if len(df_clean) > 0 else None
                if isinstance(sample, list):
                    df_clean[col] = df_clean[col].apply(
                        lambda lst: [x for x in lst if x is not None and str(x).lower() != 'nan' 
                                    and not pd.isna(x)] if isinstance(lst, list) else []
                    )
        
        # 3. Convert all numeric columns, fill NaN
        for col in df_clean.columns:
            if col not in time_series_cols:  # Don't convert time series lists
                # Convert to numeric
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                
                # Fill NaN with median
                if df_clean[col].isnull().any():
                    median_val = df_clean[col].median()
                    if pd.isna(median_val):
                        median_val = 0.0
                    df_clean[col] = df_clean[col].fillna(median_val)
        
        # 4. Drop columns that are still all NaN
        cols_to_drop = [col for col in df_clean.columns if df_clean[col].isnull().all()]
        if cols_to_drop:
            print(f"  Dropping columns that are all NaN: {cols_to_drop}")
            df_clean = df_clean.drop(columns=cols_to_drop)
        
        nan_count = df_clean.isnull().sum().sum()
        print(f"  NaN count after cleaning: {nan_count}")
        
        return df_clean
    
    def create_features_with_global_stats(self, df):
        """Create engineered features using global statistics"""
        if not self.registry or not self.registry.global_stats:
            return self.create_features_with_local_stats(df)
            
        print(f"[Client {self.client_id}] Creating features with global stats...")
        
        def get_global_stat(stat_name, default=None):
            if stat_name in self.registry.global_stats:
                return self.registry.global_stats[stat_name]
            elif default is not None:
                return default
            else:
                if 'mean' in stat_name:
                    return 0.0
                elif 'std' in stat_name or 'max' in stat_name or 'q' in stat_name:
                    return 1.0
                else:
                    return 1.0
        
        hr_rmssd_max = get_global_stat('hr_rmssd_max', 1.0)
        hr_recovery_rate_max = get_global_stat('hr_recovery_rate_max', 1.0)
        hr_rmssd_min = get_global_stat('hr_rmssd_min', df['hr_rmssd'].min() if 'hr_rmssd' in df.columns else 0.0)
        hr_recovery_rate_min = get_global_stat('hr_recovery_rate_min', df['hr_recovery_rate'].min() if 'hr_recovery_rate' in df.columns else 0.0)
        hr_rmssd_q25 = get_global_stat('hr_rmssd_q25', 0.25)
        high_stress_pct_q75 = get_global_stat('high_stress_pct_q75', 0.75)
        deep_sleep_ratio_q25 = get_global_stat('deep_sleep_ratio_q25', 0.25)
        
        if 'hr_recovery_rate' in df.columns and 'stress_recovery_rate' in df.columns:
            df['recovery_stress_balance'] = df['hr_recovery_rate'] / (df['stress_recovery_rate'] + 1)
        
        if 'deep_sleep_ratio' in df.columns and 'high_stress_pct' in df.columns:
            df['sleep_stress_efficiency'] = df['deep_sleep_ratio'] * (1 - df['high_stress_pct'])
        
        if 'hr_rmssd' in df.columns and 'stress_mean' in df.columns:
            df['hrv_stress_ratio'] = df['hr_rmssd'] / (df['stress_mean'] + 1)
        
        if 'hr_rmssd' in df.columns and 'resp_cv' in df.columns:
            df['ans_balance'] = df['hr_rmssd'] / (df['resp_cv'] + 1e-10)
        
        if 'stress_mean' in df.columns and 'hr_std' in df.columns:
            df['sympathetic_tone'] = df['stress_mean'] * df['hr_std']
        
        # if 'hr_mean' in df.columns:
        #     df['hr_mean_squared'] = df['hr_mean'] ** 2
        
        # if 'stress_mean' in df.columns:
        #     df['stress_mean_squared'] = df['stress_mean'] ** 2
        
        # if 'sleep_sleepTimeSeconds' in df.columns:
        #     df['sleep_squared'] = df['sleep_sleepTimeSeconds'] ** 2
        
        if 'act_totalCalories' in df.columns and 'hr_recovery_rate' in df.columns:
            df['calories_per_recovery'] = df['act_totalCalories'] / (df['hr_recovery_rate'] + 1)
        
        if 'stress_mean' in df.columns and 'sleep_sleepTimeSeconds' in df.columns:
            df['stress_per_sleep'] = df['stress_mean'] / (df['sleep_sleepTimeSeconds'] / 3600 + 1)
        
        if 'hr_awake_pct' in df.columns and 'hr_deep_sleep_pct' in df.columns:
            df['awake_to_sleep_transition'] = df['hr_awake_pct'] - df['hr_deep_sleep_pct']
        
        if 'stress_first_third' in df.columns and 'stress_last_third' in df.columns:
            df['stress_to_recovery'] = df['stress_first_third'] - df['stress_last_third']
        
        if 'hr_change_point_density' in df.columns and 'stress_volatility' in df.columns:
            df['transition_variability'] = df['hr_change_point_density'] * df['stress_volatility']
        
        if 'hr_rmssd' in df.columns and 'stress_mean' in df.columns:
            df['hrv_stress_interaction'] = df['hr_rmssd'] * df['stress_mean']
        
        if 'deep_sleep_ratio' in df.columns and 'stress_mean' in df.columns:
            df['sleep_stress_interaction'] = df['deep_sleep_ratio'] * df['stress_mean']
        
        if 'hr_recovery_rate' in df.columns and 'act_totalCalories' in df.columns:
            df['recovery_activity_interaction'] = df['hr_recovery_rate'] * df['act_totalCalories']
        
        # if 'hr_rmssd' in df.columns and 'stress_mean' in df.columns:
        #     df['hrv_stress_squared'] = df['hr_rmssd'] * (df['stress_mean'] ** 2)
        
        if 'hr_rmssd' in df.columns:
            hr_rmssd_range = hr_rmssd_max - hr_rmssd_min + 1e-10
            df['hr_rmssd_normalized'] = (df['hr_rmssd'] - hr_rmssd_min) / hr_rmssd_range
        
        if 'hr_recovery_rate' in df.columns:
            hr_recovery_rate_range = hr_recovery_rate_max - hr_recovery_rate_min + 1e-10
            df['hr_recovery_rate_normalized'] = (df['hr_recovery_rate'] - hr_recovery_rate_min) / hr_recovery_rate_range
        
        if all(col in df.columns for col in ['hr_rmssd_normalized', 'high_stress_pct', 'deep_sleep_ratio', 'hr_recovery_rate_normalized']):
            df['resilience_score'] = (
                0.25 * df['hr_rmssd_normalized'] +
                0.25 * (1 - df['high_stress_pct']) +
                0.25 * df['deep_sleep_ratio'] +
                0.25 * df['hr_recovery_rate_normalized']
            )
        
        if 'hr_rmssd' in df.columns:
            df['low_hrv_flag'] = (df['hr_rmssd'] < hr_rmssd_q25).astype(int)
        
        if 'high_stress_pct' in df.columns:
            df['high_stress_flag'] = (df['high_stress_pct'] > high_stress_pct_q75).astype(int)
        
        if 'deep_sleep_ratio' in df.columns:
            df['poor_sleep_flag'] = (df['deep_sleep_ratio'] < deep_sleep_ratio_q25).astype(int)
        
        print(f"[Client {self.client_id}] Created {len([c for c in df.columns if '_flag' in c])} flag features")
        
        return df
    
    def save_client_artifacts(self):
        """Save this client's statistics and metadata"""
        client_dir = f"client_artifacts/client_{self.client_id}"
        os.makedirs(client_dir, exist_ok=True)
        
        if hasattr(self, 'local_stats'):
            stats_path = os.path.join(client_dir, "local_stats.pkl")
            with open(stats_path, 'wb') as f:
                pickle.dump(self.local_stats, f)
        
        if hasattr(self, 'transformation_metadata'):
            meta_path = os.path.join(client_dir, "transformation_metadata.pkl")
            with open(meta_path, 'wb') as f:
                pickle.dump(self.transformation_metadata, f)
        
        features_path = os.path.join(client_dir, "feature_names.pkl")
        with open(features_path, 'wb') as f:
            pickle.dump(self.feature_names, f)
        
        print(f"[Client {self.client_id}] Saved artifacts to {client_dir}")
    
    def save_model_with_features(self, model, path):
        """Save model with feature names - IMPROVED"""
        try:
            model_dir = os.path.dirname(path)
            os.makedirs(model_dir, exist_ok=True)
            
            print(f"[Client {self.client_id}] 💾 Saving model to {path}")
            print(f"  Model features: {len(self.feature_names) if hasattr(self, 'feature_names') else 'unknown'}")
            print(f"  Model directory exists: {os.path.exists(model_dir)}")
            
            # Save the model
            model.save_model(path)
            
            # Also save feature names
            feature_info_path = path.replace('.txt', '_features.json')
            if hasattr(self, 'feature_names'):
                feature_info = {
                    'feature_names': self.feature_names,
                    'client_id': self.client_id,
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'num_features': len(self.feature_names)
                }
                with open(feature_info_path, 'w') as f:
                    json.dump(feature_info, f, indent=2)
                print(f"  ✅ Saved feature info to {feature_info_path}")
            
            print(f"  ✅ Model saved successfully")
            
        except Exception as e:
            print(f"  ❌ Error saving model: {e}")
            import traceback
            traceback.print_exc()

    def diagnose_data_quality(self):
        """Diagnose if data is suitable for modeling"""
        print(f"\n[Client {self.client_id}] 🔍 DIAGNOSING DATA QUALITY...")
        
        issues = []
        
        # 1. Check target variance
        y_std = self.y_train.std()
        y_mean = self.y_train.mean()
        cv = y_std / y_mean if y_mean != 0 else 0
        
        print(f"  Target variable:")
        print(f"    Mean: {y_mean:.2f}")
        print(f"    Std: {y_std:.2f}")
        print(f"    CV: {cv:.2f}")
        
        if y_std < 1.0:
            issues.append("Target has very low variance")
        
        # 2. Check feature-target correlations
        correlations = []
        for i in range(min(self.X_train.shape[1], 30)):
            if np.std(self.X_train[:, i]) > 0:
                corr = np.corrcoef(self.X_train[:, i], self.y_train)[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
        
        if correlations:
            max_corr = max(correlations)
            avg_corr = np.mean(correlations)
            
            print(f"  Feature correlations:")
            print(f"    Max |corr|: {max_corr:.4f}")
            print(f"    Avg |corr|: {avg_corr:.4f}")
            
            if max_corr < 0.1:
                issues.append(f"Strongest correlation is only {max_corr:.4f}")
            if avg_corr < 0.05:
                issues.append(f"Average correlation is only {avg_corr:.4f}")
        
        # 3. Check for constant features
        constant_features = 0
        for i in range(self.X_train.shape[1]):
            if np.std(self.X_train[:, i]) < 1e-10:
                constant_features += 1
        
        if constant_features > 0:
            issues.append(f"{constant_features} constant features")
            print(f"  ⚠️ {constant_features} features are constant")
        
        # 4. Test baseline prediction
        baseline_mae = np.mean(np.abs(self.y_test - self.y_train.mean()))
        print(f"  Baseline (predicting mean):")
        print(f"    MAE: {baseline_mae:.4f}")
        
        # 5. Summary
        if issues:
            print(f"\n  ❌ CRITICAL ISSUES FOUND:")
            for issue in issues:
                print(f"     - {issue}")
            print(f"\n  ⚠️ Your model is likely to fail with these issues!")
            return False
        else:
            print(f"  ✅ Data quality looks acceptable")
            return True
    
    def get_conservative_params(self):
        """Get BETTER parameters for small datasets"""
        return {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'n_jobs': -1,
            'seed': 42 + self.client_id,
            
            # SMALLER, SIMPLER TREES
            'num_leaves': 7,           # Much smaller (prevents overfitting)
            'max_depth': 3,            # Very shallow
            'min_data_in_leaf': 20,    # Require minimum data
            
            # SLOWER LEARNING
            'learning_rate': 0.02,     # Very slow
            
            # STRONG REGULARIZATION
            'reg_alpha': 2.0,          # Strong L1
            'reg_lambda': 2.0,         # Strong L2
            'feature_fraction': 0.7,   # Random feature subset
            'bagging_fraction': 0.7,   # Random data subset
            'bagging_freq': 1,
            
            # FEWER ITERATIONS WITH EARLY STOPPING
            'num_iterations': 100,
        }
    def select_features_by_correlation(self, min_correlation=0.1):
        """Select only features that correlate with the target - RETURNS dataframes"""
        print(f"\n[Client {self.client_id}] Selecting features by correlation (min={min_correlation})...")
        
        if 'label' not in self.train_processed.columns:
            print("  ⚠️ 'label' column not found in train data")
            return self.train_processed, self.test_processed  # FIX: Return what we have
        
        correlations = []
        for col in self.train_processed.columns:
            if col != 'label' and pd.api.types.is_numeric_dtype(self.train_processed[col]):
                corr = self.train_processed['label'].corr(self.train_processed[col])
                if not np.isnan(corr):
                    correlations.append((col, corr))
        
        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Select features with meaningful correlation
        selected_features = []
        for col, corr in correlations:
            if abs(corr) >= min_correlation:
                selected_features.append(col)
                print(f"  ✓ {col}: {corr:.4f}")
            else:
                print(f"  ✗ {col}: {corr:.4f} (too weak)")
        
        if len(selected_features) < 5:
            print(f"  ⚠️ Only {len(selected_features)} features have correlation > {min_correlation}")
            print(f"  Keeping top 10 by absolute correlation instead")
            selected_features = [col for col, _ in correlations[:10]]
        
        # Add label back
        selected_features.append('label')
        
        # Apply selection
        train_processed = self.train_processed[selected_features]
        test_processed = self.test_processed[selected_features]
        self.feature_names = [f for f in selected_features if f != 'label']
        
        print(f"  Selected {len(self.feature_names)} features")
        
        # FIX: RETURN the processed dataframes
        return train_processed, test_processed
 

    def get_parameters(self, config):
        """Get model parameters (required by Flower)"""
        if self.model is None:
            return []
        return []
    
    def set_parameters(self, parameters):
        """Set model parameters from server (required by Flower)"""
        print(f"[Client {self.client_id}] Received parameters from server")
        print(f"  Parameter length: {len(parameters) if parameters else 0}")
        
        if parameters and len(parameters) > 0:
            print(f"  First parameter shape: {parameters[0].shape if hasattr(parameters[0], 'shape') else 'unknown'}")
        
        return True
   
    
    def optimize_with_optuna(self, n_trials=30):
        """Optimize LightGBM hyperparameters with Optuna"""
        print(f"\n[Client {self.client_id}] Starting Optuna optimization ({n_trials} trials)...")
        
        import optuna
        from sklearn.model_selection import KFold
        
        def objective(trial):
            params = {
                'objective': 'regression',
                'metric': 'mae',
                'verbosity': -1,
                'boosting_type': 'gbdt',
                'n_jobs': -1,
                
                # Hyperparameters to tune
                'num_leaves': trial.suggest_int('num_leaves', 10, 150),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            }
            
            # Cross-validation
            kf = KFold(n_splits=3, shuffle=True, random_state=42 + self.client_id)
            mae_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(self.X_train)):
                X_fold_train = self.X_train[train_idx]
                y_fold_train = self.y_train[train_idx]
                X_fold_val = self.X_train[val_idx]
                y_fold_val = self.y_train[val_idx]
                
                train_data = lgb.Dataset(X_fold_train, label=y_fold_train)
                val_data = lgb.Dataset(X_fold_val, label=y_fold_val, reference=train_data)
                
                model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=200,
                    valid_sets=[val_data],
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=30, verbose=False),
                        lgb.log_evaluation(0)
                    ]
                )
                
                y_pred = model.predict(X_fold_val)
                mae = mean_absolute_error(y_fold_val, y_pred)
                mae_scores.append(mae)
            
            return np.mean(mae_scores)
        
        # Create study
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42 + self.client_id)
        )
        
        # Optimize
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        print(f"\n✅ Optuna optimization complete!")
        print(f"  Best MAE: {study.best_value:.4f}")
        
        # Train final model with best params
        best_params = {
            'objective': 'regression',
            'metric': 'mae',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'n_jobs': -1,
        }
        best_params.update(study.best_params)
        
        # Train on full training data
        train_data = lgb.Dataset(self.X_train, label=self.y_train)
        self.model = lgb.train(
            best_params,
            train_data,
            num_boost_round=300,
            callbacks=[lgb.log_evaluation(0)]
        )
        
        return study.best_params, study.best_value
    

    def validate_feature_quality(self, df):
        """Check if engineered features are actually good AND select best ones"""
        if 'label' not in df.columns:
            return
        
        print(f"\n[Client {self.client_id}] 🔍 FEATURE QUALITY CHECK & SELECTION:")
        
        # Calculate correlations
        correlations = []
        for col in df.columns:
            if col != 'label' and pd.api.types.is_numeric_dtype(df[col]):
                corr = df['label'].corr(df[col])
                if not np.isnan(corr):
                    correlations.append((col, abs(corr)))
        
        # Sort by correlation strength
        correlations.sort(key=lambda x: x[1], reverse=True)
        
        # Select top 30 features
        top_k = 30
        top_features = [col for col, _ in correlations[:top_k]]
        
        # Ensure label is included
        if 'label' not in top_features:
            top_features.append('label')
        
        print(f"Selected {len(top_features)-1} best features based on correlation")
        
        # Return selected features (caller should filter dataframe)
        return top_features

    def select_best_features(self, n_features=20):
        """Select only the most predictive features"""
        print(f"[Client {self.client_id}] Selecting best {n_features} features...")
        
        if 'label' not in self.train_processed.columns:
            return self.train_processed, self.test_processed
        
        # Calculate correlations
        correlations = []
        for col in self.train_processed.columns:
            if col != 'label' and pd.api.types.is_numeric_dtype(self.train_processed[col]):
                corr = self.train_processed['label'].corr(self.train_processed[col])
                if not np.isnan(corr):
                    correlations.append((col, abs(corr)))
        
        # Sort by correlation strength
        correlations.sort(key=lambda x: x[1], reverse=True)
        
        # Select top features
        top_features = [col for col, _ in correlations[:n_features]]
        
        print(f"Selected {len(top_features)} best features:")
        for i, col in enumerate(top_features[:10]):
            for orig_col, corr in correlations:
                if orig_col == col:
                    print(f"  {i+1}. {col}: {corr:.4f}")
                    break
        
        # Keep label + top features
        columns_to_keep = top_features + ['label']
        
        self.train_processed = self.train_processed[columns_to_keep]
        self.test_processed = self.test_processed[columns_to_keep]
        self.feature_names = top_features
        
        return self.train_processed, self.test_processed

    def diagnose_core_issue(self):
        """Find out WHY features don't correlate with target"""
        print(f"\n{'='*60}")
        print(f"[Client {self.client_id}] 🔍 ROOT CAUSE ANALYSIS")
        print(f"{'='*60}")
        
        # 1. First, check what the target variable ACTUALLY IS
        print(f"\n1. TARGET VARIABLE ANALYSIS:")
        print(f"   Column name: 'label'")
        print(f"   Data type: {type(self.y_train[0])}")
        print(f"   Sample values: {self.y_train[:5]}")
        print(f"   Statistics:")
        print(f"     Min: {self.y_train.min():.2f}")
        print(f"     Max: {self.y_train.max():.2f}")
        print(f"     Mean: {self.y_train.mean():.2f}")
        print(f"     Std: {self.y_train.std():.2f}")
        print(f"     % of unique values: {(len(np.unique(self.y_train))/len(self.y_train))*100:.1f}%")
        
        # 2. Check if target is constant or has very little variance
        if self.y_train.std() < 1.0:
            print(f"   ⚠️ CRITICAL: Target has very low variance (std={self.y_train.std():.4f})")
            print(f"   This means there's almost nothing to predict!")
        
        # 3. Check the actual feature values
        print(f"\n2. FEATURE ANALYSIS:")
        print(f"   Number of features: {len(self.feature_names)}")
        
        # Check top 5 features
        for i, feature_name in enumerate(self.feature_names[:5]):
            if i < self.X_train.shape[1]:
                feature_data = self.X_train[:, i]
                print(f"\n   Feature: {feature_name}")
                print(f"     Min: {feature_data.min():.2f}")
                print(f"     Max: {feature_data.max():.2f}")
                print(f"     Mean: {feature_data.mean():.2f}")
                
                # Calculate correlation
                if np.std(feature_data) > 0 and np.std(self.y_train) > 0:
                    corr = np.corrcoef(feature_data, self.y_train)[0, 1]
                    print(f"     Correlation with target: {corr:.4f}")
                    
                    if abs(corr) > 0.3:
                        print(f"     ✓ DECENT correlation")
                    elif abs(corr) > 0.1:
                        print(f"     ⚠️ Weak correlation")
                    else:
                        print(f"     ❌ NO correlation")
                else:
                    print(f"     ⚠️ No variance in feature or target")
        
        # 4. Check the actual raw data columns
        print(f"\n3. RAW DATA COLUMNS (from CSV):")
        raw_columns = []
        for col in self.train_processed.columns:
            if col not in ['label'] + self.feature_names:
                raw_columns.append(col)
        
        print(f"   Raw columns that aren't being used as features: {len(raw_columns)}")
        if raw_columns:
            print(f"   First 5: {raw_columns[:5]}")
        
        # 5. Check if label is actually predictable from sleep data
        print(f"\n4. DOMAIN SANITY CHECK:")
        print(f"   Based on sleep/HR/stress data, what SHOULD we be predicting?")
        print(f"   Common targets in sleep studies:")
        print(f"   - Sleep quality score (0-100)")
        print(f"   - Recovery score (0-100)")
        print(f"   - Next day performance")
        print(f"   - Fatigue level")
        print(f"   - Mood score")
        
        # 6. Check baseline prediction
        baseline_pred = np.full_like(self.y_test, self.y_train.mean())
        baseline_mae = mean_absolute_error(self.y_test, baseline_pred)
        
        print(f"\n5. BASELINE PERFORMANCE:")
        print(f"   Predicting mean ({self.y_train.mean():.2f}) gives:")
        print(f"   MAE: {baseline_mae:.2f}")
        
        # If baseline MAE is already very low, model can't improve much
        if baseline_mae < (self.y_train.max() - self.y_train.min()) * 0.1:
            print(f"   ⚠️ Baseline is already good - hard to improve")
        
        print(f"\n{'='*60}")
        print(f"RECOMMENDATIONS:")
        
        if self.y_train.std() < 1.0:
            print(f"1. Your target has NO VARIANCE. Fix this first!")
            print(f"   Check if 'label' column is correct")
        
        # Check feature correlations
        if len(self.feature_names) > 0:
            correlations = []
            for i in range(min(20, self.X_train.shape[1])):
                if np.std(self.X_train[:, i]) > 0 and np.std(self.y_train) > 0:
                    corr = np.corrcoef(self.X_train[:, i], self.y_train)[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
            
            if correlations:
                avg_corr = np.mean(correlations)
                if avg_corr < 0.1:
                    print(f"2. Features have NO correlation with target (avg={avg_corr:.4f})")
                    print(f"   Your features don't predict your target at all!")
                    print(f"   Either:")
                    print(f"   a) Wrong features for this target")
                    print(f"   b) Wrong target for these features")
                    print(f"   c) Need completely different feature engineering")
        
        print(f"{'='*60}")

    def fit(self, parameters, config):
        """Train LightGBM model locally (required by Flower)"""
        server_round = config.get("server_round", 1)
        print(f"\n{'='*60}")
        print(f"[Client {self.client_id}] Starting FIT round {server_round}")
        print(f"{'='*60}")
        print(f"[Client {self.client_id}] Validating feature names before training...")
        # Ensure feature_names is a list
        if not hasattr(self, 'feature_names') or self.feature_names is None:
            print("  ⚠️ Feature names not found, creating default names")
            self.feature_names = [f'feature_{i}' for i in range(self.X_train.shape[1])]
        
        # Clean feature names
        self.feature_names = self.clean_feature_names(self.feature_names)
        
        # Remove duplicates
        unique_features = []
        seen = set()
        for name in self.feature_names:
            if name not in seen:
                unique_features.append(name)
                seen.add(name)
            else:
                # Make it unique
                idx = 1
                new_name = f"{name}_{idx}"
                while new_name in seen:
                    idx += 1
                    new_name = f"{name}_{idx}"
                unique_features.append(new_name)
                seen.add(new_name)
        
        self.feature_names = unique_features
        
        # Check length matches
        if len(self.feature_names) != self.X_train.shape[1]:
            print(f"  ⚠️ Feature count mismatch: {len(self.feature_names)} names for {self.X_train.shape[1]} features")
            print(f"  Using default feature names")
            self.feature_names = [f'feature_{i}' for i in range(self.X_train.shape[1])]
        
        print(f"  Final feature names: {len(self.feature_names)} unique names")
        print(f"  First 3: {self.feature_names[:3]}")
        
        # Use conservative parameters
        best_params = self.get_conservative_params()
        learning_rate = config.get("learning_rate", 0.05)
        best_params['learning_rate'] = learning_rate
        
        if parameters:
            self.set_parameters(parameters)
        
        try:
            # Create feature name mapping to ensure consistency
            feature_name_map = {}
            for i, name in enumerate(self.feature_names):
                feature_name_map[f'feature_{i}'] = name
            
            train_data = lgb.Dataset(
                self.X_train,
                label=self.y_train,
                feature_name=self.feature_names,  # Use validated names
                params={'verbose': -1}
            )
            
            # Use test as validation for early stopping
            valid_data = lgb.Dataset(
                self.X_test,
                label=self.y_test,
                reference=train_data
            )
            
            num_boost_round = 100 if server_round == 1 else 50
            
            # Debug: print first few feature names
            print(f"  First 5 feature names: {self.feature_names[:5]}")
            
            self.model = lgb.train(
                best_params,
                train_data,
                num_boost_round=num_boost_round,
                valid_sets=[train_data, valid_data],
                valid_names=['train', 'valid'],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=30),
                    lgb.log_evaluation(period=10)
                ]
            )
            
            # Evaluate
            y_train_pred = self.model.predict(self.X_train)
            y_test_pred = self.model.predict(self.X_test)
            
            train_mae = mean_absolute_error(self.y_train, y_train_pred)
            train_r2 = r2_score(self.y_train, y_train_pred)
            test_mae = mean_absolute_error(self.y_test, y_test_pred)
            test_r2 = r2_score(self.y_test, y_test_pred)
            
            # In your fit method, replace R² focus with MAE focus:
                
            baseline_pred = np.full_like(self.y_test, self.y_train.mean())
            baseline_mae = mean_absolute_error(self.y_test, baseline_pred)
            baseline_r2 = r2_score(self.y_test, baseline_pred)
            print(f"\n[Client {self.client_id}] Round {server_round} Results:")
            print(f"  Training MAE: {train_mae:.4f}")
            print(f"  Testing MAE:  {test_mae:.4f} (Baseline: {baseline_mae:.4f})")
            print(f"  Improvement:  {((baseline_mae - test_mae)/baseline_mae*100):.1f}%")
          
            print(f"  Baseline (predict mean): MAE={baseline_mae:.4f}, R²={baseline_r2:.4f}")
            
            # If baseline already has low MAE, model will struggle
            if baseline_mae < 5.0:  # Adjust threshold based on your data
                print(f"  ⚠️ Baseline MAE is already low ({baseline_mae:.4f})")
                print(f"  Need stronger features to beat baseline")

            # Save model
            if self.model is not None:
                model_dir = f"models/client_{self.client_id}"
                os.makedirs(model_dir, exist_ok=True)
                
                lgbm_path = os.path.join(model_dir, f"round_{server_round:03d}.txt")
                self.save_model_with_features(self.model, lgbm_path)
                
            # Save best model
            if test_mae < self.best_test_mae:
                self.best_test_mae = test_mae
                best_path = os.path.join(model_dir, "best_model.txt")
                self.save_model_with_features(self.model, best_path)
                self.best_model_path = best_path
            
            print(f"{'='*60}")
            
            self.round_history.append({
                'round': server_round,
                'train_mae': train_mae,
                'train_r2': train_r2,
                'test_mae': test_mae,
                'test_r2': test_r2,
            })
            
            return self.get_parameters(config), len(self.X_train), {
                "train_mae": float(train_mae),
                "train_r2": float(train_r2),
                "test_mae": float(test_mae),  # Add this
                "test_r2": float(test_r2),
                "client_id": self.client_id  # Add this
            }
                    
        except Exception as e:
            print(f"[Client {self.client_id}] Error during training: {e}")
            import traceback
            traceback.print_exc()
            return [], len(self.X_train), {}
    
    def evaluate(self, parameters, config):
        """Evaluate - report only MAE for competition"""
        if self.model is None:
            return 0.0, len(self.X_test), {"mae": 10.0, "client_id": self.client_id}
        
        y_pred = self.model.predict(self.X_test)
        mae = mean_absolute_error(self.y_test, y_pred)
        
        baseline_pred = np.full_like(self.y_test, self.y_train.mean())
        baseline_mae = mean_absolute_error(self.y_test, baseline_pred)
        
        print(f"[Client {self.client_id}] MAE: {mae:.4f} (Baseline: {baseline_mae:.4f})")
        print(f"  Improvement: {((baseline_mae - mae)/baseline_mae*100):.1f}%")
        
        # CRITICAL: Include client_id in metrics so strategy can identify the client
        return float(mae), len(self.X_test), {
            "mae": float(mae),
            "test_mae": float(mae),  # Add this for easier access
            "client_id": self.client_id
        }

# ========== END OF LightGBMClient CLASS ==========

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python client_federated.py <csv_file_path> [client_id] [optuna_trials] [use_global_registry]")
        print("Example: python client_federated.py data/client1.csv 1 20 True")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    client_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    optuna_trials = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    use_global_registry = sys.argv[4].lower() == 'true' if len(sys.argv) > 4 else True
    
    print(f"[Client {client_id}] Starting LightGBM client with Optuna tuning (Federated Version)")
    print(f"  Data: {csv_path}")
    print(f"  Optuna trials: {optuna_trials}")
    print(f"  Use global registry: {use_global_registry}")
    
    # Initialize client
    client = LightGBMClient(
        csv_path=csv_path,
        client_id=client_id,
        use_optuna=True,
        optuna_trials=optuna_trials,
        use_global_registry=use_global_registry,
        registry_path="global_feature_registry.pkl"
    )
    
    try:
        # Start Flower client
        fl.client.start_numpy_client(
            server_address="127.0.0.1:8080",  # Server address
            client=client
        )
    except Exception as e:
        print(f"[Client {client_id}] Error: {e}")
        import traceback
        traceback.print_exc()
