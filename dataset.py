import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import GroupShuffleSplit
import json
import pickle
from pathlib import Path
from utils import save_stats

TARGET_COL = "label"

DROP_COLS = [
    "Unnamed: 0",
    "day",
    "act_activeTime",
    "hr_time_series",
    "resp_time_series",
    "stress_time_series"
]

def safe_preprocess(func):
    """Decorator to handle NaN in preprocessing functions"""
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            
            # Check for NaN
            if hasattr(result, 'isnull'):
                nan_count = result.isnull().sum().sum()
                if nan_count > 0:
                    print(f"⚠️ {func.__name__} produced {nan_count} NaN values")
                    # Fill NaN with column median
                    numeric_cols = result.select_dtypes(include=[np.number]).columns
                    for col in numeric_cols:
                        if result[col].isnull().any():
                            median_val = result[col].median()
                            result[col] = result[col].fillna(median_val)
            
            return result
        except Exception as e:
            print(f"❌ Error in {func.__name__}: {e}")
            return args[0]  # Return original dataframe
    return wrapper

# Add this function to dataset.py (after the existing functions)
def save_transformation_metadata(metadata, filepath="transformation_metadata.pkl"):
    """Save transformation metadata for later use on test data"""
    with open(filepath, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"Saved transformation metadata to {filepath}")

def load_transformation_metadata(filepath="transformation_metadata.pkl"):
    """Load transformation metadata"""
    with open(filepath, 'rb') as f:
        metadata = pickle.load(f)
    print(f"Loaded transformation metadata from {filepath}")
    return metadata

def fit_transform_features(data, feature_list, target_column, train_idx=None):
    """Process features and save metadata for test transformation"""
    if train_idx is not None:
        train_data = data.iloc[train_idx]
    else:
        train_data = data
    
    processed_df = data.copy()
    transformation_metadata = {}
    
    print(f"Processing {len(feature_list)} features...")
    print(f"Target column: {target_column}")
    print("=" * 60)
    
    dangerous_features = [
        'sleep_sleepTimeSeconds', 'sleep_napTimeSeconds', 
        'sleep_unmeasurableSleepSeconds', 'sleep_deepSleepSeconds',
        'sleep_lightSleepSeconds', 'sleep_remSleepSeconds', 
        'sleep_awakeSleepSeconds', 'act_totalCalories',
        'act_activeKilocalories', 'act_distance'
    ]
    
    safe_feature_list = []
    for feature in feature_list:
        is_dangerous = any(danger_feat in feature for danger_feat in dangerous_features)
        if is_dangerous:
            print(f"⚠️ Skipping {feature} (avoid squared/cubic transforms)")
            # Keep the original feature
            if feature in processed_df.columns:
                # Just keep it as is, don't transform
                continue
        safe_feature_list.append(feature)
    
    print(f"Processing {len(safe_feature_list)} safe features (skipped {len(feature_list) - len(safe_feature_list)} dangerous features)")

    for i, feature in enumerate(feature_list, 1):
        print(f"[{i}/{len(feature_list)}] Processing: {feature}")
        
        try:
            # Fit transformations on TRAINING data only
            results, transforms = test_all_transformations(
                train_data,
                feature,
                target_column
            )
            
            # Get top 3 transformation names
            top3_names = results.iloc[:3]['transform'].tolist()
            
            # Store metadata for this feature (using training data stats)
            feature_stats = {
                'selected_transforms': top3_names,
                'mean': train_data[feature].mean(),
                'std': train_data[feature].std(),
                'median': train_data[feature].median(),
                'q01': train_data[feature].quantile(0.01),
                'q99': train_data[feature].quantile(0.99),
                'q25': train_data[feature].quantile(0.25),
                'q75': train_data[feature].quantile(0.75),
                'iqr': train_data[feature].quantile(0.75) - train_data[feature].quantile(0.25),
                'upper_bound': train_data[feature].quantile(0.75) + 1.5 * (train_data[feature].quantile(0.75) - train_data[feature].quantile(0.25)),
                'bins_5': pd.cut(train_data[feature], 5, retbins=True)[1],
                'bins_10': pd.cut(train_data[feature], 10, retbins=True)[1]
            }
            transformation_metadata[feature] = feature_stats
            
            # Create new columns with top 3 transformations
            for rank, trans_name in enumerate(top3_names, 1):
                col_name = f"{feature}_{rank}"
                processed_df[col_name] = transforms[trans_name]
                print(f"  Added: {col_name} ({trans_name})")
            
            # Drop the original column
            if feature in processed_df.columns:
                processed_df = processed_df.drop(feature, axis=1)
                print(f"  Dropped: {feature}")
                
        except Exception as e:
            print(f"  ⚠️ Error processing {feature}: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE!")
    print(f"Original columns: {len(data.columns)}")
    print(f"New columns: {len(processed_df.columns)}")
    
    return processed_df, transformation_metadata

def apply_transformations_to_test(test_data, transformation_metadata):
    """Apply transformations to test data using training metadata"""
    processed_df = test_data.copy()
    
    for feature, meta in transformation_metadata.items():
        if feature not in processed_df.columns:
            continue
            
        series = processed_df[feature]
        
        # Calculate all transforms using TRAINING stats
        all_transforms = {}
        
        # Basic transforms
        all_transforms['original'] = series
        all_transforms['log'] = np.log1p(series)
        all_transforms['sqrt'] = np.sqrt(series)
        all_transforms['squared'] = series ** 2
        all_transforms['cubic'] = series ** 3
        
        # Statistical transforms using training statistics
        all_transforms['z_score'] = (series - meta['mean']) / meta['std']
        all_transforms['deviation'] = series - meta['median']
        
        # Outlier handling using training percentiles
        all_transforms['winsorized'] = series.clip(lower=meta['q01'], upper=meta['q99'])
        all_transforms['iqr_capped'] = series.clip(upper=meta['upper_bound'])
        
        # Binning using training bin edges
        try:
            all_transforms['binned_5'] = pd.cut(series, bins=meta['bins_5'], labels=False, include_lowest=True)
        except:
            all_transforms['binned_5'] = pd.Series([0] * len(series))
        
        try:
            all_transforms['binned_10'] = pd.cut(series, bins=meta['bins_10'], labels=False, include_lowest=True)
        except:
            all_transforms['binned_10'] = pd.Series([0] * len(series))
        
        # Percentile ranking on test data (this is the only transform that uses test data)
        all_transforms['percentile'] = series.rank(pct=True)
        
        # Apply only the top 3 selected during training
        for rank, trans_name in enumerate(meta['selected_transforms'], 1):
            col_name = f"{feature}_{rank}"
            # Fill NaNs using training median
            transformed = all_transforms[trans_name]
            if hasattr(transformed, 'fillna'):
                processed_df[col_name] = transformed.fillna(meta['median'])
            else:
                processed_df[col_name] = transformed
        
        # Drop original feature
        if feature in processed_df.columns:
            processed_df = processed_df.drop(feature, axis=1)
    
    return processed_df
@safe_preprocess
def simple_preprocess_heart_rate(df):
    """
    Simple preprocessing for heart rate time series
    """
    df_processed = df.copy()

    # Debug: Check what we're working with
    print("Data types in hr_time_series column:")
    print(df_processed['hr_time_series'].apply(type).value_counts())

    # Direct conversion approach
    def convert_to_list(value):
        if isinstance(value, list):
            return value

        # Convert string representation of list to actual list
        try:
            # If it looks like a Python list string
            if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
                # Use ast.literal_eval for safety
                import ast
                return ast.literal_eval(value)

            # If it's already numeric or array-like
            if hasattr(value, '__iter__') and not isinstance(value, str):
                return list(value)

            # Last resort: split and convert
            str_val = str(value)
            # Remove brackets and split
            str_val = str_val.replace('[', '').replace(']', '')
            # Split by comma or space
            parts = []
            for part in str_val.replace(',', ' ').split():
                try:
                    parts.append(float(part))
                except:
                    continue
            return parts

        except Exception as e:
            print(f"Warning: Could not parse value: {str(value)[:50]}... Error: {e}")
            return []

    df_processed['hr_time_series_list'] = df_processed['hr_time_series'].apply(convert_to_list)

    # Basic cleaning
    df_processed['hr_time_series_clean'] = df_processed['hr_time_series_list'].apply(
        lambda x: [v for v in x if isinstance(v, (int, float)) and 30 <= v <= 200]
    )

    return df_processed
@safe_preprocess
def extract_hr_features(df):
    """
    Extract meaningful features from heart rate time series
    """
    df_features = df.copy()

    # Basic statistical features
    df_features['hr_mean'] = df_features['hr_time_series_clean'].apply(
        lambda x: np.mean(x) if len(x) > 0 else np.nan
    )
    df_features['hr_median'] = df_features['hr_time_series_clean'].apply(
        lambda x: np.median(x) if len(x) > 0 else np.nan
    )
    df_features['hr_std'] = df_features['hr_time_series_clean'].apply(
        lambda x: np.std(x) if len(x) > 0 else np.nan
    )
    df_features['hr_variance'] = df_features['hr_time_series_clean'].apply(
        lambda x: np.var(x) if len(x) > 0 else np.nan
    )

    # Heart Rate Variability (HRV) related features
    def calculate_rmssd(series):
        if len(series) < 2:
            return np.nan
        differences = np.diff(series)
        return np.sqrt(np.mean(differences**2))

    df_features['hr_rmssd'] = df_features['hr_time_series_clean'].apply(calculate_rmssd)
    '''
    What is RMSSD?
    RMSSD measures how much your heart rate naturally varies between beats.
    It's one of the most important HRV metrics because it primarily reflects parasympathetic (vagal)
    nervous system activity - your body's "rest and digest" system.
    Higher RMSSD = better recovery, deeper sleep, less stress
    '''

    # Sleep stage estimation features
    def sleep_stage_features(series):
        if len(series) < 10:
            return np.nan, np.nan, np.nan

        # Assuming data points are evenly spaced
        # Low HR periods (likely deep sleep)
        low_hr_threshold = np.percentile(series, 25)
        deep_sleep_ratio = np.sum(series < low_hr_threshold) / len(series)

        # High HR periods (likely REM/light sleep)
        high_hr_threshold = np.percentile(series, 75)
        rem_sleep_ratio = np.sum(series > high_hr_threshold) / len(series)

        # HR stability (low variability suggests deeper sleep)
        rolling_std = pd.Series(series).rolling(window=5, min_periods=3).std().mean()
        '''
        Low rolling_std = very stable heart rate (characteristic of deep sleep)
        High rolling_std = fluctuating heart rate (lighter sleep or awake)
        '''
        return deep_sleep_ratio, rem_sleep_ratio, rolling_std

    df_features[['deep_sleep_ratio', 'rem_sleep_ratio', 'hr_stability']] = pd.DataFrame(
        df_features['hr_time_series_clean'].apply(sleep_stage_features).tolist(),
        index=df_features.index
    )


    # Time domain features
    df_features['hr_range'] = df_features['hr_time_series_clean'].apply(
        lambda x: np.max(x) - np.min(x) if len(x) > 0 else np.nan
    )

    # Percentile features
    df_features['hr_25th'] = df_features['hr_time_series_clean'].apply(
        lambda x: np.percentile(x, 25) if len(x) > 0 else np.nan
    )
    df_features['hr_75th'] = df_features['hr_time_series_clean'].apply(
        lambda x: np.percentile(x, 75) if len(x) > 0 else np.nan
    )

    # Slope features (how HR changes during sleep)
    def calculate_trend(series):
        if len(series) < 5:
            return np.nan
        x = np.arange(len(series))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, series)
        return slope

    df_features['hr_trend_slope'] = df_features['hr_time_series_clean'].apply(calculate_trend)
    '''
    Negative slope = HR decreasing through the night (good, normal)
    Positive slope = HR increasing through the night (possibly restless)
    Near zero = HR stays constant
    '''
    # Night segmentation (first half vs second half)
    def night_segmentation(series):
        if len(series) < 4:
            return np.nan, np.nan

        split_point = len(series) // 2
        first_half = series[:split_point]
        second_half = series[split_point:]

        return np.mean(first_half), np.mean(second_half)

    df_features[['hr_first_half_mean', 'hr_second_half_mean']] = pd.DataFrame(
        df_features['hr_time_series_clean'].apply(night_segmentation).tolist(),
        index=df_features.index
    )
    '''
    Compares first half vs second half of sleep:

    1. First half: Usually more deep sleep (should have lower HR)
    2. Second half: Usually more REM sleep (might have higher HR)
    '''
    # Recovery features
    df_features['hr_recovery_rate'] = df_features.apply(
        lambda row: (row['hr_maxHeartRate'] - row['hr_restingHeartRate']) /
                   (row['hr_maxHeartRate'] - row['hr_minHeartRate'])
        if not pd.isna(row['hr_maxHeartRate']) and row['hr_maxHeartRate'] != row['hr_minHeartRate']
        else np.nan,
        axis=1
    )
    '''
    What it measures: How close your resting HR is to your minimum HR.
    1. Higher value (closer to 1) = Resting HR is close to max HR (poor recovery)
    2. Lower value (closer to 0) = Resting HR is close to min HR (good recovery)

    Better recovery example:
    Max HR = 100, Min HR = 40, Resting HR = 42
    Recovery Rate = (100-42)/(100-40) = 58/60 = 0.97

    '''
    return df_features

'''
This function standardizes all heart rate time series to the same
length and smooths out noise. Think of it like making all sleep recordings
the same "duration" regardless of how long you actually slept, and cleaning up small fluctuations.
'''
@safe_preprocess
def resample_and_smooth_series(df, target_length=360):
    """
    Resample time series to consistent length and apply smoothing
    """
    df_resampled = df.copy()

    def process_series(series, target_len):
        if len(series) == 0:
            return np.array([])

        # Linear interpolation to target length
        x_old = np.linspace(0, 1, len(series))
        x_new = np.linspace(0, 1, target_len)
        series_resampled = np.interp(x_new, x_old, series)
        '''
        Example to understand:

        Imagine you recorded:

        Night 1: 4 hours of sleep (240 minutes = 240 data points)
        Night 2: 8 hours of sleep (480 minutes = 480 data points)

        Problem: They have different lengths, so we can't compare them directly.
        Solution: Make both have 360 points using linear interpolation.
        '''
        # Apply smoothing (moving average)
        window_size = max(3, target_len // 120)  # Adaptive window
        if window_size % 2 == 0:
            window_size += 1

        series_smoothed = pd.Series(series_resampled).rolling(
            window=window_size, center=True, min_periods=1
        ).mean().values

        '''
        Moving average calculation:

        1. Takes average of several points around each position
        2. center=True → uses points before AND after the current point
        3. min_periods=1 → works even at edges (with fewer points)

        Example of smoothing:
        Original noisy data: [50, 52, 48, 46, 60, 58, 56]
        With window_size=3:
        Point 1: (50+52+48)/3 = 50.0
        Point 2: (52+48+46)/3 = 48.7
        Point 3: (48+46+60)/3 = 51.3
        ... etc.

        Result: [50.0, 48.7, 51.3, ...]  # Much smoother!
        '''
        return series_smoothed

    df_resampled['hr_time_series_resampled'] = df_resampled['hr_time_series_clean'].apply(
        lambda x: process_series(x, target_length)
    )

    return df_resampled
@safe_preprocess
def detect_hr_outliers(df):
    """
    Detect and handle outliers in heart rate data
    """
    df_clean = df.copy()

    def remove_hr_outliers(series, method='iqr', threshold=3.0):
        if len(series) < 10:
            return series

        if method == 'iqr':
            Q1 = np.percentile(series, 25)
            Q3 = np.percentile(series, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            mask = (series >= lower_bound) & (series <= upper_bound)
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(series))
            mask = z_scores < threshold

        # Instead of removing, interpolate outliers
        series_clean = pd.Series(series).copy()
        if not mask.all():
            # Find indices of outliers
            outlier_idx = np.where(~mask)[0]
            # Replace outliers with NaN
            series_clean.iloc[outlier_idx] = np.nan
            # Interpolate
            series_clean = series_clean.interpolate(method='linear', limit_direction='both')
        '''
        Original: [45, 46, 120, 47, 48]
               ↑
           Outlier (120)

        Step 1: Mark outlier as NaN: [45, 46, NaN, 47, 48]
        Step 2: Interpolate between 46 and 47
        Step 3: Result: [45, 46, 46.5, 47, 48]
        '''
        return series_clean.values

    df_clean['hr_time_series_no_outliers'] = df_clean['hr_time_series_resampled'].apply(
        lambda x: remove_hr_outliers(x, method='iqr')
    )

    return df_clean
@safe_preprocess
def classify_sleep_stages(df):
    """
    Create simplified sleep stage labels based on heart rate patterns
    """
    df_stages = df.copy()
    # Needs at least 20 data points (minutes) to classify.
    def segment_sleep_stages(series):
        if len(series) < 20:
            return np.array([])

        stages = []
        window_size = 5  # 5-minute windows

        for i in range(0, len(series), window_size):
            window = series[i:i+window_size]
            if len(window) < 3:
                break

            window_mean = np.mean(window)
            window_std = np.std(window)
            '''
            Window 1: [50, 49, 48, 47, 46]
            Mean = (50+49+48+47+46)/5 = 48.0
            Std = 1.6 (very stable)

            Window 2: [60, 62, 61, 59, 58]
            Mean = 60.0
            Std = 1.6 (also stable but higher mean)

            Window 3: [55, 70, 50, 65, 60]
            Mean = 60.0
            Std = 7.9 (very variable!)
            '''
            # Simple rules based on your data patterns
            # CAN BE EDITED!
            if window_mean < 45 and window_std < 5:
                stage = 3  # Deep sleep
            elif window_mean > 65 and window_std > 10:
                stage = 4  # REM sleep
            elif window_mean < 55 and window_std < 8:
                stage = 2  # Light sleep
            else:
                stage = 1  # Awake/transition

            stages.extend([stage] * min(window_size, len(window)))

        return np.array(stages[:len(series)])

    df_stages['hr_sleep_stages'] = df_stages['hr_time_series_no_outliers'].apply(segment_sleep_stages)

    # Calculate stage percentages
    def calculate_stage_percentages(stages):
        if len(stages) == 0:
            return [np.nan] * 4

        unique, counts = np.unique(stages, return_counts=True)
        percentages = np.zeros(4)

        for stage, count in zip(unique, counts):
            if 1 <= stage <= 4:
                percentages[stage-1] = count / len(stages)

        return percentages

    stage_percentages = df_stages['hr_sleep_stages'].apply(calculate_stage_percentages)
    df_stages[['hr_awake_pct', 'hr_light_sleep_pct', 'hr_deep_sleep_pct', 'hr_rem_sleep_pct']] = pd.DataFrame(
        stage_percentages.tolist(),
        index=df_stages.index
    )

    return df_stages
@safe_preprocess
def extract_advanced_features(df):
    """
    Extract more sophisticated features from time series for ML
    """
    df_advanced = df.copy()

    # Frequency domain features (using FFT)
    def frequency_features(series):
        if len(series) < 10:
            return np.nan, np.nan, np.nan

        # Fast Fourier Transform
        fft_values = np.fft.fft(series)
        fft_freq = np.fft.fftfreq(len(series))

        # Power spectral density
        psd = np.abs(fft_values) ** 2

        # Dominant frequency
        dominant_freq = np.abs(fft_freq[np.argmax(psd[1:]) + 1])

        # Spectral entropy
        psd_norm = psd / psd.sum()
        spectral_entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-10))

        return dominant_freq, spectral_entropy, np.mean(psd)

    df_advanced[['hr_dominant_freq', 'hr_spectral_entropy', 'hr_avg_power']] = pd.DataFrame(
        df_advanced['hr_time_series_no_outliers'].apply(frequency_features).tolist(),
        index=df_advanced.index
    )

    # Autocorrelation features
    def autocorr_features(series, lag=10):
        if len(series) < 20:
            return np.nan
        autocorr = pd.Series(series).autocorr(lag=lag)
        return autocorr

    df_advanced['hr_autocorr_lag10'] = df_advanced['hr_time_series_no_outliers'].apply(
        lambda x: autocorr_features(x, lag=10)
    )

    # Change point detection
    def count_changepoints(series, threshold=5):
        if len(series) < 10:
            return np.nan
        changes = np.sum(np.abs(np.diff(series)) > threshold)
        return changes / len(series)  # Normalize by length

    df_advanced['hr_change_point_density'] = df_advanced['hr_time_series_no_outliers'].apply(
        lambda x: count_changepoints(x, threshold=5)
    )

    return df_advanced
@safe_preprocess
def preprocess_respiration_data(df):
    """
    Preprocess Garmin respiration time series data
    """
    df_processed = df.copy()

    # Parse time series (similar to HR parsing)
    def parse_resp_series(series_data):
        if series_data is None or (isinstance(series_data, float) and np.isnan(series_data)):
            return []

        # If already a list
        if isinstance(series_data, (list, np.ndarray)):
            return list(series_data)

        # Convert string to list
        series_str = str(series_data).strip()
        if not series_str or series_str in ['nan', 'None', '[]']:
            return []

        try:
            # Try JSON parsing
            import json
            return json.loads(series_str)
        except:
            try:
                # Try Python literal evaluation
                import ast
                return ast.literal_eval(series_str)
            except:
                # Manual parsing
                series_str = series_str.replace('[', '').replace(']', '')
                values = []
                for val in series_str.split(','):
                    val = val.strip()
                    if val:
                        try:
                            values.append(float(val))
                        except:
                            continue
                return values

    df_processed['resp_time_series_parsed'] = df_processed['resp_time_series'].apply(parse_resp_series)

    # Clean the time series
    def clean_resp_series(series):
        if series is None or len(series) == 0:
            return np.array([])

        series_array = np.array(series, dtype=float)

        # Remove NaN
        series_array = series_array[~np.isnan(series_array)]

        # Remove physiologically impossible values (0-30 breaths/min is plausible)
        # Negative values are definitely errors
        mask = (series_array > 0) & (series_array <= 30)
        series_clean = series_array[mask]

        return series_clean

    df_processed['resp_time_series_clean'] = df_processed['resp_time_series_parsed'].apply(clean_resp_series)

    return df_processed
@safe_preprocess
def extract_respiration_features(df):
    """
    Extract meaningful features from respiration time series
    """
    df_features = df.copy()

    # Basic statistical features
    df_features['resp_mean'] = df_features['resp_time_series_clean'].apply(
        lambda x: np.mean(x) if len(x) > 0 else np.nan
    )
    df_features['resp_median'] = df_features['resp_time_series_clean'].apply(
        lambda x: np.median(x) if len(x) > 0 else np.nan
    )
    df_features['resp_std'] = df_features['resp_time_series_clean'].apply(
        lambda x: np.std(x) if len(x) > 0 else np.nan
    )
    df_features['resp_cv'] = df_features['resp_time_series_clean'].apply(
        lambda x: np.std(x)/np.mean(x) if len(x) > 0 and np.mean(x) > 0 else np.nan
    )  # Coefficient of variation

    # Breathing pattern features
    def breathing_pattern_features(series):
        if len(series) < 10:
            return np.nan, np.nan, np.nan

        # Breathing regularity (autocorrelation at lag 1)
        autocorr = pd.Series(series).autocorr(lag=1)
        '''
        What it calculates: How similar breathing is to itself one minute later
        Interpretation:

        1. High autocorrelation (close to 1): Very regular, predictable breathing
        2. Low autocorrelation (close to 0): Irregular, unpredictable breathing
        3. Negative autocorrelation: Breathing alternates high/low

        Example:
        Breathing: [14, 15, 14, 15, 14, 15]  # Alternating pattern
        Autocorrelation will be NEGATIVE (breathing alternates)
        '''
        # Breathing depth variability
        diff_std = np.std(np.diff(series))
        '''
        What it calculates: How much your breathing CHANGES from one minute to the next

        Example:
        Breathing: [14, 16, 13, 15, 14]
        Differences: [2, -3, 2, -1]  (16-14, 13-16, 15-13, 14-15)
        Std of differences: ~2.16
        '''
        # Periodic breathing detection (Cheyne-Stokes like patterns)
        # Count significant oscillations
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(series, height=np.mean(series))
        valleys, _ = find_peaks(-series, height=-np.mean(series))

        oscillation_count = min(len(peaks), len(valleys))
        oscillation_ratio = oscillation_count / len(series)

        return autocorr, diff_std, oscillation_ratio
        '''
        What it detects: Waxing-waning breathing patterns (like Cheyne-Stokes)

        How it works:

        1. Find peaks: Breathing rate above average
        2. Find valleys: Breathing rate below average
        3. Count oscillations: Pairs of peaks and valleys
        4. Calculate ratio: Oscillations per minute

        Breathing: [10, 12, 15, 12, 10, 13, 16, 13, 10]
                  valley  peak  valley  peak  valley

        Peaks at: 15, 16
        Valleys at: 10, 10, 10
        Oscillations: min(2 peaks, 3 valleys) = 2
        Oscillation ratio: 2 oscillations / 9 minutes = 0.22

        Medical significance:

        1. High oscillation ratio: Possible sleep-disordered breathing
        2. Normal: Occasional oscillations during sleep transitions
        '''
    df_features[['resp_autocorr', 'resp_diff_std', 'resp_oscillation_ratio']] = pd.DataFrame(
        df_features['resp_time_series_clean'].apply(breathing_pattern_features).tolist(),
        index=df_features.index
    )

    # Sleep stage breathing patterns
    def sleep_breathing_features(series):
        if len(series) < 20:
            return np.nan, np.nan, np.nan

        # Low breathing rate periods (likely deep sleep)
        low_threshold = np.percentile(series, 25)
        slow_breathing_ratio = np.sum(series < low_threshold) / len(series)

        # High breathing rate periods (likely light/REM sleep)
        high_threshold = np.percentile(series, 75)
        fast_breathing_ratio = np.sum(series > high_threshold) / len(series)

        # Breathing stability (low variability suggests deeper sleep)
        rolling_mean = pd.Series(series).rolling(window=5, min_periods=3).mean()
        stability = 1 / (1 + np.std(rolling_mean.dropna()))
        '''
        What it calculates: How stable breathing is over 5-minute windows

        Step-by-step:

        1. Calculate 5-minute rolling average
        2. Find standard deviation of those averages
        3. Convert to stability score (higher = more stable)

        If rolling means are very consistent: [13.2, 13.1, 13.3, 13.2]
        Std is low (0.08) → stability = 1/(1+0.08) = 0.93 (high stability)

        If rolling means vary: [12, 15, 11, 16]
        Std is high (2.4) → stability = 1/(1+2.4) = 0.29 (low stability)
        '''
        return slow_breathing_ratio, fast_breathing_ratio, stability

    df_features[['resp_slow_ratio', 'resp_fast_ratio', 'resp_stability']] = pd.DataFrame(
        df_features['resp_time_series_clean'].apply(sleep_breathing_features).tolist(),
        index=df_features.index
    )

    # Apnea/Hypopnea detection (simplified)
    def detect_breathing_events(series):
        if len(series) < 30:
            return np.nan, np.nan

        # Detect significant drops in respiration (possible apneas)
        mean_resp = np.mean(series)
        std_resp = np.std(series)

        # Apnea-like events: respiration < 50% of mean for at least 2 consecutive points
        apnea_threshold = 0.5 * mean_resp
        apnea_mask = series < apnea_threshold

        # Count apnea events (consecutive points below threshold)
        apnea_events = 0
        in_apnea = False
        apnea_duration = 0

        for i in range(len(apnea_mask)):
            if apnea_mask[i]:
                apnea_duration += 1
                if apnea_duration >= 2 and not in_apnea:
                    apnea_events += 1
                    in_apnea = True
            else:
                in_apnea = False
                apnea_duration = 0

        apnea_index = apnea_events / (len(series) / 60)  # Events per hour
        '''
        Apnea Index

        What it detects: Possible breathing stoppages (apneas)

        Medical definition: Apnea = breathing stops for ≥10 seconds
        Our approximation: Breathing rate drops below 50% of average for ≥2 minutes

        Your mean breathing: ~14 breaths/min
        Apnea threshold: 0.5 × 14 = 7 breaths/min

        Look for consecutive minutes with breathing <7
        '''
        # Hypopnea-like events: respiration < 70% of mean
        hypopnea_threshold = 0.7 * mean_resp
        hypopnea_mask = series < hypopnea_threshold
        hypopnea_events = np.sum(hypopnea_mask) / 2  # Approximate count
        '''
        Hypopnea Events

        What it detects: Shallow breathing events (hypopneas)

        Difference from apnea: Breathing reduced but not stopped
        Our threshold: Breathing <70% of average

        Your data check: Looking at your values, you have some 7-9 breaths/min periods that might be flagged here.

        '''
        return apnea_index, hypopnea_events

    df_features[['apnea_index', 'hypopnea_events']] = pd.DataFrame(
        df_features['resp_time_series_clean'].apply(detect_breathing_events).tolist(),
        index=df_features.index
    )

    # Breathing-heart rate coupling (if both datasets aligned)
    def hr_resp_coupling(row):
        hr_series = row.get('hr_time_series_clean', [])
        resp_series = row.get('resp_time_series_clean', [])

        if len(hr_series) < 10 or len(resp_series) < 10:
            return np.nan

        # Resample to common length
        min_len = min(len(hr_series), len(resp_series))
        hr_aligned = hr_series[:min_len]
        resp_aligned = resp_series[:min_len]

        # Calculate correlation
        correlation = np.corrcoef(hr_aligned, resp_aligned)[0, 1]

        '''
        What it calculates: How heart rate and breathing move together

        Normal physiology: When you breathe in, heart rate increases slightly
        Example correlation: Around 0.3-0.6 is normal
        '''

        # Respiratory sinus arrhythmia (RSA) approximation
        # Heart rate should increase with inspiration, decrease with expiration
        hr_diff = np.diff(hr_aligned)
        resp_diff = np.diff(resp_aligned)

        if len(hr_diff) > 0 and len(resp_diff) > 0:
            rsa_corr = np.corrcoef(hr_diff[:len(resp_diff)], resp_diff[:len(hr_diff)])[0, 1]
        else:
            rsa_corr = np.nan
        '''
        RSA Strength

        What it calculates: Respiratory Sinus Arrhythmia - healthy heart rate variation with breathing

        Medical importance: Strong RSA = healthy autonomic nervous system
        Weak RSA: Associated with stress, poor sleep, cardiovascular issues

        How it works: Correlates CHANGES in HR with CHANGES in breathing
        '''
        return correlation, rsa_corr

    # Apply if both HR and Resp data exist
    if 'hr_time_series_clean' in df_features.columns:
        coupling_results = df_features.apply(hr_resp_coupling, axis=1)
        df_features[['hr_resp_correlation', 'rsa_strength']] = pd.DataFrame(
            coupling_results.tolist(),
            index=df_features.index
        )

    return df_features
@safe_preprocess
def combined_sleep_stage_classification(df):
    """
    Improved sleep stage classification using BOTH heart rate and respiration
    """
    df_combined = df.copy()

    def classify_with_both(row):
        hr_series = row.get('hr_time_series_no_outliers', [])
        resp_series = row.get('resp_time_series_clean', [])

        if len(hr_series) < 20 or len(resp_series) < 20:
            return np.array([])

        # Align series
        min_len = min(len(hr_series), len(resp_series))
        hr_aligned = hr_series[:min_len]
        resp_aligned = resp_series[:min_len]
        '''
        Problem: Heart rate and respiration might have different lengths
        Solution: Use only the overlapping portion of both series
        '''
        stages = []
        window_size = 5  # 5-minute windows

        for i in range(0, min_len, window_size):
            hr_window = hr_aligned[i:i+window_size]
            resp_window = resp_aligned[i:i+window_size]

            if len(hr_window) < 3 or len(resp_window) < 3:
                break

            hr_mean = np.mean(hr_window)
            hr_std = np.std(hr_window)
            resp_mean = np.mean(resp_window)
            resp_std = np.std(resp_window)

            # Combined rules
            # Deep sleep: low HR, low variability, slow steady breathing
            if (hr_mean < 45 and hr_std < 5 and
                resp_mean < 13 and resp_std < 2):
                stage = 3  # Deep sleep

            # REM sleep: higher variable HR, faster variable breathing
            elif (hr_mean > 65 and hr_std > 8 and
                  resp_mean > 15 and resp_std > 3):
                stage = 4  # REM sleep

            # Light sleep: moderate HR, moderate breathing
            elif (45 <= hr_mean <= 60 and hr_std < 8 and
                  13 <= resp_mean <= 16 and resp_std < 3):
                stage = 2  # Light sleep

            # Awake/transition
            else:
                stage = 1  # Awake/transition

            stages.extend([stage] * min(window_size, len(hr_window)))

        return np.array(stages[:min_len])

    df_combined['combined_sleep_stages'] = df_combined.apply(classify_with_both, axis=1)

    # Calculate percentages
    def calculate_combined_percentages(stages):
        if len(stages) == 0:
            return [np.nan] * 4

        unique, counts = np.unique(stages, return_counts=True)
        percentages = np.zeros(4)

        for stage, count in zip(unique, counts):
            if 1 <= stage <= 4:
                percentages[stage-1] = count / len(stages)

        return percentages

    combined_percentages = df_combined['combined_sleep_stages'].apply(calculate_combined_percentages)
    df_combined[['combined_awake_pct', 'combined_light_pct',
                 'combined_deep_pct', 'combined_rem_pct']] = pd.DataFrame(
        combined_percentages.tolist(),
        index=df_combined.index
    )

    return df_combined
@safe_preprocess
def preprocess_stress_data(df):
    """
    Preprocess Garmin stress time series data
    """
    df_processed = df.copy()

    # Parse time series
    def parse_stress_series(series_data):
        if series_data is None or (isinstance(series_data, float) and np.isnan(series_data)):
            return []

        # If already a list
        if isinstance(series_data, (list, np.ndarray)):
            return list(series_data)

        # Convert string to list
        series_str = str(series_data).strip()
        if not series_str or series_str in ['nan', 'None', '[]']:
            return []

        try:
            # Try JSON parsing
            import json
            return json.loads(series_str)
        except:
            try:
                # Try Python literal evaluation
                import ast
                return ast.literal_eval(series_str)
            except:
                # Manual parsing
                series_str = series_str.replace('[', '').replace(']', '')
                values = []
                for val in series_str.split(','):
                    val = val.strip()
                    if val:
                        try:
                            values.append(float(val))
                        except:
                            continue
                return values

    df_processed['stress_time_series_parsed'] = df_processed['stress_time_series'].apply(parse_stress_series)

    # Clean the time series
    def clean_stress_series(series):
        if series is None or len(series) == 0:
            return np.array([])

        series_array = np.array(series, dtype=float)

        # Remove NaN
        series_array = series_array[~np.isnan(series_array)]

        # Remove negative values (Garmin uses -1, -2 for "no data")
        # Keep only plausible stress values (0-100)
        mask = (series_array >= 0) & (series_array <= 100)
        series_clean = series_array[mask]

        return series_clean

    df_processed['stress_time_series_clean'] = df_processed['stress_time_series_parsed'].apply(clean_stress_series)

    return df_processed
@safe_preprocess
def extract_training_statistics(df_train):
    """Extract comprehensive training statistics"""
    stats = {}
    
    numeric_cols = df_train.select_dtypes(include=['float64', 'int64']).columns
    
    for col in numeric_cols:
        try:
            # Basic stats
            stats[f'{col}_mean'] = float(df_train[col].mean())
            stats[f'{col}_median'] = float(df_train[col].median())
            stats[f'{col}_std'] = float(df_train[col].std())
            stats[f'{col}_min'] = float(df_train[col].min())
            stats[f'{col}_max'] = float(df_train[col].max())
            
            # Percentiles
            for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
                stats[f'{col}_q{p}'] = float(df_train[col].quantile(p/100.0))
                
        except:
            continue
    
    # Also save statistics for engineered features
    engineered_features = [
        'recovery_stress_balance', 'sleep_stress_efficiency', 'hrv_stress_ratio',
        'ans_balance', 'sympathetic_tone', 'resilience_score'
    ]
    
    for feat in engineered_features:
        if feat in df_train.columns:
            stats[f'{feat}_mean'] = float(df_train[feat].mean())
            stats[f'{feat}_std'] = float(df_train[feat].std())
            stats[f'{feat}_min'] = float(df_train[feat].min())
            stats[f'{feat}_max'] = float(df_train[feat].max())
    
    return stats
@safe_preprocess
def extract_stress_features(df):
    """
    Extract meaningful features from stress time series
    """
    df_features = df.copy()

    # Basic statistical features
    df_features['stress_mean'] = df_features['stress_time_series_clean'].apply(
        lambda x: np.mean(x) if len(x) > 0 else np.nan
    )
    df_features['stress_median'] = df_features['stress_time_series_clean'].apply(
        lambda x: np.median(x) if len(x) > 0 else np.nan
    )
    df_features['stress_std'] = df_features['stress_time_series_clean'].apply(
        lambda x: np.std(x) if len(x) > 0 else np.nan
    )

    # Stress level distribution features
    def stress_distribution_features(series):
        if len(series) < 10:
            return np.nan, np.nan, np.nan, np.nan

        # Time in different stress zones
        low_stress = np.sum((series >= 0) & (series <= 25)) / len(series)
        medium_stress = np.sum((series > 25) & (series <= 50)) / len(series)
        high_stress = np.sum((series > 50) & (series <= 75)) / len(series)
        very_high_stress = np.sum(series > 75) / len(series)

        return low_stress, medium_stress, high_stress, very_high_stress

    df_features[['low_stress_pct', 'medium_stress_pct',
                 'high_stress_pct', 'very_high_stress_pct']] = pd.DataFrame(
        df_features['stress_time_series_clean'].apply(stress_distribution_features).tolist(),
        index=df_features.index
    )

    # Stress pattern features
    def stress_pattern_features(series):
        if len(series) < 20:
            return np.nan, np.nan, np.nan

        # Stress volatility (how much stress changes)
        stress_changes = np.diff(series)
        volatility = np.std(stress_changes)

        # Stress spikes (sudden increases)
        spike_threshold = np.mean(series) + 2 * np.std(series)
        stress_spikes = np.sum(series > spike_threshold) / len(series)

        # Stress recovery (ability to return to low stress)
        # Count transitions from high (>50) to low (<25)
        recovery_events = 0
        was_high = False

        for i in range(len(series)):
            if series[i] > 50:
                was_high = True
            elif series[i] < 25 and was_high:
                recovery_events += 1
                was_high = False

        recovery_rate = recovery_events / (len(series) / 60)  # Events per hour

        return volatility, stress_spikes, recovery_rate

    df_features[['stress_volatility', 'stress_spike_ratio', 'stress_recovery_rate']] = pd.DataFrame(
        df_features['stress_time_series_clean'].apply(stress_pattern_features).tolist(),
        index=df_features.index
    )

    # Sleep-specific stress features
    def sleep_stress_features(series):
        if len(series) < 30:
            return np.nan, np.nan, np.nan

        # Split night into thirds
        third_len = len(series) // 3
        if third_len == 0:
            return np.nan, np.nan, np.nan

        first_third = series[:third_len]
        middle_third = series[third_len:2*third_len]
        last_third = series[2*third_len:]

        # Calculate average stress for each third
        stress_first = np.mean(first_third) if len(first_third) > 0 else np.nan
        stress_middle = np.mean(middle_third) if len(middle_third) > 0 else np.nan
        stress_last = np.mean(last_third) if len(last_third) > 0 else np.nan

        return stress_first, stress_middle, stress_last

    df_features[['stress_first_third', 'stress_middle_third', 'stress_last_third']] = pd.DataFrame(
        df_features['stress_time_series_clean'].apply(sleep_stress_features).tolist(),
        index=df_features.index
    )

    # Stress-heart rate relationship
    def stress_hr_relationship(row):
        stress_series = row.get('stress_time_series_clean', [])
        hr_series = row.get('hr_time_series_clean', [])

        if len(stress_series) < 10 or len(hr_series) < 10:
            return np.nan, np.nan

        # Align series
        min_len = min(len(stress_series), len(hr_series))
        stress_aligned = stress_series[:min_len]
        hr_aligned = hr_series[:min_len]

        # Correlation between stress and HR
        correlation = np.corrcoef(stress_aligned, hr_aligned)[0, 1]

        # Stress efficiency: low stress with moderate HR (good recovery)
        # vs high stress with high HR (poor recovery)
        stress_hr_ratio = np.mean(stress_aligned) / np.mean(hr_aligned) if np.mean(hr_aligned) > 0 else np.nan

        return correlation, stress_hr_ratio

    if 'hr_time_series_clean' in df_features.columns:
        stress_hr_results = df_features.apply(stress_hr_relationship, axis=1)
        df_features[['stress_hr_correlation', 'stress_hr_ratio']] = pd.DataFrame(
            stress_hr_results.tolist(),
            index=df_features.index
        )

    # Stress-respiration relationship
    def stress_resp_relationship(row):
        stress_series = row.get('stress_time_series_clean', [])
        resp_series = row.get('resp_time_series_clean', [])

        if len(stress_series) < 10 or len(resp_series) < 10:
            return np.nan

        # Align series
        min_len = min(len(stress_series), len(resp_series))
        stress_aligned = stress_series[:min_len]
        resp_aligned = resp_series[:min_len]

        # Correlation between stress and respiration
        correlation = np.corrcoef(stress_aligned, resp_aligned)[0, 1]

        return correlation

    if 'resp_time_series_clean' in df_features.columns:
        df_features['stress_resp_correlation'] = df_features.apply(stress_resp_relationship, axis=1)

    return df_features
@safe_preprocess
def triple_combined_sleep_classification(df):
    """
    Ultimate sleep stage classification using ALL THREE data sources
    """
    df_triple = df.copy()

    def classify_with_all_three(row):
        hr_series = row.get('hr_time_series_no_outliers', [])
        resp_series = row.get('resp_time_series_clean', [])
        stress_series = row.get('stress_time_series_clean', [])

        # Need all three datasets
        if len(hr_series) < 20 or len(resp_series) < 20 or len(stress_series) < 20:
            return np.array([])

        # Align all three series
        min_len = min(len(hr_series), len(resp_series), len(stress_series))
        hr_aligned = hr_series[:min_len]
        resp_aligned = resp_series[:min_len]
        stress_aligned = stress_series[:min_len]

        stages = []
        window_size = 5  # 5-minute windows

        for i in range(0, min_len, window_size):
            hr_window = hr_aligned[i:i+window_size]
            resp_window = resp_aligned[i:i+window_size]
            stress_window = stress_aligned[i:i+window_size]

            if len(hr_window) < 3 or len(resp_window) < 3 or len(stress_window) < 3:
                break

            hr_mean = np.mean(hr_window)
            hr_std = np.std(hr_window)
            resp_mean = np.mean(resp_window)
            resp_std = np.std(resp_window)
            stress_mean = np.mean(stress_window)
            stress_std = np.std(stress_window)

            # TRIPLE COMBINATION RULES

            # DEEP SLEEP: Low HR, low variability, slow steady breathing, LOW stress
            if (hr_mean < 45 and hr_std < 5 and
                resp_mean < 13 and resp_std < 2 and
                stress_mean < 25):
                stage = 3  # Deep sleep

            # REM SLEEP: Variable HR, variable breathing, MODERATE-HIGH stress (dreams!)
            elif (hr_mean > 65 and hr_std > 8 and
                  resp_mean > 15 and resp_std > 3 and
                  stress_mean > 40):
                stage = 4  # REM sleep

            # LIGHT SLEEP: Moderate everything
            elif (45 <= hr_mean <= 60 and hr_std < 8 and
                  13 <= resp_mean <= 16 and resp_std < 3 and
                  25 <= stress_mean <= 50):
                stage = 2  # Light sleep

            # AWAKE/TRANSITION with HIGH STRESS
            elif stress_mean > 60:
                stage = 1  # Awake/high stress

            # AWAKE/TRANSITION (catch-all)
            else:
                stage = 1  # Awake/transition

            stages.extend([stage] * min(window_size, len(hr_window)))

        return np.array(stages[:min_len])

    df_triple['triple_sleep_stages'] = df_triple.apply(classify_with_all_three, axis=1)

    # Calculate percentages
    def calculate_triple_percentages(stages):
        if len(stages) == 0:
            return [np.nan] * 4

        unique, counts = np.unique(stages, return_counts=True)
        percentages = np.zeros(4)

        for stage, count in zip(unique, counts):
            if 1 <= stage <= 4:
                percentages[stage-1] = count / len(stages)

        return percentages

    triple_percentages = df_triple['triple_sleep_stages'].apply(calculate_triple_percentages)
    df_triple[['triple_awake_pct', 'triple_light_pct',
               'triple_deep_pct', 'triple_rem_pct']] = pd.DataFrame(
        triple_percentages.tolist(),
        index=df_triple.index
    )

    return df_triple

def calculate_score(row):
        if row['sig']:
            # For significant results: R² * (1 - p_val)
            # This rewards both high R² and low p-value
            return row['r2'] * (1 - min(row['p_val'], 0.05))  # Cap p_val at 0.05
        else:
            # Penalize non-significant results
            return row['r2'] * 0.1

def test_all_transformations(data, feature, target):
    """
    Test ALL transformations on one feature and return results
    """
    series = data[feature]
    target_series = data[target]

    # Prepare transformations
    transforms = {}

    # Original
    transforms['original'] = series

    # Basic math
    transforms['log'] = np.log1p(series)
    transforms['sqrt'] = np.sqrt(series)
    # transforms['squared'] = series ** 2
    # transforms['cubic'] = series ** 3

    if any(term in feature for term in ['SleepSeconds', 'TimeSeconds', 'Calories', 'distance']):
        # Don't use squared/cubic for large-value features
        transforms['squared'] = np.sqrt(series)  # Use sqrt instead of square!
        transforms['cubic'] = np.log1p(series)   # Use log instead of cubic!
    else:
        transforms['squared'] = series ** 2
        transforms['cubic'] = series ** 3

    # Statistical
    transforms['percentile'] = series.rank(pct=True)
    transforms['z_score'] = (series - series.mean()) / series.std()
    transforms['deviation'] = series - series.median()

    # Outlier handling
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr

    transforms['winsorized'] = series.clip(
        lower=series.quantile(0.01),
        upper=series.quantile(0.99)
    )
    transforms['iqr_capped'] = series.clip(upper=upper_bound)

    # Binning
    transforms['binned_5'] = pd.cut(series, 5, labels=False)
    transforms['binned_10'] = pd.cut(series, 10, labels=False)

    # Test each
    results = []
    for name, values in transforms.items():
        clean_vals = values.fillna(values.median())
        clean_target = target_series.fillna(target_series.median())

        if name in ['binned_5', 'binned_10']:
            corr, p_val = stats.spearmanr(clean_vals, clean_target)
        else:
            corr, p_val = stats.pearsonr(clean_vals, clean_target)

        results.append({
            'transform': name,
            'corr': corr,
            'r2': corr ** 2,
            'p_val': p_val,
            'sig': p_val < 0.05
        })

    # Create results dataframe
    df_results = pd.DataFrame(results)
    df_results['score'] = df_results.apply(calculate_score, axis=1)
    df_results = df_results.sort_values('score', ascending=False)

    return df_results, transforms

def process_all_features(data, feature_list, target_column, train_idx=None):
    """Process features using only training data for transformation fitting"""
    
    if train_idx is not None:
        train_data = data.iloc[train_idx]
    else:
        train_data = data  # Fallback
    
    processed_df = data.copy()
    
    print(f"Processing {len(feature_list)} features...")
    print(f"Target column: {target_column}")
    print("=" * 60)
    
    for i, feature in enumerate(feature_list, 1):
        print(f"[{i}/{len(feature_list)}] Processing: {feature}")
        
        try:
            # Fit transformations on TRAINING data only
            results, transforms = test_all_transformations(
                train_data,  # Only training data
                feature,
                target_column
            )
            
            # Get top 3 transformation names
            top3_names = results.iloc[:3]['transform'].tolist()
            
            # Create new columns with top 3 transformations
            for rank, trans_name in enumerate(top3_names, 1):
                col_name = f"{feature}_{rank}"
                processed_df[col_name] = transforms[trans_name]
                print(f"  Added: {col_name} ({trans_name})")
            
            # Drop the original column (OUTSIDE the inner loop!)
            if feature in processed_df.columns:
                processed_df = processed_df.drop(feature, axis=1)
                print(f"  Dropped: {feature}")
                
        except Exception as e:
            print(f"  ⚠️ Error processing {feature}: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE!")
    print(f"Original columns: {len(data.columns)}")
    print(f"New columns: {len(processed_df.columns)}")
    print(f"Total features processed: {len(feature_list)}")
    
    return processed_df

def extract_training_statistics(df_train):
    """
    Extract statistics from training data to use for test preprocessing
    """
    stats = {}
    
    print("Extracting training statistics...")
    
    # Basic statistics for ALL numeric columns (including transformed ones)
    numeric_cols = df_train.select_dtypes(include=['float64', 'int64']).columns
    
    for col in numeric_cols:
        try:
            stats[f'{col}_mean'] = float(df_train[col].mean())
            stats[f'{col}_median'] = float(df_train[col].median())
            stats[f'{col}_std'] = float(df_train[col].std())
            stats[f'{col}_q01'] = float(df_train[col].quantile(0.01))
            stats[f'{col}_q25'] = float(df_train[col].quantile(0.25))
            stats[f'{col}_q75'] = float(df_train[col].quantile(0.75))
            stats[f'{col}_q99'] = float(df_train[col].quantile(0.99))
            stats[f'{col}_min'] = float(df_train[col].min())
            stats[f'{col}_max'] = float(df_train[col].max())
        except Exception as e:
            print(f"Warning: Could not extract stats for {col}: {e}")
            continue
    
    # Special statistics for engineered features
    # Check for transformed versions of act_totalCalories
    act_cal_cols = [col for col in df_train.columns if 'act_totalCalories' in col]
    if act_cal_cols:
        # Use the first transformed column
        act_col = act_cal_cols[0]
        stats['act_totalCalories_mean'] = float(df_train[act_col].mean())
        stats['act_totalCalories_std'] = float(df_train[act_col].std())
        stats['act_totalCalories_max'] = float(df_train[act_col].max())
        stats['act_totalCalories_min'] = float(df_train[act_col].min())
        print(f"Found transformed act_totalCalories column: {act_col}")
    
    # Check for transformed versions of sleep_sleepTimeSeconds
    sleep_cols = [col for col in df_train.columns if 'sleep_sleepTimeSeconds' in col]
    if sleep_cols:
        sleep_col = sleep_cols[0]
        stats['sleep_sleepTimeSeconds_mean'] = float(df_train[sleep_col].mean())
        stats['sleep_sleepTimeSeconds_std'] = float(df_train[sleep_col].std())
        stats['sleep_sleepTimeSeconds_max'] = float(df_train[sleep_col].max())
        stats['sleep_sleepTimeSeconds_min'] = float(df_train[sleep_col].min())
        print(f"Found transformed sleep_sleepTimeSeconds column: {sleep_col}")
    
    # Original features (if they still exist)
    if 'hr_rmssd' in df_train.columns:
        stats['hr_rmssd_max'] = float(df_train['hr_rmssd'].max())
        stats['hr_rmssd_q25'] = float(df_train['hr_rmssd'].quantile(0.25))
    
    if 'hr_recovery_rate' in df_train.columns:
        stats['hr_recovery_rate_max'] = float(df_train['hr_recovery_rate'].max())
        stats['hr_recovery_rate_min'] = float(df_train['hr_recovery_rate'].min())
    
    if 'high_stress_pct' in df_train.columns:
        stats['high_stress_pct_q75'] = float(df_train['high_stress_pct'].quantile(0.75))
    
    if 'deep_sleep_ratio' in df_train.columns:
        stats['deep_sleep_ratio_q25'] = float(df_train['deep_sleep_ratio'].quantile(0.25))
    
    print(f"Extracted and saved {len(stats)} training statistics")
    return stats

def load_data(csv_path, is_training=True, transformation_metadata=None):
    df = pd.read_csv(csv_path, sep=";")
    print(f"1. After loading: {len(df.columns)} columns")
    
    # Drop time series columns and other unwanted columns
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    print(f"2. After dropping columns: {len(df.columns)} columns")
    
    # Time Series Processing
    df = simple_preprocess_heart_rate(df)
    print(f"3. After HR preprocessing: {len(df.columns)} columns")
    df = extract_hr_features(df)
    print(f"4. After HR features: {len(df.columns)} columns")
    df = resample_and_smooth_series(df, target_length=360)
    df = detect_hr_outliers(df)
    df = classify_sleep_stages(df)
    df = extract_advanced_features(df)
    
    df_resp_processed = preprocess_respiration_data(df)
    df_with_resp_features = extract_respiration_features(df_resp_processed)
    df = combined_sleep_stage_classification(df_with_resp_features)
    
    df_stress_processed = preprocess_stress_data(df)
    df_with_stress_features = extract_stress_features(df_stress_processed)
    df = triple_combined_sleep_classification(df_with_stress_features)
    
    print(f"5. After time series processing: {len(df.columns)} columns")
    
    # Select only numeric columns
    df = df.select_dtypes(include=['int', 'float'])
    print(f"6. After selecting numeric only: {len(df.columns)} columns")
    
    # Impute missing values
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
    if 'columns_to_impute' in locals():
        # Filter to only columns that actually exist
        existing_columns = [col for col in columns_to_impute if col in df.columns]
        if existing_columns:
            df[existing_columns] = df[existing_columns].fillna(
                df[existing_columns].median()
            )
    else:
        # If no specific columns, fill all NaN
        numeric_cols = df.select_dtypes(include=['int', 'float']).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Create engineered features
    df['recovery_stress_balance'] = df['hr_recovery_rate'] / (df['stress_recovery_rate'] + 1)
    df['sleep_stress_efficiency'] = df['deep_sleep_ratio'] * (1 - df['high_stress_pct'])
    df['hrv_stress_ratio'] = df['hr_rmssd'] / (df['stress_mean'] + 1)
    # ... (add ALL your engineered features here - remove duplicates!)
    
    print(f"7. After engineered features: {len(df.columns)} columns")
    
    # Your feature list - NOW includes ALL numeric columns except target and IDs
    all_numeric_cols = [col for col in df.columns 
                       if col not in [TARGET_COL, 'user_id', 'group_id'] 
                       and df[col].dtype in ['int64', 'float64']]
    
    print(f"Found {len(all_numeric_cols)} numeric features for transformation")
    print(f"Features: {all_numeric_cols}")
    
    # Process features using the CORRECT method
    if is_training:
        # Use fit_transform_features (which saves metadata)
        processed_df, transformation_metadata = fit_transform_features(
            data=df,
            feature_list=all_numeric_cols,  # Use actual numeric columns
            target_column=TARGET_COL
        )
        # Save metadata for later use on test data
        save_transformation_metadata(transformation_metadata, "transformation_metadata.pkl")
    else:
        # Load saved metadata and apply to test data
        if transformation_metadata is None:
            transformation_metadata = load_transformation_metadata("transformation_metadata.pkl")
        
        # Apply transformations using saved metadata
        processed_df = apply_transformations_to_test(
            test_data=df,
            transformation_metadata=transformation_metadata
        )
    
    print(f"8. After feature transformation: {len(processed_df.columns)} columns")
    
    # Split data
    splitter = GroupShuffleSplit(test_size=0.20, n_splits=1, random_state=42)
    train_idx, test_idx = next(splitter.split(processed_df, groups=processed_df['user_id']))
    
    train_df = processed_df.iloc[train_idx]
    test_df = processed_df.iloc[test_idx]
    
    # Drop ID columns AFTER split
    X_train = train_df.drop(columns=[TARGET_COL, 'user_id', 'group_id'])
    y_train = train_df[TARGET_COL]
    
    X_test = test_df.drop(columns=[TARGET_COL, 'user_id', 'group_id'])
    y_test = test_df[TARGET_COL]
    
    print(f"9. Final training features: {len(X_train.columns)}")
    print(f"10. Final test features: {len(X_test.columns)}")
    
    # Debug: Check feature mismatch
    if len(X_train.columns) != len(X_test.columns):
        print(f"WARNING: Feature mismatch! Train: {len(X_train.columns)}, Test: {len(X_test.columns)}")
        
        train_features = set(X_train.columns)
        test_features = set(X_test.columns)
        
        print(f"Features in train but not test: {train_features - test_features}")
        print(f"Features in test but not train: {test_features - train_features}")
    
    return X_train, y_train, X_test, y_test