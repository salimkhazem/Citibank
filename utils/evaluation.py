import numpy as np
import pandas as pd
import datetime
from scipy.stats import wasserstein_distance
from typing import Dict, Union, Any, Optional
from .encoding import compute_order_signature

def compute_wasserstein_distance(original_df, gen_df, column):
    """
    Compute the 1D Wasserstein distance between a column in the original and generated orders.
    
    Parameters:
        original_df: Original orders DataFrame
        gen_df: Generated orders DataFrame
        column: Column name to compare
        
    Returns:
        float: Wasserstein distance
    """
    try:
        return wasserstein_distance(original_df[column], gen_df[column])
    except Exception as e:
        print("Error in compute_wasserstein_distance:", e)
        return np.nan

def compute_total_variation_distance(orig_series, gen_series):
    """
    Compute the Total Variation (TV) distance between two categorical distributions.
    
    Parameters:
        orig_series: Original categorical series
        gen_series: Generated categorical series
        
    Returns:
        float: Total variation distance (between 0 and 1)
    """
    try:
        orig_counts = orig_series.value_counts(normalize=True).sort_index()
        gen_counts = gen_series.value_counts(normalize=True).sort_index()
        all_categories = set(orig_counts.index).union(set(gen_counts.index))
        orig_probs = np.array([orig_counts.get(cat, 0) for cat in all_categories])
        gen_probs = np.array([gen_counts.get(cat, 0) for cat in all_categories])
        tv_distance = 0.5 * np.sum(np.abs(orig_probs - gen_probs))
        return tv_distance
    except Exception as e:
        print("Error in compute_total_variation_distance:", e)
        return np.nan

def compute_side_lag_correlation(df, lag=5):
    """
    Compute the correlation between current and lagged side values for each symbol.
    
    Parameters:
        df: Orders DataFrame
        lag: Lag value for correlation
        
    Returns:
        float: Average correlation across symbols
    """
    try:
        correlations = []
        for sym in df['symbol'].unique():
            symbol_data = df[df['symbol'] == sym]
            side_numeric = symbol_data['side'].map({'buy': 1, 'sell': 0}).to_numpy()
            if len(side_numeric) > lag:
                correlations.append(np.mean(side_numeric[:-lag] == side_numeric[lag:]))
        return np.mean(correlations) if correlations else np.nan
    except Exception as e:
        print("Error in compute_side_lag_correlation:", e)
        return np.nan

def compute_lead_lag_distance(orig_df, gen_df):
    """
    Compute a lead–lag distance metric as the Euclidean distance between
    the vectors of average order times (in seconds) per security.
    
    Parameters:
        orig_df: Original orders DataFrame
        gen_df: Generated orders DataFrame
        
    Returns:
        float: Lead-lag distance metric
    """
    try:
        def time_to_seconds(t):
            if isinstance(t, str):
                try:
                    t_obj = datetime.datetime.strptime(t, "%H:%M:%S").time()
                except Exception as e:
                    t_obj = datetime.time(0,0,0)
                return t_obj.hour*3600 + t_obj.minute*60 + t_obj.second
            else:
                return t.hour*3600 + t.minute*60 + t.second

        symbols = sorted(orig_df['symbol'].unique())
        orig_means = []
        gen_means = []
        for sym in symbols:
            orig_times = orig_df[orig_df['symbol'] == sym]['time'].apply(time_to_seconds)
            gen_times = gen_df[gen_df['symbol'] == sym]['time'].apply(time_to_seconds)
            if len(orig_times) == 0 or len(gen_times) == 0:
                continue
            orig_means.append(orig_times.mean())
            gen_means.append(gen_times.mean())
        orig_vec = np.array(orig_means)
        gen_vec = np.array(gen_means)
        return np.linalg.norm(orig_vec - gen_vec)
    except Exception as e:
        print("Error in compute_lead_lag_distance:", e)
        return np.nan

def evaluate_generated_orders(original_df, gen_df, side_lag=5):
    """
    Evaluate generated orders against the original orders using multiple metrics.
    
    Parameters:
        original_df: Original orders DataFrame
        gen_df: Generated orders DataFrame
        side_lag: Lag for side correlation metric
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    try:
        metrics = {}
        # Wasserstein distances for continuous variables
        metrics['wasserstein_quantity'] = compute_wasserstein_distance(original_df, gen_df, 'quantity')
        metrics['wasserstein_price'] = compute_wasserstein_distance(original_df, gen_df, 'price')
        
        # Total variation distances for categorical variables
        metrics['tv_symbol'] = compute_total_variation_distance(original_df['symbol'], gen_df['symbol'])
        metrics['tv_side'] = compute_total_variation_distance(original_df['side'], gen_df['side'])
        
        # Signature-based distance - use safer implementation
        try:
            from utils.safe_compute_signature import safe_compute_signature
            orig_sig = safe_compute_signature(original_df, level=2)
            gen_sig = safe_compute_signature(gen_df, level=2)
            
            # Make sure both signatures have the same length
            min_len = min(len(orig_sig), len(gen_sig))
            orig_sig = orig_sig[:min_len]
            gen_sig = gen_sig[:min_len]
            
            # Check for NaN or Inf values
            mask = np.isfinite(orig_sig) & np.isfinite(gen_sig)
            if np.any(mask) and min_len > 0:
                metrics['signature_distance'] = np.linalg.norm(orig_sig[mask] - gen_sig[mask])
            else:
                metrics['signature_distance'] = np.nan
        except Exception as e:
            print(f"Error computing signature distance: {e}")
            metrics['signature_distance'] = np.nan
        
        # Side lag correlation
        metrics['side_lag_corr_original'] = compute_side_lag_correlation(original_df, lag=side_lag)
        metrics['side_lag_corr_generated'] = compute_side_lag_correlation(gen_df, lag=side_lag)
        metrics['side_lag_corr_diff'] = abs(metrics['side_lag_corr_original'] - metrics['side_lag_corr_generated'])
        
        # Lead-lag distance
        metrics['lead_lag_distance'] = compute_lead_lag_distance(original_df, gen_df)
        
        # Overall combined metric
        valid_metrics = []
        for key, value in metrics.items():
            if key != 'overall' and np.isfinite(value):
                valid_metrics.append(value)
                
        if valid_metrics:
            metrics['overall'] = sum(valid_metrics)
        else:
            metrics['overall'] = np.nan
        
        return metrics
    except Exception as e:
        print("Error in evaluate_generated_orders:", e)
        return {}
