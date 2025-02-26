"""
This module provides a pure Python implementation for computing signature-like features
without relying on any external libraries that might cause segmentation faults.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union, Any

def safe_compute_signature(df, level=2):
    """
    Compute signature-like features from a dataframe using only NumPy.
    This implementation is deliberately simple and robust to avoid segmentation faults.
    
    Parameters:
        df: DataFrame containing order data
        level: Truncation level for feature complexity (1 or 2)
        
    Returns:
        numpy.ndarray: Feature vector approximating signature terms
    """
    try:
        # Make sure we have a dataframe with sufficient data
        if df is None or len(df) < 2:
            print("Warning: Not enough data for signature computation")
            return np.array([1.0, 0.0, 0.0, 0.0, 0.0])  # Default minimal signature
            
        # Define the numeric columns we'll use
        numeric_cols = ['price', 'quantity']
        
        # Extract and validate numerical columns
        data = []
        for col in numeric_cols:
            if col in df.columns:
                values = df[col].values.astype(float)
                # Replace any invalid values
                values[~np.isfinite(values)] = 0.0
                data.append(values)
        
        if not data:
            print("Warning: No valid numerical columns found for signature computation")
            return np.array([1.0, 0.0, 0.0, 0.0, 0.0])  # Default minimal signature
            
        # Stack columns to get a 2D array (n_samples, n_features)
        X = np.column_stack(data)
        
        # Start building feature vector
        features = []
        
        # Level 0: constant term
        features.append(1.0)
        
        # Level 1: basic statistics for each dimension
        for i in range(X.shape[1]):
            col_data = X[:, i]
            
            # Basic statistics (safe operations)
            try:
                mean = np.mean(col_data)
                std = np.std(col_data)
                if not np.isfinite(std) or std < 1e-10:
                    std = 1.0  # Avoid division by zero
                
                # Add level 1 features
                features.append(mean)                      # Level 1: mean
                features.append(std)                       # Level 1: std
                features.append(np.min(col_data))         # Level 1: min
                features.append(np.max(col_data))         # Level 1: max
                
                # Normalized increments (similar to level 1 signature terms)
                increments = np.diff(col_data)
                features.append(np.sum(increments) / std)  # Level 1: like signature
            except Exception as e:
                print(f"Error computing level 1 features: {e}")
                # Add placeholders for failed computations
                features.extend([0.0, 1.0, 0.0, 0.0, 0.0])
        
        # Add basic categorical features 
        if 'side' in df.columns:
            try:
                buy_ratio = np.mean(df['side'] == 'buy')
                features.append(buy_ratio)
            except:
                features.append(0.5)  # Default if computation fails
        
        # Level 2: Interactions between dimensions (if requested and possible)
        if level >= 2 and X.shape[1] > 1:
            for i in range(X.shape[1]):
                for j in range(i+1, X.shape[1]):
                    try:
                        # Extract the two dimensions
                        dim_i = X[:, i]
                        dim_j = X[:, j]
                        
                        # Correlation
                        corr = np.corrcoef(dim_i, dim_j)[0, 1]
                        features.append(corr if np.isfinite(corr) else 0.0)
                        
                        # Level 2 signature-like term: sum of increment products
                        # This is the simplest proxy for second level sig terms
                        inc_i = np.diff(dim_i)
                        inc_j = np.diff(dim_j)
                        level2_term = np.sum(inc_i * inc_j)
                        
                        # Normalize to avoid extreme values
                        std_i = np.std(dim_i)
                        std_j = np.std(dim_j)
                        if std_i > 1e-10 and std_j > 1e-10:
                            level2_term /= (std_i * std_j)
                            
                        features.append(level2_term if np.isfinite(level2_term) else 0.0)
                    except Exception as e:
                        print(f"Error computing level 2 features: {e}")
                        # Add placeholders for failed level 2 computations
                        features.extend([0.0, 0.0])
        
        # Ensure all features are finite
        features = np.array(features, dtype=np.float32)
        features[~np.isfinite(features)] = 0.0
        
        return features
        
    except Exception as e:
        print(f"Safe signature computation failed with error: {e}")
        # Return a minimal valid feature vector as fallback
        return np.array([1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)
