import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from typing import Dict, Optional, Any

from utils.encoding import encode_orders, decode_orders
from utils.data_generation import adjust_orders_cointegration

def generate_orders_from_hmm(original_df, num_orders, hyperparams, seed=None):
    """
    Generate synthetic orders using a Gaussian HMM model.
    
    Parameters:
        original_df: Original orders DataFrame to learn from
        num_orders: Number of orders to generate
        hyperparams: Dictionary of hyperparameters
        seed: Random seed for reproducibility
        
    Returns:
        pd.DataFrame: Generated orders
    """
    try:
        if seed is not None:
            np.random.seed(seed)
            
        # Define the list of symbols
        symbols = [f"SEC{i}" for i in range(1, 11)]
        
        # Encode the orders into a numeric array
        data = encode_orders(original_df, symbols)
        
        # Extract hyperparameters for the HMM
        n_components = hyperparams.get('n_components', 4)
        cov_type = hyperparams.get('covariance_type', 'full')
        
        # Initialize and fit the HMM model
        model = GaussianHMM(
            n_components=n_components, 
            covariance_type=cov_type,
            random_state=seed, 
            n_iter=100
        )
        model.fit(data)
        
        # Sample a new sequence of orders from the HMM
        gen_matrix, _ = model.sample(num_orders)
        
        # Decode the generated numeric data back into a DataFrame
        gen_df = decode_orders(gen_matrix, original_df, symbols)
        
        # Apply cointegration adjustment if requested
        if hyperparams.get('cointegration_adjustment', False):
            margin = hyperparams.get('cointegration_margin', 0.05)
            prob_adj = hyperparams.get('cointegration_prob', 0.8)
            gen_df = adjust_orders_cointegration(gen_df, original_df, margin, prob_adj)
            
        return gen_df
        
    except Exception as e:
        print("Error in generate_orders_from_hmm:", e)
        return None
