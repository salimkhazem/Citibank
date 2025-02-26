import numpy as np
import pandas as pd
from typing import Dict, Optional, Any

from utils.data_generation import adjust_orders_cointegration

def generate_orders_from_baseline(original_df, num_orders, hyperparams=None, seed=None):
    """
    Generate synthetic orders using a simple baseline model.
    
    For continuous features (quantity and price), fits a Gaussian model.
    For categorical features (date, time, symbol, side), samples from the empirical distribution.
    
    Parameters:
        original_df: Original orders DataFrame to learn from
        num_orders: Number of orders to generate
        hyperparams: Dictionary of hyperparameters (optional)
        seed: Random seed for reproducibility
        
    Returns:
        pd.DataFrame: Generated orders
    """
    try:
        if seed is not None:
            np.random.seed(seed)
        
        # Fit Gaussian for quantity and price
        quantity_mean = original_df['quantity'].mean()
        quantity_std = original_df['quantity'].std()
        price_mean = original_df['price'].mean()
        price_std = original_df['price'].std()
        
        # Generate from fitted distributions
        gen_quantity = np.random.normal(quantity_mean, quantity_std, num_orders)
        gen_price = np.random.normal(price_mean, price_std, num_orders)
        
        # Sample categorical fields from the original distribution
        gen_date = np.random.choice(original_df['date'], size=num_orders)
        gen_time = np.random.choice(original_df['time'], size=num_orders)
        gen_symbol = np.random.choice(original_df['symbol'], size=num_orders)
        gen_side = np.random.choice(original_df['side'], size=num_orders)
        
        # Sample operations (add, modify, delete) from the original data
        operations = original_df[['add', 'modify', 'delete']].values
        op_indices = np.random.choice(len(operations), size=num_orders)
        gen_ops = operations[op_indices]
        gen_add = gen_ops[:, 0]
        gen_modify = gen_ops[:, 1]
        gen_delete = gen_ops[:, 2]
        
        # Create DataFrame with generated data
        df = pd.DataFrame({
            'date': gen_date,
            'time': gen_time,
            'symbol': gen_symbol,
            'quantity': gen_quantity,
            'price': gen_price,
            'side': gen_side,
            'add': gen_add,
            'modify': gen_modify,
            'delete': gen_delete
        })
        
        # Apply cointegration adjustment if requested
        if hyperparams and hyperparams.get('cointegration_adjustment', False):
            margin = hyperparams.get('cointegration_margin', 0.05)
            prob_adj = hyperparams.get('cointegration_prob', 0.8)
            df = adjust_orders_cointegration(df, original_df, margin, prob_adj)
            
        return df
        
    except Exception as e:
        print("Error in generate_orders_from_baseline:", e)
        return None
