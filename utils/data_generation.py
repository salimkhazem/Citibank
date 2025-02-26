import numpy as np
import pandas as pd
import datetime
from typing import List, Dict, Tuple, Optional, Union, Any

def generate_random_orders(
    num_orders=1000, 
    distribution='normal', 
    serial_corr=False, 
    seed=None,
    buy_prob=0.5, 
    side_corr_lag=5, 
    side_corr_prob=0.0,
    symbol_probs=None
):
    """
    Generate a random orders DataFrame with columns:
      date, time, symbol, quantity, price, side, add, modify, delete.
    Includes stochastic lead–lag features via side correlation.
    
    Parameters:
      num_orders (int): Number of orders to generate.
      distribution (str): Distribution for quantity/price. Options: 'normal', 'uniform', 'exponential'.
      serial_corr (bool): If True, simulate serial correlation in quantity and price.
      seed (int): Optional seed for reproducibility.
      buy_prob (float): Probability of generating a buy order.
      side_corr_lag (int): Lag for side correlation.
      side_corr_prob (float): Probability of using lagged side value.
      symbol_probs (list): Probability distribution for symbols.
      
    Returns:
      pd.DataFrame: Generated orders.
    """
    try:
        if seed is not None:
            np.random.seed(seed)
            
        start_date = pd.Timestamp('2025-01-01')
        symbols = [f"SEC{i}" for i in range(1, 11)]
        
        if symbol_probs is None:
            symbol_probs = [1/10] * 10
        else:
            total = sum(symbol_probs)
            symbol_probs = [p/total for p in symbol_probs]
            
        dates, times, syms = [], [], []
        quantities, prices = [], []
        sides = []
        adds, modifies, deletes = [], [], []
        
        if serial_corr:
            q_prev, p_prev, rho = 100.0, 50.0, 0.8
            
        last_sides = {sym: [] for sym in symbols}
        
        for i in range(num_orders):
            rand_day = np.random.randint(0, 30)
            current_date = start_date + pd.Timedelta(days=rand_day)
            dates.append(current_date.date())
            
            seconds_in_day = np.random.randint(0, 24*60*60)
            tdelta = pd.to_timedelta(seconds_in_day, unit='s')
            times.append((pd.Timestamp("2025-01-01") + tdelta).time())
            
            sym = np.random.choice(symbols, p=symbol_probs)
            syms.append(sym)
            
            # Handle side correlation
            if len(last_sides[sym]) >= side_corr_lag and np.random.rand() < side_corr_prob:
                side = last_sides[sym][-side_corr_lag]
            else:
                side = 'buy' if np.random.rand() < buy_prob else 'sell'
            sides.append(side)
            last_sides[sym].append(side)
            
            # Generate quantity and price
            if serial_corr:
                if distribution == 'normal':
                    noise_q = np.random.normal(0, 5)
                    noise_p = np.random.normal(0, 1)
                elif distribution == 'uniform':
                    noise_q = np.random.uniform(-5, 5)
                    noise_p = np.random.uniform(-1, 1)
                elif distribution == 'exponential':
                    noise_q = np.random.exponential(5) - 5
                    noise_p = np.random.exponential(1) - 1
                else:
                    noise_q = np.random.normal(0, 5)
                    noise_p = np.random.normal(0, 1)
                    
                q = rho * q_prev + (1 - rho) * noise_q
                p = rho * p_prev + (1 - rho) * noise_p
                q_prev, p_prev = q, p
            else:
                if distribution == 'normal':
                    q = np.random.normal(100, 10)
                    p = np.random.normal(50, 5)
                elif distribution == 'uniform':
                    q = np.random.uniform(80, 120)
                    p = np.random.uniform(45, 55)
                elif distribution == 'exponential':
                    q = np.random.exponential(100)
                    p = np.random.exponential(50)
                else:
                    q = np.random.normal(100, 10)
                    p = np.random.normal(50, 5)
                    
            quantities.append(q)
            prices.append(p)
            
            # Randomly assign operation type
            op = np.random.choice(['add', 'modify', 'delete'])
            adds.append(1 if op=='add' else 0)
            modifies.append(1 if op=='modify' else 0)
            deletes.append(1 if op=='delete' else 0)
            
        df = pd.DataFrame({
            'date': dates,
            'time': times,
            'symbol': syms,
            'quantity': quantities,
            'price': prices,
            'side': sides,
            'add': adds,
            'modify': modifies,
            'delete': deletes
        })
        return df
        
    except Exception as e:
        print("Error in generate_random_orders:", e)
        return pd.DataFrame()

def adjust_orders_cointegration(orders_df, original_df, margin=0.05, prob_adjust=0.8):
    """
    Adjust the side of generated orders based on cointegration:
    For each security, if its generated price is below (1 - margin)*avg_price,
    then set side to 'buy' with probability prob_adjust; if above (1+margin)*avg_price,
    set side to 'sell' with probability prob_adjust.
    
    Parameters:
        orders_df (pd.DataFrame): Generated orders to adjust
        original_df (pd.DataFrame): Original orders to compute average prices from
        margin (float): Margin around average price for adjustment threshold
        prob_adjust (float): Probability of making the adjustment
        
    Returns:
        pd.DataFrame: Adjusted orders
    """
    try:
        avg_prices = original_df.groupby('symbol')['price'].mean().to_dict()
        adjusted = orders_df.copy()
        
        for idx, row in adjusted.iterrows():
            sym = row['symbol']
            avg = avg_prices.get(sym, np.nan)
            
            if np.isnan(avg):
                continue
                
            price = row['price']
            if price < avg * (1 - margin) and np.random.rand() < prob_adjust:
                adjusted.at[idx, 'side'] = 'buy'
            elif price > avg * (1 + margin) and np.random.rand() < prob_adjust:
                adjusted.at[idx, 'side'] = 'sell'
                
        return adjusted
        
    except Exception as e:
        print("Error in adjust_orders_cointegration:", e)
        return orders_df
