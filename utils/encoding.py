import numpy as np
import pandas as pd
import datetime
from typing import List, Dict, Tuple, Optional, Union, Any

# Import the safe signature implementation
from .safe_compute_signature import safe_compute_signature

def compute_order_signature(df, level=2, epsilon=1e-6):
    """
    Compute the signature of orders. This is a wrapper around the safe implementation.
    
    Parameters:
        df: DataFrame of orders
        level: Level of signature truncation
        epsilon: Small value to avoid numerical instability
        
    Returns:
        numpy array: Signature features vector or empty array if computation fails
    """
    try:
        # Use the safe implementation
        return safe_compute_signature(df, level)
    except Exception as e:
        print(f"Signature computation error: {e}")
        # Return minimum viable signature on failure
        return np.array([1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)

def encode_order(row, symbols):
    """
    Encode a single order row into a numeric vector.
    
    Parameters:
        row: Single DataFrame row representing an order
        symbols: List of symbol strings for one-hot encoding
        
    Returns:
        np.ndarray: Encoded order vector
    """
    try:
        vec = []
        t = row['time']
        if isinstance(t, str):
            try:
                t = datetime.datetime.strptime(t, "%H:%M:%S").time()
            except Exception as e:
                t = datetime.time(0,0,0)
        vec.append(t.hour*3600 + t.minute*60 + t.second)
        
        # One-hot encoding for symbol
        one_hot = [1 if row['symbol'] == s else 0 for s in symbols]
        vec.extend(one_hot)
        
        vec.append(row['quantity'])
        vec.append(row['price'])
        vec.append(1 if row['side']=='buy' else 0)
        vec.append(row['add'])
        vec.append(row['modify'])
        vec.append(row['delete'])
        
        return np.array(vec, dtype=np.float32)
        
    except Exception as e:
        print("Error in encode_order:", e)
        return np.zeros(17, dtype=np.float32)

def encode_orders(df, symbols=None):
    """
    Encode all orders in a DataFrame into a matrix.
    
    Parameters:
        df: DataFrame of orders
        symbols: List of symbol strings (default generates SEC1-SEC10)
        
    Returns:
        np.ndarray: Matrix of encoded orders
    """
    try:
        if symbols is None:
            symbols = [f"SEC{i}" for i in range(1,11)]
        encoded = df.apply(lambda row: encode_order(row, symbols), axis=1)
        return np.stack(encoded.to_numpy())
        
    except Exception as e:
        print("Error in encode_orders:", e)
        return np.zeros((len(df), 17), dtype=np.float32)

def decode_order(vector, original_df, symbols):
    """
    Decode a numeric vector back into an order.
    
    Parameters:
        vector: Encoded order vector
        original_df: Reference DataFrame for sampling dates
        symbols: List of symbols for decoding
        
    Returns:
        dict: Order attributes
    """
    try:
        seconds = vector[0]
        hour = int(seconds // 3600) % 24
        minute = int((seconds % 3600) // 60)
        second = int(seconds % 60)
        time_str = f"{hour:02d}:{minute:02d}:{second:02d}"
        
        sym_idx = np.argmax(vector[1:11])
        symbol = symbols[sym_idx]
        
        quantity = vector[11]
        price = vector[12]
        side = 'buy' if vector[13]>=0.5 else 'sell'
        add = 1 if vector[14]>=0.5 else 0
        modify = 1 if vector[15]>=0.5 else 0
        delete = 1 if vector[16]>=0.5 else 0
        
        return {'time': time_str, 'symbol': symbol, 'quantity': quantity,
                'price': price, 'side': side, 'add': add, 'modify': modify, 'delete': delete}
                
    except Exception as e:
        print("Error in decode_order:", e)
        return {'time': "00:00:00", 'symbol': symbols[0], 'quantity': 0, 'price': 0,
                'side':'buy', 'add':0, 'modify':0, 'delete':0}

def decode_orders(matrix, original_df, symbols=None):
    """
    Decode a matrix of encoded orders back into a DataFrame.
    
    Parameters:
        matrix: Matrix of encoded orders
        original_df: Reference DataFrame for sampling dates
        symbols: List of symbols (default generates SEC1-SEC10)
        
    Returns:
        pd.DataFrame: DataFrame of decoded orders
    """
    try:
        if symbols is None:
            symbols = [f"SEC{i}" for i in range(1,11)]
        orders = []
        dates = original_df['date'].to_numpy()
        
        for vec in matrix:
            order = decode_order(vec, original_df, symbols)
            order['date'] = np.random.choice(dates)
            orders.append(order)
            
        return pd.DataFrame(orders)
        
    except Exception as e:
        print("Error in decode_orders:", e)
        return pd.DataFrame()

def prepare_sequence_data(data, seq_len):
    """
    Prepare sequence data for training recurrent models.
    
    Parameters:
        data: Input data matrix
        seq_len: Sequence length for inputs
        
    Returns:
        tuple: (X, Y) where X contains input sequences and Y contains target values
    """
    try:
        if len(data) <= seq_len:
            raise ValueError("Not enough data to prepare sequence")
        X, Y = [], []
        for i in range(len(data)-seq_len):
            X.append(data[i:i+seq_len])
            Y.append(data[i+seq_len])
        return np.array(X), np.array(Y)
        
    except Exception as e:
        print("Error in prepare_sequence_data:", e)
        return np.array([]), np.array([])
