import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import copy
from typing import Dict, Optional, Any

from utils.encoding import encode_orders, decode_orders, prepare_sequence_data, compute_order_signature
from utils.data_generation import adjust_orders_cointegration

class LSTMGenerator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, use_signature=False, sig_dim=0):
        super(LSTMGenerator, self).__init__()
        self.use_signature = use_signature
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        fc_in = hidden_size + (sig_dim if use_signature else 0)
        self.fc = nn.Linear(fc_in, output_size)
        
    def forward(self, x, global_sig=None):
        out, _ = self.lstm(x)  # out: (batch, seq_len, hidden_size)
        out = out[:, -1, :]    # get final states only: (batch, hidden_size)
        
        if self.use_signature and global_sig is not None:
            out = torch.cat([out, global_sig], dim=1)
            
        out = self.fc(out)
        return out

def generate_orders_from_lstm(original_df, num_orders, hyperparams, seed=None, global_sig_override=None):
    """
    Generate synthetic orders using an LSTM model, optionally with signature conditioning.
    
    Parameters:
        original_df: Original orders DataFrame to learn from
        num_orders: Number of orders to generate
        hyperparams: Dictionary of hyperparameters
        seed: Random seed for reproducibility
        global_sig_override: Optional override for the signature conditioning
        
    Returns:
        pd.DataFrame: Generated orders
    """
    try:
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        print("Starting LSTM order generation...")
        
        # Make a defensive copy of dataframe to prevent any modifications to original
        df_copy = original_df.copy(deep=True)
            
        symbols = [f"SEC{i}" for i in range(1,11)]
        print("Encoding orders...")
        data = encode_orders(df_copy, symbols)
        
        seq_len = hyperparams.get('seq_len', 10)
        print(f"Preparing sequence data with length {seq_len}...")
        X, Y = prepare_sequence_data(data, seq_len)
        
        if X.size == 0 or Y.size == 0:
            raise ValueError("Empty sequence data")
            
        feature_dim = data.shape[1]
        hidden_size = hyperparams.get('num_units', 32)
        epochs = hyperparams.get('epochs', 10)
        batch_size = hyperparams.get('batch_size', 32)
        learning_rate = hyperparams.get('learning_rate', 0.001)
        use_sig = hyperparams.get('use_signature', False)
        
        sig = None
        sig_dim = 0
        
        if use_sig:
            # Try to compute the signature with error handling
            try:
                print("Computing signature for conditioning...")
                # Import here to ensure we're using the latest implementation
                from utils.safe_compute_signature import safe_compute_signature
                
                # Use the safe implementation directly
                sig = safe_compute_signature(df_copy, level=2)
                
                # Check if we got a valid signature
                if sig is None or sig.size == 0:
                    print("Warning: Signature computation returned empty array, disabling signature conditioning")
                    use_sig = False
                else:
                    print(f"Signature computed successfully with dimension: {sig.size}")
                    sig_dim = sig.size
                    
                    if global_sig_override is not None:
                        print("Using provided signature override")
                        sig = global_sig_override
                        sig_dim = len(sig)
                        
                    sig = torch.tensor(sig, dtype=torch.float32)
                    # Sanity check the tensor
                    if not torch.isfinite(sig).all():
                        print("Warning: Signature contains non-finite values, replacing with zeros")
                        sig[~torch.isfinite(sig)] = 0.0
            except Exception as e:
                print(f"Error in signature computation: {e}")
                print("Disabling signature conditioning")
                use_sig = False
        
        # Prepare training data
        print("Preparing training data...")
        X_tensor = torch.tensor(X, dtype=torch.float32)
        Y_tensor = torch.tensor(Y, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, Y_tensor)
        dataloader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)
        
        # Initialize model
        print(f"Initializing LSTM model (hidden_size={hidden_size}, use_signature={use_sig})...")
        model = LSTMGenerator(
            input_size=feature_dim, 
            hidden_size=hidden_size, 
            output_size=feature_dim,
            use_signature=use_sig, 
            sig_dim=sig_dim
        )
        
        # Train model
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        print(f"Training model for {epochs} epochs...")
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                
                try:
                    if use_sig:
                        bs = batch_x.size(0)
                        global_sig_batch = sig.unsqueeze(0).expand(bs, -1)
                        output = model(batch_x, global_sig_batch)
                    else:
                        output = model(batch_x)
                        
                    loss = criterion(output, batch_y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                except Exception as e:
                    print(f"Error in training iteration: {e}")
                    continue
            
            if (epoch + 1) % max(1, epochs // 5) == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/max(1,len(dataloader)):.6f}")
        
        # Generation phase
        print("Generating synthetic orders...")
        seed_seq = torch.tensor(data[-seq_len:], dtype=torch.float32)
        generated = []
        
        model.eval()
        with torch.no_grad():
            for i in range(num_orders):
                inp = seed_seq.unsqueeze(0)  # add batch dimension
                
                try:
                    if use_sig:
                        global_sig_batch = sig.unsqueeze(0)
                        pred = model(inp, global_sig_batch)
                    else:
                        pred = model(inp)
                    
                    # Check for NaN/Inf values
                    if not torch.isfinite(pred).all():
                        print(f"Warning: Non-finite prediction at step {i}, replacing with zeros")
                        pred = torch.zeros_like(pred)
                        
                    pred_np = pred.squeeze(0).cpu().numpy()
                    generated.append(pred_np)
                    
                    # Update seed sequence for next prediction
                    seed_seq = torch.cat([seed_seq[1:], pred], dim=0)
                except Exception as e:
                    print(f"Error in generation step {i}: {e}")
                    # If generation fails, add a copy of the last successful generated item or a zero vector
                    if len(generated) > 0:
                        generated.append(generated[-1])
                    else:
                        generated.append(np.zeros(feature_dim))
                
                # Print progress
                if (i + 1) % max(1, num_orders // 10) == 0:
                    print(f"Generated {i+1}/{num_orders} orders")
        
        print("Decoding generated orders...")
        gen_matrix = np.stack(generated)
        gen_df = decode_orders(gen_matrix, df_copy, symbols)
        
        # Apply cointegration adjustment if requested
        if hyperparams.get('cointegration_adjustment', False):
            margin = hyperparams.get('cointegration_margin', 0.05)
            prob_adj = hyperparams.get('cointegration_prob', 0.8)
            try:
                print("Applying cointegration adjustment...")
                gen_df = adjust_orders_cointegration(gen_df, df_copy, margin, prob_adj)
            except Exception as e:
                print(f"Error in cointegration adjustment: {e}")
        
        print("LSTM order generation complete!")
        return gen_df
        
    except Exception as e:
        print(f"Error in generate_orders_from_lstm: {e}")
        import traceback
        traceback.print_exc()
        return None
