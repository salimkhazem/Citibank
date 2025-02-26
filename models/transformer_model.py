import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Dict, Optional, Any

from utils.encoding import encode_orders, decode_orders, prepare_sequence_data, compute_order_signature
from utils.data_generation import adjust_orders_cointegration

class TransformerGenerator(nn.Module):
    def __init__(self, input_size, num_units, num_heads, num_layers=1, use_signature=False, sig_dim=0):
        super(TransformerGenerator, self).__init__()
        self.use_signature = use_signature
        self.input_proj = nn.Linear(input_size, num_units)
        encoder_layer = nn.TransformerEncoderLayer(d_model=num_units, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        fc_in = num_units + (sig_dim if use_signature else 0)
        self.fc = nn.Linear(fc_in, input_size)

    def forward(self, x, global_sig=None):
        x = self.input_proj(x)               # (batch, seq_len, num_units)
        x = self.transformer_encoder(x)      # (batch, seq_len, num_units)
        x = x[:, -1, :]                      # take the output at the last time-step: (batch, num_units)
        
        if self.use_signature and global_sig is not None:
            x = torch.cat([x, global_sig], dim=1)
            
        x = self.fc(x)
        return x

def generate_orders_from_transformer(original_df, num_orders, hyperparams, seed=None, global_sig_override=None):
    """
    Generate synthetic orders using a Transformer model, optionally with signature conditioning.
    
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
            
        symbols = [f"SEC{i}" for i in range(1,11)]
        data = encode_orders(original_df, symbols)
        
        seq_len = hyperparams.get('seq_len', 10)
        X, Y = prepare_sequence_data(data, seq_len)
        
        if X.size == 0 or Y.size == 0:
            raise ValueError("Empty sequence data")
            
        feature_dim = data.shape[1]
        num_units = hyperparams.get('num_units', 32)
        num_heads = hyperparams.get('num_heads', 2)
        num_layers = hyperparams.get('num_layers', 1)
        epochs = hyperparams.get('epochs', 10)
        batch_size = hyperparams.get('batch_size', 32)
        learning_rate = hyperparams.get('learning_rate', 0.001)
        use_sig = hyperparams.get('use_signature', False)
        
        sig = None
        sig_dim = 0
        
        if use_sig:
            # Compute or use the signature for conditioning
            sig = compute_order_signature(original_df, level=2)
            sig_dim = len(sig)
            
            if global_sig_override is not None:
                sig = global_sig_override
                
            sig = torch.tensor(sig, dtype=torch.float32)
            
        # Prepare training data
        X_tensor = torch.tensor(X, dtype=torch.float32)
        Y_tensor = torch.tensor(Y, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, Y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model
        model = TransformerGenerator(
            input_size=feature_dim, 
            num_units=num_units, 
            num_heads=num_heads,
            num_layers=num_layers,
            use_signature=use_sig, 
            sig_dim=sig_dim
        )
        
        # Train model
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        model.train()
        for epoch in range(epochs):
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                
                if use_sig:
                    bs = batch_x.size(0)
                    global_sig_batch = sig.unsqueeze(0).expand(bs, -1)
                    output = model(batch_x, global_sig_batch)
                else:
                    output = model(batch_x)
                    
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
        
        # Generation phase
        seed_seq = torch.tensor(data[-seq_len:], dtype=torch.float32)
        generated = []
        
        model.eval()
        with torch.no_grad():
            for _ in range(num_orders):
                inp = seed_seq.unsqueeze(0)  # add batch dimension
                
                if use_sig:
                    global_sig_batch = sig.unsqueeze(0)
                    pred = model(inp, global_sig_batch)
                else:
                    pred = model(inp)
                    
                pred_np = pred.squeeze(0).cpu().numpy()
                generated.append(pred_np)
                
                # Update seed sequence for next prediction
                seed_seq = torch.cat([seed_seq[1:], pred], dim=0)
                
        gen_matrix = np.stack(generated)
        gen_df = decode_orders(gen_matrix, original_df, symbols)
        
        # Apply cointegration adjustment if requested
        if hyperparams.get('cointegration_adjustment', False):
            margin = hyperparams.get('cointegration_margin', 0.05)
            prob_adj = hyperparams.get('cointegration_prob', 0.8)
            gen_df = adjust_orders_cointegration(gen_df, original_df, margin, prob_adj)
            
        return gen_df
        
    except Exception as e:
        print("Error in generate_orders_from_transformer:", e)
        return None