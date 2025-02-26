import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from typing import Dict, Optional, Any

from utils.encoding import compute_order_signature
from .lstm_model import generate_orders_from_lstm

class VAEEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super(VAEEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x):
        h = torch.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class VAEDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dim):
        super(VAEDecoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, z):
        h = torch.relu(self.fc1(z))
        out = self.fc_out(h)
        return out

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super(VAE, self).__init__()
        self.encoder = VAEEncoder(input_dim, latent_dim, hidden_dim)
        self.decoder = VAEDecoder(latent_dim, input_dim, hidden_dim)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

def generate_orders_from_signature_vae(original_df, num_orders, hyperparams, seed=None):
    """
    Generate synthetic orders using a Signature VAE model.
    
    Splits original_df into segments, computes the signature for each,
    trains a VAE on these signatures, and then samples a synthetic global signature.
    Finally, uses the synthetic signature to condition an LSTM generator.
    
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
            torch.manual_seed(seed)
            
        # Extract hyperparameters for the VAE
        seg_len = hyperparams.get('segment_length', 100)
        latent_dim = hyperparams.get('latent_dim', 10)
        hidden_dim_vae = hyperparams.get('hidden_dim_vae', 64)
        epochs_vae = hyperparams.get('epochs_vae', 50)
        batch_size_vae = hyperparams.get('batch_size_vae', 16)
        
        print("Splitting data into segments...")
        # Split original_df into segments (non-overlapping)
        segments = []
        for i in range(0, len(original_df), seg_len):
            segment = original_df.iloc[i:i+seg_len].copy()
            if len(segment) == seg_len:
                segments.append(segment)
        
        if len(segments) == 0:
            raise ValueError("Not enough data to create segments")
        
        print(f"Created {len(segments)} segments of length {seg_len}")
        
        # Compute signatures for each segment
        print("Computing signatures for each segment...")
        sig_list = []
        
        # First, test signature computation on one segment
        try:
            test_sig = compute_order_signature(segments[0], level=2)
            if test_sig.size == 0:
                print("Warning: Test signature computation failed, using simplified features")
                # Use simplified feature computation if signature fails
                for i, seg in enumerate(segments):
                    features = []
                    # Simple numerical features
                    for col in ['price', 'quantity']:
                        if col in seg.columns:
                            features.extend([
                                seg[col].mean(),
                                seg[col].std(),
                                seg[col].min(),
                                seg[col].max()
                            ])
                    # Side distribution
                    if 'side' in seg.columns:
                        buy_ratio = (seg['side'] == 'buy').mean()
                        features.append(buy_ratio)
                    
                    # Make sure we have some features
                    if len(features) > 0:
                        sig_list.append(np.array(features))
            else:
                # Test was successful, compute signatures for all segments
                print(f"Test signature successful with dimension: {len(test_sig)}")
                sig_list.append(test_sig)  # Add the test signature
                
                # Compute for remaining segments
                for i, seg in enumerate(segments[1:], 1):
                    try:
                        sig = compute_order_signature(seg, level=2)
                        if sig.size > 0:
                            sig_list.append(sig)
                        else:
                            print(f"Warning: Failed to compute signature for segment {i}, skipping")
                    except Exception as e:
                        print(f"Error computing signature for segment {i}: {e}")
                        continue
        except Exception as e:
            print(f"Error in test signature computation: {e}")
            print("Falling back to LSTM without signatures")
            new_hyperparams = hyperparams.copy()
            new_hyperparams['use_signature'] = False
            return generate_orders_from_lstm(original_df, num_orders, new_hyperparams, seed=seed)
                
        if len(sig_list) == 0:
            print("Error: No valid signatures could be computed, falling back to LSTM without signature")
            # Fall back to standard LSTM
            new_hyperparams = hyperparams.copy()
            new_hyperparams['use_signature'] = False
            return generate_orders_from_lstm(original_df, num_orders, new_hyperparams, seed=seed)
            
        # Find the minimum dimension of signatures (they could have different lengths)
        min_dim = min(sig.shape[0] for sig in sig_list)
        
        # Truncate all signatures to this minimum dimension
        sig_list = [sig[:min_dim] for sig in sig_list]
        sig_data = np.stack(sig_list)  # shape: (num_segments, min_dim)
        input_dim = sig_data.shape[1]
        
        print(f"Training VAE on {len(sig_list)} signatures with dimension {input_dim}")
        
        # Initialize the VAE model
        vae = VAE(input_dim, latent_dim, hidden_dim_vae)
        optimizer = optim.Adam(vae.parameters(), lr=0.001)
        
        # Prepare data for training
        dataset = torch.tensor(sig_data, dtype=torch.float32)
        vae_dataloader = DataLoader(dataset, batch_size=batch_size_vae, shuffle=True)
        
        # Train the VAE
        vae.train()
        for epoch in range(epochs_vae):
            total_loss = 0
            for batch in vae_dataloader:
                optimizer.zero_grad()
                recon, mu, logvar = vae(batch)
                
                # Loss = reconstruction loss + KL divergence
                recon_loss = nn.MSELoss()(recon, batch)
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + kl_loss
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            if epoch % max(1, epochs_vae // 5) == 0:  # Print only a few updates
                print(f"VAE Epoch {epoch+1}/{epochs_vae}, Loss: {total_loss/len(vae_dataloader):.6f}")
        
        print("Generating synthetic signature...")
        # Generate synthetic signature
        vae.eval()
        with torch.no_grad():
            # Sample from the latent space
            z_sample = torch.randn(1, latent_dim)
            # Decode to get a synthetic signature
            synthetic_sig = vae.decoder(z_sample).squeeze(0).cpu().numpy()
        
        print("Using LSTM with synthetic signature for final generation...")
        # Use LSTM generator with the synthetic signature
        # We require that the hyperparameters include 'use_signature': True
        new_hyperparams = hyperparams.copy()
        new_hyperparams['use_signature'] = True
        
        gen_df = generate_orders_from_lstm(
            original_df, 
            num_orders, 
            new_hyperparams, 
            seed=seed,
            global_sig_override=synthetic_sig
        )
        
        return gen_df
        
    except Exception as e:
        print(f"Error in generate_orders_from_signature_vae: {e}")
        import traceback
        traceback.print_exc()
        
        # Fall back to LSTM without signature
        try:
            new_hyperparams = hyperparams.copy()
            new_hyperparams['use_signature'] = False
            print("Falling back to LSTM without signature due to error")
            return generate_orders_from_lstm(original_df, num_orders, new_hyperparams, seed=seed)
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            return None
