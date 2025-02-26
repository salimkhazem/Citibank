import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Dict, Optional, Any

from utils.encoding import encode_orders, decode_orders, compute_order_signature
from utils.data_generation import adjust_orders_cointegration

class GANGenerator(nn.Module):
    def __init__(self, noise_dim, hidden_dim, output_dim, use_signature=False, sig_dim=0):
        super(GANGenerator, self).__init__()
        input_dim = noise_dim + (sig_dim if use_signature else 0)
        self.use_signature = use_signature
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, z):
        return self.model(z)

class GANDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GANDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

def generate_orders_from_gan(original_df, num_orders, hyperparams, seed=None, global_sig_override=None):
    """
    Generate synthetic orders using a GAN model, optionally with signature conditioning.
    
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
        data_tensor = torch.tensor(data, dtype=torch.float32)
        input_dim = data.shape[1]
        
        noise_dim = hyperparams.get('noise_dim', 100)
        hidden_dim = hyperparams.get('hidden_dim', 64)
        epochs = hyperparams.get('epochs', 100)
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
            
        # Initialize models
        generator = GANGenerator(noise_dim, hidden_dim, input_dim, use_signature=use_sig, sig_dim=sig_dim)
        discriminator = GANDiscriminator(input_dim, hidden_dim)

        # Loss and optimizers
        criterion = nn.BCELoss()
        optim_G = optim.Adam(generator.parameters(), lr=learning_rate)
        optim_D = optim.Adam(discriminator.parameters(), lr=learning_rate)

        # Prepare dataset
        dataset = torch.utils.data.TensorDataset(data_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Training loop
        for epoch in range(epochs):
            for real_batch, in dataloader:
                batch_size_curr = real_batch.size(0)
                real_labels = torch.ones(batch_size_curr, 1)
                fake_labels = torch.zeros(batch_size_curr, 1)

                # Train discriminator
                optim_D.zero_grad()
                outputs_real = discriminator(real_batch)
                loss_real = criterion(outputs_real, real_labels)

                # Generate fake data
                z = torch.randn(batch_size_curr, noise_dim)
                if use_sig:
                    global_sig_batch = sig.unsqueeze(0).expand(batch_size_curr, -1)
                    z = torch.cat([z, global_sig_batch], dim=1)
                
                fake_data = generator(z)
                outputs_fake = discriminator(fake_data.detach())
                loss_fake = criterion(outputs_fake, fake_labels)

                loss_D = loss_real + loss_fake
                loss_D.backward()
                optim_D.step()

                # Train generator
                optim_G.zero_grad()
                outputs_fake = discriminator(fake_data)
                loss_G = criterion(outputs_fake, real_labels)
                loss_G.backward()
                optim_G.step()

        # Generate orders
        generator.eval()
        with torch.no_grad():
            z = torch.randn(num_orders, noise_dim)
            if use_sig:
                global_sig_batch = sig.unsqueeze(0).expand(num_orders, -1)
                z = torch.cat([z, global_sig_batch], dim=1)
            
            gen_data = generator(z).cpu().numpy()
            
        gen_df = decode_orders(gen_data, original_df, symbols)
        
        # Apply cointegration adjustment if requested
        if hyperparams.get('cointegration_adjustment', False):
            margin = hyperparams.get('cointegration_margin', 0.05)
            prob_adj = hyperparams.get('cointegration_prob', 0.8)
            gen_df = adjust_orders_cointegration(gen_df, original_df, margin, prob_adj)
            
        return gen_df
        
    except Exception as e:
        print("Error in generate_orders_from_gan:", e)
        return None
