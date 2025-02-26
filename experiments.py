import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple

from utils.data_generation import generate_random_orders
from utils.evaluation import evaluate_generated_orders
from models import (
    generate_orders_from_lstm,
    generate_orders_from_transformer,
    generate_orders_from_gan,
    generate_orders_from_hmm,
    generate_orders_from_baseline,
    generate_orders_from_signature_vae
)

# Define hyperparameter grids for each model
HP_GRID_LSTM = {
    'seq_len': [5, 10],
    'num_units': [32, 64],
    'epochs': [5, 10],
    'batch_size': [32],
    'learning_rate': [0.001, 0.0005],
    'use_signature': [True, False],
    'cointegration_adjustment': [True, False],
    'cointegration_margin': [0.05],
    'cointegration_prob': [0.8]
}

HP_GRID_TRANSFORMER = {
    'seq_len': [5, 10],
    'num_units': [32, 64],
    'num_heads': [2, 4],
    'num_layers': [1, 2],
    'epochs': [5, 10],
    'batch_size': [32],
    'learning_rate': [0.001, 0.0005],
    'use_signature': [True, False],
    'cointegration_adjustment': [True, False],
    'cointegration_margin': [0.05],
    'cointegration_prob': [0.8]
}

HP_GRID_HMM = {
    'n_components': [2, 4, 6],
    'covariance_type': ['full', 'diag'],
    'cointegration_adjustment': [True, False],
    'cointegration_margin': [0.05],
    'cointegration_prob': [0.8]
}

HP_GRID_GAN = {
    'noise_dim': [100],
    'hidden_dim': [64, 128],
    'epochs': [50, 100],
    'batch_size': [32],
    'learning_rate': [0.001, 0.0005],
    'use_signature': [True, False],
    'cointegration_adjustment': [True, False],
    'cointegration_margin': [0.05],
    'cointegration_prob': [0.8]
}

HP_GRID_BASELINE = {
    'cointegration_adjustment': [True, False],
    'cointegration_margin': [0.05],
    'cointegration_prob': [0.8]
}

HP_GRID_SIGNATURE_VAE = {
    'segment_length': [100],
    'latent_dim': [10],
    'hidden_dim_vae': [64],
    'epochs_vae': [50],
    'batch_size_vae': [16],
    # Also pass LSTM hyperparameters for generation:
    'seq_len': [5],
    'num_units': [32],
    'epochs': [5],
    'batch_size': [32],
    'learning_rate': [0.001],
    'cointegration_adjustment': [True],
    'cointegration_margin': [0.05],
    'cointegration_prob': [0.8]
}

def tune_model(model_func, original_df, num_orders, hyperparams_grid, seed=None):
    """
    Tune hyperparameters for a model by trying different combinations.
    
    Parameters:
        model_func: Function that generates orders
        original_df: Original orders DataFrame
        num_orders: Number of orders to generate
        hyperparams_grid: Grid of hyperparameter values to try
        seed: Random seed for reproducibility
        
    Returns:
        tuple: (best_params, best_metrics, best_generated_df)
    """
    best_score = np.inf
    best_params = None
    best_metrics = None
    best_generated = None
    
    try:
        if not hyperparams_grid:
            try:
                gen_df = model_func(original_df, num_orders, {}, seed=seed)
                metrics = evaluate_generated_orders(original_df, gen_df)
                return {}, metrics, gen_df
            except Exception as e:
                print(f"Error in model_func call with empty hyperparameters: {e}")
                return {}, None, None
        
        keys, values = zip(*hyperparams_grid.items())
        for combination in itertools.product(*values):
            params = dict(zip(keys, combination))
            try:
                gen_df = model_func(original_df, num_orders, params, seed=seed)
                if gen_df is None:
                    raise ValueError("Generated DataFrame is None")
                metrics = evaluate_generated_orders(original_df, gen_df)
                score = metrics.get('overall', np.inf)
                if score < best_score:
                    best_score = score
                    best_params = params
                    best_metrics = metrics
                    best_generated = gen_df
            except Exception as e:
                print(f"Error with parameters {params}: {e}")
        
        if best_metrics is None:
            print(f"All hyperparameter combinations failed for model {model_func.__name__}")
            return {}, None, None
            
        return best_params, best_metrics, best_generated
        
    except Exception as e:
        print(f"Error in tune_model: {e}")
        return {}, None, None

def run_experiments():
    """
    Run experiments comparing different order generation models.
    """
    # Define experiments with different data generation parameters
    experiments = [
        {
            'name': 'Exp1: Baseline balanced, uniform symbols, no side corr',
            'distribution': 'normal',
            'serial_corr': False,
            'buy_prob': 0.5,
            'side_corr_prob': 0.0,
            'symbol_probs': [0.1]*10
        },
        {
            'name': 'Exp2: More buys, uniform symbols',
            'distribution': 'normal',
            'serial_corr': False,
            'buy_prob': 0.7,
            'side_corr_prob': 0.0,
            'symbol_probs': [0.1]*10
        },
        {
            'name': 'Exp3: Skewed symbols, balanced buys',
            'distribution': 'normal',
            'serial_corr': False,
            'buy_prob': 0.5,
            'side_corr_prob': 0.0,
            'symbol_probs': [0.3, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05]
        },
        {
            'name': 'Exp4: Strong side correlation',
            'distribution': 'normal',
            'serial_corr': False,
            'buy_prob': 0.5,
            'side_corr_prob': 0.8,
            'symbol_probs': [0.1]*10
        },
        {
            'name': 'Exp5: High buys + serial corr in qty/price',
            'distribution': 'normal',
            'serial_corr': True,
            'buy_prob': 0.7,
            'side_corr_prob': 0.8,
            'symbol_probs': [0.1]*10
        }
    ]
    
    # Define models to test
    models = [
        {'name': 'Baseline', 'func': generate_orders_from_baseline, 'hp_grid': HP_GRID_BASELINE},
        {'name': 'LSTM', 'func': generate_orders_from_lstm, 'hp_grid': HP_GRID_LSTM},
        {'name': 'Transformer', 'func': generate_orders_from_transformer, 'hp_grid': HP_GRID_TRANSFORMER},
        {'name': 'HMM', 'func': generate_orders_from_hmm, 'hp_grid': HP_GRID_HMM},
        {'name': 'GAN', 'func': generate_orders_from_gan, 'hp_grid': HP_GRID_GAN},
        {'name': 'SignatureVAE', 'func': generate_orders_from_signature_vae, 'hp_grid': HP_GRID_SIGNATURE_VAE}
    ]
    
    num_orders = 1000   # Number of orders for original and generated data
    side_lag = 5        # For computing side lag correlation
    plot_results = False  # Set True to display graphs
    
    summary_results = []
    
    for exp in experiments:
        print(f"\n--- {exp['name']} ---")
        try:
            orig_df = generate_random_orders(
                num_orders=num_orders,
                distribution=exp['distribution'],
                serial_corr=exp['serial_corr'],
                seed=42,
                buy_prob=exp['buy_prob'],
                side_corr_lag=side_lag,
                side_corr_prob=exp['side_corr_prob'],
                symbol_probs=exp['symbol_probs']
            )
        except Exception as e:
            print(f"Error generating original orders for {exp['name']}: {e}")
            continue
            
        for model_info in models:
            model_name = model_info['name']
            func = model_info['func']
            hp_grid = model_info['hp_grid']
            
            print(f"Running model: {model_name}")
            best_hp, metrics, gen_df = tune_model(func, orig_df, num_orders, hp_grid, seed=101)
            
            if metrics is None or gen_df is None:
                print(f"Model {model_name} failed for experiment {exp['name']}. Skipping...")
                continue
                
            result = {
                'Experiment': exp['name'],
                'Model': model_name,
                'Hyperparameters': best_hp,
                'Wasserstein_quantity': metrics.get('wasserstein_quantity', np.nan),
                'Wasserstein_price': metrics.get('wasserstein_price', np.nan),
                'TV_symbol': metrics.get('tv_symbol', np.nan),
                'TV_side': metrics.get('tv_side', np.nan),
                'Signature_distance': metrics.get('signature_distance', np.nan),
                'Side_lag_corr_diff': metrics.get('side_lag_corr_diff', np.nan),
                'Overall_metric': metrics.get('overall', np.nan)
            }
            
            summary_results.append(result)
            print(f"  Best HP: {best_hp}")
            print(f"  Metrics: {metrics}")
            
            if plot_results:
                try:
                    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                    axes[0].hist(orig_df['quantity'], bins=20, alpha=0.6, label='Original')
                    axes[0].hist(gen_df['quantity'], bins=20, alpha=0.6, label='Generated')
                    axes[0].set_title(f"{model_name} - Quantity")
                    axes[0].legend()
                    
                    axes[1].hist(orig_df['price'], bins=20, alpha=0.6, label='Original')
                    axes[1].hist(gen_df['price'], bins=20, alpha=0.6, label='Generated')
                    axes[1].set_title(f"{model_name} - Price")
                    axes[1].legend()
                    
                    plt.suptitle(exp['name'])
                    plt.tight_layout()
                    plt.show()
                except Exception as e:
                    print(f"Error during plotting: {e}")
    
    if summary_results:
        summary_df = pd.DataFrame(summary_results)
        print("\nSummary of Experiments and Model Performance:")
        print(summary_df.sort_values(by=['Experiment', 'Overall_metric']).to_string(index=False))
        return summary_df
    else:
        print("No successful experiments to summarize.")
        return None

if __name__ == "__main__":
    results_df = run_experiments()
