import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional, Any

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
from experiments import run_experiments

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Order Generation Framework')
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'experiments'],
                        help='Mode: "single" for one model or "experiments" for all experiments')
    parser.add_argument('--model', type=str, default='baseline',
                        choices=['baseline', 'lstm', 'transformer', 'hmm', 'gan', 'signature_vae'],
                        help='Model to use for generating orders')
    parser.add_argument('--num-orders', type=int, default=1000,
                        help='Number of orders to generate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path (CSV) for generated orders')
    parser.add_argument('--use-signature', action='store_true',
                        help='Whether to use signature conditioning (for applicable models)')
    parser.add_argument('--cointegration', action='store_true',
                        help='Whether to apply cointegration adjustment')
                        
    return parser.parse_args()

def get_model_function(model_name):
    """Map model name to the corresponding generation function."""
    model_mapping = {
        'baseline': generate_orders_from_baseline,
        'lstm': generate_orders_from_lstm,
        'transformer': generate_orders_from_transformer,
        'hmm': generate_orders_from_hmm,
        'gan': generate_orders_from_gan,
        'signature_vae': generate_orders_from_signature_vae
    }
    return model_mapping.get(model_name.lower())

def get_default_hyperparams(model_name, use_signature=False, use_cointegration=False):
    """Get default hyperparameters for specified model."""
    base_params = {
        'baseline': {
            'cointegration_adjustment': use_cointegration,
            'cointegration_margin': 0.05,
            'cointegration_prob': 0.8
        },
        'lstm': {
            'seq_len': 10,
            'num_units': 64,
            'epochs': 10,
            'batch_size': 32,
            'learning_rate': 0.001,
            'use_signature': use_signature,
            'cointegration_adjustment': use_cointegration,
            'cointegration_margin': 0.05,
            'cointegration_prob': 0.8
        },
        'transformer': {
            'seq_len': 10,
            'num_units': 64,
            'num_heads': 4,
            'num_layers': 2,
            'epochs': 10,
            'batch_size': 32,
            'learning_rate': 0.001,
            'use_signature': use_signature,
            'cointegration_adjustment': use_cointegration,
            'cointegration_margin': 0.05,
            'cointegration_prob': 0.8
        },
        'hmm': {
            'n_components': 4,
            'covariance_type': 'full',
            'cointegration_adjustment': use_cointegration,
            'cointegration_margin': 0.05,
            'cointegration_prob': 0.8
        },
        'gan': {
            'noise_dim': 100,
            'hidden_dim': 128,
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 0.001,
            'use_signature': use_signature,
            'cointegration_adjustment': use_cointegration,
            'cointegration_margin': 0.05,
            'cointegration_prob': 0.8
        },
        'signature_vae': {
            'segment_length': 100,
            'latent_dim': 10,
            'hidden_dim_vae': 64,
            'epochs_vae': 50,
            'batch_size_vae': 16,
            'seq_len': 5,
            'num_units': 32,
            'epochs': 5,
            'batch_size': 32,
            'learning_rate': 0.001,
            'use_signature': True,  # Always true for SignatureVAE
            'cointegration_adjustment': use_cointegration,
            'cointegration_margin': 0.05,
            'cointegration_prob': 0.8
        }
    }
    return base_params.get(model_name.lower(), {})

def run_single_model(args):
    """Run a single model with specified parameters."""
    print(f"Generating {args.num_orders} orders using {args.model} model...")
    
    try:
        # Generate original data
        original_df = generate_random_orders(
            num_orders=args.num_orders,
            distribution='normal',
            serial_corr=False,
            seed=args.seed,
            buy_prob=0.5,
            side_corr_lag=5,
            side_corr_prob=0.0
        )
        
        # Get model function and hyperparameters
        model_func = get_model_function(args.model)
        if model_func is None:
            print(f"Error: Model '{args.model}' not found")
            return
        
        hyperparams = get_default_hyperparams(
            args.model, 
            use_signature=args.use_signature,
            use_cointegration=args.cointegration
        )
        
        # Print additional debug info when using signatures
        if args.use_signature:
            print("Using signature conditioning. Computing test signature...")
            try:
                # Use our safe implementation directly
                from utils.safe_compute_signature import safe_compute_signature
                
                # Test on just a small subset of data
                sig = safe_compute_signature(original_df.iloc[:min(100, len(original_df))], level=1)
                print(f"Test signature computed successfully with dimension: {len(sig)}")
                
                # Validate the signature
                if not np.all(np.isfinite(sig)):
                    print("Warning: Test signature contains non-finite values")
                    sig[~np.isfinite(sig)] = 0.0
                    
            except Exception as e:
                print(f"Warning: Test signature computation failed with error: {e}")
                print("Continuing with fallback mechanisms...")
        
        # Generate orders using the specified model with timeout protection (if platform supports it)
        try:
            import signal
            timeout_supported = True
        except ImportError:
            timeout_supported = False
            
        if timeout_supported:
            def timeout_handler(signum, frame):
                raise TimeoutError("Model execution timed out")
            
            # Set timeout of 300 seconds (5 minutes)
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(300)
            
            try:
                generated_df = model_func(original_df, args.num_orders, hyperparams, seed=args.seed)
            except TimeoutError as e:
                print(f"Error: {e}")
                print("Model took too long to execute. Try with fewer orders or different parameters.")
                return None
            finally:
                # Restore old signal handler
                signal.signal(signal.SIGALRM, old_handler)
                signal.alarm(0)
        else:
            # No timeout support, just run directly
            generated_df = model_func(original_df, args.num_orders, hyperparams, seed=args.seed)
        
        if generated_df is None:
            print(f"Error: Failed to generate orders with {args.model}")
            return None
        
        # Evaluate the generated orders
        metrics = evaluate_generated_orders(original_df, generated_df)
        print("\nEvaluation Metrics:")
        for metric, value in metrics.items():
            try:
                print(f"  {metric}: {value:.4f}")
            except:
                print(f"  {metric}: {value}")
        
        # Save to file if specified
        if args.output:
            generated_df.to_csv(args.output, index=False)
            print(f"Generated orders saved to {args.output}")
        
        # Plot comparison
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].hist(original_df['quantity'], bins=20, alpha=0.6, label='Original')
            axes[0].hist(generated_df['quantity'], bins=20, alpha=0.6, label='Generated')
            axes[0].set_title(f"{args.model.upper()} - Quantity")
            axes[0].legend()
            
            axes[1].hist(original_df['price'], bins=20, alpha=0.6, label='Original')
            axes[1].hist(generated_df['price'], bins=20, alpha=0.6, label='Generated')
            axes[1].set_title(f"{args.model.upper()} - Price")
            axes[1].legend()
            
            plt.suptitle(f"Order Generation using {args.model.upper()}")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error displaying plots: {e}")
        
        return generated_df, metrics
    
    except Exception as e:
        print(f"Unexpected error in run_single_model: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main entry point for the application."""
    args = parse_arguments()
    
    if args.mode == 'experiments':
        print("Running all experiments...")
        results_df = run_experiments()
        if args.output and results_df is not None:
            results_df.to_csv(args.output, index=False)
            print(f"Experiment results saved to {args.output}")
    else:
        run_single_model(args)

if __name__ == "__main__":
    main()
