# Order Generation Framework

This repository contains a framework for generating synthetic trading orders using various generative models. The framework employs several techniques, including signature-based methods, to ensure the synthetic orders maintain the statistical properties of the original data.
This repository in inspired by : https://github.com/thegallier/timeseries/blob/main/ordergeneration.ipynb

## Features

- Multiple generative models implemented (LSTM, Transformer, GAN, HMM, etc.)
- Signature-based conditioning for improved fidelity
- Cointegration adjustment for realistic price-side relationships
- Comprehensive evaluation metrics for model comparison
- Hyperparameter tuning capabilities

## Structure

- `main.py`: Entry point script to run experiments
- `models/`: Implementation of different generative models
- `utils/`: Utility functions for data processing, evaluation, etc.
- `experiments.py`: Definition of experimental setups

## Requirements

```
numpy
pandas
torch
scipy
matplotlib
esig
hmmlearn
```

## Usage

```bash
# Install requirements
pip install -r requirements.txt

# Run the main script
python main.py
```

### Run with specific model

```bash
# Run a specific model (baseline, lstm, transformer, hmm, gan, signature_vae)
python main.py --model lstm --num-orders 1000 --use-signature

# Apply cointegration adjustment
python main.py --model transformer --num-orders 1000 --cointegration

# Run all experiments
python main.py --mode experiments --output results.csv
```

## Models

- **Baseline**: Simple generative model that fits Gaussians to continuous features
- **LSTM**: Long Short-Term Memory network for sequence modeling
- **Transformer**: Attention-based architecture for sequence modeling
- **HMM**: Hidden Markov Model for capturing hidden states in order patterns
- **GAN**: Generative Adversarial Network for mimicking the distribution
- **SignatureVAE**: VAE trained on rough-path signatures for enhanced conditioning

## Example Output

The framework will output a summary table comparing different models across several evaluation metrics:

- Wasserstein distance for continuous features
- Total variation distance for categorical features
- Signature distance for path-level similarity
- Lead-lag correlation preservation
- Overall combined metric

## License

MIT
