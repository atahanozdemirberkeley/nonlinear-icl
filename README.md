# In-Context Learning for Sinusoidal Functions

This repository contains an implementation of a transformer-based model for in-context learning (ICL) of sinusoidal functions.

## Configuration-Based Training and Evaluation

Training and evaluation are based on configuration files located in the `configs/` directory. This allows for better reproducibility and easier experimentation.

### Available Configurations

- `configs/default.yaml`: Default configuration
- `configs/sinusoidal.yaml`: Configuration for sinusoidal function tasks
- `configs/linear.yaml`: Configuration for linear function tasks

### Training

To train a model using a specific configuration file:

```bash
python src/train.py --config configs/sinusoidal.yaml
```

You can override parameters while keeping other configuration options:

```bash
python src/train.py --config configs/sinusoidal.yaml --freq_min 0.2 --freq_max 5.0
```

### Evaluation

To evaluate a trained model:

```bash
python src/eval.py --config configs/sinusoidal.yaml --model_path experiments/sinusoidal_d1_s1.0_x5.0_f0.5-2.0/[timestamp]/best_model.pt
```

## Available Tasks

- `sinusoidal`: Sinusoidal functions (y = A*sin(ωx + φ))
- `linear`: Linear functions
- `quadratic`: Quadratic functions
- `relu`: ReLU network functions

## Model Architecture

The model uses a GPT2-style transformer architecture to perform in-context learning. The key components are:

1. Input embeddings to project the input-output pairs to a higher-dimensional space
2. Transformer layers (12 layers, 256 embedding dimension, 8 attention heads)
3. Output projection to produce predictions

## Evaluating In-Context Learning Performance

The evaluation compares the transformer model against several baselines:
- 1-NN: 1-nearest neighbor
- 3-NN: 3-nearest neighbors
- Linear Regression: Least squares model

Results are summarized and visualized to show how the model improves with more context examples.

## 📊 Research Question

Can transformers learn to predict sinusoidal functions through in-context learning? How does performance vary with sinusoidal parameters like frequency and amplitude?

This investigation builds upon the framework from [Garg et al. (2022)](https://arxiv.org/pdf/2208.01066), focusing specifically on sinusoidal functions.

## Key Sinusoidal Parameters

- **Amplitude (A)**: Controls the height of the sine wave, sampled from Uniform(0.5, 1.5) * scale
- **Frequency (ω)**: Controls the periodicity, sampled from Uniform(freq_min, freq_max)
- **Phase (φ)**: Controls the horizontal shift, sampled from Uniform(0, 2π)
- **x_range**: Range of x values, uniformly sampled from (-x_range, x_range)

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/prob-icl.git
cd prob-icl

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

```bash
# Train with sinusoidal configuration
python src/train.py --config configs/sinusoidal.yaml

# Train with custom parameters
python src/train.py --config configs/sinusoidal.yaml --batch_size 128 --steps 10000
```

## 📁 Project Structure

```
prob-icl/
├── src/
│   ├── train.py         # Main training script
│   ├── model.py         # Transformer model implementations
│   ├── tasks.py         # Task implementations (sinusoidal, linear, etc.)
│   ├── eval.py          # Evaluation utilities
│   └── config.py        # Configuration management
├── configs/             # Configuration files
│   └── sinusoidal.yaml  # Sinusoidal task configuration
└── experiments/         # Training outputs (created during training)
    └── sinusoidal_d1_s1.0_x5.0_f0.5-2.0/ # Experiment directories
```

## 📝 Methodology

Our approach:

1. **Generate Sinusoidal Functions**: We use parameterized sine waves with different amplitudes, frequencies, and phases
2. **In-Context Learning**: The transformer learns to predict function values based on context examples
3. **Evaluation**: We compare transformer performance with nearest-neighbor and linear regression baselines

## 📈 Results

The key metrics we analyze:

- MSE between transformer predictions and true sinusoidal values
- How performance varies with frequency ranges and input ranges
- Impact of context length on prediction accuracy

## 🔍 Parameter Selection

- **Frequency range**: Controls the complexity of sinusoidal functions (freq_min to freq_max)
- **Input range**: Range of x values (-x_range to x_range)
- **Context length**: Number of examples provided for in-context learning (n_positions)
- **Amplitude scale**: Scale factor for the amplitude of sine waves

## 📚 Related Work

- [In-context Learning and Induction Heads](https://arxiv.org/pdf/2208.01066) (Garg et al., 2022)
- [Transformers Learn In-Context by Gradient Descent](https://arxiv.org/abs/2212.07677) (Hahn et al., 2022)

## 🤝 Contributing

Contributions welcome! Please feel free to submit a Pull Request.