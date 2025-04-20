# Kernel-ICL: In-Context Learning for Probability Distributions

This repository contains an implementation of a transformer-based model for in-context learning (ICL) of probability distributions.

## Configuration-Based Training and Evaluation

Training and evaluation are based on configuration files located in the `configs/` directory. This allows for better reproducibility and easier experimentation.

### Available Configurations

- `configs/default.yaml`: Default configuration
- `configs/gaussian.yaml`: Configuration for Gaussian distribution tasks
- `configs/gamma.yaml`: Configuration for Gamma distribution tasks (more complex)

### Training

To train a model using a specific configuration file:

```bash
python src/train.py --config configs/gaussian.yaml
```

You can override the task name while keeping other configuration options:

```bash
python src/train.py --config configs/gaussian.yaml --task_name poisson
```

### Evaluation

To evaluate a trained model:

```bash
python src/eval.py --config configs/gaussian.yaml --model_path outputs/[timestamp]/best_model.pt
```

You can evaluate on a different task than what the model was trained on:

```bash
python src/eval.py --config configs/gaussian.yaml --model_path outputs/[timestamp]/best_model.pt --task_name exponential
```

## Available Tasks

- `gaussian`: Gaussian distribution
- `poisson`: Poisson distribution
- `bernoulli`: Bernoulli distribution
- `exponential`: Exponential distribution
- `gamma`: Gamma distribution

## Model Architecture

The model uses a transformer architecture based on GPT-2 to perform in-context learning. The key components are:

1. Input embeddings to project the input-output pairs to a higher-dimensional space
2. Positional encoding to maintain sequence order
3. Transformer layers for contextual processing
4. Output projection to produce distribution parameter predictions

## Evaluating In-Context Learning Performance

The evaluation compares the transformer model against several baselines:
- 1-NN: 1-nearest neighbor
- 3-NN: 3-nearest neighbors
- Averaging: Simple averaging of previous outputs

Results are summarized and visualized to show how the model improves with more context examples.

## 📊 Research Question

Can transformers implicitly approximate kernelized models such as RBF-kernel ridge regression purely through in-context learning, without explicit kernel feature computation?

This investigation builds upon the framework from [Garg et al. (2022)](https://arxiv.org/pdf/2208.01066), extending their function class from linear to kernelized (non-linear) functions.

## 🧩 Key Components

- **Random Fourier Features (RFF)**: Used to approximate RBF kernels, bridging classical kernel theory with modern transformer models
- **In-context Learning Transformer**: A transformer model that learns to predict function values from context examples
- **Kernel Ridge Regression**: Serves as a baseline for comparison

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/kernel-icl.git
cd kernel-icl

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

```bash
# Train with default settings
python src/train.py

# Train with a specific configuration
python src/train.py --config configs/default.yaml
```

## 📁 Project Structure

```
kernel-icl/
├── src/
│   ├── train.py         # Main training script
│   ├── model.py         # Transformer model implementations
│   ├── kernels.py       # RFF implementations and kernel methods
│   ├── data.py          # Dataset creation utilities
│   ├── eval.py          # Evaluation utilities
│   └── config.py        # Configuration management
├── configs/             # Configuration files
│   └── default.yaml     # Default configuration
└── outputs/             # Training outputs (created during training)
    └── YYYYMMDD_HHMMSS/ # Timestamped experiment directories
```

## 📝 Methodology

Our approach:

1. **Generate RFF Functions**: We use Random Fourier Features to approximate RBF kernels
2. **In-Context Learning**: The transformer learns to predict function values based on context examples
3. **Evaluation**: We compare transformer performance with kernel ridge regression baselines

## 📈 Results

The key metrics we analyze:

- MSE between transformer predictions and true function values
- Comparison with kernel ridge regression at various regularization strengths
- How performance varies with kernel parameters (e.g., σ)

## 🔍 Parameter Selection

- **RFF dimension**: Controls the approximation quality of the kernel
- **Sigma**: Controls the smoothness of the generated functions
- **Context length**: Number of examples provided for in-context learning
- **Input dimension**: Dimension of the input space

## 📚 Related Work

- [In-context Learning and Induction Heads](https://arxiv.org/pdf/2208.01066) (Garg et al., 2022)
- [Transformers Learn In-Context by Gradient Descent](https://arxiv.org/abs/2212.07677) (Hahn et al., 2022)

## 🤝 Contributing

Contributions welcome! Please feel free to submit a Pull Request.