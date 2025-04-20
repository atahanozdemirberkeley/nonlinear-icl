# Kernel In-Context Learning

This repository investigates whether Transformers can learn kernelized functions in-context, with a focus on RBF kernels approximated using Random Fourier Features (RFF).

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