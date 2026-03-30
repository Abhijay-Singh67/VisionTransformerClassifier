# Vision Transformer Classifier — Built from Scratch in NumPy

A complete implementation of the [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929) architecture using **only NumPy** — no PyTorch, no TensorFlow, no autograd. Every component, from patch embeddings and multi-head self-attention to layer normalization and backpropagation, is written by hand with fully derived gradients.

Trained and evaluated on the **CIFAR-10** dataset (32×32 RGB images, 10 classes).

---

## Motivation

Most ViT tutorials wrap a few `nn.Module` calls and move on. This project strips away the framework entirely to expose the raw mechanics: how attention gradients flow backward through softmax, how layer norm is differentiated, and how positional embeddings actually learn. If you want to understand what a Vision Transformer is *doing* — not just how to instantiate one — this is for you.

**No autograd. No computational graph. No framework.** Every partial derivative is derived on paper and translated into NumPy operations.

---

## Architecture Overview

```
Input Image (32×32×3)
  │
  ├── 1. Patch Extraction
  │       Slice into P×P patches → flatten each to a vector
  │       (32×32×3) → (N, patch_dim) where N = (32/P)², patch_dim = P²·3
  │
  ├── 2. Linear Projection + Scaling
  │       Learned matrix (patch_dim → D), scaled by √D
  │
  ├── 3. + Positional Embeddings
  │       Learned (N × D) matrix, added element-wise
  │
  ├── 4. Transformer Encoder Block
  │   ├── Layer Norm 1
  │   ├── Multi-Head Self-Attention (H heads) + Residual
  │   │     ├── Per head: Q = E·W_Q,  K = E·W_K,  V = E·W_V
  │   │     ├── Attention = softmax(Q·Kᵀ / √d) · V
  │   │     ├── Concatenate all heads
  │   │     └── Output projection W_O + residual connection
  │   ├── Layer Norm 2
  │   └── Feed-Forward MLP + Residual
  │         └── Linear(D → M·D, ReLU) → Linear(M·D → D, linear) + residual
  │
  ├── 5. Global Average Pooling
  │       Mean across all N patch embeddings → single (1, D) vector
  │
  └── 6. Classification Head
          Linear(D → M·D, ReLU) → Linear(M·D → 10, Softmax)
          → class probabilities
```

---

## What's Implemented from Scratch

Every layer listed below includes both its forward pass and its complete, manually-derived backward pass:

| Component | Forward | Backward |
|---|---|---|
| **Patch Embedding** | Slice image into non-overlapping patches, flatten to vectors | Gradient reshaped back to patch grid (not updated — input layer) |
| **Linear Projection** | `X @ W + b` with configurable activation | `dW = Aᵀ @ δ`, `db = Σδ`, `dX = δ @ Wᵀ` |
| **Positional Embeddings** | Additive learned position encoding `E + P` | Gradient flows directly; P updated via Adam |
| **Layer Normalization** | `γ · (x - μ) / √(σ² + ε) + β` per token | Full LN gradient: accounts for mean/variance dependencies via `∂x̂/∂x`, plus `dγ`, `dβ` |
| **Scaled Dot-Product Attention** | `softmax(Q·Kᵀ / √d) · V` | `dA = dZ·Vᵀ`, `dV = Aᵀ·dZ`, softmax Jacobian via `S ⊙ (dA - Σ(dA ⊙ S))`, then `dQ`, `dK` from the bilinear form |
| **Multi-Head Attention** | H parallel heads, concatenate, project through `W_O`, add residual | Split gradient by head dim, backprop each head independently, sum input gradients, add residual gradient |
| **Feed-Forward MLP** | Two-layer network with ReLU (transformer) or Softmax (classifier) | Chain rule through activations; `backward_delta` returns input gradient for upstream layers |
| **Categorical Cross-Entropy** | `-Σ yᵢ log(pᵢ)` with numerical clipping | Combined softmax-CCE gradient: `(pred - y) / batch_size` for stability |
| **AdamW Optimizer** | — | Bias-corrected first/second moment estimates, decoupled weight decay, built-in gradient accumulation |

---

## Detailed Component Descriptions

### Patch Embedding (`patch_embeddings`)

The input image `(H, W, 3)` is divided into a grid of non-overlapping `P×P` patches. Each patch is flattened into a vector of dimension `P² · 3`. For a 32×32 image with P=4, this produces 64 patches of dimension 48.

```
Image (32, 32, 3)  →  64 patches  →  (64, 48) matrix
```

Unlike convolutional tokenizers, this uses simple array slicing — no learned kernels at the patch extraction stage. The learning happens in the subsequent linear projection.

### Linear Projection + Embedding Scaling

Patches are projected from `patch_dim` into the model's embedding dimension D through a learned `Linear` layer. The output is then scaled by `√D` before adding positional embeddings. This scaling (borrowed from the original Transformer paper) prevents the positional signal from being overwhelmed by the magnitude of the projected embeddings.

### Positional Embeddings

A learned `(N, D)` matrix is added element-wise to the patch embeddings. Each of the N positions gets its own D-dimensional vector, trained end-to-end via backpropagation. No sinusoidal encoding — the model discovers spatial relationships from data.

### Layer Normalization

Normalizes each token independently across its embedding dimension:

```
x̂ = (x - μ) / √(σ² + ε)
output = γ · x̂ + β
```

The backward pass computes the full Jacobian of the normalization, accounting for the fact that μ and σ² are themselves functions of x. The gradient through LayerNorm is:

```
dx = (1/√(σ²+ε)) · (dx̂ - mean(dx̂) - x̂ · mean(dx̂ · x̂))
```

where `dx̂ = dout · γ`. Learnable parameters `γ` (scale) and `β` (shift) are updated independently.

### Attention Head

Each head independently computes queries, keys, and values by projecting the input embeddings through separate weight matrices `W_Q`, `W_K`, `W_V` of shape `(D, d)` where `d = D/H`:

```
Q = E · W_Q       (N, d)
K = E · W_K       (N, d)
V = E · W_V       (N, d)
scores = Q · Kᵀ / √d    (N, N)
A = softmax(scores)      (N, N)   — the attention pattern
Z = A · V                (N, d)   — the attended output
```

Weight initialization uses Xavier scaling: `randn / √D`.

**Backward pass:** The gradient through attention requires differentiating through the softmax (which has an N×N Jacobian per row), the bilinear `QKᵀ` product, and three separate projection matrices. The softmax gradient is computed efficiently using the identity:

```
dscores = S ⊙ (dA - (dA ⊙ S) · 1)
```

which avoids materializing the full Jacobian.

### Multi-Head Attention Block

Runs H attention heads in parallel, concatenates their outputs along the feature dimension, and projects through `W_O`:

```
Z_concat = [Z₁ | Z₂ | ... | Zₕ]    (N, D)
output = Z_concat · W_O + E          (N, D)   — includes residual
```

During backprop, the gradient through `W_O` is computed first, then sliced by head dimension and distributed to each head's individual backward pass. The residual connection adds the incoming gradient directly to the attention gradient.

### Transformer Feed-Forward MLP

A two-layer network with expansion factor M:

```
hidden = ReLU(x · W₁ + b₁)      (N, M·D)
output = hidden · W₂ + b₂ + x   (N, D)     — includes residual
```

The second layer uses a linear activation (identity) since the residual addition follows immediately.

### Classification Pipeline

After the transformer block produces `(N, D)` feature embeddings:

1. **Global average pooling** — mean across all N patch tokens → `(1, D)` vector
2. **Classification MLP** — `Linear(D → M·D, ReLU) → Linear(M·D → 10, Softmax)`
3. **Loss** — Categorical cross-entropy

During backprop, the gradient from the classification head is distributed equally to all N patches (since pooling averaged them), then flows backward through the entire transformer.

### Adam Optimizer with Gradient Accumulation

Each learnable parameter gets its own `AdamOptimizer` instance that maintains:

- First moment estimate `m` (momentum)
- Second moment estimate `v` (adaptive learning rate)
- Bias correction via timestep `t`
- Decoupled weight decay (AdamW formulation)
- **Gradient accumulator** for simulated mini-batch training

Since the model processes one image at a time (no batched matrix operations), the optimizer accumulates gradients internally and only fires the actual weight update every `batch_size` steps. This gives mini-batch SGD behavior without requiring batch dimensions in the forward pass.

```
accumulate grad for batch_size steps
    ↓
avg_grad = accumulated / count
    ↓
m = β₁·m + (1-β₁)·avg_grad
v = β₂·v + (1-β₂)·avg_grad²
    ↓
m̂ = m / (1-β₁ᵗ)        — bias correction
v̂ = v / (1-β₂ᵗ)
    ↓
w = w - lr·λ·w          — weight decay (decoupled)
w = w - lr·m̂ / (√v̂+ε)  — Adam step
```

### Gradient Stability

Training a transformer from scratch without a framework is numerically fragile. This implementation uses a multi-level gradient clipping strategy:

| Location | Clip range | Rationale |
|---|---|---|
| Weight gradients (`dW_Q`, `dW_K`, etc.) | [-5, 5] | Prevents catastrophic weight updates |
| Inter-layer deltas (between LN, attention, MLP) | [-1, 1] | Keeps signal magnitude bounded through the residual stream |
| LayerNorm `dγ`, `dβ` | [-5, 5] | Norm parameters are sensitive to outlier gradients |
| Linear layer `dW`, `db` | [-5, 5] | Standard gradient clipping |

### Gradient Checking

`Implementation.py` includes a numerical gradient verification utility that perturbs a single weight by ±ε and compares the finite-difference gradient against the analytically computed gradient:

```python
numerical  = (L(w+ε) - L(w-ε)) / 2ε
analytical = (w_after_update - w_before) / (-lr)
```

This validates that the manual backprop implementation is mathematically correct.

---

## Parameter Count

**Default configuration** (embedding_dim=64, patch_size=4, heads=8, MLP expansion=4):

| Component | Shape | Parameters |
|---|---|---|
| Patch projection | Linear(48 → 64) + bias | 3,136 |
| Positional embeddings | 64 × 64 | 4,096 |
| Layer norms (×2) | γ + β each, dim 64 | 256 |
| Multi-head attention | 8 × (W_Q + W_K + W_V) + W_O | 16,384 |
| Transformer MLP | Linear(64 → 256) + Linear(256 → 64) | 33,088 |
| Classification head | Linear(64 → 256) + Linear(256 → 10) | 19,210 |
| **Total** | | **76,170** |

---

## Project Structure

```
├── NetworkCore.py       # Core ViT architecture
│   ├── patch_embeddings()     — image → patch vectors
│   ├── LayerNorm              — forward + backward with learnable γ, β
│   ├── AttentionHead          — single Q/K/V head with full backprop
│   ├── MultiAttentionBlock    — H parallel heads + W_O + residual
│   ├── VisionTransformer      — full encoder: patch proj → LN → MHSA → LN → MLP
│   └── ClassificationVIT      — ViT + pooling + classification head + training loop
│
├── MLP.py               # Neural network primitives
│   ├── Linear                 — fully connected layer with activation
│   └── Sequential             — layer container with backProp + backward_delta
│
├── helper.py            # Mathematical foundations
│   ├── Activations            — relu, sigmoid, softmax, linear (+ all gradients)
│   ├── Loss functions         — MSE, RMSE, MAE, CCE (+ all gradients)
│   ├── softCCEgrad()          — fused softmax + CCE gradient
│   ├── softgrad()             — efficient softmax Jacobian-vector product
│   └── AdamOptimizer          — AdamW with gradient accumulation
│
├── Implementation.py    # Training & evaluation script
│   ├── gradient_check()       — numerical vs analytical gradient verification
│   ├── Data pipeline          — CIFAR-10 loading, one-hot encoding, normalization
│   └── Training loop          — shuffled epochs, batch training, test evaluation
│
└── requirements.txt     # Dependencies
```

---

## Getting Started

### Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** PyTorch and torchvision are used *only* for downloading and loading the CIFAR-10 dataset. The model itself uses nothing but NumPy.

### Train the model

```bash
python Implementation.py
```

This will:
1. Download CIFAR-10 (if not already cached)
2. Load 2,000 training images and 500 test images
3. Normalize pixel values to [-1, 1]
4. Train for 50 epochs with batch size 32 and learning rate 1e-4
5. Save the trained model to `vit_model.pkl`
6. Evaluate test accuracy

### Customize hyperparameters

Edit the configuration in `Implementation.py`:

```python
model = ClassificationVIT(
    image_dim=32,           # Input image size (square)
    embedding_dim=64,       # Model width D — try 128 for more capacity
    patch_size=4,           # Patch size P — 4×4 gives 64 tokens, 8×8 gives 16
    num_of_heads=8,         # Attention heads H — must divide embedding_dim evenly
    MLP_hidden_param=4,     # Expansion factor M — hidden MLP dim = M × D
    output_dim=10,          # Number of classes
    learning_rate=0.0001    # Base learning rate (scaled by 1/batch_size internally)
)
```

### Load and use a trained model

```python
from NetworkCore import ClassificationVIT
import numpy as np

model = ClassificationVIT(32, 64, 4, 8, 4, 10)
model.load("vit_model.pkl")

# Input: (32, 32, 3) NumPy array, pixel values normalized to [-1, 1]
prediction = model.forward(image)       # → (1, 10) softmax probabilities
predicted_class = np.argmax(prediction)  # → integer label 0-9
```

### Run gradient checking

The gradient check function in `Implementation.py` verifies backprop correctness by comparing against numerical finite differences:

```python
gradient_check(model, x_sample, y_sample)
# Expected output: Difference should be < 1e-4
```

---

## Design Decisions & Trade-offs

### Single-image forward pass (no batch dimension)

The model processes one image at a time. The attention matrices are `(64, 64)` — one token per row — rather than `(B, 64, 64)`. This keeps the gradient derivations clean and easy to verify, at the cost of not leveraging NumPy's batch parallelism. Mini-batch training is achieved via gradient accumulation in the optimizer.

### No CLS token

The original ViT prepends a learnable `[CLS]` token and reads the classifier output from that position. This implementation instead uses global average pooling across all patch tokens. This is a common simplification (used in DeiT and others) that avoids introducing an asymmetric token into the sequence.

### Embedding scaling by √D

After the linear projection, embeddings are multiplied by `√D`. This follows the original Transformer convention from "Attention Is All You Need," ensuring the projected embeddings and positional embeddings contribute at similar magnitudes.

### Pre-norm transformer

Layer normalization is applied *before* attention and *before* the MLP (Pre-LN), rather than after (Post-LN). Pre-norm transformers are known to be more stable during training, which matters especially here since there's no framework-level gradient scaling or automatic mixed precision to help.

### One transformer block

The current implementation uses a single transformer encoder layer rather than stacking multiple blocks. This keeps the parameter count manageable (~76K) and training feasible on CPU. Extending to deeper architectures would be straightforward — instantiate multiple `VisionTransformer` blocks and chain them.

### Learning rate scaling

The base learning rate is divided by `batch_size` before being passed to the model. Since gradients are accumulated (not averaged at the model level) over `batch_size` samples before the Adam update averages them, this scaling ensures the effective step size remains consistent regardless of batch size.

---

## Limitations

- **Speed** — Pure NumPy on CPU. Training on the full 50K CIFAR-10 dataset is slow. The 2K subset is used for practical training times.
- **Single block** — One transformer layer limits representational depth. Stacking blocks would improve accuracy.
- **No data augmentation** — Random flips, crops, and mixup would significantly help but add complexity outside the core ViT implementation.
- **No dropout** — The model relies on weight decay and small model size for regularization.
- **No learning rate scheduling** — A warmup + cosine decay schedule would improve convergence.

---

## Possible Extensions

- Stack multiple transformer encoder blocks for deeper representations
- Add a learnable `[CLS]` token as an alternative to global average pooling
- Implement dropout in the attention weights and MLP layers
- Add learning rate warmup + cosine annealing
- Visualize the learned attention patterns and positional embeddings
- Train on the full CIFAR-10 (50K samples) with patience
- Port the architecture to CIFAR-100 or Tiny ImageNet

---

## Requirements

- Python 3.8+
- NumPy
- PyTorch + torchvision (dataset loading only)

---

## References

- Dosovitskiy et al., ["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"](https://arxiv.org/abs/2010.11929) (2020)
- Vaswani et al., ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) (2017)
- Loshchilov & Hutter, ["Decoupled Weight Decay Regularization"](https://arxiv.org/abs/1711.05101) (AdamW, 2017)
- Ba et al., ["Layer Normalization"](https://arxiv.org/abs/1607.06450) (2016)
