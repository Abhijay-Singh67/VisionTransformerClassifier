# Vision Transformer (ViT) in Pure NumPy 🚀



A complete, working implementation of a Vision Transformer (ViT) built entirely from scratch using only `numpy`. No PyTorch, no TensorFlow, no Keras—just raw linear algebra, calculus, and Python.

This project trains a ViT to classify 32x32 RGB images from the CIFAR-10 dataset, serving as a transparent, educational deep dive into how Self-Attention and Transformer architectures actually work under the hood.

## 🌟 Key Features

Building a Transformer from scratch requires overcoming massive mathematical instability. This repository includes custom-built solutions for standard deep learning hurdles:
* **Custom AdamW Optimizer:** Features true gradient accumulation (mini-batching) and weight decay to prevent forward-pass weight explosions.
* **Numerically Stable Softmax:** Implements the max-shift trick to prevent `np.exp()` overflows during Query-Key dot products.
* **Global Gradient Clipping:** Safely truncates exploding error signals during backpropagation across LayerNorm and MLP blocks.
* **Learning Rate Step Decay:** Dynamically scales down the learning rate to help the model settle into narrow local minima.
* **Pickle Serialization:** Save and load trained model weights and optimizer states seamlessly.

## 🧠 Architecture Overview

The model closely follows the original "An Image is Worth 16x16 Words" paper, scaled down for CIFAR-10:
* **Image Size:** 32x32x3
* **Patch Size:** 4x4 (Yields 64 patches per image)
* **Embedding Dimension:** 64
* **Multi-Head Attention:** 8 Heads (Head Dimension = 8)
* **MLP Hidden Dimension:** 256 (4x Expansion)
* **Positional Embeddings:** Learnable 1D parameters added to patch projections.

*(Note: `torchvision` is strictly used to download and parse the CIFAR-10 dataset. All neural network forward passes, backpropagation, and weight updates are purely NumPy).*

## 📂 Repository Structure

* `NetworkCore.py`: Contains the core Transformer components (`AttentionHead`, `MultiAttentionBlock`, `LayerNorm`, `VisionTransformer`, and the wrapper `ClassificationVIT`).
* `MLP.py`: Contains the `Linear` layer and `Sequential` blocks used for the internal Transformer MLP and the final classification head.
* `helper.py`: Contains activation functions (ReLU, Softmax), Loss functions (Categorical Cross Entropy), their respective derivatives, and the `AdamOptimizer`.
* `Implementation.py`: The main training loop, dataset preparation, and testing logic.

## 🚀 Getting Started

### Prerequisites
```bash
pip install numpy torchvision
```
### Training the Model
You can adjust the dataset limit in the main script to train on a small subset (e.g., 2,000 images) for quick testing, or the full 50,000 for maximum accuracy.

```python
from NetworkCore import ClassificationVIT

# Initialize the model
model = ClassificationVIT(
    image_dim=32,
    embedding_dim=64, 
    patch_size=4,
    num_of_heads=8,
    MLP_hidden_param=4,
    output_dim=10,
    learning_rate=0.0001
)

# Train the model
model.fit(x_train, y_train, epochs=50, batch_size=32)

# Save the weights
model.save("vit_model.pkl")
```
### 📊 Results & Scaling
On CIFAR-10, this NumPy implementation successfully converges, achieving ~63% test accuracy in 150 epochs (over 6x better than random guessing), proving the mathematical soundness of the custom attention mechanism and backpropagation logic.
The accuracy was less because only one Attention+MLP Block instead of multiple blocks due to limitation of compute.
