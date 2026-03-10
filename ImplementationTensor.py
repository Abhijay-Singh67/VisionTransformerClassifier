import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms

from NetworkCoreTensor import ClassificationVIT

# 🚀 1. Setup global device (Auto-detects GPU)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
    
print(f"Loading data to: {device}")

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

def dataset_to_tensor(dataset, limit=None):
    images = []
    labels = []
    for i, (img, label) in enumerate(dataset):
        # PyTorch uses .permute() instead of NumPy's .transpose()
        # Changes shape from (3, 32, 32) to (32, 32, 3)
        img = img.permute(1, 2, 0).to(device)
        images.append(img)

        # Create one-hot labels directly on the GPU
        onehot = torch.zeros(10, device=device)
        onehot[label] = 1.0
        labels.append(onehot)

        if limit and i >= limit - 1:
            break

    # torch.stack smoothly combines the list of tensors into a batch
    return torch.stack(images), torch.stack(labels)

# We can keep the limits small for a quick test to ensure it runs!
x_train, y_train = dataset_to_tensor(train_dataset, limit=2000)
x_test, y_test = dataset_to_tensor(test_dataset, limit=500)

x_train = (x_train - 0.5) * 2.0
x_test = (x_test - 0.5) * 2.0

original_lr = 0.0001
batch_size = 32

model = ClassificationVIT(
    image_dim=32,
    embedding_dim=64,
    patch_size=4,
    num_of_heads=8,
    MLP_hidden_param=4,
    output_dim=10,
    learning_rate=original_lr
)

# NOTE: The manual learning rate scaling hack has been DELETED 
# because your AdamOptimizer now handles batch averaging perfectly!

epochs = 50
N = len(x_train)

# PyTorch's version of random permutation
perm = torch.randperm(N, device=device)
x_train_shuffled = x_train[perm]
y_train_shuffled = y_train[perm]

print("\nStarting main training loop...")

# Uncomment this if you have a PyTorch weights file saved!
# model.load() 

model.fit(x_train_shuffled, y_train_shuffled, epochs, batch_size)
model.save("vit_model_pytorch.pkl")

correct = 0

for i in range(len(x_test)):
    x = x_test[i]
    y = y_test[i]

    # Temporarily disable gradient tracking for inference (saves memory/time)
    with torch.no_grad():
        pred = model.forward(x)

    # torch.argmax replaces np.argmax
    if torch.argmax(pred) == torch.argmax(y):
        correct += 1

accuracy = correct / len(x_test)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")