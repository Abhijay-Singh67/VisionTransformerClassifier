from NetworkCore import *
import numpy as np

def gradient_check(model, x, y, eps=1e-5):

    # pick parameter
    head = model.VIT.AttentionBlock.heads[0]

    i, j = 0, 0

    original = head.weight_Q[i,j]

    # forward pass
    pred = model.forward(x)
    loss = CCE(y,pred)

    # run backprop
    model.backprop(x,y,pred)

    # analytic gradient
    analytic = (head.weight_Q[i,j] - original) / (-model.lr)

    # numerical gradient
    head.weight_Q[i,j] = original + eps
    loss_plus = CCE(y, model.forward(x))

    head.weight_Q[i,j] = original - eps
    loss_minus = CCE(y, model.forward(x))

    numerical = (loss_plus - loss_minus)/(2*eps)

    # restore weight
    head.weight_Q[i,j] = original

    print("Analytic:", analytic)
    print("Numerical:", numerical)
    print("Difference:", abs(analytic-numerical))

model = ClassificationVIT(
    image_dim=32,
    embedding_dim=128,
    patch_size=4,
    num_of_heads=4,
    MLP_hidden_param=2,
    output_dim=10
)

import numpy as np
from torchvision.datasets import CIFAR10
from torchvision import transforms

from NetworkCore import ClassificationVIT
from helper import CCE

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

def dataset_to_numpy(dataset, limit=None):
    images = []
    labels = []
    for i, (img, label) in enumerate(dataset):
        img = img.numpy().transpose(1, 2, 0)
        images.append(img)

        onehot = np.zeros(10)
        onehot[label] = 1
        labels.append(onehot)

        if limit and i >= limit - 1:
            break

    return np.array(images), np.array(labels)

x_train, y_train = dataset_to_numpy(train_dataset, limit=2000)
x_test, y_test = dataset_to_numpy(test_dataset, limit=500)

x_train = (x_train - 0.5) * 2
x_test = (x_test - 0.5) * 2

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

scaled_lr = original_lr / batch_size
model.lr = scaled_lr
model.VIT.lr = scaled_lr
model.MLP.learning_rate = scaled_lr

model.lr = scaled_lr
model.VIT.lr = scaled_lr
model.MLP.learning_rate = scaled_lr

epochs = 50
N = len(x_train)
perm = np.random.permutation(N)
x_train_shuffled = x_train[perm]
y_train_shuffled = y_train[perm]
print("\nStarting main training loop...")
model.load()
model.fit(x_train_shuffled,y_train_shuffled,epochs,batch_size)
model.save()

correct = 0

for i in range(len(x_test)):
    x = x_test[i]
    y = y_test[i]

    pred = model.forward(x)

    if np.argmax(pred) == np.argmax(y):
        correct += 1

accuracy = correct / len(x_test)
print("\nTest Accuracy:", accuracy*100,"%")
