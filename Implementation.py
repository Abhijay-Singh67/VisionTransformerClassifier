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
    patch_size=4,
    num_of_heads=4,
    MLP_hidden_param=2,
    output_dim=10
)

x = np.random.randn(32,32,3)

y = np.zeros((1,10))
y[0,3] = 1

gradient_check(model,x,y)
