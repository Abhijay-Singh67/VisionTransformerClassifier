import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def siggrad(x):
    return sigmoid(x)*(1-sigmoid(x))

def relu(x):
    return np.maximum(0,x)

def relugrad(x):
    return (x > 0).astype(float)

def lin(x):
    return x

def lingrad(x):
    return np.ones(x.shape)

def MSE(y, pred):
    return np.mean((y - pred) ** 2) / 2

def RMSE(y,pred):
    return np.sqrt(np.mean((y-pred) ** 2)/2)

def MSEgrad(y,pred):
    return (pred-y)/y.size

def MAE(y,pred):
    return np.mean(np.abs(y-pred))

def MAEgrad(y, pred):
    m = y.shape[0]
    return np.sign(pred - y) / m

def softmax(x):
    #Takes the row wise softmax of the attention pattern
    x = x - np.max(x, axis=1, keepdims=True)
    x_exp = np.exp(x)
    return x_exp / np.sum(x_exp, axis=1, keepdims=True)

def softgrad(grad, softmax_out):
    dot = np.sum(grad * softmax_out, axis=1, keepdims=True)
    return softmax_out * (grad - dot)

def CCE(y,pred):
    eps = 1e-12
    pred = np.clip(pred, eps, 1 - eps)
    return -np.mean(np.sum(y * np.log(pred), axis=1))

def CCEgrad(y,pred):
    eps = 1e-12
    pred = np.clip(pred, eps, 1 - eps)
    return -y / pred / pred.shape[0]

def softCCEgrad(preds, targets):
    return (preds - targets) / preds.shape[0]


grads = {MSE:MSEgrad,relu:relugrad,sigmoid:siggrad,lin:lingrad,MAE:MAEgrad,softmax:softgrad,CCE:CCEgrad}

def grad(y,pred,loss):
    return grads[loss](y,pred)

def actigrad(x,act):
    return grads[act](x)

def adam(prev_loss,loss,lr):
    if (((prev_loss-loss)/prev_loss)<0.5):
        return loss,lr/1000
    return prev_loss,lr
    
