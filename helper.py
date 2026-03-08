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

def softCCEgrad(y,pred):
    return (pred - y) / pred.shape[0]


def _softmax_grad_dummy(x):
    # gradient of softmax is handled explicitly in MLP; this placeholder
    # prevents a KeyError if someone accidentally calls ``actigrad`` with
    # softmax.  Multiplying by ones is a no-op.
    return np.ones_like(x)


grads = {
    MSE: MSEgrad,
    relu: relugrad,
    sigmoid: siggrad,
    lin: lingrad,
    MAE: MAEgrad,
    CCE: CCEgrad,
    softmax: _softmax_grad_dummy,
}

def grad(y,pred,loss):
    return grads[loss](y,pred)

def actigrad(x,act):
    return grads[act](x)

class AdamOptimizer:
    def __init__(self, shape, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.weight_decay = weight_decay
        
        # Internal accumulators for True Batching
        self.grad_accum = np.zeros(shape)
        self.accum_count = 0

    def update(self, w, grad, lr, batch_size=32):
        # 1. Add gradient to the accumulator
        self.grad_accum += grad
        self.accum_count += 1
        
        # 2. If we haven't reached the batch size, return the weights unchanged!
        if self.accum_count < batch_size:
            return w
            
        # 3. Once we hit the batch size, do the Adam update using the AVERAGED gradient
        self.t += 1
        avg_grad = self.grad_accum / self.accum_count
        
        self.m = self.beta1 * self.m + (1 - self.beta1) * avg_grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (avg_grad ** 2)
        
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        w = w - (lr * self.weight_decay * w)
        w -= lr * m_hat / (np.sqrt(v_hat) + self.eps)
        
        # 4. Reset accumulators for the next batch
        self.grad_accum.fill(0)
        self.accum_count = 0
        
        return w
