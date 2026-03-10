import torch

def sigmoid(x):
    # torch natively handles this with better numerical stability
    return torch.sigmoid(x)

def siggrad(x):
    sig = torch.sigmoid(x)
    return sig * (1 - sig)

def relu(x):
    return torch.relu(x)

def relugrad(x):
    # .float() converts the boolean mask to 1.0s and 0.0s
    return (x > 0).float()

def lin(x):
    return x

def lingrad(x):
    return torch.ones_like(x)

def MSE(y, pred):
    return torch.mean((y - pred) ** 2) / 2

def RMSE(y, pred):
    return torch.sqrt(torch.mean((y - pred) ** 2) / 2)

def MSEgrad(y, pred):
    # .numel() is PyTorch's equivalent to NumPy's .size
    return (pred - y) / y.numel()

def MAE(y, pred):
    return torch.mean(torch.abs(y - pred))

def MAEgrad(y, pred):
    m = y.shape[0]
    return torch.sign(pred - y) / m

def softmax(x):
    # Takes the row-wise softmax of the attention pattern
    # Note: torch.max returns (values, indices), so we grab .values
    x = x - torch.max(x, dim=1, keepdim=True).values
    x_exp = torch.exp(x)
    return x_exp / torch.sum(x_exp, dim=1, keepdim=True)

def softgrad(grad, softmax_out):
    dot = torch.sum(grad * softmax_out, dim=1, keepdim=True)
    return softmax_out * (grad - dot)

def CCE(y, pred):
    eps = 1e-12
    pred = torch.clamp(pred, eps, 1.0 - eps)
    return -torch.mean(torch.sum(y * torch.log(pred), dim=1))

def CCEgrad(y, pred):
    eps = 1e-12
    pred = torch.clamp(pred, eps, 1.0 - eps)
    return -y / pred / pred.shape[0]

def softCCEgrad(y, pred):
    return (pred - y) / pred.shape[0]

def _softmax_grad_dummy(x):
    # dummy softmax
    return torch.ones_like(x)

grads = {
    MSE: MSEgrad,
    relu: relugrad,
    sigmoid: siggrad,
    lin: lingrad,
    MAE: MAEgrad,
    CCE: CCEgrad,
    softmax: _softmax_grad_dummy,
}

def grad(y, pred, loss):
    return grads[loss](y, pred)

def actigrad(x, act):
    return grads[act](x)

class AdamOptimizer:
    # Notice the new device parameter to put Adam's memory on the GPU!
    def __init__(self, shape, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01, device='cpu'):
        self.m = torch.zeros(shape, device=device)
        self.v = torch.zeros(shape, device=device)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.weight_decay = weight_decay
        
        # Internal accumulators for True Batching
        self.grad_accum = torch.zeros(shape, device=device)
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
        w -= lr * m_hat / (torch.sqrt(v_hat) + self.eps)
        
        # 4. Reset accumulators for the next batch using PyTorch's in-place zero
        self.grad_accum.zero_()
        self.accum_count = 0
        
        return w