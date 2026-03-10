import torch
from helperTensor import relu, MSE, grad, actigrad, softmax, CCE, softCCEgrad, softgrad, AdamOptimizer

# 🚀 Setup global device (Auto-detects GPU)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

class Linear:
    def __init__(self, input_features: int, output_features: int, activation=relu):
        self.inp = input_features
        self.out = output_features
        # Initialize directly on the GPU
        self.__weights = torch.randn((input_features, output_features), device=device) * 0.02
        self.__bias = torch.randn((1, output_features), device=device) * 0.02
        self.id = 0
        self.activation = activation
        self.current_output = torch.zeros(self.__bias.shape, device=device)
        self.current_activated_output = torch.zeros(self.__bias.shape, device=device)

        # Pass the device to Adam
        self.opt_W = AdamOptimizer(self.__weights.shape, device=device)
        self.opt_B = AdamOptimizer(self.__bias.shape, device=device)
    
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            raise Exception(f"The input vector to layer {self.id} is not a PyTorch Tensor!!")
        if x.shape[1] != self.inp:
            raise Exception(f"The input shape to layer {self.id} is incorrect!! Expected ({self.inp},{self.out}) but {x.shape} was provided")
        try:
            out = x @ self.__weights + self.__bias
            self.current_output = out
        except Exception as e:
            raise Exception(f"Error in propagating forward through the layer {self.id}: {e}")
        self.current_activated_output = self.activation(out)
        return self.current_activated_output
    
    def update(self, gradW, gradB, lr):
        gradW = torch.clamp(gradW, -5.0, 5.0)
        gradB = torch.clamp(gradB, -5.0, 5.0)
        
        self.__weights = self.opt_W.update(self.__weights, gradW, lr)
        self.__bias = self.opt_B.update(self.__bias, gradB, lr)
    
    def weights(self):
        return self.__weights

    def bias(self):
        return self.__bias

class Sequential:
    def __init__(self, *layers, loss=MSE, learning_rate=1e-3):
        self.layers = list(layers)
        self.num_layers = len(layers)
        for i in range(len(layers)):
            self.layers[i].id = i
        self.__lr = learning_rate
        self.__loss = loss
    
    def forwardPass(self, x):
        out = x
        for i in self.layers:
            out = i.forward(out)
        self.currentOutput = out
        return out

    def fit(self, x, y, epochs: int, batch_size=1):
        N = x.shape[0]
        for i in range(epochs):
            for j in range(0, N, batch_size):
                x_batch = x[j:j+batch_size]
                y_batch = y[j:j+batch_size]
                pred = self.forwardPass(x_batch)
                self.backProp(x_batch, y_batch, pred)
            predFull = self.forwardPass(x)
            
            # .item() extracts the float from the 0D tensor
            print(f"Epoch {i+1}/{epochs} Training Loss: {self.__loss(y, predFull).item():.6f}")

    def backProp(self, x, y, pred):
        grads = []
        delta = torch.zeros(y.shape, device=device)
        for i in range(self.num_layers - 1, -1, -1):
            layer = self.layers[i]
            if i == self.num_layers - 1:
                if layer.activation == softmax:
                    if self.__loss == CCE:
                        delta = softCCEgrad(y, pred)
                    else:
                        delta = softgrad(grad(y, pred, self.__loss), layer.current_activated_output)
                else:
                    delta = grad(y, pred, self.__loss) * actigrad(layer.current_output, layer.activation)
            else:
                next_layer = self.layers[i+1]
                delta = (delta @ next_layer.weights().T) * actigrad(layer.current_output, layer.activation)
            
            delta = torch.clamp(delta, -1.0, 1.0)
            
            if i == 0:
                A_prev = x
            else:
                A_prev = self.layers[i-1].current_activated_output
            
            gradW = (A_prev.T @ delta)
            # axis=0 -> dim=0, keepdims -> keepdim
            gradB = torch.sum(delta, dim=0, keepdim=True)
            grads.append((layer, gradW, gradB))
        
        for layer, gradW, gradB in grads:
            layer.update(gradW, gradB, self.__lr)

    def backward_delta(self, x, delta):
        grads = []

        for i in range(self.num_layers - 1, -1, -1):
            layer = self.layers[i]
            if layer.activation is not softmax:
                delta = delta * actigrad(layer.current_output, layer.activation)

            if i == 0:
                A_prev = x
            else:
                A_prev = self.layers[i-1].current_activated_output

            gradW = A_prev.T @ delta
            gradB = torch.sum(delta, dim=0, keepdim=True)

            grads.append((layer, gradW, gradB))
            delta = delta @ layer.weights().T
            delta = torch.clamp(delta, -1.0, 1.0)

        for layer, gradW, gradB in grads:
            layer.update(gradW, gradB, self.__lr)
        return delta
    
    def predict(self, x):
        return self.forwardPass(x)
    
    def dump(self):
        with open("weights.txt", "w") as file:
            text = ""
            for i in self.layers:
                text += f"Layer: {i.id}\n"
                # Move to CPU and convert back to numpy just for clean text formatting
                text += str(i.weights().cpu().numpy().flatten()) + "\n"
                text += str(i.bias().cpu().numpy().flatten()) + "\n"
            file.write(text)
    
    def summary(self):
        print(f"Layers: {self.num_layers}")
        params = 0
        for i in self.layers:
            print(f"Layer: {i.id}, inputs: {i.inp}, outputs: {i.out}")
            params += (i.inp * i.out) + i.out
        print(f"Trainable Params: {params}")