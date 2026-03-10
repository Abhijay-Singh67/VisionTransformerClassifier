import torch
import math
import pickle
from MLPTensor import Sequential, Linear
from helperTensor import lin, softmax, CCE, softCCEgrad, softgrad, AdamOptimizer

# 🚀 1. Setup global device (Auto-detects GPU)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

def patch_embeddings(x, image_dim, patch_size):
    # Generates patch embeddings
    # Dim:(32,32,3)->(64,48) [N,D]
    embeddings = []
    for i in range(0, image_dim, patch_size):
        for j in range(0, image_dim, patch_size):
            embeddings.append(x[i:i+patch_size, j:j+patch_size].flatten())
    # torch.stack smoothly converts a list of tensors into a single tensor
    return torch.stack(embeddings)

class LayerNorm: 
    def __init__(self, embedding_dim, eps=1e-5):
        # Move weights directly to the GPU
        self.gamma = torch.ones(embedding_dim, device=device)
        self.beta = torch.zeros(embedding_dim, device=device)
        self.eps = eps

        self.opt_gamma = AdamOptimizer(self.gamma.shape, device=device)
        self.opt_beta = AdamOptimizer(self.beta.shape, device=device)

    def forward(self, x):
        self.mean = torch.mean(x, dim=1, keepdim=True)
        # CRITICAL: unbiased=False ensures it matches NumPy's exact var() math
        self.var = torch.var(x, dim=1, unbiased=False, keepdim=True) 
        self.x_hat = (x - self.mean) / torch.sqrt(self.var + self.eps)
        return self.gamma * self.x_hat + self.beta

    def backward(self, grad_out, lr=None):
        grad_gamma = torch.sum(grad_out * self.x_hat, dim=0)
        grad_beta = torch.sum(grad_out, dim=0)
        grad_gamma = torch.clamp(grad_gamma, -5.0, 5.0)
        grad_beta = torch.clamp(grad_beta, -5.0, 5.0)
        grad_xhat = grad_out * self.gamma
        inv_std = 1.0 / torch.sqrt(self.var + self.eps)
        grad_x = inv_std * (grad_xhat - torch.mean(grad_xhat, dim=1, keepdim=True) - self.x_hat * torch.mean(grad_xhat * self.x_hat, dim=1, keepdim=True))

        if lr is not None:
            self.gamma = self.opt_gamma.update(self.gamma, grad_gamma, lr)
            self.beta = self.opt_beta.update(self.beta, grad_beta, lr)
        return grad_x

class AttentionHead:
    def __init__(self, embedding_dim, head_dim):
        self.head_dim = head_dim 
        self.embedding_dim = embedding_dim 
        
        self.weight_Q = (torch.randn((embedding_dim, head_dim), device=device) / math.sqrt(embedding_dim))
        self.weight_K = (torch.randn((embedding_dim, head_dim), device=device) / math.sqrt(embedding_dim))
        self.weight_V = (torch.randn((embedding_dim, head_dim), device=device) / math.sqrt(embedding_dim))
        
        self.opt_Q = AdamOptimizer(self.weight_Q.shape, device=device)
        self.opt_K = AdamOptimizer(self.weight_K.shape, device=device)
        self.opt_V = AdamOptimizer(self.weight_V.shape, device=device)

    def forward(self, E):
        self.E = E
        self.Q = E @ self.weight_Q
        self.K = E @ self.weight_K
        self.V = E @ self.weight_V
        
        self.scores = (self.Q @ self.K.T) / math.sqrt(self.head_dim)
        self.A = softmax(self.scores)

        self.Z = self.A @ self.V
        return self.Z 
    
    def backprop(self, grad_Z, lr):
        dA = grad_Z @ self.V.T
        dV = self.A.T @ grad_Z

        dpattern = softgrad(dA, self.A)
        dpattern /= math.sqrt(self.head_dim)

        dQ = dpattern @ self.K
        dK = dpattern.T @ self.Q
        
        grad_WQ = torch.clamp(self.E.T @ dQ, -5.0, 5.0)
        grad_WK = torch.clamp(self.E.T @ dK, -5.0, 5.0)
        grad_WV = torch.clamp(self.E.T @ dV, -5.0, 5.0)

        grad_E = dQ @ self.weight_Q.T + dK @ self.weight_K.T + dV @ self.weight_V.T
        
        self.weight_Q = self.opt_Q.update(self.weight_Q, grad_WQ, lr)
        self.weight_K = self.opt_K.update(self.weight_K, grad_WK, lr)
        self.weight_V = self.opt_V.update(self.weight_V, grad_WV, lr)

        return grad_E

class MultiAttentionBlock:
    def __init__(self, embedding_dim, num_of_heads):
        self.embedding_dim = embedding_dim
        self.num_of_heads = num_of_heads
        self.head_dim = int(embedding_dim / num_of_heads)
        
        self.weight_O = (torch.randn((self.embedding_dim, self.embedding_dim), device=device) / math.sqrt(self.embedding_dim))
        self.opt_O = AdamOptimizer(self.weight_O.shape, device=device)
        self.heads = list(AttentionHead(self.embedding_dim, self.head_dim) for i in range(num_of_heads))
    
    def forward(self, E):
        self.E = E
        output = self.heads[0].forward(E)
        for i in range(1, self.num_of_heads):
            # torch.cat replaces np.concatenate
            output = torch.cat((output, self.heads[i].forward(E)), dim=1)
        self.concat = output
        self.output = self.concat @ self.weight_O
        return self.output + E 
    
    def backprop(self, E_norm, grad_A, lr):
        grad_proj = grad_A
        grad_res = grad_A

        grad_concat = grad_proj @ self.weight_O.T
        grad_WO = torch.clamp(self.concat.T @ grad_proj, -5.0, 5.0)

        self.weight_O = self.opt_O.update(self.weight_O, grad_WO, lr)

        head_dim = self.head_dim
        grad_E_total = torch.zeros_like(E_norm)

        for i, head in enumerate(self.heads):
            start = i * head_dim
            end = (i + 1) * head_dim
            grad_head = grad_concat[:, start:end]
            grad_E_total += head.backprop(grad_head, lr)
        
        grad_E_total += grad_res
        return grad_E_total

class VisionTransformer:
    def __init__(self, image_dim, embedding_dim, patch_size, num_of_heads, MLP_hidden_param, learning_rate=1e-3):
        self.image_dim = image_dim
        self.patch_size = patch_size
        self.num_of_heads = num_of_heads
        self.lr = learning_rate
        self.MLP_hidden_param = MLP_hidden_param
        self.patch_dim = (self.patch_size**2) * 3
        self.embedding_dim = embedding_dim
        self.num_patches = (self.image_dim // self.patch_size) ** 2
        
        self.pos_embedding = torch.randn((self.num_patches, self.embedding_dim), device=device) * 0.02
        self.opt_pos = AdamOptimizer(self.pos_embedding.shape, device=device)
        
        self.patch_proj = Linear(self.patch_dim, self.embedding_dim)

        self.LN1 = LayerNorm(self.embedding_dim)
        self.LN2 = LayerNorm(self.embedding_dim)
        self.AttentionBlock = MultiAttentionBlock(self.embedding_dim, self.num_of_heads)
        self.MLP = Sequential(Linear(self.embedding_dim, self.MLP_hidden_param * self.embedding_dim), Linear(self.MLP_hidden_param * self.embedding_dim, self.embedding_dim, lin), learning_rate=self.lr)
    
    def forward(self, x):
        patches = patch_embeddings(x, self.image_dim, self.patch_size)
        self.current_patches = patches
        embeddings = self.patch_proj.forward(patches)
        embeddings *= math.sqrt(self.embedding_dim)
        embeddings = embeddings + self.pos_embedding
        self.current_embeddings = embeddings
        embeddings_norm = self.LN1.forward(embeddings)
        self.current_embeddings_norm = embeddings_norm
        output = self.AttentionBlock.forward(embeddings_norm)
        self.current_attention_output = output
        output_norm = self.LN2.forward(output)
        self.current_attention_output_norm = output_norm
        self.current_MLP_output = self.MLP.forwardPass(output_norm)
        return self.current_MLP_output + output 
    
    def backprop(self, grad_feature_embeddings):
        grad_C = grad_feature_embeddings
        grad_A = grad_feature_embeddings
        
        grad_B = self.MLP.backward_delta(self.current_attention_output_norm, grad_C)
        grad_B = torch.clamp(grad_B, -1.0, 1.0)
        
        grad_ln2 = self.LN2.backward(grad_B, self.lr)
        grad_A_total = grad_A + grad_ln2
        
        grad_embeddings_norm = self.AttentionBlock.backprop(self.current_embeddings_norm, grad_A_total, self.lr)
        grad_embeddings_norm = torch.clamp(grad_embeddings_norm, -1.0, 1.0)
        
        grad_embeddings = self.LN1.backward(grad_embeddings_norm, self.lr)
        delta = torch.clamp(grad_embeddings, -1.0, 1.0) 
        self.pos_embedding = self.opt_pos.update(self.pos_embedding, delta, self.lr)
        
        self.patch_proj.update(self.current_patches.T @ delta, torch.sum(delta, dim=0, keepdim=True), self.lr)
        grad_patches = delta @ self.patch_proj.weights().T
        return grad_patches

class ClassificationVIT:
    def __init__(self, image_dim, embedding_dim, patch_size, num_of_heads, MLP_hidden_param, output_dim, learning_rate=1e-3):
        self.image_dim = image_dim
        self.output_dim = output_dim
        self.patch_size = patch_size
        self.num_of_heads = num_of_heads
        self.lr = learning_rate
        self.MLP_hidden_param = MLP_hidden_param
        self.patch_dim = (self.patch_size**2) * 3
        self.embedding_dim = embedding_dim
        self.num_of_patches = (self.image_dim**2) // (self.patch_size**2)

        self.VIT = VisionTransformer(image_dim, embedding_dim, patch_size, num_of_heads, MLP_hidden_param, learning_rate=self.lr)
        self.MLP = Sequential(Linear(self.embedding_dim, self.MLP_hidden_param * self.embedding_dim), Linear(self.MLP_hidden_param * self.embedding_dim, self.output_dim, softmax), loss=CCE, learning_rate=self.lr)
    
    def forward(self, x):
        feature_embeddings = self.VIT.forward(x)
        logits = torch.mean(feature_embeddings, dim=0).reshape(1, -1)
        self.current_feature_embeddings = feature_embeddings
        self.logits = logits
        return self.MLP.forwardPass(logits)
    
    def backprop(self, x, y, pred):
        delta = softCCEgrad(y, pred)
        grad_logits = self.MLP.backward_delta(self.logits, delta)
        grad_logits = torch.clamp(grad_logits, -1.0, 1.0)
        
        # PyTorch uses .repeat() instead of np.repeat for copying rows
        grad_feature_embeddings = grad_logits.repeat(self.num_of_patches, 1) / self.num_of_patches
        self.VIT.backprop(grad_feature_embeddings)

    def fit(self, X, Y, epochs: int, batch_size: int = 1):
        N = X.shape[0]

        for ep in range(epochs):
            total_loss = 0.0
            
            # Use PyTorch's random permutation
            indices = torch.randperm(N)
            X_shuffled = X[indices]
            Y_shuffled = Y[indices]

            for i in range(0, N, batch_size):
                xb = X_shuffled[i : i + batch_size]
                yb = Y_shuffled[i : i + batch_size]
                
                batch_loss = 0.0
                
                for xi, yi in zip(xb, yb):
                    pred = self.forward(xi)
                    batch_loss += CCE(yi, pred).item() # .item() extracts the float from the tensor
                    self.backprop(xi, yi, pred) 
                    
                total_loss += batch_loss

            avg_loss = total_loss / N
            print(f"Epoch {ep+1}/{epochs} Loss: {avg_loss:.6f}")

    def save(self, filename="vit_model.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self, f)
        print(f"Model successfully saved to {filename}")

    def load(self, filename="vit_model.pkl"):
        try:
            with open(filename, "rb") as f:
                loaded_model = pickle.load(f)
            self.__dict__.update(loaded_model.__dict__)
            print(f"Model successfully loaded from {filename}")
        except FileNotFoundError:
            print(f"Error: Could not find {filename}. Are you sure it exists?")