import numpy as np

#The goal of the network is to classify 32x32 RGB images from CIFAR-10 dataset

def patch_embeddings(x,image_dim,patch_size):
    #Generates patch embeddings
    #Dim:(32,32,3)->(64,48) [N,D]
    embeddings=[]
    for i in range(0,image_dim,patch_size):
        for j in range(0,image_dim,patch_size):
            embeddings.append(x[i:i+patch_size,j:j+patch_size].flatten())
    return np.array(embeddings)

def softmax(x):
    #Takes the row wise softmax of the attention pattern
    x = x - np.max(x, axis=1, keepdims=True)
    x_exp = np.exp(x)
    return x_exp / np.sum(x_exp, axis=1, keepdims=True)

import numpy as np

class LayerNorm: 
    def __init__(self, embedding_dim, eps=1e-5):
        self.gamma = np.ones(embedding_dim)
        self.beta = np.zeros(embedding_dim)
        self.eps = eps

    def forward(self, x):
        mean = np.mean(x, axis=1, keepdims=True)
        var = np.var(x, axis=1, keepdims=True)
        x_hat = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta


class AttentionHead:
    def __init__(self,embedding_dim,head_dim):
        self.head_dim=head_dim #d
        self.embedding_dim=embedding_dim #D
        #All of these have dimensions D,d
        self.weight_Q=np.random.randn(embedding_dim,head_dim)
        self.weight_K=np.random.randn(embedding_dim,head_dim)
        self.weight_V=np.random.randn(embedding_dim,head_dim)

    def forward(self,E):
        #All of these have dimensions N,d
        Q = E@self.weight_Q
        K = E@self.weight_K
        V = E@self.weight_V
        #The attention pattern has dimension N,N
        A = softmax((Q@K.T)/np.sqrt(self.head_dim))

        return A@V #Dimension N,d

class MultiAttentionBlock:
    def __init__(self,embedding_dim,num_of_heads):
        self.embedding_dim=embedding_dim
        self.num_of_heads=num_of_heads
        self.head_dim=int(embedding_dim/num_of_heads)
        self.weight_O=np.random.randn(self.embedding_dim,self.embedding_dim)
        self.heads=list(AttentionHead(self.embedding_dim,self.head_dim) for i in range(num_of_heads))
    
    def forward(self,E):
        output = self.heads[0].forward(E)
        for i in range(1,self.num_of_heads):
            output=np.concatenate((output,self.heads[i].forward(E)),axis=1)
        return (output@self.weight_O)+E #Dimensions: N,D
    

    

        









    