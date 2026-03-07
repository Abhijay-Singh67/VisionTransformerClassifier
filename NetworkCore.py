import numpy as np
from MLP import Sequential,Linear
from helper import lin,softmax,CCE

#The goal of the network is to classify 32x32 RGB images from CIFAR-10 dataset

def patch_embeddings(x,image_dim,patch_size):
    #Generates patch embeddings
    #Dim:(32,32,3)->(64,48) [N,D]
    embeddings=[]
    for i in range(0,image_dim,patch_size):
        for j in range(0,image_dim,patch_size):
            embeddings.append(x[i:i+patch_size,j:j+patch_size].flatten())
    return np.array(embeddings)

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

class VisionTransformer:
    def __init__(self,image_dim,patch_size,num_of_heads,MLP_hidden_param,learning_rate=1e-3):
        self.image_dim=image_dim
        self.patch_size=patch_size
        self.num_of_heads=num_of_heads
        self.lr=learning_rate
        self.MLP_hidden_param=MLP_hidden_param
        self.embedding_dim = (self.patch_size**2)*3

        self.LN1 = LayerNorm(self.embedding_dim)
        self.LN2 = LayerNorm(self.embedding_dim)
        self.AttentionBlock = MultiAttentionBlock(self.embedding_dim,self.num_of_heads)
        self.MLP = Sequential(Linear(self.embedding_dim,self.MLP_hidden_param*self.embedding_dim),Linear(self.MLP_hidden_param*self.embedding_dim,self.embedding_dim,lin),learning_rate=self.lr)
    
    def forward(self,x):
        embeddings = patch_embeddings(x,self.image_dim,self.patch_size)
        embeddings_norm = self.LN1.forward(embeddings)
        output = self.AttentionBlock.forward(embeddings_norm)
        output_norm = self.LN2.forward(output)
        return self.MLP.forwardPass(output_norm)+embeddings #Dimensions: N,D
    
class ClassificationVIT:
    def __init__(self,image_dim,patch_size,num_of_heads,MLP_hidden_param,output_dim,learning_rate=1e-3):
        self.image_dim=image_dim
        self.output_dim = output_dim
        self.patch_size=patch_size
        self.num_of_heads=num_of_heads
        self.lr=learning_rate
        self.MLP_hidden_param=MLP_hidden_param
        self.embedding_dim = (self.patch_size**2)*3

        self.VIT = VisionTransformer(image_dim,patch_size,num_of_heads,MLP_hidden_param,learning_rate=1e-3)
        self.MLP = Sequential(Linear(self.embedding_dim,self.MLP_hidden_param*self.embedding_dim),Linear(self.MLP_hidden_param*self.embedding_dim,self.output_dim,softmax),loss=CCE,learning_rate=self.lr)
    
    def forward(self,x):
        feature_embeddings = self.VIT.forward(x)
        logits = np.mean(feature_embeddings,axis=0).reshape(1,-1) #feature pooling
        return self.MLP.forwardPass(logits)



    

        









    