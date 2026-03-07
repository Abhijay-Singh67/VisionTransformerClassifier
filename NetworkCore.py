import numpy as np
from MLP import Sequential,Linear
from helper import lin,softmax,CCE,softCCEgrad,softgrad

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
        self.mean = np.mean(x, axis=1, keepdims=True)
        self.var = np.var(x, axis=1, keepdims=True)
        self.x_hat = (x - self.mean) / np.sqrt(self.var + self.eps)
        return self.gamma * self.x_hat + self.beta

    def backward(self, grad_out, lr=None):
        grad_gamma = np.sum(grad_out * self.x_hat, axis=0)
        grad_beta = np.sum(grad_out, axis=0)
        grad_xhat = grad_out * self.gamma
        inv_std = 1.0 / np.sqrt(self.var + self.eps)
        grad_x = inv_std * (grad_xhat- np.mean(grad_xhat, axis=1, keepdims=True)- self.x_hat * np.mean(grad_xhat * self.x_hat, axis=1, keepdims=True))

        if lr is not None:
            self.gamma -= lr * grad_gamma
            self.beta -= lr * grad_beta
        return grad_x


class AttentionHead:
    def __init__(self,embedding_dim,head_dim):
        self.head_dim=head_dim #d
        self.embedding_dim=embedding_dim #D
        #All of these have dimensions D,d
        self.weight_Q=np.random.randn(embedding_dim,head_dim)*0.02
        self.weight_K=np.random.randn(embedding_dim,head_dim)*0.02
        self.weight_V=np.random.randn(embedding_dim,head_dim)*0.02

    def forward(self,E):
        #All of these have dimensions N,d
        self.E=E
        self.Q = E@self.weight_Q
        self.K = E@self.weight_K
        self.V = E@self.weight_V
        #The attention pattern has dimension N,N
        self.scores=(self.Q@self.K.T)/np.sqrt(self.head_dim)
        self.A = softmax(self.scores)

        self.Z=self.A@self.V
        return self.Z #Dimension N,d
    
    def backprop(self,grad_Z,lr):
        #Z=A @ V
        dA = grad_Z@self.V.T
        dV = self.A.T@grad_Z

        dpattern = softgrad(dA,self.A)
        dpattern/=np.sqrt(self.head_dim)

        #pattern = QK'
        dQ = dpattern@self.K
        dK = dpattern.T@self.Q
        grad_WQ = self.E.T@dQ
        grad_WQ=np.clip(grad_WQ,-5,5)
        grad_WK = self.E.T@dK
        grad_WK=np.clip(grad_WK,-5,5)
        grad_WV = self.E.T@dV
        grad_WV=np.clip(grad_WV,-5,5)

        grad_E = dQ@self.weight_Q.T + dK@self.weight_K.T + dV@self.weight_V.T
        self.weight_Q-=lr*grad_WQ
        self.weight_K-=lr*grad_WK
        self.weight_V-=lr*grad_WV

        return grad_E


class MultiAttentionBlock:
    def __init__(self,embedding_dim,num_of_heads):
        self.embedding_dim=embedding_dim
        self.num_of_heads=num_of_heads
        self.head_dim=int(embedding_dim/num_of_heads)
        self.weight_O=np.random.randn(self.embedding_dim,self.embedding_dim)*0.02
        self.heads=list(AttentionHead(self.embedding_dim,self.head_dim) for i in range(num_of_heads))
    
    def forward(self,E):
        self.E=E
        output = self.heads[0].forward(E)
        for i in range(1,self.num_of_heads):
            output=np.concatenate((output,self.heads[i].forward(E)),axis=1)
        self.concat=output
        self.output=self.concat@self.weight_O
        return self.output+E #Dimensions: N,D
    
    def backprop(self,E_norm,grad_A,lr):
        grad_proj=grad_A
        grad_res=grad_A

        grad_concat = grad_proj@self.weight_O.T
        grad_WO = self.concat.T@grad_proj
        grad_WO = np.clip(grad_WO,-5,5)

        self.weight_O-=lr*grad_WO

        head_dim = self.head_dim

        grad_E_total = np.zeros_like(E_norm)

        for i,head in enumerate(self.heads):
            start = i*head_dim
            end = (i+1)*head_dim

            grad_head = grad_concat[:,start:end]

            grad_E_total += head.backprop(grad_head,lr)
        
        grad_E_total+=grad_res

        return grad_E_total
class VisionTransformer:
    def __init__(self,image_dim,patch_size,num_of_heads,MLP_hidden_param,learning_rate=1e-3):
        self.image_dim=image_dim
        self.patch_size=patch_size
        self.num_of_heads=num_of_heads
        self.lr=learning_rate
        self.MLP_hidden_param=MLP_hidden_param
        self.embedding_dim = (self.patch_size**2)*3
        self.num_patches = (self.image_dim // self.patch_size) ** 2
        self.pos_embedding = np.random.randn(self.num_patches, self.embedding_dim) * 0.02

        self.LN1 = LayerNorm(self.embedding_dim)
        self.LN2 = LayerNorm(self.embedding_dim)
        self.AttentionBlock = MultiAttentionBlock(self.embedding_dim,self.num_of_heads)
        self.MLP = Sequential(Linear(self.embedding_dim,self.MLP_hidden_param*self.embedding_dim),Linear(self.MLP_hidden_param*self.embedding_dim,self.embedding_dim,lin),learning_rate=self.lr)
    
    def forward(self,x):
        embeddings = patch_embeddings(x,self.image_dim,self.patch_size)
        embeddings = embeddings + self.pos_embedding
        self.current_embeddings=embeddings
        embeddings_norm = self.LN1.forward(embeddings)
        self.current_embeddings_norm=embeddings_norm
        output = self.AttentionBlock.forward(embeddings_norm)
        self.current_attention_output=output
        output_norm = self.LN2.forward(output)
        self.current_attention_output_norm=output_norm
        self.current_MLP_output=self.MLP.forwardPass(output_norm)
        return self.current_MLP_output+output #Dimensions: N,D
    
    def backprop(self,grad_feature_embeddings):
        #We define
        #E=Embeddings+Positional 
        #A=Attention output
        #B=LN2 output
        #C=MLP output
        #Z=C+A (feature_embeddings)
        grad_C=grad_feature_embeddings
        grad_A=grad_feature_embeddings
        grad_B=self.MLP.backward_delta(self.current_attention_output_norm,grad_C)
        grad_ln2=self.LN2.backward(grad_B,self.lr)
        grad_A_total = grad_A+grad_ln2
        grad_embeddings_norm = self.AttentionBlock.backprop(self.current_embeddings_norm,grad_A_total,self.lr)
        grad_embeddings = self.LN1.backward(grad_embeddings_norm,self.lr)
        return grad_embeddings

    
class ClassificationVIT:
    def __init__(self,image_dim,patch_size,num_of_heads,MLP_hidden_param,output_dim,learning_rate=1e-3):
        self.image_dim=image_dim
        self.output_dim = output_dim
        self.patch_size=patch_size
        self.num_of_heads=num_of_heads
        self.lr=learning_rate
        self.MLP_hidden_param=MLP_hidden_param
        self.embedding_dim = (self.patch_size**2)*3
        self.num_of_patches = (self.image_dim**2)//(self.patch_size**2)

        self.VIT = VisionTransformer(image_dim,patch_size,num_of_heads,MLP_hidden_param,learning_rate=1e-3)
        self.MLP = Sequential(Linear(self.embedding_dim,self.MLP_hidden_param*self.embedding_dim),Linear(self.MLP_hidden_param*self.embedding_dim,self.output_dim,softmax),loss=CCE,learning_rate=self.lr)
    
    def forward(self,x):
        feature_embeddings = self.VIT.forward(x)
        logits = np.mean(feature_embeddings,axis=0).reshape(1,-1) #feature pooling
        self.current_feature_embeddings=feature_embeddings
        self.logits=logits
        return self.MLP.forwardPass(logits)
    
    def backprop(self,x,y,pred):
        delta = softCCEgrad(y,pred)
        self.MLP.backProp(self.logits,y,pred)
        W2 = self.MLP.layers[1]
        W1 = self.MLP.layers[0]
        grad_hidden = delta@W2.weights().T
        grad_logits = grad_hidden@W1.weights().T
        grad_feature_embeddings = np.repeat(grad_logits/self.num_of_patches,self.num_of_patches,axis=0)
        self.VIT.backprop(grad_feature_embeddings)


