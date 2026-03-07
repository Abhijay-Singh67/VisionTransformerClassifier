from NetworkCore import *
import numpy as np

input = np.random.randn(32,32,3)
LN = LayerNorm(48)
AttentionBlock = MultiAttentionBlock(48,6)
embeddings = patch_embeddings(input,32,4)
embeddings = LN.forward(embeddings)
out = AttentionBlock.forward(embeddings)
out = LN.forward(out)
print(out.shape)
print(out)