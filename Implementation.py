from NetworkCore import *
import numpy as np

input = np.random.randn(32,32,3)
AttentionBlock = MultiAttentionBlock(48,6)
embeddings = patch_embeddings(input,32,4)
out = AttentionBlock.forward(embeddings)
print(out.shape)
print(out)