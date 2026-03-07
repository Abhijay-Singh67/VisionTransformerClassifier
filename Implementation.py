from NetworkCore import *
import numpy as np

input = np.random.randn(32,32,3)
VIT = ClassificationVIT(32,4,8,4,10)
out = VIT.forward(input)
print(out.shape)
print(out)
