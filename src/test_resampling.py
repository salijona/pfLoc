import numpy as np
from mobpf.base import Engine

weights = np.array([0.081, 0.16, 0,4,  0.19, 0.5,  0.02, 0.08, 0.15, 0.19])
weights =np.random.dirichlet(np.ones(100),size=1)
weights = weights[0,:]
def sampling(weights):
    "Resampling process."
    N = len(weights)
    cumulative_sum= np.cumsum(weights)
    print(cumulative_sum)
    cumulative_sum[-1]=1.0  # to avoid round-off error
    np.random.seed(4)
    rd = np.random.random(N)
    print(rd)
    indexes = np.searchsorted(cumulative_sum, rd)
    return indexes

inds = sampling(weights)    

print(inds)
print(np.unique(inds))