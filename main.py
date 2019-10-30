from sklearn.datasets import load_svmlight_file
import os
import numpy as np
DATA_DIR='./data'

data_files = os.listdir(DATA_DIR)

f = data_files[0]
print('Filename ', f)


print('Getting data')

X, y = load_svmlight_file(os.path.join(DATA_DIR, f))

print('X shape ', X.shape)
print('Y shape ', y.shape)

print('y min', y.min(), ' max ', y.max())
print('X mean ', np.mean(X))#, ' std', np.std(X))
W = np.random.normal(size=X.shape[1])

def loss_logistic(X, y, W, lam):
    return np.mean(np.log(1 + np.exp(-y * (W @ X.T)))) + lam * np.sum(np.abs(W))

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def shrink(x, lam):
    return np.sign(x) * np.max(x - lam, 0)

def update_coord(W, i, X, y, lam):
    partial = -((y+1)/2-sigmoid(W.T @ X.T)) 
    partial = partial @ X[:, i]
    
    W[i] = shrink(W[i] - 1e-6 * 4 * partial, 4 * lam)

lam = 0.1
for i in range(1000):
    i = np.random.randint(W.shape[0])
    print(loss_logistic(X, y, W, lam))
    update_coord(W, i, X, y, lam)