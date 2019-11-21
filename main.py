from sklearn.datasets import load_svmlight_file
import os
import numpy as np
from tqdm import tqdm
DATA_DIR='./data'

data_files = os.listdir(DATA_DIR)

f = data_files[0]
print('Filename ', f)


print('Getting data')

X, y = load_svmlight_file(os.path.join(DATA_DIR, f))

X = X.toarray()

print('X shape ', X.shape)
print('Y shape ', y.shape)

print('y min', y.min(), ' max ', y.max())
print('X mean ', np.mean(X))#, ' std', np.std(X))
print('y min', ((y+1)/2).min(), ' max ', ((y+1)/2).max())

W = np.random.normal(size=X.shape[1], loc=0, scale=1)
def loss_logistic(X, y, W, lam):
    pred = sigmoid(W.T @ X.T)
    yp = (y+1)/2

    loss = -yp * np.log(pred)
    loss += -(1-yp) * np.log(1 - pred)

    return np.mean(loss) + lam * np.sum(np.abs(W))
    # return np.mean(np.log(1 + np.exp(y * (W @ X.T)))) + lam * np.sum(np.abs(W))

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def shrink(x, lam):
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0)

def partial(W, i, X, y):
    p = -((y+1)/2-sigmoid(W.T @ X.T)) 
    # print('partal shape ', p.shape)
    # print('y shape', y.shape)
    p = p @ X[:, i] / y.shape[0]
    # print('partal shape 2', p.shape)
    return p

def update_coord(W, i, X, y, lam):
    W[i] = shrink(W[i] - 4*partial(W, i, X, y), 4*lam)

# Coordinate-wise duality gap
def gap(i, W, X, y, lam, B):
    part = partial(W, i, X, y)
    return B * np.maximum(np.abs(part) - lam, 0) + lam * np.abs(W[i]) + W[i] * part

# Dual residual - a.k.a. kappa
def res(i, W, X, y, lam, B):
    part = partial(W, i, X, y)
    return np.abs(W[i] - B * shrink(part, lam))

def reward(i, W, X, y, lam, B):
    beta = 1
    Gi = gap(i, W, X, y, lam, B)
    kappai = res(i, W, X, y, lam, B)
    norma = np.sum(X[:, i]**2)
    s = Gi / (kappai**2 * norma / beta)
    if s < 1:
        return s * Gi / 2
    else:
        return Gi - norma * kappai**2 / (2*beta)

lam = 0.01
vals = []
coords_chosen = []
W_init = np.array(W, copy=True)

NUM_ITERATIONS = 10000
for j in tqdm(range(NUM_ITERATIONS)):
    # Use the 'Lipschitzing' trick - we know that the iterates are contained in ball of radius B
    if j == 0:
        B = loss_logistic(X, y, W, lam) / lam
    # if j % 10 == 0:
    i = np.random.randint(W.shape[0])
    # i = j % W.shape[0]
    # l_before = loss_logistic(X, y, W, lam)
    print('GAP i ', gap(i, W, X, y, lam, B), 'residual ', res(i, W, X, y, lam, B), 'reward ', reward(i, W, X, y, lam, B))
    coords_chosen.append(i)
    update_coord(W, i, X, y, lam)
    l_after = loss_logistic(X, y, W, lam)
    print(l_after)

# print('Result', loss_logistic(X, y, W, lam))
# for _ in range(500):
#     W = np.array(W_init, copy=True)
#     np.random.shuffle(coords_chosen)
#     for j in tqdm(range(NUM_ITERATIONS)):
#         i = coords_chosen[j]
#         update_coord(W, i, X, y, lam)
#     print('Result2', loss_logistic(X, y, W, lam))