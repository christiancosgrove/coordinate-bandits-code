#%%
from sklearn.datasets import load_svmlight_file
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import policy
import problem

#%%
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

W_init = np.random.normal(size=X.shape[1], loc=0, scale=1)
logistic = problem.LogisticL1(X, (y+1)/2, 0.1)

#%%
for pol in [policy.RandomPolicy(logistic), policy.MaxRPolicy(logistic), policy.MaxEtaPolicy(logistic), policy.BMaxRPolicy(logistic)]:
    s = policy.Solver(np.array(W_init, copy=True), pol, logistic).train(10)
    plt.plot(s)
plt.legend(['random', 'max_r', 'max_eta', 'B_max_r'])
plt.xlabel('iterations')
plt.ylabel('Logistic loss')
plt.yscale('log')
plt.title('Comparison of different coordinate policies')
plt.show()

# %%
