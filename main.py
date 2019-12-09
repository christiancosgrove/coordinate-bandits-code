#%%
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import policy
import problem
from dataset import load_dataset

DATA_DIR='./data'
data_files = os.listdir(DATA_DIR)

def experiment(dataset_name, problem_type):
    X, y = load_dataset(os.path.join(DATA_DIR, dataset_name))

    W_init = np.random.normal(size=X.shape[1], loc=0, scale=0.01)
    logistic = problem_type(X, y, 0.1, W_init)
    
    samps = []
    for pol in [
        policy.RandomPolicy(logistic),
        # policy.MaxRPolicy(logistic),
        # policy.MaxEtaPolicy(logistic),
        policy.BMaxRPolicy(logistic, 1e-2),
        policy.BMaxEtaPolicy(logistic),
        policy.BMaxEtaPolicy2(logistic)]:
        s = policy.Solver(np.array(W_init, copy=True), pol, logistic).train(500)
        samps.append(s)
    for s in samps:
        # plt.plot(np.array(s) - samps[2][-1]) # Max_eta converges the fastest -- use this as the min loss
        plt.plot(np.array(s)) # Max_eta converges the fastest -- use this as the min loss
    # plt.legend(['random', 'max_r', 'max_eta', 'B_max_r', 'B_max_eta'])
    plt.legend(['random', 'B_max_r', 'B_max_eta', 'B_max_eta_2'])
    plt.xlabel('iterations')
    plt.ylabel('Logistic loss')
    plt.yscale('log')
    plt.title('Comparison of different coordinate policies')
    plt.show()
#%%
# Logistic regression experiment
experiment(data_files[0], problem.LogisticL1)

#%%

def experiment_epsilon(dataset_name, problem_type):
    X, y = load_dataset(os.path.join(DATA_DIR, dataset_name))

    W_init = np.random.normal(size=X.shape[1], loc=0, scale=0.01)
    logistic = problem_type(X, y, 0.1, W_init)
    eps = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]
    samps = []
    for pol in [
        policy.BMaxRPolicy(logistic, epsilon) for epsilon in eps]:
        s = policy.Solver(np.array(W_init, copy=True), pol, logistic).train(500)
        samps.append(s)
    for s in samps:
        plt.plot(np.array(s))
    plt.legend(['$\\varepsilon={}$'.format(e) for e in eps])
    plt.xlabel('iterations')
    plt.ylabel('Logistic loss')
    # plt.yscale('log')
    plt.title('Comparison of B_max_r varying $\\varepsilon$')
    plt.show()

#%%
experiment_epsilon(data_files[0], problem.LogisticL1)
# %%

# Lasso experiment
experiment(data_files[1], problem.Lasso)

#%%