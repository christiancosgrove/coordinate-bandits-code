import numpy as np
from numba import njit


class Problem(object):
    def __init__(self, A):
        super().__init__()
        self.A = A

    def loss(self, x):
        return self.f(self.A @ x) + np.sum([self.g(i, x[i]) for i in range(x.shape[0])])

    def g(self, i, xi):
        raise NotImplementedError

    def gstar(self, w):
        raise NotImplementedError

    def f(self, Ax):
        raise NotImplementedError

    def partial_f(self, i, Ax):
        raise NotImplementedError

    def kappa(self, i):
        raise NotImplementedError

    def gap_i(self, i, x):
        p = self.partial_f(i, self.A @ x)
        return self.gstar(-p) + self.g(i, x[i]) + x[i] * p

    def gap(self, x):
        return np.sum([self.gap_i(i, x) for i in range(x.shape[0])])

    def reward(self, i, x):
        Gi = self.gap_i(i, x)
        ki = self.kappa(i, x)
        Ai2 = np.sum(self.A[:,i]**2)
        s = Gi / (ki**2 * Ai2 / self.beta)
        if s < 1:
            return s * Gi / 2
        else:
            return Gi - Ai2 * ki**2 / (2*self.beta)

    def etas(self, x):
        gaps = [self.gap_i(i, x) for i in range(x.shape[0])]
        return np.array(gaps)/np.sum(gaps)

    def update(self, i, x):
        raise NotImplementedError

@njit
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

@njit
def shrink(x, lam):
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0)

class L1Problem(Problem):
    def __init__(self, A, lam, x):
        super().__init__(A)
        self.lam = lam
        self.B = self.loss(x) / self.lam

    def g(self, i, xi):
        return self.lam * np.abs(xi)

    def gstar(self, aw):
        return self.B * np.maximum(np.abs(aw) - self.lam, 0)

    def kappa(self, i, x):
        part = self.partial_f(i, self.A @ x)
        return np.abs(x[i] - self.B * shrink(part, self.lam))

    def update(self, i, x):
        part = self.partial_f(i, self.A @ x)
        x[i] = shrink(x[i] - part, self.lam)
        # assert (x < self.B).all()

class LogisticL1(L1Problem):
    def __init__(self, A, y, lam, x):
        self.y = y
        self.beta = 1
        super().__init__(A, lam, x)

    def f(self, Ax):
        return self._f(Ax, self.y)

    @staticmethod
    # @njit
    def _f(Ax, y):
        pred = sigmoid(Ax)
        loss = -y * np.log(pred)
        loss += -(1-y) * np.log(1 - pred)
        return np.mean(loss)
    
    def partial_f(self, i, Ax):
        return self._partial_f(i, Ax, self.y, self.A)
    
    @staticmethod
    # @njit
    def _partial_f(i, Ax, y, A):
        p = -(y-sigmoid(Ax)) 
        p = p @ A[:, i] / y.shape[0]
        return p

class Lasso(L1Problem):
    def __init__(self, A, y, lam, x):
        self.y = y
        self.beta = 1
        super().__init__(A, lam, x)

    def f(self, Ax):
        return self._f(Ax, self.y)
    @staticmethod
    @njit
    def _f(Ax, y):
        print('f', np.mean((y - Ax)**2))
        return np.mean((y - Ax)**2)

    def update(self, i, x):
        x_not = np.array(x)
        x_not[i] = 0
        norm = (self.A[:, i] ** 2).sum()
        x[i] = shrink(self.A[:, i].T @ (self.y - self.A @ x_not) / norm, self.lam / norm)

    # @njit
    def partial_f(self, i, Ax):
        return self._partial_f(i, Ax, self.A, self.y)
    @staticmethod
    @njit
    def _partial_f(i, Ax, A, y):
        p = (y-Ax)
        p = -p @ A[:, i] / y.shape[0]
        return p

