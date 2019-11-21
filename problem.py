import numpy as np

class Problem(object):
    def __init__(self, A):
        super().__init__()
        self.A = A

    def loss(self, x):
        return self.f(self.A @ x) + np.sum([self.g(i, x) for i in range(x.shape[0])])

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


def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def shrink(x, lam):
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0)

class LogisticL1(Problem):
    def __init__(self, A, y, lam):
        super().__init__(A)
        self.y = y
        self.B = None
        self.lam = lam
        self.beta = 1

    def g(self, i, xi):
        return self.lam * np.abs(xi)

    def gstar(self, aw):
        return self.B * np.maximum(np.abs(aw) - self.lam, 0)

    def f(self, Ax):
        pred = sigmoid(Ax)
        loss = -self.y * np.log(pred)
        loss += -(1-self.y) * np.log(1 - pred)
        return np.mean(loss)

    def partial_f(self, i, Ax):
        p = -(self.y-sigmoid(Ax)) 
        p = p @ self.A[:, i] / self.y.shape[0]
        return p

    def kappa(self, i, x):
        part = self.partial_f(i, self.A @ x)
        return np.abs(x[i] - self.B * shrink(part, self.lam))

    def update(self, i, x):
        if self.B is None:
            self.B = self.loss(x) / self.lam
        part = self.partial_f(i, self.A @ x)
        x[i] = shrink(x[i] - 4*part, 4*self.lam)