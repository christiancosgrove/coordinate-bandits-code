from problem import *
import numpy as np

class CoordinateDescentPolicy(object):
    def __init__(self, problem: Problem):
        self.problem = problem
    def get(self, x) -> int:
        raise NotImplementedError


class RandomPolicy(CoordinateDescentPolicy):

    def __init__(self, problem):
        super().__init__(problem)

    def get(self, x):
        return np.random.randint(x.shape[0])


class MaxRPolicy(CoordinateDescentPolicy):
    def __init__(self, problem):
        super().__init__(problem)

    def get(self, x):
        rs = [self.problem.reward(i,x) for i in range(x.shape[0])]
        imax = np.argmax(rs)

        return imax

class MaxEtaPolicy(CoordinateDescentPolicy):
    def __init__(self, problem):
        super().__init__(problem)

    def get(self, x):
        etas = self.problem.etas(x)
        imax = np.argmax(etas)
        return imax

class BMaxRPolicy(CoordinateDescentPolicy):
    def __init__(self, problem):
        super().__init__(problem)

        self.E = None
        self.epsilon=1e-2
        self.iter = 0

    def get(self, x):
        if self.E is None:
            # Default: set E=d/2
            self.E = x.shape[0] //2

        if np.random.uniform() < self.epsilon:
            return np.random.randint(x.shape[0])
        
        if self.iter % self.E == 0:
            self.rs = [self.problem.reward(i,x) for i in range(x.shape[0])]

        imax = np.argmax(rs)
        rs[imax] = self.problem.reward(imax,x)

        return imax

class Solver(object):
    def __init__(self, x, policy: CoordinateDescentPolicy, problem: Problem):
        self.x = x
        self.policy = policy
        self.problem = problem

    def train(self, iterations):
        losses = []
        for i in range(iterations):
            coord = self.policy.get(self.x)
            self.problem.update(coord, self.x)
            losses.append(self.problem.loss(self.x))

        return losses