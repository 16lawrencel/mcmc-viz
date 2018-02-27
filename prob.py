import numpy as np

class MultNormalDist:
    def __init__(self, mu = np.zeros(2), Sigma = np.ones(2)):
        self.mu = np.reshape(mu, -1)
        if len(Sigma.shape) == 1: Sigma = np.diag(Sigma)
        self.Sigma = Sigma
        self.J = np.linalg.inv(Sigma)

    def sample(self):
        return np.random.multivariate_normal(self.mu, self.Sigma)

    def prob(self, x):
        x = np.reshape(x, (-1, 1))
        mu = np.reshape(self.mu, (-1, 1))
        return np.exp(-0.5 * (x - mu).T @ self.J @ (x - mu))

class CondNormalDist:
    def __init__(self, Sigma):
        if len(Sigma.shape) == 1: Sigma = np.diag(Sigma)
        self.Sigma = Sigma
        self.J = np.linalg.inv(Sigma)

    def cond_sample(self, x):
        return np.random.multivariate_normal(x, self.Sigma)
    
    def cond_prob(self, xp, x):
        xp = np.reshape(xp, (-1, 1))
        x = np.reshape(x, (-1, 1))
        return np.exp(-0.5 * (xp - x).T @ self.J @ (xp - x))

