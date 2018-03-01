import numpy as np

class MultNormalDist:
    def __init__(self, mu = np.zeros(2), Sigma = np.ones(2)):
        self.mu = np.reshape(mu, -1)
        self.dim = len(self.mu)
        if len(Sigma.shape) == 1: Sigma = np.diag(Sigma)
        self.Sigma = Sigma
        self.J = np.linalg.inv(Sigma)

    def sample(self):
        return np.random.multivariate_normal(self.mu, self.Sigma)

    def prob(self, x):
        x = np.reshape(x, (-1, 1))
        mu = np.reshape(self.mu, (-1, 1))
        return np.exp(-0.5 * (x - mu).T @ self.J @ (x - mu))
    
    def gibbs_sample(self, x, i):
        """
        i is the dimension that we're changing
        """
        x = np.reshape(x, (-1, 1))
        mu = np.reshape(self.mu, (-1, 1))
        Sigma = self.Sigma

        ix1 = np.array([i])
        ix2 = np.array([j for j in range(self.dim) if j != i])

        # formulas for conditional normal
        Sigma11 = Sigma[np.ix_(ix1, ix1)]
        Sigma12 = Sigma[np.ix_(ix1, ix2)]
        Sigma21 = Sigma[np.ix_(ix2, ix1)]
        Sigma22_inv = np.linalg.inv(Sigma[np.ix_(ix2, ix2)])

        mu_ = mu[ix1] + Sigma12 @ Sigma22_inv @ (x[ix2] - mu[ix2])
        mu_ = np.reshape(mu_, -1) # reshape for sampling
        Sigma_ = Sigma11 - Sigma12 @ Sigma22_inv @ Sigma21

        return np.random.multivariate_normal(mu_, Sigma_)

class CondNormalDist:
    def __init__(self, Sigma):
        if len(Sigma.shape) == 1: Sigma = np.diag(Sigma)
        self.dim = Sigma.shape[0]
        self.Sigma = Sigma
        self.J = np.linalg.inv(Sigma)

    def cond_sample(self, x):
        return np.random.multivariate_normal(x, self.Sigma)
    
    def cond_prob(self, xp, x):
        xp = np.reshape(xp, (-1, 1))
        x = np.reshape(x, (-1, 1))
        return np.exp(-0.5 * (xp - x).T @ self.J @ (xp - x))

