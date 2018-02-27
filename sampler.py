import numpy as np

class MCMCSampler:
    def __init__(self, p):
        self.p = p
    
    def sample(self, x):
        pass

class MHSampler(MCMCSampler):
    def __init__(self, p, q):
        super(MHSampler, self).__init__(p)
        self.q = q

    def sample(self, x):
        xp = self.q.cond_sample(x) # proposed x
        a = np.min(self.p.prob(xp) / self.p.prob(x) * self.q.cond_prob(x, xp) / self.q.cond_prob(xp, x), 1)
        if np.random.uniform() < a: # accept
            return xp
        else: # reject
            return x

