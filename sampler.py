import numpy as np
import prob

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

class RWSampler(MHSampler):
    def __init__(self, p, eps):
        dim = p.dim
        q = prob.CondNormalDist(np.full(dim, eps))
        super(RWSampler, self).__init__(p, q)

class GibbsSampler(MCMCSampler):
    def __init__(self, p):
        super(GibbsSampler, self).__init__(p)
        self.i = 0

    def sample(self, x):
        dim = self.p.dim
        x_ = x.copy()
        x_[self.i] = self.p.gibbs_sample(x, self.i)
        self.i = (self.i + 1) % dim
        return x_

