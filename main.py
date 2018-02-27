import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import viz
import prob
import sampler

fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], 'ro', animated=True, markersize=3)

T = 100
pdist = prob.MultNormalDist()
eps = 0.1
qdist = prob.CondNormalDist(np.full(2, eps))
mcmc = sampler.MHSampler(pdist, qdist)
x = np.random.randn(2)

def init():
    viz.plot_dist(pdist, ax)
    return ln,

def update(frame):
    global x
    x = mcmc.sample(x)
    xdata.append(x[0])
    ydata.append(x[1])
    ln.set_data(xdata, ydata)
    return ln,

ani = FuncAnimation(fig, update, frames=np.arange(0, T),
                    init_func=init, blit=True)
plt.show()
