import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import viz
import prob
import sampler

SHOW_LINES = False
MAX_PTS = 100000

fig, ax = plt.subplots()
xdata, ydata = [], []
lxdata, lydata = [], []
pts, = plt.plot([], [], 'ro', animated=True, markersize=3)
ln, = plt.plot([], [], 'k-', animated=True)

T = 10000
L = np.array([[1, 3], [-0.5, 0.5]])
Sigma = L.T @ L
pdist = prob.MultNormalDist(Sigma = Sigma)
eps = 0.01
#mcmc = sampler.RWSampler(pdist, eps)
mcmc = sampler.GibbsSampler(pdist)
x = np.random.randn(2)

def init():
    viz.plot_dist(pdist, ax)
    if SHOW_LINES: return pts, ln,
    else: return pts,

def update(frame):
    global x
    xp = mcmc.sample(x)
    xdata.append(xp[0])
    ydata.append(xp[1])

    while len(xdata) > MAX_PTS: xdata.pop(0)
    while len(ydata) > MAX_PTS: ydata.pop(0)

    pts.set_data(xdata, ydata)
    ln.set_data(xdata, ydata)

    x = xp
    if SHOW_LINES: return pts, ln,
    else: return pts,

ani = FuncAnimation(fig, update, frames=np.arange(0, T),
                    init_func=init, blit=True, interval = 200)
plt.show()

