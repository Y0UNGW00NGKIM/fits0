import numpy as np
from numpy.linalg import inv
from matplotlib import pyplot as plt

xmin=1.0
xmax=20.0
npoints=12
sigma=0.2
lx=np.zeros(npoints)
ly=np.zeros(npoints)
ley=np.zeros(npoints)
pars=[0.5,1.3,0.5]

from math import log
def f(x,par):
    return par[0]+par[1]*log(x)+par[2]*log(x)*log(x)

from random import gauss
def getX(x):  # x = array-like
    step=(xmax-xmin)/npoints
    for i in range(npoints):
        x[i]=xmin+i*step
        
def getY(x,y,ey):  # x,y,ey = array-like
    for i in range(npoints):
        y[i]=f(x[i],pars)+gauss(0,sigma)
        ey[i]=sigma

# get a random sampling of the (x,y) data points, rerun to generate different data sets for the plot below
getX(lx)
getY(lx,ly,ley)

A = np.column_stack([np.ones_like(lx), np.log(lx), np.log(lx)**2])
W = np.diag(1.0/(ley**2))
N = A.T @ W @ A
beta = inv(N) @ (A.T @ W @ ly)
xs = np.linspace(xmin, xmax, 400)
Agrid = np.column_stack([np.ones_like(xs), np.log(xs), np.log(xs)**2])
yfit = Agrid @ beta
res = ly - (A @ beta)
chi2 = float(res.T @ (W @ res))
chi2_red = chi2/(npoints-3)

fig, ax = plt.subplots()
ax.errorbar(lx, ly, yerr=ley, fmt='o', label='data', capsize=3)
ax.plot(xs, yfit, label=f'fit: a={beta[0]:.3f}, b={beta[1]:.3f}, c={beta[2]:.3f}; $\\chi^2_\\nu$={chi2_red:.2f}')
ax.set_title('Pseudoexperiment')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
fig.tight_layout()
fig.savefig('data_fit_py.png', dpi=150)


# *** modify and add your code here ***
nexperiments = 1000  # for example


# perform many least squares fits on different pseudo experiments here
# fill histograms w/ required data

par_a = np.empty(nexperiments)
par_b = np.empty(nexperiments)
par_c = np.empty(nexperiments)
chi2_reduced = np.empty(nexperiments)

                        
A_fixed = np.column_stack([np.ones_like(lx), np.log(lx), np.log(lx)**2])
W_fixed = np.diag(np.full(npoints, 1.0/sigma**2))
N_fixed = A_fixed.T @ W_fixed @ A_fixed
Ninv_fixed = inv(N_fixed)

for t in range(nexperiments):
    getY(lx, ly, ley)
    beta_t = Ninv_fixed @ (A_fixed.T @ W_fixed @ ly)
    r = ly - (A_fixed @ beta_t)
    chi2_t = float(r.T @ (W_fixed @ r))
    par_a[t], par_b[t], par_c[t] = beta_t
    chi2_reduced[t] = chi2_t/(npoints-3)

fig2, axs = plt.subplots(2, 2, figsize=(8,8))
axs[0, 0].hist2d(par_a, par_b, bins=60)
axs[0, 0].set_title('Parameter b vs a')
axs[0, 0].set_xlabel('a')
axs[0, 0].set_ylabel('b')

axs[0, 1].hist2d(par_a, par_c, bins=60)
axs[0, 1].set_title('Parameter c vs a')
axs[0, 1].set_xlabel('a')
axs[0, 1].set_ylabel('c')

axs[1, 0].hist2d(par_b, par_c, bins=60)
axs[1, 0].set_title('Parameter c vs b')
axs[1, 0].set_xlabel('b')
axs[1, 0].set_ylabel('c')

axs[1, 1].hist(chi2_reduced, bins=40)
axs[1, 1].set_title('Reduced $\\chi^2$ distribution')
axs[1, 1].set_xlabel('$\\chi^2_\\nu$')

fig2.tight_layout()
fig2.savefig('study_panels_py.png', dpi=150)

# **************************************
  

input("hit Enter to exit")
