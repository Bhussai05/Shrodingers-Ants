import numpy as np 
import matplotlib.pyplot as plt 
import diptest as diptest
from scipy.special import gamma


epsilon = 10
mu = 1
alpha = epsilon / mu
time = 50
x0 = 0.2


dt = 0.001

tarray = np.arange(0, time, dt)



xarray = np.zeros(len(tarray))
xarray[0] = x0
burn = int(0.3 * len(xarray))



for i in range(1, len(tarray)):
    prevx = xarray[i-1]
    dW = np.sqrt(dt) * np.random.normal()

    drift = epsilon * (1-2*prevx)
    diffusion = np.sqrt(2*mu*prevx*(1-prevx))

    xarray[i] = prevx + drift*dt + diffusion*dW
    xarray[i] = np.clip(xarray[i], 0, 1)


x_grid = np.linspace(0.001, 0.999, 1000)
C = gamma(2 * alpha) / gamma(alpha)**2
paper_pdf = C * (x_grid * (1-x_grid))**(alpha-1)

samples = xarray[burn:]

fig1 = plt.plot(tarray, xarray)
plt.show()
fig2 = plt.hist(samples, bins=500, density= True)
plt.show()
fig3 = plt.plot(x_grid, paper_pdf, linewidth=2)
plt.show()

print(paper_pdf[0], paper_pdf[1])
    


