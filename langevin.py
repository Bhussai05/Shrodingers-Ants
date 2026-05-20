import numpy as np 
import matplotlib.pyplot as plt 

epsilon = 0.1
mu = 0.2
time = 50
x0 = 0.2

dt = 0.001

tarray = np.arange(0, time, dt)

xarray = np.zeros(len(tarray))
xarray[0] = x0


for i in range(1, len(tarray)):
    prevx = xarray[i-1]
    drift = epsilon * (1-2*prevx)
    diffusion = np.sqrt(2*mu*(1-prevx))
    total = drift + diffusion


print(total)





