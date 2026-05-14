import numpy as np 
import signalz
import matplotlib.pyplot as plt

epsilon = 1
muu = 2
t0 = np.linspace(0, 10000, 1000)
x0= 6



etaa = signalz.brownian_noise(1000, leak=0.1, start=0, std=1, source="gaussian")


def langevin(x_0, e, mu, t, eta):
    xdot = np.zeros(len(t))
    xdot =  e*(1-2*x_0) + np.sqrt(2*mu*x_0*(1-x_0))*eta

    for i in range(len(t)):
        xdot[i] = e*(1-2*xdot[i-1]) + np.sqrt(2*mu*xdot[i-1]*(1-xdot[i-1]))*eta

langevin(x0, epsilon, muu, t0, etaa)





    