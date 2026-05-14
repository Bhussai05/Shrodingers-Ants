import numpy as np 
import signalz
import matplotlib.pyplot as plt

epsilon = 1
mu = 2
t = np.linspace(0, 10000, 10)
x_0 = np.zeros(len(t))


eta = signalz.brownian_noise(10, leak=0.1, start=0, std=1, source="gaussian")




    


    