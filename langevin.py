import numpy as np 
import matplotlib.pyplot as plt 
from diptest import diptest
from scipy.special import gamma
from tqdm import tqdm



def langevin(epsilon, mu, time=50, dt = 0.001, x0 = 0.5, seed = None):
    rng = np.random.default_rng(seed)
    tarray = np.arange(0, time, dt)
    xarray = np.zeros(len(tarray))
    xarray[0] = x0


    for i in range(1, len(tarray)):
        prevx = xarray[i-1]
        dW = np.sqrt(dt) * rng.normal()

        drift = epsilon * (1-2*prevx)
        diffusion = np.sqrt(2*mu*prevx*(1-prevx))

        xarray[i] = prevx + drift*dt + diffusion*dW
        xarray[i] = np.clip(xarray[i], 0, 1)

    return tarray, xarray


# single run

epsilon = 1
mu = 10
alpha = epsilon / mu

tarray, xarray = langevin(epsilon, mu)

burn = int(0.3 * len(xarray))
samples = xarray[burn:]

x_grid = np.linspace(0.001, 0.999, 1000)
C = gamma(2 * alpha) / gamma(alpha)**2
paper_pdf = C * (x_grid * (1-x_grid))**(alpha-1)


fig1 = plt.plot(tarray, xarray)
plt.show()
fig2 = plt.hist(samples, bins=500, density= True)
plt.show()
fig3 = plt.plot(x_grid, paper_pdf, linewidth=2)
plt.show()

#multiple runs 

from tqdm import tqdm

epsilon_vals = np.linspace(0.05, 2.0, 10)
mu_vals = np.linspace(0.05, 2.0, 10)

pval_grids = np.zeros((len(mu_vals), len(epsilon_vals)))
alpha_grids = np.zeros_like(pval_grids)

total_runs = len(mu_vals) * len(epsilon_vals)

with tqdm(total=total_runs, desc="Running heatmap") as pbar:
    for i, mu_val in enumerate(mu_vals):
        for j, eps_val in enumerate(epsilon_vals):

            _, xarray = langevin(
                epsilon=eps_val,
                mu=mu_val,
                time=100,
                dt=0.005,
                x0=0.5,
                seed=1000 + i * len(epsilon_vals) + j
            )

            burn = int(0.3 * len(xarray))
            samples = xarray[burn:]

            dip, p_value = diptest(samples)

            pval_grids[i, j] = p_value
            alpha_grids[i, j] = eps_val / mu_val

            pbar.set_postfix({
                "eps": f"{eps_val:.2f}",
                "mu": f"{mu_val:.2f}",
                "p": f"{p_value:.2g}"
            })
            pbar.update(1)

plt.figure(figsize=(8, 6))
plt.imshow(
    pval_grids,
    origin="lower",
    aspect="auto",
    extent=[
        epsilon_vals[0],
        epsilon_vals[-1],
        mu_vals[0],
        mu_vals[-1],
    ],
)

plt.colorbar(label="Hartigan dip-test p-value")
plt.xlabel(r"$\epsilon$")
plt.ylabel(r"$\mu$")
plt.title("Dip-test p-value heatmap")

plt.show()




