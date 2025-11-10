import numpy as np
import matplotlib.pyplot as plt

# === Grid parameters ===
L = 40e-6     
N = 300       
x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, y)
r = np.sqrt(X**2 + Y**2)
phi = np.arctan2(Y, X)


r_inner = 5e-6
r_outer = 12e-6


n_o = 1.45
n_e = 1.50
eps_o = n_o**2
eps_e = n_e**2


eps_xx = np.ones_like(r) * eps_o
eps_yy = np.ones_like(r) * eps_o
eps_xy = np.zeros_like(r)


mask = (r >= r_inner) & (r <= r_outer)

eps_xx[mask] = eps_e * np.cos(phi[mask])**2 + eps_o * np.sin(phi[mask])**2
eps_yy[mask] = eps_e * np.sin(phi[mask])**2 + eps_o * np.cos(phi[mask])**2
eps_xy[mask] = (eps_e - eps_o) * np.sin(phi[mask]) * np.cos(phi[mask])


fig, axs = plt.subplots(1, 3, figsize=(15, 4))
im0 = axs[0].imshow(eps_xx, extent=[-L/2*1e6, L/2*1e6, -L/2*1e6, L/2*1e6])
axs[0].set_title(r'$\epsilon_{xx}$')
plt.colorbar(im0, ax=axs[0])

im1 = axs[1].imshow(eps_yy, extent=[-L/2*1e6, L/2*1e6, -L/2*1e6, L/2*1e6])
axs[1].set_title(r'$\epsilon_{yy}$')
plt.colorbar(im1, ax=axs[1])

im2 = axs[2].imshow(eps_xy, extent=[-L/2*1e6, L/2*1e6, -L/2*1e6, L/2*1e6])
axs[2].set_title(r'$\epsilon_{xy}$ (coupling term)')
plt.colorbar(im2, ax=axs[2])

for ax in axs:
    ax.set_xlabel('x (µm)')
    ax.set_ylabel('y (µm)')

plt.suptitle('Anisotropic Dielectric Tensor of Myelin Tube (Radial Uniaxial)')
plt.tight_layout()
plt.show()
