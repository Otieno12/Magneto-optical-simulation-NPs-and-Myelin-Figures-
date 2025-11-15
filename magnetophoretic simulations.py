from fipy import *
import numpy as np
import pandas as pd

# ==========================================================
# 1. LOAD COMSOL MAGNETIC FIELD
# ==========================================================
df = pd.read_csv("B_export.csv")   # must contain columns: x, y, Bx, By, Bz

# Build regular grid from COMSOL points
xvals = np.unique(df['x'])
yvals = np.unique(df['y'])
nx = len(xvals)
ny = len(yvals)
dx = xvals[1] - xvals[0]
dy = yvals[1] - yvals[0]

mesh = Grid2D(dx=dx, dy=dy, nx=nx, ny=ny)

# Map COMSOL data into arrays matching mesh ordering
Bx = df['Bx'].values.reshape((nx,ny))
By = df['By'].values.reshape((nx,ny))
Bz = df['Bz'].values.reshape((nx,ny))

B2 = (Bx**2 + By**2 + Bz**2).reshape((nx,ny))

# Make FiPy cell variables
B2_fi = CellVariable(name="B2", mesh=mesh, value=B2.T)

# ∇(B²/2)
dB2dx, dB2dy = np.gradient(B2, dx, dy)
gradB2 = (dB2dx**2 + dB2dy**2)**0.5
gradB2_fi = CellVariable(name="gradB2", mesh=mesh, value=gradB2.T)

# ==========================================================
# 2. DEFINE PHYSICAL PARAMETERS
# ==========================================================
Dn = 1e-12       # nanoparticle diffusion coefficient (m²/s)
D0 = 5e-12       # lipid hydration diffusion base rate
mu_m = 2e-13     # magnetophoretic mobility (depends on size)
kappa = 50       # slowdown coefficient for lipid diffusion

dt = 0.01        # time step (s)
steps = 2000     # number of iterations

# ==========================================================
# 3. DEFINE FIELDS: nanoparticles (n) and hydration (c)
# ==========================================================
n = CellVariable(name="NP concentration", mesh=mesh, value=0.001)
c = CellVariable(name="hydration", mesh=mesh, value=0.01)

# ==========================================================
# 4. DIFFUSION COEFFICIENT FOR LIPIDS
# ==========================================================
Deff = D0 / (1 + kappa * n)

# ==========================================================
# 5. PDE EQUATIONS
# ==========================================================

# 5a. Nanoparticle Transport (diffusion + magnetophoresis)
n_eq = (TransientTerm(var=n)
        == DiffusionTerm(coeff=Dn, var=n)
        - ConvectionTerm(coeff=mu_m * gradB2_fi, var=n))

# 5b. Lipid Hydration (nonlinear diffusion)
c_eq = (TransientTerm(var=c)
        == DiffusionTerm(coeff=Deff, var=c))

# ==========================================================
# 6. INITIAL CONDITIONS — MYELIN GEOMETRY
# ==========================================================
# cylinder shell region (simplified 2D cross section)
X, Y = np.meshgrid(xvals, yvals, indexing='ij')
R = np.sqrt((X)**2 + (Y)**2)

R_inner = 1.0e-6
R_outer = 3.0e-6

shell_region = (R > R_inner) & (R < R_outer)

# nanoparticles initially only inside the shell
n.value = 0.01 * shell_region.T

# hydration initially zero outside
c.value = 0.2 * shell_region.T

# ==========================================================
# 7. TIME LOOP
# ==========================================================
viewer = Viewer(vars=(n, c))

for t in range(steps):
    n_eq.solve(var=n, dt=dt)
    Deff = D0 / (1 + kappa * n)
    c_eq.solve(var=c, dt=dt)

    if t % 50 == 0:
        print("Step", t)
        viewer.plot()

# ==========================================================
# 8. SAVE OUTPUT
# ==========================================================
np.savetxt("n_final.csv", n.value)
np.savetxt("c_final.csv", c.value)
