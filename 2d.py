from mpi4py import MPI
from slepc4py import SLEPc
from petsc4py import PETSc, init
import dolfinx.fem.petsc
from dolfinx import fem
from dolfinx.plot import vtk_mesh
from dolfinx.io import gmshio
import os, shutil
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import sys
import scipy

from constants import *
import ufl
from ufl import inner, dot, conj, grad, dx, Dx, ds, TrialFunction, TestFunction, SpatialCoordinate

cache_dir = os.path.expanduser("~/.cache/fenics")
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    print(f"Cleared FEniCS cache: {cache_dir}")

# Read mesh
domain, cell_tags, facet_tags = gmshio.read_from_msh("mesh/resonator.msh", MPI.COMM_WORLD, gdim=2)
dim = domain.topology.dim

# Function space
V = fem.FunctionSpace(domain, ("CG", 1, (2,)))

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Relative permittivity
epsilon_r = fem.Function(V.sub(0).collapse()[0])
air_dofs = fem.locate_dofs_topological(V, dim, cell_tags.find(2))
ceramic_dofs = fem.locate_dofs_topological(V, dim, cell_tags.find(1))
epsilon_r.x.array[air_dofs] = 1
epsilon_r.x.array[ceramic_dofs] = EPSILON_R

# Trial and test function
uz, ur = ufl.split(u)
vz, vr = ufl.split(v)
vr = conj(vr)
vz = conj(vz)

# Coordinate system
x = SpatialCoordinate(domain)
z = x[0]
r = x[1]

ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
daxis = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)

# Define the weak form
a = r * inner(grad(u), grad(v)) * dx + \
    1 / r * ur * vr * dx
    
b = -1j * epsilon_r ** 0.5 * r * inner(u, v) * ds(3) # TODO: Fix this boundary condition

m = r * epsilon_r * inner(u, v) * dx

# Dirichlet boundary along cylinder's axis
Vr = V.sub(1)
boundary_dofs = fem.locate_dofs_topological((Vr, V), dim-1, facet_tags.find(4))
u_bc = fem.Function(V)
u_bc.x.array[:] = 0

bc = fem.dirichletbc(u_bc, boundary_dofs, V.sub(1))
bcs = [bc]

A = fem.petsc.assemble_matrix(fem.form(a), bcs=bcs)
A.assemble()
B = fem.petsc.assemble_matrix(fem.form(b), bcs=bcs)
B.assemble()
M = fem.petsc.assemble_matrix(fem.form(m), bcs=bcs)
M.assemble()

pep = SLEPc.PEP().create(domain.comm)
pep.setOperators([A, B, M])  # Mλ^2 + Bλ + A = 0.

# Set solver options
pep.setType(SLEPc.PEP.Type.LINEAR) # Good results: LINEAR, QARNOLDI, TOAR
pep.setDimensions(nev=10, ncv=100, mpd=100) # num eigenvalues, num column vectors, max projection dimension

# Search around f = 3 GHz
target = (3e9 * 2 * np.pi / C)
pep.setWhichEigenpairs(SLEPc.PEP.Which.TARGET_MAGNITUDE)
pep.setTarget(target)

# Shifted-inverse spectral preconditioner to magnify around the target
st = pep.getST()
st.setType(SLEPc.ST.Type.SINVERT)
st.setShift(target)

# Use a direct solver with the spectral preconditioner
ksp = st.getKSP()
ksp.setType(PETSc.KSP.Type.PREONLY)
pc = ksp.getPC()
pc.setType(PETSc.PC.Type.LU)

# Solve the eigenvalue problem
pep.solve()

nconv = pep.getConverged()
print("Total converged", nconv)

vr, vi = pep.getOperators()[0].createVecs()

# Plot wave modes
for i in range(nconv):
    eigval = pep.getEigenpair(i, vr, vi)
    f = eigval.imag * C / (2 * np.pi)
    print(f"Frequency {i}: f = {f/1e9:.4f} GHz")
    if abs(f) < 2e9 or abs(f) > 8e9: continue

    H = vr.array.real.reshape(-1, 2)
    H /= H.max()
    H = np.pad(H, ((0, 0), (0, 1)))

    topology, cell_types, geometry = vtk_mesh(V)
    grid = pv.UnstructuredGrid(topology, cell_types, geometry)
    plotter = pv.Plotter()
    plotter.add_title(f"{f/1e9:.2f} GHz")
    
    # Magnitude
    sign = np.sign(np.where(np.abs(H[:, 0]) > np.abs(H[:, 1]), H[:, 0], H[:, 1]))
    H_mag = sign * np.linalg.norm(H, axis=-1)
    
    grid.point_data["u"] = H_mag
    grid.set_active_scalars("u")
    
    plotter.add_mesh(grid, show_edges=False, cmap="jet")
    
    # Arrows
    x, y, z = grid.points.T
    x_coarse = np.linspace(x.min(), x.max(), 25)
    y_coarse = np.linspace(y.min(), y.max(), 25)
    X, Y = np.meshgrid(x_coarse, y_coarse)
    points_coarse = np.column_stack((X.ravel(), Y.ravel(), np.zeros_like(X.ravel())))
    H_coarse = scipy.interpolate.griddata(grid.points[:, :2], H, points_coarse[:, :2], method='linear')
    
    plotter.add_arrows(points_coarse, H_coarse, mag=1e-3, color="white")
    
    # Add in reflection
    grid = grid.reflect(normal=(0, 1, 0))
    H[:, 1] *= -1
    grid.point_data["u"] = H_mag
    grid.set_active_scalars("u")
    
    plotter.add_mesh(grid, show_edges=False, cmap="jet")
    
    points_coarse[:, 1] *= -1
    H_coarse[:, 1] *= -1
    plotter.add_arrows(points_coarse, H_coarse, mag=1e-3, color="white")
    
    # Remove annoying legend
    plotter.remove_scalar_bar()

    # Show
    plotter.view_xy()
    plotter.show()

# Clean up
A.destroy()
B.destroy()
M.destroy()