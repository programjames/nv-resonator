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

from constants import *
import ufl
from ufl import inner, dot, conj, grad, dx, Dx, ds, TrialFunction, TestFunction, SpatialCoordinate

cache_dir = os.path.expanduser("~/.cache/fenics")
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    print(f"Cleared FEniCS cache: {cache_dir}")

# Read mesh
domain, cell_tags, facet_tags = gmshio.read_from_msh("mesh/resonator.msh", MPI.COMM_WORLD, gdim=2)

# Function space
V = fem.FunctionSpace(domain, ("CG", 1, (2,)))

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Relative permittivity
epsilon_r = fem.Function(V.sub(0).collapse()[0])
air_dofs = fem.locate_dofs_topological(V, domain.topology.dim, cell_tags.find(2))
ceramic_dofs = fem.locate_dofs_topological(V, domain.topology.dim, cell_tags.find(1))
epsilon_r.x.array[air_dofs] = 1
epsilon_r.x.array[ceramic_dofs] = EPSILON_R

# Trial and test function
uz, ur = ufl.split(u)
vz, vr = ufl.split(v)
vz = conj(vz)
vr = conj(vr)

# Coordinate system
x = SpatialCoordinate(domain)
z = x[0]
r = x[1]

ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
daxis = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)

# Define the weak form
a = r * inner(grad(u), grad(v)) * dx + \
    1 / r * ur * vr * dx
    
b = -1j * epsilon_r**0.5 * r * inner(u, v) * ds(3)

m = r * epsilon_r * inner(u, v) * dx

# Dirichlet boundary
boundary_dofs = fem.locate_dofs_topological(V, domain.topology.dim-1, entities=facet_tags.find(3))
u_bc = fem.Function(V)
u_bc.x.array[:] = 0

bc = fem.dirichletbc(u_bc, boundary_dofs)
# bcs = [bc]
bcs = []

A = fem.petsc.assemble_matrix(fem.form(a), bcs=bcs)
A.assemble()
B = fem.petsc.assemble_matrix(fem.form(b), bcs=bcs)
B.assemble()
M = fem.petsc.assemble_matrix(fem.form(m), bcs=bcs)
M.assemble()

target = (3e9 * 2 * np.pi / C)
pep = SLEPc.PEP().create(domain.comm)

# Set the matrices for the quadratic eigenvalue problem
pep.setOperators([M, B, A])  # Order: M*lambda^2 + B*lambda + A

# Set solver options
pep.setProblemType(SLEPc.PEP.ProblemType.GENERAL)
pep.setType(SLEPc.PEP.Type.QARNOLDI) #TOAR, LINEAR, QARNOLDI, STOAR
pep.setDimensions(nev=100, ncv=200, mpd=100)
pep.setWhichEigenpairs(SLEPc.PEP.Which.TARGET_MAGNITUDE)  # Find eigenvalues closest to target
pep.setTarget(target)

# Set up the spectral transformation
st = pep.getST()
st.setType(SLEPc.ST.Type.SINVERT)
st.setShift(target)

# Configure the linear solver for the transformation
ksp = st.getKSP()
ksp.setType(PETSc.KSP.Type.PREONLY)
pc = ksp.getPC()
pc.setType(PETSc.PC.Type.LU)

# Set convergence test and tolerance
# pep.setConvergenceTest(SLEPc.PEP.Conv.REL)
# pep.setTolerances(tol=1e-10, max_it=1000)

# Solve the eigenvalue problem
pep.solve()

nconv = pep.getConverged()
print("Total converged", nconv)

vr, vi = pep.getOperators()[0].createVecs()

# Extract eigenvalues and eigenvectors
for i in range(nconv):
    eigval = pep.getEigenpair(i, vr, vi)
    f = (1 / eigval).imag * C / (2 * np.pi)
    print(f"Frequency {i}: f = {f/1e9:.4f} GHz")


    H = vr.array.real.reshape(-1, 2)
    H /= H.max()
    H = np.pad(H, ((0, 0), (0, 1)))
    
    print(np.linalg.norm(H))
    
    
    if abs(f) < 1e9 or abs(f) > 8e9: continue

    topology, cell_types, geometry = vtk_mesh(V)
    grid = pv.UnstructuredGrid(topology, cell_types, geometry)
    plotter = pv.Plotter()
    
    sign = np.sign(np.where(np.abs(H[:, 0]) > np.abs(H[:, 1]), H[:, 0], H[:, 1]))
    H_mag = sign * np.linalg.norm(H, axis=-1)
    
    grid.point_data["u"] = H_mag
    grid.set_active_scalars("u")
    
    plotter.add_mesh(grid, show_edges=False, cmap="jet")
    plotter.add_arrows(grid.points, H, mag=1e-3, color="white")
    
    # Add in reflection
    grid = grid.reflect(normal=(0, 1, 0))
    H[:, 1] *= -1
    grid.point_data["u"] = H_mag
    grid.set_active_scalars("u")
    
    plotter.add_mesh(grid, show_edges=False, cmap="jet")
    plotter.add_arrows(grid.points, H, mag=1e-3, color="white")

    # Show
    plotter.view_xy()
    if not pv.OFF_SCREEN:
        plotter.show()
    else:
        figure = plotter.screenshot("fundamentals_mesh.png")

# Clean up
A.destroy()
M.destroy()