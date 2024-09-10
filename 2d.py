from mpi4py import MPI
from slepc4py import SLEPc
import dolfinx.fem.petsc
from dolfinx import fem
from dolfinx.plot import vtk_mesh
from dolfinx.io import gmshio
import os, shutil
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

from constants import *
import ufl
from ufl import inner, dot, grad, dx, Dx, ds, TrialFunction, TestFunction, SpatialCoordinate

cache_dir = os.path.expanduser("~/.cache/fenics")
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    print(f"Cleared FEniCS cache: {cache_dir}")

# Read mesh
domain, cell_tags, facet_tags = gmshio.read_from_msh("resonator.msh", MPI.COMM_WORLD, gdim=2)

# Function space
V = fem.FunctionSpace(domain, ("CG", 1))

# Relative permittivity
epsilon_r = fem.Function(V)
air_dofs = fem.locate_dofs_topological(V, domain.topology.dim, cell_tags.find(2))
ceramic_dofs = fem.locate_dofs_topological(V, domain.topology.dim, cell_tags.find(1))
epsilon_r.x.array[air_dofs] = 1
epsilon_r.x.array[ceramic_dofs] = EPSILON_R

# Trial and test functions
u = TrialFunction(V)
v = TestFunction(V)

# Coordinate system
x = SpatialCoordinate(domain)
r = x[1]

# Define the weak form
a = r * dot(grad(u), grad(v)) * dx + \
    1/r * u * v * dx + \
    0#r * u * v * ds

m = r * epsilon_r * u * v * dx

# Assemble matrices
A = fem.petsc.assemble_matrix(fem.form(a))
A.assemble()
M = fem.petsc.assemble_matrix(fem.form(m))
M.assemble()

# Solve eigenproblem
eps = SLEPc.EPS().create(domain.comm)
eps.setOperators(A, M)
eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
eps.setDimensions(10)
eps.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_MAGNITUDE)
eps.solve()


nconv = eps.getConverged()
print(f"Number of converged eigenvalues: {nconv}")

# Extract eigenvalues and eigenvectors
for i in range(nconv):
    eigval = eps.getEigenvalue(i)
    f = np.sqrt(eigval.real) * C / (2 * np.pi)
    print(f"Frequency {i}: f = {f/1e9:.4f} GHz")

    # Get eigenvector
    vr, vi = A.createVecs()
    eps.getEigenvector(i, vr, vi)

    topology, cell_types, geometry = vtk_mesh(domain)
    grid = pv.UnstructuredGrid(topology, cell_types, geometry)
    plotter = pv.Plotter()
    grid.point_data["u"] = vr.array
    grid.set_active_scalars("u")
    plotter.add_mesh(grid, show_edges=True)
    plotter.view_xy()
    if not pv.OFF_SCREEN:
        plotter.show()
    else:
        figure = plotter.screenshot("fundamentals_mesh.png")

# Clean up
A.destroy()
M.destroy()