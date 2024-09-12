from mpi4py import MPI
from slepc4py import SLEPc
from petsc4py import PETSc
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
from ufl import inner, dot, conj, grad, dx, Dx, ds, TrialFunction, TestFunction, SpatialCoordinate

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

# Trial and test function
u = TrialFunction(V)
v = conj(TestFunction(V))

# Coordinate system
x = SpatialCoordinate(domain)
z = x[0]
r = x[1]

ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
daxis = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)

# Define the weak form
a = r * dot(grad(u), grad(v)) * dx + \
    1 / r * u * v * dx - \
    1j * r * u * v * ds(3)

m = r * epsilon_r * u * v * dx

# Dirichlet boundary
boundary_dofs = fem.locate_dofs_topological(V, domain.topology.dim-1, entities=facet_tags.find(3))
u_bc = fem.Function(V)
u_bc.x.array[:] = 0

bc = fem.dirichletbc(u_bc, boundary_dofs)
# bcs = [bc]
bcs = []

# Assemble matrices
A = fem.petsc.assemble_matrix(fem.form(a), bcs=bcs)
A.assemble()
M = fem.petsc.assemble_matrix(fem.form(m), bcs=bcs)
M.assemble()

# Target eigenvalue
target = (3e9 * 2 * np.pi / C)**2

# Solve eigenproblem
eps = SLEPc.EPS().create(domain.comm)
eps.setOperators(A, M)
eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
eps.setDimensions(50)
eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
eps.setTarget(target)
# eps.setTolerances(1e-10, 20)
# eps.setType(SLEPc.EPS.Type.POWER)

st = eps.getST()
st.setType(SLEPc.ST.Type.SINVERT)

# Set the shift to your target value
st.setShift(target)

# Configure the linear solver for the transformation
ksp = st.getKSP()
ksp.setType(PETSc.KSP.Type.PREONLY)
pc = ksp.getPC()
pc.setType(PETSc.PC.Type.LU)

# If you're dealing with a large problem, you might want to use an iterative solver instead:
# ksp.setType(PETSc.KSP.Type.GMRES)
# pc.setType(PETSc.PC.Type.ILU)

# Optionally, you can set a convergence test
eps.setConvergenceTest(SLEPc.EPS.Conv.REL)

# Now you can call eps.solve()
eps.solve()

nconv = eps.getConverged()
print("Total converged", nconv)

# Extract eigenvalues and eigenvectors
# for i in range(nconv):
i = 0
eigval = eps.getEigenvalue(i)
f = np.sqrt(eigval.real) * C / (2 * np.pi)
print(f"Frequency {i}: f = {f/1e9:.4f} GHz")

# Get eigenvector
vr, _ = A.createVecs()
eps.getEigenvector(i, vr, _)

Hr = fem.Function(V)
Hr.x.array[:] = vr.array

Hz = ufl.TrialFunction(V)

# Weak form of the equation
a = r * Hz.dx(0) * v * dx
L = r * Hr * v.dx(1) * dx - 1j * r * Hr * v * ds(3)

a = r * dot(grad(u), grad(v)) * dx + \
    1 / r * u * v * dx - \
    1j * r * u * v * ds(3)

L = eigval * r * epsilon_r * u * v * dx

boundary_dofs = fem.locate_dofs_topological(V, domain.topology.dim-1, entities=facet_tags.find(3))
u_bc = fem.Function(V)
u_bc.x.array[:] = 0

bc = fem.dirichletbc(u_bc, boundary_dofs)
# bcs = [bc]
bcs = []

A = fem.petsc.assemble_matrix(fem.form(a))
A.assemble()

b = fem.petsc.assemble_vector(fem.form(L))
b.assemble()

ksp = PETSc.KSP().create(V.mesh.comm)
ksp.setOperators(A)

# Set solver options
ksp.setType(PETSc.KSP.Type.PREONLY)
pc = ksp.getPC()
pc.setType(PETSc.PC.Type.LU)
pc.setFactorSolverType("mumps")

Hz = fem.Function(V)
ksp.solve(b, Hz.vector)

# # Create the problem
# problem = fem.petsc.LinearProblem(a, L, bcs=bcs, petsc_options={
#     "ksp_type": "preonly",
#     "pc_type": "lu",
#     "pc_factor_mat_solver_type": "mumps",
#     "mat_type": "aij",  # Use this for complex matrices
# })

# # Solve the problem
# Hz = problem.solve()

# # Derivative of Hz
# Hz_dz = ufl.TrialFunction(V)
# a = r * Hz_dz * v * dx
# L = r * Hz.dx(0) * v * dx

# problem = fem.petsc.LinearProblem(a, L)
# Hz_dz = problem.solve()

# # Derivative of Hr
# Hr_dr = ufl.TrialFunction(V)
# a = r * Hr_dr * v * dx
# L = (r * Hr).dx(1) * v * dx

# problem = fem.petsc.LinearProblem(a, L)
# Hr_dr = problem.solve()

# print(np.column_stack((Hr_dr.x.array, Hz_dz.x.array)).tolist())

topology, cell_types, geometry = vtk_mesh(V)
grid = pv.UnstructuredGrid(topology, cell_types, geometry)
plotter = pv.Plotter()
grid.point_data["u"] = Hz.x.array# Hr_dr.x.array + Hz_dz.x.array
grid.set_active_scalars("u")
plotter.add_mesh(grid, show_edges=False)

# Arrows
H = np.zeros((Hr.x.array.shape[0], 3), dtype=np.complex64)
H[:, 0] = Hz.x.array
H[:, 1] = Hr.x.array


# print(np.column_stack((Hz.x.array, Hr.x.array)).tolist())

plotter.add_arrows(grid.points, H, mag=1e-15, color="white")  # Adjust scale as needed

# Show
plotter.view_xy()
if not pv.OFF_SCREEN:
    plotter.show()
else:
    figure = plotter.screenshot("fundamentals_mesh.png")
# break

# Clean up
# A.destroy()
# M.destroy()
# b.destroy()