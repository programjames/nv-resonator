import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc
from dolfinx import mesh, fem, io
from dolfinx.fem import petsc
import dolfinx.plot
import pyvista

# Initialize MPI
comm = MPI.COMM_WORLD

# Define the mesh
N = 20
domain = mesh.create_box(comm, [np.array([0, 0, 0]), np.array([1, 1, 1])], [N, N, N])

# Define function space (Nédélec elements of the first kind)
V = fem.functionspace(domain, ("N1curl", 1))

# Define the weak form
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Material parameters
epsilon_r = 10.0  # relative permittivity of the ceramic
mu_r = 1.0  # relative permeability (assuming non-magnetic material)
c = 3e8  # speed of light in vacuum

a = (1/mu_r) * ufl.inner(ufl.curl(u), ufl.curl(v)) * ufl.dx
m = epsilon_r * ufl.inner(u, v) * ufl.dx

# Assemble matrices
A = fem.petsc.assemble_matrix(fem.form(a))
A.assemble()
M = fem.petsc.assemble_matrix(fem.form(m))
M.assemble()

# Create eigensolver
eps = SLEPc.EPS().create(comm)
eps.setOperators(A, M)
eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
eps.setDimensions(10)
eps.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)
eps.solve()

# Extract eigenvalues and eigenvectors
nconv = eps.getConverged()
eigenvalues = []
eigenvectors = []

for i in range(nconv):
    eigenvalue = eps.getEigenvalue(i)
    # Convert eigenvalue to frequency
    frequency = np.sqrt(eigenvalue.real) * c / (2 * np.pi)
    eigenvalues.append(frequency)
    
    # Extract eigenvector
    vr, vi = A.getVecs()
    eps.getEigenvector(i, vr, vi)
    eigenvectors.append(vr)

# Print eigenvalues (resonant frequencies)
print("Resonant frequencies (Hz):")
for i, f in enumerate(eigenvalues):
    print(f"Mode {i+1}: {f:.2e}")

# Visualize the first mode
if comm.rank == 0:
    mode_function = fem.Function(V)
    mode_function.x.array[:] = eigenvectors[0].array
    
    topology, cell_types, geometry = dolfinx.plot.create_vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    
    # For vector functions, we need to reshape the data
    mode_data = mode_function.x.array.reshape((-1, 3))
    grid.point_data["mode"] = mode_data
    
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, scalars="mode", cmap="coolwarm")
    plotter.add_arrows(grid.points, mode_data, mag=0.1)
    plotter.view_xy()
    plotter.show()