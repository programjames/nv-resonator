from mpi4py import MPI
from dolfinx import mesh, fem, plot
import dolfinx.fem.petsc
from slepc4py import SLEPc
from dolfinx.io import gmshio
import numpy as np
import ufl
import pyvista as pv
import argparse

from constants import C, EPSILON_R


# Read args
parser = argparse.ArgumentParser(description='Find resonate frequencies of a cavity.')
parser.add_argument('filename', type=str, help='Mesh file')
parser.add_argument('--no-popup', action='store_true', help='Suppress wave mode visual')

args = parser.parse_args()

# Read mesh
domain, cell_tags, facet_tags = gmshio.read_from_msh(
    args.filename, MPI.COMM_WORLD, gdim=3)

# Create function space and test/trial functions
V = fem.FunctionSpace(domain, ("CG", 1))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Weak formulation of the Helmholtz equation
# ∇²u + k²u = 0 = 0, where k is the wavenumber
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
m = EPSILON_R * ufl.inner(u, v) * ufl.dx
    
def boundary(x):
    r_squared = x[0]**2 + x[1]**2
    return np.isclose(r_squared, r_squared.min()) | \
        np.isclose(r_squared, r_squared.max()) | \
        np.isclose(x[2], 0.0) | \
        np.isclose(x[2], domain.geometry.x[2].max())

# Apply PEC boundary condition
facets = mesh.locate_entities_boundary(domain, dim=2, marker=boundary)
dofs = fem.locate_dofs_topological(V, entity_dim=2, entities=facets)

# Create a vector-valued constant for the boundary condition
u_bc = fem.Function(V)
u_bc.x.array[:] = 0

bc = fem.dirichletbc(u_bc, dofs)

# Assemble matrices
A = fem.petsc.assemble_matrix(fem.form(a), bcs=[bc])
A.assemble()
M = fem.petsc.assemble_matrix(fem.form(m), bcs=[bc])
M.assemble()

# Create eigensolver
eps = SLEPc.EPS().create(domain.comm)
eps.setOperators(A, M)
eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
eps.setDimensions(10)  # Number of eigenvalues to compute
eps.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)
eps.solve()

# Extract eigenvalues and eigenvectors
nconv = eps.getConverged()
frequencies = []
eigenvectors = []

for i in range(nconv):
    eigenvalue = eps.getEigenvalue(i)
    # Convert eigenvalue to frequency
    frequency = np.sqrt(eigenvalue.real) * C / (2 * np.pi)
    frequencies.append(frequency)
    
    # Extract eigenvector
    vr, vi = A.getVecs()
    eps.getEigenvector(i, vr, vi)
    eigenvectors.append(vr)
    
# Print frequencies
if domain.comm.rank == 0:
    print("Resonant frequencies (GHz):")
    for i, freq in enumerate(frequencies):
        print(f"Mode {i+1}: {freq/1e9:.4f} GHz")
        
def plot_mode(frequencies, eigenvectors):
    u_mode = fem.Function(V)
    for i, freq in enumerate(frequencies):
        if freq > 1e9:
            u_mode.x.array[:] = eigenvectors[i].array
            break

    # Get the mesh data for visualization
    u_topology, u_cell_types, u_geometry = plot.vtk_mesh(V)

    # Create a pv UnstructuredGrid for visualization
    u_grid = pv.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
    u_grid.point_data["u"] = u_mode.x.array.real

    # Visualize the interpolated solution
    u_plotter = pv.Plotter()
    u_plotter.add_mesh(u_grid, show_edges=True, opacity=0.5, cmap='coolwarm')
    u_plotter.view_isometric()
    if not pv.OFF_SCREEN:
        u_plotter.show()
    
if not args.no_popup:
    plot_mode(frequencies, eigenvectors)