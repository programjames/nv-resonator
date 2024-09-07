from mpi4py import MPI
from dolfinx import mesh, fem, plot
import dolfinx.fem.petsc
from slepc4py import SLEPc
from dolfinx.io import gmshio
import numpy as np
import ufl
import pyvista as pv
import argparse

from constants import C, EPSILON_R, MU_R


# Read args
parser = argparse.ArgumentParser(description="Find resonate frequencies of a cavity.")
parser.add_argument("type", type=str, help="'ring' or 'cylinder'")
parser.add_argument("--no-popup", action="store_true", help="Suppress wave mode visual")

args = parser.parse_args()

# Read mesh
domain, cell_tags, facet_tags = gmshio.read_from_msh(
    args.type + ".msh", MPI.COMM_WORLD, gdim=3)

num_3d_cells = domain.topology.index_map(domain.topology.dim).size_local
print(f"Number of 3D cells: {num_3d_cells}")

# Create function space and test/trial functions
V = fem.FunctionSpace(domain, ("CG", 1))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Weak formulation
a = 1/MU_R * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
m = EPSILON_R * ufl.inner(u, v) * ufl.dx

boundary_dofs = fem.locate_dofs_topological(V, entity_dim=2, entities=facet_tags.find(1))

# n = ufl.FacetNormal(domain)

# Add boundary term to the weak form
# a += ufl.inner(ufl.cross(n, ufl.grad(u)), ufl.cross(n, ufl.grad(v))) * ufl.ds

# Create a vector-valued constant for the boundary condition
u_bc = fem.Function(V)
u_bc.x.array[:] = 0

bc = fem.dirichletbc(u_bc, boundary_dofs)

# # Assemble matrices
A = fem.petsc.assemble_matrix(fem.form(a), bcs=[bc])
A.assemble()
M = fem.petsc.assemble_matrix(fem.form(m), bcs=[bc])
M.assemble()

# A = fem.petsc.assemble_matrix(fem.form(a))
# A.assemble()
# M = fem.petsc.assemble_matrix(fem.form(m))
# M.assemble()

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


def save_slice(grid):
    slice = grid.slice(normal="x")
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(slice, show_edges=False, cmap="coolwarm")
    plotter.set_background([255,255,255,0])
    plotter.view_yz()
    plotter.remove_scalar_bar()
    scalar_bar = plotter.add_scalar_bar(
        title="Normalized magnetic field\n",
        position_x=0.2,
        position_y=0.075,
        title_font_size=18,
        label_font_size=12
    )
    plotter.screenshot(f"images/{args.type}_slice.png", transparent_background=True)

def plot_mode(grid):
    plotter = pv.Plotter()
    plotter.add_mesh(grid, show_edges=False, opacity=1.0, cmap="coolwarm")
    plotter.view_isometric()
    if not pv.OFF_SCREEN:
        plotter.show()
        
def plot_boundary():
    pass
    # boundary_function = fem.Function(V)
    # bottom_dofs = fem.locate_dofs_topological(V, entity_dim=2, entities=facet_tags.find(1))
    # top_dofs = fem.locate_dofs_topological(V, entity_dim=2, entities=facet_tags.find(2))
    # outer_dofs = fem.locate_dofs_topological(V, entity_dim=2, entities=facet_tags.find(3))
    # inner_dofs = fem.locate_dofs_topological(V, entity_dim=2, entities=facet_tags.find(4))
    # boundary_function.x.array[top_dofs] = 1.0
    # boundary_function.x.array[bottom_dofs] = 10.0
    # boundary_function.x.array[outer_dofs] = -10.0
    # boundary_function.x.array[inner_dofs] = -5.0
    
    # boundary_grid = pv.UnstructuredGrid(topology, cell_types, geometry)
    # boundary_grid.point_data["boundary"] = boundary_function.x.array
    # plotter.add_mesh(boundary_grid, opacity=1.0, cmap="coolwarm", label="Boundary")
    
mode = fem.Function(V)
for i, freq in enumerate(frequencies):
    if freq > 1e9:
        mode.x.array[:] = eigenvectors[i].array
        break

grid = pv.UnstructuredGrid(*plot.vtk_mesh(V))
x = mode.x.array.real
x -= x.min()
x /= x.max()
grid.point_data["u"] = x
save_slice(grid)

if not args.no_popup:
    plot_mode(grid)