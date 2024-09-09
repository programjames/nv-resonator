from mpi4py import MPI
from dolfinx import mesh, fem, plot
import dolfinx.fem.petsc
from slepc4py import SLEPc
from dolfinx.io import gmshio
import numpy as np
import ufl
import pyvista as pv
import argparse
import basix

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
V = fem.FunctionSpace(domain, ("N1curl", 1))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Weak formulation
a = ufl.inner(ufl.curl(u), ufl.curl(v)) * ufl.dx  # Stiffness matrix
m = EPSILON_R * MU_R * ufl.inner(u, v) * ufl.dx  # Mass matrix

# Boundary condition
boundary_dofs = fem.locate_dofs_topological(V, entity_dim=2, entities=facet_tags.find(1))
u_bc = fem.Function(V)
u_bc.x.array[:] = 0

bc = fem.dirichletbc(u_bc, boundary_dofs)

# Assemble matrices
A = fem.petsc.assemble_matrix(fem.form(a), bcs=[bc])
A.assemble()
M = fem.petsc.assemble_matrix(fem.form(m), bcs=[bc])
M.assemble()

# Solve eigenvalue problem
eps = SLEPc.EPS().create(domain.comm)
eps.setOperators(A, M)
eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
eps.setDimensions(10)
eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
eps.setTarget(3e9 / C * 2 * np.pi)
eps.solve()

# Extract eigenvalues and eigenvectors
nconv = eps.getConverged()
frequencies = []
modes = []

for i in range(nconv):
    # Convert eigenvalue to frequency
    eigenvalue = eps.getEigenvalue(i)
    frequency = np.sqrt(eigenvalue.real) * C / (2 * np.pi)
    frequencies.append(frequency)
    
    # Convert eigenvector to mode
    vr, vi = A.getVecs()
    eps.getEigenvector(i, vr, vi)
    modes.append(vr.array + 1j * vi.array)
    
# Print frequencies
if domain.comm.rank == 0:
    print("Resonant frequencies (GHz):")
    for i, freq in enumerate(frequencies):
        print(f"Mode {i+1}: {freq/1e9:.4f} GHz")


def save_slice(grid):
    slice = grid.slice(normal="x")
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(slice, show_edges=False, cmap="viridis")
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

def plot_mode(mode):
    # Create vector space with 3 elements
    cell_type = str(domain.ufl_cell())
    element = basix.ufl.element("Lagrange", cell_type, 1, shape=(3,))
    V_vec = dolfinx.fem.FunctionSpace(domain, element)
    grid = pv.UnstructuredGrid(*plot.vtk_mesh(V_vec))
    
    mode_vec = fem.Function(V)
    mode_vec.vector.array[:] = mode.real
    
    # Interpolate mode into vector space
    u_vec = dolfinx.fem.Function(V_vec)
    u_vec.interpolate(mode_vec)
    
    # Interpolate magnitude onto vector space
    
    mode_vec.vector.array[:] = abs(mode)
    u_mag = dolfinx.fem.Function(V_vec)
    u_mag.interpolate(mode_vec)
    
    # Create Grid
    grid.point_data["H_field"] = np.linalg.norm(u_mag.x.array.reshape(-1, 3), axis=1)
    grid.point_data["H_field_vector"] = u_vec.x.array.real.reshape(-1, 3)
    arrows = grid.glyph(
        orient="H_field_vector",
        scale="H_field",
        factor=5e-6,
    )
    
    # Save slice
    save_slice(grid)
    
    if not args.no_popup:
        # Plot and show
        plotter = pv.Plotter()
        plotter.add_mesh(grid, opacity=0.5, scalars="H_field", cmap="viridis")
        plotter.add_mesh(arrows)
        plotter.view_isometric()
        plotter.show()

for freq, mode in zip(frequencies, modes):
    if freq > 1e9:
        plot_mode(mode)

exit()

# Project the H(curl) solution onto the vector-valued space
u_vec = dolfinx.fem.Function(V_vec)
u_vec.interpolate(mode)

# Convert dolfinx mesh to pv mesh

# Add point data to the mesh
grid.point_data["H_field"] = np.linalg.norm(u_vec.x.array.reshape(-1, 3), axis=1)
plotter = pv.Plotter()
plotter.add_mesh(grid, scalars="H_field", cmap="viridis")
plotter.show()

grid.point_data["H_field_vector"] = u_vec.x.array.reshape(-1, 3)
plotter = pv.Plotter()
arrows = grid.glyph(
    orient="H_field_vector",
    scale="H_field",
    factor=5e-6,
)
plotter.add_mesh(arrows)
plotter.add_scalar_bar(title="Magnetic Field")
plotter.show()

# curl_element = basix.ufl.element("Lagrange", str(domain.ufl_cell()), 1, shape=(3,))
# V_curl = dolfinx.fem.FunctionSpace(domain, curl_element)
# curl_u = dolfinx.fem.Function(V_curl)
# curl_u.interpolate(ufl.curl(mode))

# E_field = curl_u.x.array.reshape(-1, 3)
# E_magnitude = np.linalg.norm(E_field, axis=1)
# grid.point_data["E_field"] = E_magnitude
# grid.point_data["E_field_vector"] = E_field

# plotter = pyvista.Plotter()
# plotter.add_mesh(grid, scalars="E_field", cmap="plasma")
# plotter.add_scalar_bar(title="Electric Field Magnitude (curl of H)")
# plotter.show()

# # Visualize E-field vectors
# plotter = pyvista.Plotter()
# plotter.add_mesh(grid, scalars="E_field", cmap="plasma")
# e_arrows = grid.glyph(
#     orient="E_field_vector",
#     scale="E_field",
#     factor=5e-6,  # Adjust this factor to change arrow size
# )
# plotter.add_mesh(e_arrows)
# plotter.add_scalar_bar(title="Electric Field")
# plotter.show()



# grid = pv.UnstructuredGrid(*plot.vtk_mesh(V))
# x = mode.x.array.real
# x -= x.min()
# x /= x.max()
# grid.point_data["H_field"] = mode.x.array.real
# plotter = pv.Plotter()
# plotter.add_mesh(grid, scalars="H_field", cmap="viridis")
# plotter.show()

# grid.point_data["u"] = x
# save_slice(grid)

# if not args.no_popup:
#     plot_mode(grid)
    
# grid = pv.UnstructuredGrid(*plot.vtk_mesh(V))

# grid.point_data["u"] = vr.x.array