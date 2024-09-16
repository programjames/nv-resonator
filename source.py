import mpi4py.MPI as MPI
import slepc4py.SLEPc as SLEPc
import petsc4py.PETSc as PETSc
import dolfinx.fem.petsc
import dolfinx.fem as fem
import dolfinx.plot
import dolfinx.io
import numpy as np
import pyvista as pv
import scipy
import ufl

import constants

# Read mesh
domain, cell_tags, facet_tags = dolfinx.io.gmshio.read_from_msh("mesh/source.msh", MPI.COMM_WORLD, gdim=2)
dim = domain.topology.dim


# Function space
V = fem.FunctionSpace(domain, ("N1curl", 1))

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Relative permittivity
V_eps = fem.FunctionSpace(domain, ("DG", 0))
epsilon_r = fem.Function(V_eps)
ceramic_dofs = fem.locate_dofs_topological(V_eps, dim, cell_tags.find(1))
epsilon_r.x.array[:] = 1
epsilon_r.x.array[ceramic_dofs] = constants.EPSILON_R

# Dipole source
def dipole_source_expression(x):
    return [[0.0], [1.0j],] / (abs(x[0])**3 + abs(x[1])**3 + 1e-8)

dipole_source = fem.Function(V)
dipole_source.interpolate(dipole_source_expression)

# Coordinate system
x = ufl.SpatialCoordinate(domain)
z = x[0]
r = x[1]

n = ufl.FacetNormal(domain)
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)

# Weak form
a = epsilon_r * ufl.inner(u, v) * r * ufl.dx
b = ufl.inner(ufl.inner(n, ufl.conj(u)) * n, v) * r * ds(3)
c = ufl.inner(ufl.curl(u), ufl.curl(v)) * r * ufl.dx

# NV center frequency (Hz)
f = 2.87e9
k = -2j * np.pi * f / constants.C


# Problem
a = k**2 * a + k * b + c
L = ufl.inner(dipole_source, v) * r * ufl.dx

A = fem.petsc.assemble_matrix(fem.form(a))
A.assemble()
b = fem.petsc.assemble_vector(fem.form(L))
b.assemble()

ksp = PETSc.KSP().create(MPI.COMM_WORLD)
ksp.setOperators(A)

# Set solver options
ksp.setType(PETSc.KSP.Type.GMRES)
ksp.setTolerances(max_it=1000)
ksp.getPC().setType(PETSc.PC.Type.ILU)

# Set up monitor to check convergence
ksp.setMonitor(lambda ksp, its, rnorm: print(f"Iteration {its}: residual norm {rnorm}"))

uh = fem.Function(V)
ksp.solve(b, uh.vector)

# Plot

V_plot = fem.FunctionSpace(domain, ("CG", 1, (dim,)))
u_plot = fem.Function(V_plot)
u_plot.interpolate(uh)

B = u_plot.x.array.real.reshape(-1, dim)
print(abs(B).max())
B /= abs(B).max()

topology, cell_types, geometry = dolfinx.plot.vtk_mesh(V_plot)
grid = pv.UnstructuredGrid(topology, cell_types, geometry)
plotter = pv.Plotter()
plotter.add_title(f"{f/1e9:.2f} GHz")

# Magnitude

grid.point_data["u"] = B
grid.set_active_scalars("u")

plotter.add_mesh(grid, show_edges=False, cmap="jet")

# Arrows
x, y, z = grid.points.T
x_coarse = np.linspace(x.min(), x.max(), 25)
y_coarse = np.linspace(y.min(), y.max(), 25)
X, Y = np.meshgrid(x_coarse, y_coarse)
points_coarse = np.column_stack((X.ravel(), Y.ravel(), np.zeros_like(X.ravel())))
B_coarse = scipy.interpolate.griddata(grid.points[:, :2], B, points_coarse[:, :2], method='linear')

# plotter.add_arrows(points_coarse, B_coarse, mag=1e-3, color="white")

# Add in reflection
grid = grid.reflect(normal=(0, 1, 0))
grid.point_data["u"] = B
grid.set_active_scalars("u")

plotter.add_mesh(grid, show_edges=False, cmap="jet")

points_coarse[:, 1] *= -1
B_coarse[:, 1] *= -1
# plotter.add_arrows(points_coarse, B_coarse, mag=1e-3, color="white")

# Remove annoying legend
# plotter.remove_scalar_bar()

# Save screenshots
plotter.view_xy()
plotter.show()