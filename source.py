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
V = fem.FunctionSpace(domain, ("CG", 2, (2,)))

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Relative permittivity
epsilon_r = fem.Function(V.sub(0).collapse()[0])
air_dofs = fem.locate_dofs_topological(V, dim, cell_tags.find(2))
ceramic_dofs = fem.locate_dofs_topological(V, dim, cell_tags.find(1))
epsilon_r.x.array[air_dofs] = 1
epsilon_r.x.array[ceramic_dofs] = constants.EPSILON_R

# Dipole source
def dipole_source_expression(x):
    return x[:2] / (abs(x[0])**3 + abs(x[1])**3 + 1e-8)

dipole_source = fem.Function(V)
dipole_source.interpolate(dipole_source_expression)

# Trial and test function
uz, ur = ufl.split(u)
vz, vr = map(ufl.conj, ufl.split(v))

# Coordinate system
x = ufl.SpatialCoordinate(domain)
z = x[0]
r = x[1]

ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)

# Weak form
a = r * epsilon_r * ufl.inner(u, v) * ufl.dx
b = r * ufl.inner(u, v) * ds(3)
c = r * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + \
    1 / r * ur * vr * ufl.dx

# NV center frequency (Hz)
f = 2.87e9
k = -2j * np.pi * f / constants.C

# Br = 0 along ring's axis
Vr = V.sub(1)
boundary_dofs = fem.locate_dofs_topological((Vr, V), dim-1, facet_tags.find(4))
u_bc = fem.Function(V)
u_bc.x.array[:] = 0

bc = fem.dirichletbc(u_bc, boundary_dofs, V.sub(1))
bcs = [bc]

# Problem
a = k**2 * a + k * b + c
L = ufl.inner(dipole_source, v) / epsilon_r * r * ufl.dx

A = fem.petsc.assemble_matrix(fem.form(a), bcs=bcs)
A.assemble()
b = fem.petsc.assemble_vector(fem.form(L))
b.assemble()

ksp = PETSc.KSP().create(MPI.COMM_WORLD)
ksp.setOperators(A)

# Set solver options
ksp.setType(PETSc.KSP.Type.GMRES)
ksp.getPC().setType(PETSc.PC.Type.ILU)

# Set up monitor to check convergence
ksp.setMonitor(lambda ksp, its, rnorm: print(f"Iteration {its}: residual norm {rnorm}"))

uh = fem.Function(V)
ksp.solve(b, uh.vector)

# Plot

B = uh.x.array.real.reshape(-1, 2)
B /= B.max()
B = np.pad(B, ((0, 0), (0, 1)))

topology, cell_types, geometry = dolfinx.plot.vtk_mesh(V)
grid = pv.UnstructuredGrid(topology, cell_types, geometry)
plotter = pv.Plotter()
plotter.add_title(f"{f/1e9:.2f} GHz")

# Magnitude
sign = np.sign(np.where(np.abs(B[:, 0]) > np.abs(B[:, 1]), B[:, 0], B[:, 1]))
B_mag = sign * np.linalg.norm(B, axis=-1)
B_mag = B

grid.point_data["u"] = B_mag
grid.set_active_scalars("u")

plotter.add_mesh(grid, show_edges=False, cmap="jet")

# Arrows
x, y, z = grid.points.T
x_coarse = np.linspace(x.min(), x.max(), 25)
y_coarse = np.linspace(y.min(), y.max(), 25)
X, Y = np.meshgrid(x_coarse, y_coarse)
points_coarse = np.column_stack((X.ravel(), Y.ravel(), np.zeros_like(X.ravel())))
B_coarse = scipy.interpolate.griddata(grid.points[:, :2], B, points_coarse[:, :2], method='linear')

plotter.add_arrows(points_coarse, B_coarse, mag=1e-3, color="white")

# Add in reflection
grid = grid.reflect(normal=(0, 1, 0))
B[:, 1] *= -1
grid.point_data["u"] = B_mag
grid.set_active_scalars("u")

plotter.add_mesh(grid, show_edges=False, cmap="jet")

points_coarse[:, 1] *= -1
B_coarse[:, 1] *= -1
plotter.add_arrows(points_coarse, B_coarse, mag=1e-3, color="white")

# Remove annoying legend
# plotter.remove_scalar_bar()

# Save screenshots
plotter.view_xy()
plotter.show()