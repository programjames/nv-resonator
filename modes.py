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
domain, cell_tags, facet_tags = dolfinx.io.gmshio.read_from_msh("mesh/resonator.msh", MPI.COMM_WORLD, gdim=2)
dim = domain.topology.dim

SOLVE_MAGNETIC = False

# Function space
V = ("N1curl", 2) if SOLVE_MAGNETIC else ("CG", 1, (2,))
V = fem.FunctionSpace(domain, V)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Relative permittivity
V_eps = fem.FunctionSpace(domain, ("DG", 0)) if SOLVE_MAGNETIC else V.sub(0).collapse()[0]
ceramic_dofs = fem.locate_dofs_topological(V_eps, dim, cell_tags.find(1))
epsilon_r = fem.Function(V_eps)
epsilon_r.x.array[:] = 1
epsilon_r.x.array[ceramic_dofs] = constants.EPSILON_R

# Coordinate system
x = ufl.SpatialCoordinate(domain)
z = x[0]
r = x[1]

n = ufl.FacetNormal(domain)
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)

# Weak form; note mu_r = 1
if SOLVE_MAGNETIC:
    a = ufl.inner(u, v) * r * ufl.dx
    b = ufl.inner(1 / epsilon_r * (ufl.dot(n, u) * n - u), v) * ds(3)
    c = ufl.inner(1 / epsilon_r * ufl.curl(u), ufl.curl(v)) * r * ufl.dx
else:
    a = ufl.inner(epsilon_r * u, v) * r * ufl.dx
    b = ufl.inner(u, v) * ds(3)
    c = ufl.inner(ufl.grad(u), ufl.grad(v)) * r * ufl.dx

bcs = []

"""
Uncomment for Dirichlet boundary conditions instead of radiation. Only works
for the the electric field for now (magnetic field needs curl(B) = 0, not B = 0).
"""

# b = b - b
# boundary_dofs = fem.locate_dofs_topological(V, dim-1, facet_tags.find(3))
# u_bc = fem.Function(V)
# u_bc.x.array[:] = 0
# bc = fem.dirichletbc(u_bc, boundary_dofs)
# bcs = [bc]

# Assemble matrices for eigenvalue solver
A = fem.petsc.assemble_matrix(fem.form(a), bcs=bcs)
A.assemble()
B = fem.petsc.assemble_matrix(fem.form(b), bcs=bcs)
B.assemble()
C = fem.petsc.assemble_matrix(fem.form(c), bcs=bcs)
C.assemble()

pep = SLEPc.PEP().create(domain.comm)
pep.setOperators([C, B, A])  # Aλ^2 + Bλ + C = 0.

# Set solver options
pep.setType(SLEPc.PEP.Type.TOAR) # Good results: LINEAR, QARNOLDI, TOAR
pep.setDimensions(nev=100, ncv=200, mpd=100) # num eigenvalues, num column vectors, max projection dimension

# Search around f = 3 GHz
target = (3e9 * 2 * np.pi / constants.C)
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

# Avoid memory leaks
A.destroy()
B.destroy()
C.destroy()

nconv = pep.getConverged()
vr, vi = pep.getOperators()[0].createVecs()

outer_radius = 8.17e-3
inner_radius = 2e-3
height = 1.452e-2

# Plot wave modes
for i in range(nconv):
    eigval = pep.getEigenpair(i, vr, vi)
    f = eigval.imag * constants.C / (2 * np.pi)
    if f < 2e9 or f > 6e9: continue
    
    u_eigen = fem.Function(V)
    u_eigen.x.array[:] = vr.array
    
    V_plot = fem.FunctionSpace(domain, ("CG", 1, (dim,)))
    u_plot = fem.Function(V_plot)
    u_plot.interpolate(u_eigen)

    B = u_plot.x.array.real.reshape(-1, dim)
    B /= B.max()
    B = np.pad(B, ((0, 0), (0, 1)))

    topology, cell_types, geometry = dolfinx.plot.vtk_mesh(V_plot)
    grid = pv.UnstructuredGrid(topology, cell_types, geometry)
    plotter = pv.Plotter(off_screen = True)
    
    # Magnitude
    
    grid.point_data["u"] = B
    grid.set_active_scalars("u")
    
    plotter.add_mesh(grid, show_edges=False, cmap="turbo")
    
    def add_line(start, end):
        line = pv.Line(start, end)
        plotter.add_mesh(line, color='black', line_width=2)
    
    add_line([-height/2, inner_radius, 0], [height/2, inner_radius, 0])
    add_line([-height/2, outer_radius, 0], [height/2, outer_radius, 0])
    add_line([-height/2, inner_radius, 0], [-height/2, outer_radius, 0])
    add_line([height/2, inner_radius, 0], [height/2, outer_radius, 0])
    
    # Arrows
    x, y, z = grid.points.T
    x_coarse = np.linspace(x.min(), x.max(), 25)
    y_coarse = np.linspace(y.min(), y.max(), 25)
    X, Y = np.meshgrid(x_coarse, y_coarse)
    points_coarse = np.column_stack((X.ravel(), Y.ravel(), np.zeros_like(X.ravel())))
    B_coarse = scipy.interpolate.griddata(grid.points[:, :2], B, points_coarse[:, :2], method='linear')
    B_coarse /= np.linalg.norm(B_coarse, axis=1)[:, None]
    
    plotter.add_arrows(points_coarse, B_coarse, mag=5e-4, color="white")
    
    # Add in reflection
    grid = grid.reflect(normal=(0, 1, 0))
    B[:, 1] *= -1
    grid.point_data["u"] = B
    grid.set_active_scalars("u")
    
    plotter.add_mesh(grid, show_edges=False, cmap="turbo")
    
    add_line([-height/2, -inner_radius, 0], [height/2, -inner_radius, 0])
    add_line([-height/2, -outer_radius, 0], [height/2, -outer_radius, 0])
    add_line([-height/2, -inner_radius, 0], [-height/2, -outer_radius, 0])
    add_line([height/2, -inner_radius, 0], [height/2, -outer_radius, 0])
    
    points_coarse[:, 1] *= -1
    B_coarse[:, 1] *= -1
    plotter.add_arrows(points_coarse, B_coarse, mag=5e-4, color="white")
    
    # Remove annoying legend
    plotter.remove_scalar_bar()

    # Save screenshots
    plotter.view_xy()
    plotter.camera.zoom(2.0)
    plotter.screenshot(f"modes/{f/1e9:.4f}_ghz.png")
    
    print("Saved", f"modes/{f/1e9:.4f}_ghz.png")