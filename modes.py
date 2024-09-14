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

# Function space
V = fem.FunctionSpace(domain, ("CG", 2, (2,)))

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Relative permittivity
epsilon_r = fem.Function(V.sub(0).collapse()[0])
air_dofs = fem.locate_dofs_topological(V, dim, cell_tags.find(2))
ceramic_dofs = fem.locate_dofs_topological(V, dim, cell_tags.find(1))
epsilon_r.x.array[:] = 1
epsilon_r.x.array[ceramic_dofs] = constants.EPSILON_R

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

# Br = 0 along ring's axis
Vr = V.sub(1)
boundary_dofs = fem.locate_dofs_topological((Vr, V), dim-1, facet_tags.find(4))
u_bc = fem.Function(V)
u_bc.x.array[:] = 0

bc = fem.dirichletbc(u_bc, boundary_dofs, V.sub(1))
bcs = [bc]

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

# Plot wave modes
for i in range(nconv):
    eigval = pep.getEigenpair(i, vr, vi)
    f = eigval.imag * constants.C / (2 * np.pi)
    if f < 2e9 or f > 5e9: continue

    B = vr.array.real.reshape(-1, 2)
    B /= B.max()
    B = np.pad(B, ((0, 0), (0, 1)))

    topology, cell_types, geometry = dolfinx.plot.vtk_mesh(V)
    grid = pv.UnstructuredGrid(topology, cell_types, geometry)
    plotter = pv.Plotter(off_screen = True)
    plotter.add_title(f"{f/1e9:.4f} GHz")
    
    # Magnitude
    sign = np.sign(np.where(np.abs(B[:, 0]) > np.abs(B[:, 1]), B[:, 0], B[:, 1]))
    B_mag = sign * np.linalg.norm(B, axis=-1)
    
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
    plotter.remove_scalar_bar()

    # Save screenshots
    plotter.view_xy()
    plotter.screenshot(f"modes/{f/1e9:.4f}_ghz.png")
    
    print("Saved", f"modes/{f/1e9:.4f}_ghz.png")