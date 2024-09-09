import numpy as np
import ufl
from mpi4py import MPI
import h5py
from petsc4py import PETSc
from slepc4py import SLEPc
import dolfinx.fem.petsc
from dolfinx import fem, mesh, io
from dolfinx.io import gmshio

# Read mesh
domain, cell_tags, facet_tags = gmshio.read_from_msh("resonator.msh", MPI.COMM_WORLD, gdim=2)

# Function space
V = fem.FunctionSpace(domain, ("CG", 1))

# Define problem constants
EPSILON_R_CERAMIC = 34.0  # Relative permittivity of the ceramic
EPSILON_R_AIR = 1.0  # Relative permittivity of air
MU_R = 1.0  # Relative permeability
C = 299792458  # Speed of light in vacuum

# TODO: formulate this properly

# Create a function to represent the relative permittivity
epsilon_r = fem.Function(V)
air_cells = cell_tags.find(1)
ceramic_cells = cell_tags.find(2)
air_dofs = fem.locate_dofs_topological(V, domain.topology.dim, air_cells)
ceramic_dofs = fem.locate_dofs_topological(V, domain.topology.dim, ceramic_cells)
epsilon_r.x.array[air_dofs] = EPSILON_R_AIR
epsilon_r.x.array[ceramic_dofs] = EPSILON_R_CERAMIC

# Weak formulation
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# For axisymmetric problems, we need to include the radial coordinate in the integrals
x = ufl.SpatialCoordinate(domain)
r = x[1]  # Radial coordinate (assuming y is the radial direction in your mesh)

# Corrected weak form for the TE01n modes
a = (ufl.inner(ufl.Dx(u, 1), ufl.Dx(v, 1)) +
     ufl.inner(ufl.Dx(u, 0), ufl.Dx(v, 0))) * ufl.dx
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
a += ufl.inner(r * ufl.Dx(u, 1), ufl.Dx(v, 1)) * ufl.dx - ufl.inner(ufl.Dx(u, 1), r * ufl.Dx(v, 1)) * ufl.dx
m = ufl.inner(u, v) * ufl.dx

# Boundary condition
boundary_dofs = fem.locate_dofs_topological(V, entity_dim=1, entities=facet_tags.find(1))
u_bc = fem.Function(V)
u_bc.x.array[:] = 0

bc = fem.dirichletbc(u_bc, boundary_dofs)

# Assemble matrices
A = fem.petsc.assemble_matrix(fem.form(a), bcs=[bc])
A.assemble()
M = fem.petsc.assemble_matrix(fem.form(m), bcs=[bc])
M.assemble()

# Create eigensolver
eps = SLEPc.EPS().create(MPI.COMM_WORLD)
eps.setOperators(A, M)
eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
eps.setDimensions(10)  # Number of eigenvalues to compute
eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
eps.setTarget(3e9 / C * 2 * np.pi)  # Target frequency (adjust as needed)
eps.solve()

# Extract eigenvalues and eigenvectors
nconv = eps.getConverged()
print(f"Number of converged eigenpairs: {nconv}")

with h5py.File('resonator_modes.h5', 'w') as f:
    # Store mesh data
    f.create_dataset('coordinates', data=domain.geometry.x)
    f.create_dataset('topology', data=cell_tags.values)
    f.create_dataset('topology_tags', data=cell_tags.indices)
    
    # Create a group for modes
    modes_group = f.create_group('modes')

    for i in range(nconv):
        eigenvalue = eps.getEigenvalue(i)
        k = np.sqrt(eigenvalue.real)
        f = k * C / (2 * np.pi)
        print(f"Mode {i+1}: f = {f/1e9:.4e} GHz")

        # Compute and normalize eigenvector
        vr, vi = A.createVecs()
        eps.getEigenvector(i, vr, vi)
        
        mode_group = modes_group.create_group(f'mode_{i+1}')
        
        # Store mode data and attributes
        mode_group.create_dataset('data', data=vr.array)
        mode_group.attrs['frequency'] = f

print("Simulation complete.")