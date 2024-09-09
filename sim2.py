from mpi4py import MPI
from dolfinx import mesh, fem, plot
import dolfinx.fem.petsc
from slepc4py import SLEPc
from petsc4py import PETSc
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
V = fem.FunctionSpace(domain, ("CG", 1))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
n = ufl.FacetNormal(domain)

dx = ufl.dx
ds = ufl.ds

# Define forms
m = EPSILON_R * MU_R * ufl.inner(u, v) * dx  # Mass matrix
b = ufl.inner(u, v) * ds  # Boundary integral
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx  # Stiffness matrix

# Assemble matrices
M = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(m))
M.assemble()
B = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(b))
B.assemble()
A = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(a))
A.assemble()

# Get the size of the matrices
n = M.getSize()[0]

# Create identity and zero matrices
I = PETSc.Mat().create(MPI.COMM_WORLD)
I.setSizes([n, n])
I.setType('aij')
I.setUp()
I.setDiagonal(PETSc.Vec().createWithArray(np.ones(n)))
I.assemble()

Z = PETSc.Mat().create(MPI.COMM_WORLD)
Z.setSizes([n, n])
Z.setType('aij')
Z.setUp()
Z.assemble()

# Create the block matrices
A_block = PETSc.Mat().createNest([[A, Z], [Z, I]])
A_block.assemble()

B_imag = B.copy()
B_imag.scale(1j)
B_block = PETSc.Mat().createNest([[B_imag, M], [I, Z]])
B_block.assemble()

# Create the eigenvalue solver
eps = SLEPc.EPS().create(MPI.COMM_WORLD)
eps.setOperators(A_block, B_block)
eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
eps.setDimensions(20)  # Number of eigenvalues to compute
eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
eps.setTarget(3e9 / C * 2 * np.pi)
eps.setTolerances(1e-12, 200)

# Add this line to set the solver to use a complex inner product
# eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
eps.setFromOptions()

# Solve the eigenvalue problem
eps.solve()

# Get the number of converged eigenvalues
nconv = eps.getConverged()
frequencies = []
eigenvalues = []
modes = []

for i in range(nconv):
    # Convert eigenvalue to frequency
    eigenvalue = eps.getEigenvalue(i)
    eigenvalues.append(eigenvalue)
    frequency = eigenvalue * C / (2 * np.pi)
    frequencies.append(frequency)
    
    # Convert eigenvector to mode
    vr, vi = A_block.getVecs()
    eps.getEigenvector(i, vr, vi)
    modes.append(vr.array + 1j * vi.array)
    
# Print frequencies
if domain.comm.rank == 0:
    print("Resonant frequencies (GHz):")
    for i, freq in enumerate(frequencies):
        print(f"Mode {i+1}: {freq/1e9:.4f} GHz")
        print(modes[i].shape)

# Optionally, verify the solutions
print("\nVerifying solutions:")
for k in eigenvalues[:5]:  # Check the first 5 eigenvalues
    Bs = B.copy()
    Bs.scale(1j * k)
    Ms = M.copy()
    Ms.scale(k**2)
    lhs = Ms + Bs - A
    norm = lhs.norm()
    print(f"For k = {k:.6f}, ||k^2 M + ik B - A|| = {norm:.6e}")