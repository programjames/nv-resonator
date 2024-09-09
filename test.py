import petsc4py
from petsc4py import PETSc

scalar_type = PETSc.ScalarType
print(f"PETSc Scalar Type: {scalar_type}")