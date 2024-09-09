# pip install jax jax-fem meshio pyfiglet

import meshio
import numpy as np
import jax.numpy as jnp
import jax_fem

# Load the Gmsh mesh
mesh = meshio.read("cylinder.msh")

# Convert the meshio points and cells to arrays suitable for jax-fem
points = jnp.array(mesh.points)
cells = jnp.array(mesh.cells_dict['tetra'])

def stiffness_matrix(u, v):
    # Curl(u) · Curl(v) over the domain
    return jax_fem.inner(jax_fem.curl(u), jax_fem.curl(v))

def mass_matrix(u, v):
    # u · v over the domain
    return jax_fem.inner(u, v)

fem = jax_fem.Fem(mesh=mesh, points=points, cells=cells, degree=1)

# Set up stiffness and mass matrices
stiffness = fem.assemble_matrix(stiffness_matrix)
mass = fem.assemble_matrix(mass_matrix)
