#!/bin/bash

# Select PETSc scalar type and generate environment name
SCALAR_TYPE=real
ENV_NAME="nv"


# Create list of packages to install, depending on PETSc scalar type.
# The selected packages depend on Python 3.10 and MPICH for MPI communication.
if [[ "$SCALAR_TYPE" == "real" ]]; then
  FENICSX_PKGS="fenics-basix=0.6.0=py310hdf3cbec_0 fenics-dolfinx=0.6.0=py310hf97048e_101 fenics-ffcx=0.6.0=pyh56297ac_0 fenics-libbasix=0.6.0=h1284905_0 fenics-libdolfinx=0.6.0=h4cb9d57_101 fenics-ufcx=0.6.0=h56297ac_0 fenics-ufl=2023.1.1=pyhd8ed1ab_1"
else
  FENICSX_PKGS="fenics-basix=0.6.0=py310hdf3cbec_0 fenics-dolfinx=0.6.0=py310he6dc2dd_1 fenics-ffcx=0.6.0=pyh56297ac_0 fenics-libbasix=0.6.0=h1284905_0 fenics-libdolfinx=0.6.0=hf51c956_1 fenics-ufcx=0.6.0=h56297ac_0 fenics-ufl=2023.1.1=pyhd8ed1ab_1"
fi


# Install packages and activate the new environment
mamba create -y -c conda-forge -n "${ENV_NAME}" $FENICSX_PKGS
conda activate "${ENV_NAME}"
if [[ "$?" == "0" ]]; then
  conda config --env --add channels conda-forge

  # Pin FEniCSx packages
  for PKG in $FENICSX_PKGS; do
    conda config --env --append pinned_packages $PKG
  done

  mamba install -y python-gmsh
else
  echo "Failed to create ${ENV_NAME} environment"
fi