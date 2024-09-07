# nv-resonator

## Installation

```bash
conda create -yn nv-resonator
conda activate nv-resonator
conda install -yc conda-forge fenics-dolfinx=0.7 pyvista libstdcxx-ng gmsh
pip install gmsh
```

## Simulation

First, generate a mesh:

```bash
python mesh.py ring
```

We can then find the microwave modes with

```bash
python sim.py ring.msh
```

See `python [mesh|sim].py --help` for a full list of options.