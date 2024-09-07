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

![Ring Mesh](images/ring_mesh.png)

```
Info    : Writing 'ring.msh'...
Info    : Done writing 'ring.msh'
Cavity frequency is approximately 5.7277 GHz
```

Then, to find more exact microwave modes:

```bash
python sim.py ring.msh
```

![Ring Mode](images/ring_mode.png)

```
Resonant frequencies (GHz):
Mode 1: 0.0000 GHz
Mode 2: 1.3725 GHz
Mode 3: 1.3726 GHz
Mode 4: 1.7728 GHz
Mode 5: 2.2441 GHz
Mode 6: 2.2441 GHz
Mode 7: 2.7164 GHz
Mode 8: 2.7165 GHz
Mode 9: 3.2492 GHz
Mode 10: 3.2494 GHz
Mode 11: 3.5592 GHz
```