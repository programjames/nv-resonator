# nv-resonator

## Installation

```bash
conda create -yn nv-resonator
conda activate nv-resonator
conda install -yc conda-forge fenics-dolfinx=0.7 pyvista libstdcxx-ng gmsh
pip install gmsh
```

## Running

To generate a mesh,

```bash
gmsh mesh/resonator.geo -2
```

You can view it with

```bash
gmsh mesh/resonator.msh
```

<div style="text-align: center">
    <img src="images/mesh.png" alt="Ring Mesh" width="50%"/>
</div>

We're looking for radial or axial modes, so we only need to simulate a longitudinal slice parallel to the ring's axis. Rotating around the bottom edge will give the full ring. To create images of the wave modes, run

```bash
python modes.py
```

It should save them in `images/modes/<freq>.png`. Here's an example:

<div style="text-align: center">
    <img src="images/modes/2.74_ghz.png" alt="Axial Mode of Ring at 2.74 GHz" width="50%"/>
</div>


## Theory

<div style="text-align: center">
    <img src="images/paper_simulation.png" alt="Paper Simulation" width="50%"/>
</div>

In [Cavity-Enhanced Microwave Readout of a Solid-State Spin Sensor](https://www.nature.com/articles/s41467-021-21256-7) by Dirk Englund, et al., they use dielectric resonators to couple with an NV center. Their resonators have the following parameters:

```
outer_radius (a) = 8.17e-3
inner_radius (b) ~ 4e-3
height       (L) = 7.26e-3
permittivity (ε) ~ 34
```

The readout frequency of an NV center is ~2.87 GHz, so they want to tune the cavity to have a mode ~3 GHz. Their simulation (above) places the diamond between two resonators as a source. A cylindrical resonator's $TE_{01n}$ mode is approxmately

$$\frac{0.034 (a/L + 3.45)}{a\sqrt{\epsilon}} GHz,$$

given by Kajfez & Guillon in [Dielectric resonators](https://search.worldcat.org/en/title/927557286) and readily available on [Wikipedia](https://en.wikipedia.org/wiki/Dielectric_resonator#Theory_of_operation). This approximation yields resonance ~2.86 GHz, and their simulation is ~2.90 GHz.

Our system is slightly different. THey used an aluminum shield, but that is only important for increasing the relaxation time. Since we are using ours only as a magnetometer, it is fine to leave it open to the air. In addition, we are using an LED instead of a LASER, so we can place the diamond at one end of a longer ring, rather than orienting it between two smaller rings. The simplified resonator is defined in `mesh/resonator.geo`, while a reconstruction of their double-resonator is found in `mesh/double.geo`.

## Weak Formulation

Starting from Maxwell's equations,

$$\begin{aligned}
\nabla \cdot E &= \rho = 0&\text{(in a dielectric)}\\
\nabla \cdot B &= 0\\
\nabla \times E &= -\frac{\partial B}{\partial t}\\
\nabla \times B &= \mu\left(J + \varepsilon\frac{\partial E}{\partial t}\right) = \frac{\mu_r\varepsilon_r}{c^2}\frac{\partial E}{\partial t} &\text{(in a dielectric),}\\
\end{aligned}$$

where $\mu_r, \varepsilon_r$ are the relative permeability and permittivity of the material. Assuming we have a mode

$$B, E\sim e^{i(kx - \omega t)},$$

we get

$$\nabla \times (\nabla\times B) = \mu_r\varepsilon_r k^2B.$$

Since

$$\nabla \times (\nabla\times B) = \nabla(\nabla\cdot B) - \nabla^2 B,$$

this reduces to

$$-\nabla^2 B = \mu_r\varepsilon_r k^2B.$$

In cylindrical coordinates, and assuming $B_\theta = 0$, this is

$$\begin{aligned}
\nabla^2 B_z &= \mu_r\varepsilon_r k^2B_z\\
\nabla^2 B_r - \frac{B_r}{r^2} &= \mu_r\varepsilon_r k^2B_r\\
\end{aligned}$$

The weak formulation is

$$-\langle \nabla^2 B_z, v\rangle = k^2\langle \mu_r\varepsilon_rB_z, v\rangle
\iff
\langle \nabla B_z, \nabla v\rangle - \int \frac{B_z}{\partial \hat{n}} v\mathrm{d}S = k^2\langle \mu_r\varepsilon_rB_z, v\rangle.$$

The Sommerfeld radiation boundary condition is

$$\frac{B_z}{\partial \hat{n}} = jk B_z,$$

so we're left with

$$\langle \nabla B_z, \nabla v\rangle - jk \int B_z v\mathrm{d}S = k^2 \langle \mu_r\varepsilon_rB_z, v\rangle.$$

Letting $\lambda = -jk$, this is a quadratic eigenvalue problem of the form

$$\lambda^2 A + \lambda B + C = 0,$$

where

$$\begin{aligned}
A &= \langle \mu_r\varepsilon_rB_z, v\rangle &\text{(mass matrix)}\\
B &= \int B_z v\mathrm{d}S &\text{(boundary condition)}\\
C &= \langle \nabla B_z, \nabla v\rangle &\text{(stiffness matrix)}.
\end{aligned}$$

The derivation for $B_r$ is similar, though we need to adjust

$$C = \langle \nabla B_r, \nabla v\rangle + \frac{1}{r}\langle B_r, v\rangle.$$

Finally, we only apply the radiation boundary condition along the three non-axial edges of our mesh. The axis just has the Dirichlet $B_r = 0$ condition.