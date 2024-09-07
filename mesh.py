"""
In "Cavity-enhanced microwave readout of a solid-state spin sensor"
by Dirk Englund, et al., they use a ring with

outer_radius          = 8.17e-3
inner_radius          = 4.0e-3
height                = 7.26e-3
permittivity          = 34

If you had a cylinder with that outer_radius and height, then a
decent approximation of its resonant frequency is

 3.4e7 * (R / L + 3.45)
------------------------
      R * sqrt(e)
      
which is ~3.27 GHz. A numerical simulation gives ~3.00 GHz. Dirk
mentioned they wanted it to be slightly off from the readout
frequency (~2.87GHz) to avoid trapping the microwaves. However,
drilling a hole in the middle will about double the frequency! A
numerical simulation gives ~6.4GHz for a ring with the given
parameters.
"""

import gmsh
from constants import EPSILON_R as permittivity

# Default values, can change with arguments (see bottom)
outer_radius = 8.17e-3
inner_radius = 4.0e-3
height       = 7.26e-3 * 2
mesh_size    = 1e-3

def approx_cylinder_freq(radius, height, eps):
    return 3.4e7 / radius / permittivity**0.5 * (radius / height + 3.45)

def approx_ring_freq(outer_radius, height, eps):
    return 2 * approx_cylinder_freq(outer_radius, height, eps)

def create_cylinder(radius, height, mesh_size, display=True):
    gmsh.initialize()
    gmsh.model.add("ring")

    # Create points for outer circle
    p1 = gmsh.model.geo.addPoint(0, 0, 0, mesh_size)
    p2 = gmsh.model.geo.addPoint(radius, 0, 0, mesh_size)
    p3 = gmsh.model.geo.addPoint(0, radius, 0, mesh_size)
    p4 = gmsh.model.geo.addPoint(-radius, 0, 0, mesh_size)
    p5 = gmsh.model.geo.addPoint(0, -radius, 0, mesh_size)

    # Create arcs for outer circle
    c1 = gmsh.model.geo.addCircleArc(p2, p1, p3)
    c2 = gmsh.model.geo.addCircleArc(p3, p1, p4)
    c3 = gmsh.model.geo.addCircleArc(p4, p1, p5)
    c4 = gmsh.model.geo.addCircleArc(p5, p1, p2)

    # Create curve loop
    loop = gmsh.model.geo.addCurveLoop([c1, c2, c3, c4])

    # Create surface
    bottom_surface = gmsh.model.geo.addPlaneSurface([loop])

    # Extrude to create volume
    volume = gmsh.model.geo.extrude([(2, bottom_surface)], 0, 0, height)

    gmsh.model.geo.synchronize()

    # Get the tags of the side surface and top surface
    side_surfaces = [v[1] for v in volume[2:]]
    top_surface = volume[0][1]

    # Assign physical groups (important for Dolfinx)
    gmsh.model.addPhysicalGroup(2, [bottom_surface], tag=1, name="BottomSurface")
    gmsh.model.addPhysicalGroup(2, [top_surface], tag=2, name="TopSurface")
    gmsh.model.addPhysicalGroup(2, side_surfaces, tag=3, name="SideSurfaces")
    gmsh.model.addPhysicalGroup(3, [volume[1][1]], tag=4, name="Volume")

    # Generate 3D mesh
    gmsh.model.mesh.generate(3)

    if display:
        gmsh.fltk.run()

    gmsh.write("cylinder.msh")
    gmsh.finalize()

def create_ring(outer_radius, inner_radius, height, mesh_size, display=True):
    gmsh.initialize()
    gmsh.model.add("ring")

    # Create points for outer circle
    p1 = gmsh.model.geo.addPoint(0, 0, 0, mesh_size)
    p2 = gmsh.model.geo.addPoint(outer_radius, 0, 0, mesh_size)
    p3 = gmsh.model.geo.addPoint(0, outer_radius, 0, mesh_size)
    p4 = gmsh.model.geo.addPoint(-outer_radius, 0, 0, mesh_size)
    p5 = gmsh.model.geo.addPoint(0, -outer_radius, 0, mesh_size)

    # Create arcs for outer circle
    c1 = gmsh.model.geo.addCircleArc(p2, p1, p3)
    c2 = gmsh.model.geo.addCircleArc(p3, p1, p4)
    c3 = gmsh.model.geo.addCircleArc(p4, p1, p5)
    c4 = gmsh.model.geo.addCircleArc(p5, p1, p2)

    # Create points for inner circle
    p6 = gmsh.model.geo.addPoint(inner_radius, 0, 0, mesh_size)
    p7 = gmsh.model.geo.addPoint(0, inner_radius, 0, mesh_size)
    p8 = gmsh.model.geo.addPoint(-inner_radius, 0, 0, mesh_size)
    p9 = gmsh.model.geo.addPoint(0, -inner_radius, 0, mesh_size)

    # Create arcs for inner circle
    c5 = gmsh.model.geo.addCircleArc(p6, p1, p7)
    c6 = gmsh.model.geo.addCircleArc(p7, p1, p8)
    c7 = gmsh.model.geo.addCircleArc(p8, p1, p9)
    c8 = gmsh.model.geo.addCircleArc(p9, p1, p6)

    # Create curve loops
    outer_loop = gmsh.model.geo.addCurveLoop([c1, c2, c3, c4])
    inner_loop = gmsh.model.geo.addCurveLoop([c5, c6, c7, c8])

    # Create surface
    bottom_surface = gmsh.model.geo.addPlaneSurface([outer_loop, inner_loop])

    # Extrude to create volume
    volume = gmsh.model.geo.extrude([(2, bottom_surface)], 0, 0, height)

    gmsh.model.geo.synchronize()

    # Get the tags of the side surface and top surface
    side_surfaces = [v[1] for v in volume[2:]]
    top_surface = volume[0][1]

    # Assign physical groups (important for Dolfinx)
    gmsh.model.addPhysicalGroup(2, [bottom_surface], tag=1, name="BottomSurface")
    gmsh.model.addPhysicalGroup(2, [top_surface], tag=2, name="TopSurface")
    gmsh.model.addPhysicalGroup(2, side_surfaces, tag=3, name="SideSurfaces")
    gmsh.model.addPhysicalGroup(3, [volume[1][1]], tag=4, name="Volume")

    # Generate 3D mesh
    gmsh.model.mesh.generate(3)

    if display:
        gmsh.fltk.run()

    gmsh.write("ring.msh")
    gmsh.finalize()
    
def main():
    import argparse
    global outer_radius, inner_radius, height, permittivity, mesh_size
    parser = argparse.ArgumentParser(description="Find resonate frequencies of a cavity.")
    parser.add_argument("type", type=str, default="ring", help="'ring' or 'cylinder'")
    parser.add_argument("-a", "--outer-radius", type=float, default=outer_radius, 
                        help=f"Outer radius of the cylinder in meters (default: {outer_radius})")
    parser.add_argument("-b", "--inner-radius", type=float, default=inner_radius, 
                        help="fInner radius of the cylinder in meters (default: {inner_radius})")
    parser.add_argument("-L", "--height", type=float, default=height, 
                        help=f"Height of the cylinder in meters (default: {height})")
    parser.add_argument("-e", "--permittivity", type=float, default=permittivity, 
                        help=f"Relative permittivity of the material (default: {permittivity})")
    parser.add_argument("-s", "--mesh-size", type=float, default=mesh_size, 
                        help=f"Mesh size (default: {mesh_size})")
    parser.add_argument("--no-popup", action="store_true", help="Suppress mesh visual")

    args = parser.parse_args()
    if args.type.lower() == "ring":
        outer_radius = args.outer_radius
        inner_radius = args.inner_radius
        height = args.height
        permittivity = args.permittivity
        mesh_size = args.mesh_size
        
        create_ring(outer_radius, inner_radius, height, mesh_size, display=not args.no_popup)
        freq = approx_ring_freq(outer_radius, height, permittivity)
        print(f"Cavity frequency is approximately {freq/1e9:.4f} GHz")
        
    elif args.type.lower() == "cylinder":
        radius = args.outer_radius
        height = args.height
        permittivity = args.permittivity
        mesh_size = args.mesh_size
        
        create_cylinder(radius, height, mesh_size, display=not args.no_popup)
        freq = approx_cylinder_freq(radius, height, permittivity)
        print(f"Cavity frequency is approximately {freq/1e9:.4f} GHz")
        
if __name__ == "__main__":
    main()