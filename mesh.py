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
    gmsh.model.add("cylinder")
    
    cylinder = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, height, radius)
    gmsh.model.occ.synchronize()

    # Tag surfaces/volume
    surfaces = gmsh.model.occ.getEntities(dim=2)
    gmsh.model.addPhysicalGroup(2, [s[1] for s in surfaces], tag=1, name="Surface")
    gmsh.model.addPhysicalGroup(3, [cylinder], tag=2, name="Volume")

    # Set mesh size
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)

    # Generate 3D mesh
    gmsh.model.mesh.generate(3)

    if display:
        gmsh.fltk.run()

    gmsh.write("cylinder.msh")
    gmsh.finalize()

def create_ring(outer_radius, inner_radius, height, mesh_size, display=True):
    gmsh.initialize()
    gmsh.model.add("ring")
    
    # Create ring
    outer_cylinder = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, height, outer_radius)
    inner_cylinder = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, height, inner_radius)
    ring, _ = gmsh.model.occ.cut([(3, outer_cylinder)], [(3, inner_cylinder)])
    
    gmsh.model.occ.synchronize()

    # # Tag surfaces/volume
    surfaces = gmsh.model.occ.getEntities(dim=2)
    gmsh.model.addPhysicalGroup(2, [s[1] for s in surfaces], tag=1, name="Inner")
    gmsh.model.addPhysicalGroup(3, [ring[0][1]], tag=2, name="Volume")

    # Set mesh size
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)

    # Generate 3D mesh
    gmsh.model.mesh.generate(3)

    if display:
        gmsh.fltk.run()

    gmsh.write("ring.msh")
    gmsh.finalize()
    
def main():
    import argparse
    global outer_radius, inner_radius, height, permittivity, mesh_size
    parser = argparse.ArgumentParser(description="Generate ring or cylinder mesh.")
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