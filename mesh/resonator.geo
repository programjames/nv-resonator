// Parameters

outer_radius = 8.17e-3;
inner_radius = 2e-3;
height = 1.452e-2;

// Mesh size
lc = 5e-4;

// Resonator
Point(1) = {-height/2, inner_radius, 0, lc};
Point(2) = {-height/2, outer_radius, 0, lc};
Point(3) = {height/2, outer_radius, 0, lc};
Point(4) = {height/2, inner_radius, 0, lc};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};
Physical Surface("resonator", 1) = {1};
Physical Curve("dielectric", 2) = {1, 2, 3, 4};

// Air

z_max = 2e-2;
r_max = 2e-2;
Point(5) = {-z_max, 0, 0, lc};
Point(6) = {-z_max, r_max, 0, lc};
Point(7) = {z_max, r_max, 0, lc};
Point(8) = {z_max, 0, 0, lc};

Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 5};

Curve Loop(2) = {5, 6, 7, 8};
Plane Surface(2) = {2, 1};
Physical Surface("air", 2) = {2};

// Boundary
Physical Curve("casing", 3) = {5,6,7};
Physical Curve("axis", 4) = {8};

// Set output format to mesh
Mesh.Format = 1;  // .msh format