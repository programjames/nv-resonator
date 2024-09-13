// Parameters
outer_radius = 8.17e-3;
inner_radius = 4.0e-3;
height = 1.452e-2;
gap = 1e-3;

// Mesh size
lc = 1e-3;

// Resonator - Left
Point(1) = {-height/2, inner_radius, 0, lc};
Point(2) = {-height/2, outer_radius, 0, lc};
Point(3) = {-gap/2, outer_radius, 0, lc};
Point(4) = {-gap/2, inner_radius, 0, lc};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

// Resonator - Right
Point(5) = {gap/2, inner_radius, 0, lc};
Point(6) = {gap/2, outer_radius, 0, lc};
Point(7) = {height/2, outer_radius, 0, lc};
Point(8) = {height/2, inner_radius, 0, lc};

Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 5};

Curve Loop(2) = {8, 7, 6, 5};
Plane Surface(2) = {2};
Physical Surface("resonator", 1) = {1, 2};

// Air
// z_max = 3/4 * height;
// r_max = 3/2 * outer_radius;
z_max = 1.2e-2;
r_max = 1.2e-2;
Point(9) = {-z_max, 0, 0, lc};
Point(10) = {-z_max, r_max, 0, lc};
Point(11) = {z_max, r_max, 0, lc};
Point(12) = {z_max, 0, 0, lc};

Line(9) = {9, 10};
Line(10) = {10, 11};
Line(11) = {11, 12};
Line(12) = {12, 9};

Curve Loop(3) = {9, 10, 11, 12};
Plane Surface(3) = {3, 1, 2};
Physical Surface("air", 2) = {3};

// Boundary
Physical Curve("casing", 3) = {9,10,11};
Physical Curve("axis", 4) = {12};

// Set output format to mesh
Mesh.Format = 1;  // .msh format