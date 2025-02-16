# imports
import sys
from mpi4py import MPI
import numpy as np
import ufl
from basix.ufl import element, mixed_element
from dolfinx import default_scalar_type, fem, io, plot
from dolfinx.fem.petsc import assemble_matrix
from dolfinx.mesh import CellType, create_rectangle, exterior_facet_indices, locate_entities

try:
    import pyvista
    have_pyvista = True
except ModuleNotFoundError:
    print("pyvista and pyvistaqt are required to visualise the solution")
    have_pyvista = False

try:
    from slepc4py import SLEPc
except ModuleNotFoundError:
    print("slepc4py is required for this demo")
    sys.exit(0)

# geometry (all lengths in nanometers)

l0 = 1550  # free space wavelength in nm

w_si = 400    # core width (nm)
h_si = 220    # core height (nm)

w_clad = 2940  # overall domain width (nm)
h_clad = w_clad  # overall domain height (nm)

w_dom = w_clad + 100
h_dom = h_clad + 100

nx = 300
ny = nx

msh = create_rectangle(
    MPI.COMM_WORLD, np.array([[0, 0], [w_dom, h_dom]]), np.array([nx, ny]), CellType.triangle
)
msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)

# Set up a Pyvista plotter with 1 row and 4 columns
plotter = pyvista.Plotter(shape=(2, 2))
showvectorplot = False

# ----------------------------------------
# Subplot (0,0): Mesh Geometry
# ----------------------------------------
cells, cell_types, points = plot.vtk_mesh(msh)
grid = pyvista.UnstructuredGrid(cells, cell_types, points)
plotter.subplot(0, 0)
plotter.add_mesh(grid, color="lightgray", opacity=0.3, show_edges=True)
# Add text annotation with domain dimensions (all values in nm)
mesh_info = (
    "Mesh Geometry\n"
    f"Domain: 0 - {w_clad} nm (x) × 0 - {h_clad} nm (y)\n"
    f"Core: Center = ({w_clad/2:.0f}, {h_clad/2:.0f}) nm, Size = {w_si} nm × {h_si} nm"
)
plotter.add_text(mesh_info, font_size=10)
plotter.view_xy()

# ----------------------------------------
# eps Field: Dielectric Permittivity
# ----------------------------------------
eps_d = 3.48    # core permittivity (unitless)
eps_c = 1.44    # cladding permittivity (unitless)
eps_p = 1.44 + 0.01j
# Center coordinate for core
ctr = w_dom / 2.0

def Omega_d(x):
    # Core region: centered at (ctr, ctr)
    return ((x[0] >= ctr - w_si/2) & (x[0] <= ctr + w_si/2) &
            (x[1] >= ctr - h_si/2) & (x[1] <= ctr + h_si/2))

def Omega_c(x):
    in_cladding = ((x[0] >= ctr - w_clad/2) & (x[0] <= ctr + w_clad/2) &
                   (x[1] >= ctr - h_clad/2) & (x[1] <= ctr + h_clad/2))
    return in_cladding & (~Omega_d(x))

def eps_expr(x):
    # Return eps_d in the core, eps_c elsewhere
    return np.where(Omega_d(x), eps_d, 
           np.where(Omega_c(x), eps_c, eps_p))

# Define a CG1 function space and interpolate eps
D = fem.functionspace(msh, ("CG", 1))
eps = fem.Function(D)
eps.interpolate(eps_expr)

# Get the array of epsilon values at the vertices
eps_values = eps.x.array

# Print unique values (you should see values close to eps_d, eps_c, and eps_p)
unique_vals = np.unique(eps_values)
print("Unique epsilon values at vertices:", unique_vals)

# Optionally, print min/max of the imaginary part to check for the PML
print("Imaginary part: min =", np.min(np.imag(eps_values)),
      "max =", np.max(np.imag(eps_values)))

# Create a new grid for eps display
cells, cell_types, points = plot.vtk_mesh(msh)
grid_eps = pyvista.UnstructuredGrid(cells, cell_types, points)
# For CG1 the dof values are associated with vertices
eps_values = eps.x.array
grid_eps.point_data["eps"] = np.real(eps_values)

plotter.subplot(1, 0)
plotter.add_mesh(
    grid_eps, scalars="eps", cmap="viridis", show_edges=False,
    scalar_bar_args={"title": "Dielectric Permittivity\n(unitless)"}
)
plotter.add_text("Dielectric Permittivity Field\n(Core: 3.48, Cladding: 1.44)", font_size=10)
plotter.view_xy()

# ----------------------------------------
# Problem Definition and Eigenmode Computation
# ----------------------------------------
degree = 1
NED = element("Nedelec 1st kind H(curl)", msh.basix_cell(), degree)
Q = element("Lagrange", msh.basix_cell(), degree)
V = fem.functionspace(msh, mixed_element([NED, Q]))

lmbd0 = l0  # in nm
k0 = 2 * np.pi / lmbd0  # in 1/nm

et, ez = ufl.TrialFunctions(V)
vt, vz = ufl.TestFunctions(V)

a_tt = (ufl.inner(ufl.curl(et), ufl.curl(vt)) - (k0**2) * eps * ufl.inner(et, vt)) * ufl.dx
b_tt = ufl.inner(et, vt) * ufl.dx
b_tz = ufl.inner(et, ufl.grad(vz)) * ufl.dx
b_zt = ufl.inner(ufl.grad(ez), vt) * ufl.dx
b_zz = (ufl.inner(ufl.grad(ez), ufl.grad(vz)) - (k0**2) * eps * ufl.inner(ez, vz)) * ufl.dx

a = fem.form(a_tt)
b = fem.form(b_tt + b_tz + b_zt + b_zz)

A = assemble_matrix(a)
A.assemble()
B = assemble_matrix(b)
B.assemble()

eps_eigensolver = SLEPc.EPS().create(msh.comm)
eps_eigensolver.setOperators(A, B)
eps_eigensolver.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
tol = 1e-9
eps_eigensolver.setTolerances(tol=tol)
eps_eigensolver.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
st = eps_eigensolver.getST()
st.setType(SLEPc.ST.Type.SINVERT)
eps_eigensolver.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)
eps_eigensolver.setTarget(-((0.5 * k0) ** 2))
eps_eigensolver.setDimensions(nev=1)
eps_eigensolver.solve()
eps_eigensolver.view()
eps_eigensolver.errorView()

vals = [(i, np.sqrt(-eps_eigensolver.getEigenvalue(i))) for i in range(eps_eigensolver.getConverged())]
vals.sort(key=lambda x: x[1].real)
eh = fem.Function(V)
kz_list = []

# ----------------------------------------
# Eigenmode Visualization: Transverse Electric Field (Et)
# ----------------------------------------
for i, kz in vals:
    eps_eigensolver.getEigenpair(i, eh.x.petsc_vec)
    error = eps_eigensolver.computeError(i, SLEPc.EPS.ErrorType.RELATIVE)
    if error < tol and np.isclose(kz.imag, 0, atol=tol):
        kz_list.append(kz)
        print(f"eigenvalue: {-kz**2}")
        print(f"kz: {kz}")
        print(f"kz/k0: {kz / k0}")

        eh.x.scatter_forward()
        eth, ezh = eh.split()
        eth = eh.sub(0).collapse()
        ez = eh.sub(1).collapse()

        # Transform eth, ezh into Et and Ez
        eth.x.array[:] = eth.x.array[:] / kz
        ezh.x.array[:] = ezh.x.array[:] * 1j

        gdim = msh.geometry.dim
        V_dg = fem.functionspace(msh, ("DQ", degree, (gdim,)))
        Et_dg = fem.Function(V_dg)
        Et_dg.interpolate(eth)

        # Save solutions (if needed)
        with io.VTXWriter(msh.comm, f"sols/Et_{i}.bp", Et_dg) as f:
            f.write(0.0)
        with io.VTXWriter(msh.comm, f"sols/Ez_{i}.bp", ezh) as f:
            f.write(0.0)

        if have_pyvista:
            V_cells, V_types, V_x = plot.vtk_mesh(V_dg)
            V_grid = pyvista.UnstructuredGrid(V_cells, V_types, V_x)
            Et_values = np.zeros((V_x.shape[0], 3), dtype=np.float64)
            Et_values[:, :msh.topology.dim] = Et_dg.x.array.reshape(V_x.shape[0], msh.topology.dim).real
            V_grid.point_data["u"] = Et_values

            # Place each eigenmode in its own subplot (starting at column 2)
            plotter.subplot(0, 1)
            plotter.add_mesh(V_grid.copy(), show_edges=False)
            # Annotate the eigenmode display. Here we assume Et is a transverse electric field,
            # and we note that its values are scaled (hence “arb. units”).
            plotter.add_text(f"Eigenmode {i}\nTransverse Electric Field (Et)\n[arb. units]", font_size=10)
            plotter.view_xy()
            plotter.link_views()
        
        if have_pyvista and showvectorplot:
            # Create a Pyvista grid for the DG field of Et
            V_cells, V_types, V_x = plot.vtk_mesh(V_dg)
            V_grid = pyvista.UnstructuredGrid(V_cells, V_types, V_x)
            
            # Set the point data to be the vector field (here we assume real values are sufficient)
            Et_values = np.zeros((V_x.shape[0], 3), dtype=np.float64)
            Et_values[:, :msh.topology.dim] = Et_dg.x.array.reshape(V_x.shape[0], msh.topology.dim).real
            V_grid.point_data["u"] = Et_values
            
            # Debug: print the magnitude stats of the vector field
            norms = np.linalg.norm(Et_values, axis=1)
            print("Min, max, mean of Et vector magnitudes:", norms.min(), norms.max(), norms.mean())
            
            # Use the glyph filter to display arrow glyphs representing the field vectors.
            # Increase 'factor' to make the arrows visible
            all_indices = np.arange(V_grid.n_points)
            # This gives the indices we originally extracted (every 10th point)
            extracted = np.arange(0, V_grid.n_points, 10)
            # Inverse: all indices *not* in the extracted set
            inverse_indices = np.setdiff1d(all_indices, extracted)
            inverse_grid = V_grid.extract_points(inverse_indices)

            arrow_glyphs = inverse_grid.glyph(orient="u", scale="u", factor=10000)
            
            # Place in its subplot (for instance, row 1, column 1+i)
            plotter.subplot(1, 1)
            plotter.add_mesh(arrow_glyphs, color="red")
            plotter.add_text(f"Eigenmode {i}\nTransverse Electric Field (Et)\nDisplayed as Arrows", font_size=10)
            plotter.view_xy()
            plotter.link_views()

# Finally, display all subplots in one window.
plotter.show()
