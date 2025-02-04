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

# geometry

l0 = 1550  # free space wavelength

w_si=400
h_si=220

w_clad=3040
h_clad=w_clad

'''w_pml=(w_clad+80)
h_pml=w_pml'''

#w_void=(2*l0+300)
#h_void=w_void

nx = 300
ny = nx

msh = create_rectangle(
    MPI.COMM_WORLD, np.array([[0, 0], [w_clad, h_clad]]), np.array([nx, ny]), CellType.triangle
)
msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)

# Visualize Mesh
plotter = pyvista.Plotter(shape=(1, 4)) # create plotter
cells, cell_types, points = plot.vtk_mesh(msh)
grid = pyvista.UnstructuredGrid(cells, cell_types, points) # Create a Pyvista UnstructuredGrid
plotter.subplot(0,0)
plotter.add_mesh(grid, color="lightgray", opacity=0.3, show_edges=True)# Add the grid with visible mesh edges
plotter.view_xy()  # set the view to the xy-plane

eps_d = 3.48    # dielectric permittivity
eps_c = 1.44    # cladding permittivity
'''eps_p = 1 + 1j # complex permittivity in PML
eps_v = 1'''

# Center coordinate
ctr = w_clad / 2.0

def Omega_d(x):
    in_core = (
        (x[0] >= ctr - w_si/2) & (x[0] <= ctr + w_si/2) &
        (x[1] >= ctr - h_si/2) & (x[1] <= ctr + h_si/2)
    )
    return in_core

def Omega_c(x):
    in_cladding = (
        (x[0] >= ctr - w_clad/2) & (x[0] <= ctr + w_clad/2) &
        (x[1] >= ctr - h_clad/2) & (x[1] <= ctr + h_clad/2)
    )
    return in_cladding & (~Omega_d(x))

'''def Omega_p(x):
    in_pml = (
        (x[0] >= ctr - w_pml/2) & (x[0] <= ctr + w_pml/2) &
        (x[1] >= ctr - h_pml/2) & (x[1] <= ctr + h_pml/2)
    )
    # Vacuum region is the big square minus the dielectric
    return in_pml & (~Omega_d(x)) & (~Omega_c(x))

def Omega_v(x):
    return (~Omega_d(x)) & (~Omega_c(x)) & (~Omega_p(x))
'''

def eps_expr(x):
    # Define conditions
    cond_d = Omega_d(x)
    # cond_c = Omega_c(x)
    # cond_p = Omega_p(x)
    # cond_v = Omega_v(x)
    
    # Use np.where to assign values based on conditions
    # Priority: Omega_d > Omega_c > Omega_p > Omega_v
    # Ensure that regions do not overlap ambiguously
    # In this recursive method, the third value is supposed to be the value where the second argument is not. In trying to find this third value, np.where is called again, to return different values based on where you are in looping through the np.array
    return np.where(cond_d, eps_d, eps_c)

#D = fem.functionspace(msh, ("DQ", 0))
D = fem.functionspace(msh, ("CG", 1))
eps = fem.Function(D)
eps.interpolate(eps_expr)

# Visualise Eps
# Get the mesh data as before
cells, cell_types, points = plot.vtk_mesh(msh)
grid = pyvista.UnstructuredGrid(cells, cell_types, points)

# eps.x.array contains the values of eps at the dofs.
# (For a CG1 space, these are typically associated with mesh vertices.)
eps_values = eps.x.array  # if these are complex, you may want the real part

# Add eps as point data; if eps is complex, take the real part
grid.point_data["eps"] = np.real(eps_values)

# Now visualize the grid colored by eps
plotter.subplot(0,1)
plotter.add_mesh(grid, scalars="eps", cmap="viridis", show_edges=False)
plotter.view_xy()

"""cells_p = locate_entities(msh, msh.topology.dim, Omega_p)
cells_v = locate_entities(msh, msh.topology.dim, Omega_v)
cells_d = locate_entities(msh, msh.topology.dim, Omega_d)

eps.x.array[cells_p] = np.full_like(cells_p, eps_p, dtype=default_scalar_type)
eps.x.array[cells_v] = np.full_like(cells_v, eps_v, dtype=default_scalar_type)
eps.x.array[cells_d] = np.full_like(cells_d, eps_d, dtype=default_scalar_type)"""

# definition

degree = 1
NED = element("Nedelec 1st kind H(curl)", msh.basix_cell(), degree)
Q = element("Lagrange", msh.basix_cell(), degree)
V = fem.functionspace(msh, mixed_element([NED, Q]))

lmbd0 = l0 # If this value is too large, you might not have supported modes!
k0 = 2 * np.pi / lmbd0
et, ez = ufl.TrialFunctions(V)
vt, vz = ufl.TestFunctions(V)

a_tt = (ufl.inner(ufl.curl(et), ufl.curl(vt)) - (k0**2) * eps * ufl.inner(et, vt)) * ufl.dx
b_tt = ufl.inner(et, vt) * ufl.dx
b_tz = ufl.inner(et, ufl.grad(vz)) * ufl.dx
b_zt = ufl.inner(ufl.grad(ez), vt) * ufl.dx
b_zz = (ufl.inner(ufl.grad(ez), ufl.grad(vz)) - (k0**2) * eps * ufl.inner(ez, vz)) * ufl.dx

a = fem.form(a_tt)
b = fem.form(b_tt + b_tz + b_zt + b_zz)

'''bc_facets = exterior_facet_indices(msh.topology)
bc_dofs = fem.locate_dofs_topological(V, msh.topology.dim - 1, bc_facets)
u_bc = fem.Function(V)
with u_bc.x.petsc_vec.localForm() as loc:
    loc.set(0)
bc = fem.dirichletbc(u_bc, bc_dofs)

# solving with SLEPc

A = assemble_matrix(a, bcs=[bc])
A.assemble()
B = assemble_matrix(b, bcs=[bc])
B.assemble()'''

A = assemble_matrix(a)
A.assemble()
B = assemble_matrix(b)
B.assemble()

eps = SLEPc.EPS().create(msh.comm)

eps.setOperators(A, B)

eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)

tol = 1e-9
eps.setTolerances(tol=tol)

eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)

# Get ST context from eps
st = eps.getST()

# Set shift-and-invert transformation
st.setType(SLEPc.ST.Type.SINVERT)

eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)

eps.setTarget(-((0.5*k0) ** 2))

eps.setDimensions(nev=2)

eps.solve()
eps.view()
eps.errorView()

# Save the kz
vals = [(i, np.sqrt(-eps.getEigenvalue(i))) for i in range(eps.getConverged())]

# Sort kz by real part
vals.sort(key=lambda x: x[1].real)

eh = fem.Function(V)

kz_list = []

for i, kz in vals:
    # Save eigenvector in eh
    eps.getEigenpair(i, eh.x.petsc_vec)

    # Compute error for i-th eigenvalue
    error = eps.computeError(i, SLEPc.EPS.ErrorType.RELATIVE)

    # Verify, save and visualize solution
    if error < tol and np.isclose(kz.imag, 0, atol=tol): # This chooses the eigenvalues which are close to zero
        kz_list.append(kz)

        # I deleted the assert statement which checks against the analytical solutions.

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

        # Save solutions
        with io.VTXWriter(msh.comm, f"sols/Et_{i}.bp", Et_dg) as f:
            f.write(0.0)

        with io.VTXWriter(msh.comm, f"sols/Ez_{i}.bp", ezh) as f:
            f.write(0.0)

        # Visualize solutions with Pyvista
        if have_pyvista:
            V_cells, V_types, V_x = plot.vtk_mesh(V_dg)
            V_grid = pyvista.UnstructuredGrid(V_cells, V_types, V_x)
            Et_values = np.zeros((V_x.shape[0], 3), dtype=np.float64)
            Et_values[:, : msh.topology.dim] = Et_dg.x.array.reshape(
                V_x.shape[0], msh.topology.dim
            ).real

            V_grid.point_data["u"] = Et_values

            plotter.subplot(0,2+i)
            plotter.add_mesh(V_grid.copy(), show_edges=False)
            plotter.view_xy()
            plotter.link_views()
            '''if not pyvista.OFF_SCREEN:
                plotter.show()
            else:
                pyvista.start_xvfb()
                plotter.screenshot("Et.png", window_size=[400, 400])'''
plotter.show()

'''if have_pyvista:
    V_lagr, lagr_dofs = V.sub(1).collapse()
    V_cells, V_types, V_x = plot.vtk_mesh(V_lagr)
    V_grid = pyvista.UnstructuredGrid(V_cells, V_types, V_x)
    V_grid.point_data["u"] = ezh.x.array.real[lagr_dofs]
    plotter = pyvista.Plotter()
    plotter.add_mesh(V_grid.copy(), show_edges=False)
    plotter.view_xy()
    plotter.link_views()
    if not pyvista.OFF_SCREEN:
        plotter.show()
    else:
        pyvista.start_xvfb()
        plotter.screenshot("Ez.png", window_size=[400, 400])'''