
# ------------------- 1. Importing the Gmsh module -------------------

from mpi4py import MPI

from dolfinx.io import XDMFFile, gmshio

try:
    import gmsh  # type: ignore
except ImportError:
    print("This demo requires gmsh to be installed")
    exit(0)

## works fine

# ------------------- 2. Adding a mesh model -------------------

def gmsh_sphere(model: gmsh.model, name: str) -> gmsh.model:
    # Create the sphere model

    model.add(name)
    model.setCurrent(name)
    sphere = model.occ.addSphere(0,0,0,1,tag=1) #sphere preset: xyzr,identifier

    model.occ.synchronize() # necessary for setting multiple parameters before adding the changes

    model.add_physical_group(dim=3, tags=[sphere]) # specifies the models to be added in the group
    model.mesh.generate(dim=3)

    return model

# ------------------- 2. Creating a more complicated model -------------------
def gmsh_sphere_minus_box(model: gmsh.model, name: str) -> gmsh.model:
    
    model.add(name)
    model.setCurrent(name)

    sphere_dim_tags = model.occ.addSphere(0,0,0,1)
    box_dim_tags = model.occ.addBox(0,0,0,1,1,1)
    model_dim_tags = model.occ.cut([(3, sphere_dim_tags)], [(3, box_dim_tags)])
    
    model.occ.synchronize()

    # Adding a physical tag for exterior surfaces
    boundary = model.getBoundary(model_dim_tags[0], oriented=False)
    boundary_ids = [b[1] for b in boundary]
    model.addPhysicalGroup(2, boundary_ids, tag=1)
    model.setPhysicalName(3,2,"Sphere volume")

    model.mesh.generate(dim=3)
    return model