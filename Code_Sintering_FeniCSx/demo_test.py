from petsc4py import PETSc
from mpi4py import MPI
import numpy as np
import ufl
import dolfinx
from petsc4py.PETSc import ScalarType
from petsc4py import PETSc
from mpi4py import MPI
import ufl
import numpy as np

# Define the mesh
mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 10, 10)

# Define function spaces
V1 = dolfinx.fem.FunctionSpace(mesh, ("CG", 1))  # Scalar function space
V2 = dolfinx.fem.FunctionSpace(mesh, ("CG", 1))  # Another scalar function space
V = dolfinx.fem.MixedFunctionSpace([V1, V2])

# Define initial values as expressions
u_expr = ufl.sin(ufl.SpatialCoordinate(mesh)[0])  # Example expression for u
v_expr = ufl.cos(ufl.SpatialCoordinate(mesh)[1])  # Example expression for v

# Interpolate initial values
u_init = dolfinx.fem.Function(V.sub(0))
v_init = dolfinx.fem.Function(V.sub(1))
u_init.interpolate(u_expr)
v_init.interpolate(v_expr)

# Combine initial functions into a mixed function
u_init_combined = dolfinx.fem.Function(V)
u_init_combined.interpolate(lambda x: np.vstack([u_init.eval(x), v_init.eval(x)]))

# Output for verification
print("Initial values interpolated.")
