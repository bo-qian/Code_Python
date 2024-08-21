# ---------------------------------------------------------------------------------------------------------
# %%
from mpi4py import MPI
from dolfinx import mesh
from dolfinx.fem import functionspace
from dolfinx import fem
import numpy
import ufl
from dolfinx import default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import pyvista
from dolfinx import plot
from dolfinx import io
from pathlib import Path

# ---------------------------------------------------------------------------------------------------------
domain = mesh.create_unit_square(MPI.COMM_WORLD, 8, 8, mesh.CellType.quadrilateral)
# 可以终端运用命令 mpirun -n 2 python3 FeniCSx Practice.ipynb运行，其中-n 2表示分配两个核心运行

# ---------------------------------------------------------------------------------------------------------
# 定义函数空间
V = functionspace(domain, ("Lagrange", 1))

uD = fem.Function(V)
uD.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)
# 其中lambda函数是一个匿名函数，x是一个变量，1 + x[0] + x[1]是函数表达式。接受输入的x，返回1 + x[0] + x[1]的值。

# ---------------------------------------------------------------------------------------------------------
# Creat facet to cell connectivity required to determine boundary facets
# 创建确定边界切面所需的切面到单元的连通性
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)

boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = fem.dirichletbc(uD, boundary_dofs)

# ---------------------------------------------------------------------------------------------------------
# 定义Trail and Test函数
# 在数学中，我们区分试验空间和测试空间以及 .目前问题的唯一区别是边界条件。 在 FEniCSx 中，我们没有将边界条件指定为函数空间的一部分，
# 因此为试验和测试函数使用公共空间就足够了
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# ---------------------------------------------------------------------------------------------------------
# 定义源项
# 源项在域上是恒定的，因此我们使用一个常数表达式。`dolfinx.constant`
f = fem.Constant(domain, default_scalar_type(-6))

# ---------------------------------------------------------------------------------------------------------
# 定义变分问题
a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx

# ---------------------------------------------------------------------------------------------------------
# 线性系统求解
problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

# ---------------------------------------------------------------------------------------------------------
# 计算误差
# 最后，我们要计算误差以检查解决方案的准确性。我们通过将有限元解与精确解进行比较来做到这一点。我们通过将精确解插值到 -function 空间中来做到这一点
V2 = fem.functionspace(domain, ("Lagrange", 2))
uex = fem.Function(V2)
uex.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)

L2_error = fem.form(ufl.inner(uh - uex, uh - uex) * ufl.dx)
error_local = fem.assemble_scalar(L2_error)
error_l2 = numpy.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))

# 其次计算任意自由度下的最大误差
error_max = numpy.max(numpy.abs(uD.x.array - uh.x.array))
# Only print error on rank 0
if domain.comm.rank == 0:
    print(f"L2 error: {error_l2}")
    print(f"Max error: {error_max}")

# ---------------------------------------------------------------------------------------------------------
# 使用pyvista绘制网格
# print(pyvista.global_theme.jupyter_backend)
#
# pyvista.start_xvfb()
# domain.topology.create_connectivity(tdim, fdim)
# topology, cell_types, geometry = plot.vtk_mesh(domain, tdim)
# grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
#
# plotter = pyvista.Plotter()
# plotter.add_mesh(grid, show_edges=True)
# plotter.view_xy()
# if not pyvista.OFF_SCREEN:
#     plotter.show()
# else:
#     figure = plotter.screenshot("fundamentals_mesh.png")

# ---------------------------------------------------------------------------------------------------------
# 使用pyvista绘制解
# u_topology, u_cell_types, u_geometry = plot.vtk_mesh(V)
#
# u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
# u_grid.point_data["u"] = uh.x.array.real
# u_grid.set_active_scalars("u")
# u_plotter = pyvista.Plotter()
# u_plotter.add_mesh(u_grid, show_edges=True)
# u_plotter.view_xy()
# if not pyvista.OFF_SCREEN:
#     u_plotter.show()
#
# warped = u_grid.warp_by_scalar()
# plotter2 = pyvista.Plotter()
# plotter2.add_mesh(warped, show_edges=True, show_scalar_bar=True)
# if not pyvista.OFF_SCREEN:
#     plotter2.show()

# ---------------------------------------------------------------------------------------------------------
# 外部后处理
results_folder = Path("results")
results_folder.mkdir(exist_ok=True, parents=True)
filename = results_folder / "fundamentals"
# with io.VTXWriter(domain.comm, filename.with_suffix(".bp"), [uh]) as vtx:
#     vtx.write(0.0)
with io.XDMFFile(domain.comm, filename.with_suffix(".xdmf"), "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(uh)

# =========================================================================================================
# %%
from mpi4py import MPI
from petsc4py.PETSc import ScalarType  # type: ignore

import numpy as np
import ufl
from dolfinx import fem, io, mesh, plot, cpp
from dolfinx.fem.petsc import LinearProblem
from ufl import ds, dx, grad, inner

# Create mesh and function space
msh = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0), (2.0, 1.0)),
    n=(32, 16),
    cell_type=mesh.CellType.triangle,
    diagonal=cpp.mesh.DiagonalType.crossed
)
V = fem.functionspace(msh, ("Lagrange", 1))

# 在边界上找到网格平面
facets = mesh.locate_entities_boundary(
    msh,
    dim=(msh.topology.dim - 1), # 边界的维度，例如，对于三维网格，边界实体是二维面。
    marker=lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 2.0), # 边界的标记函数，如果实体满足条件（即上述检查返回 True）
)

# 在有限元空间 V 中定位与这些边界实体相关的自由度（dofs）
dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=facets)

# 定义边界条件
bc = fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=V)

# 定义变分问题
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(msh) #创建一个与网格 msh 关联的空间坐标对象。该对象用于在网格的几何定义中表达点的坐标。它可以用于定义各种有限元公式，例如积分、边界条件和载荷等。
f = 10 * ufl.exp(-((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) / 0.02)
g = ufl.sin(5 * x[0])
a = inner(grad(u), grad(v)) * dx
L = inner(f, v) * dx + inner(g, v) * ds

# 解决变分问题
problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()
with io.XDMFFile(msh.comm, "out_poisson/poisson.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_function(uh)

