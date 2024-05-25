from dolfin import *
import numpy as np


# ---------制作网格以及定义有限元函数空间-----------
nx = 8
ny = 8
mesh = UnitSquareMesh(8, 8)
# ----------------定义函数空间-------------------
V = FunctionSpace(mesh, "P", 1)
# ----------------定义边界条件-------------------
u_D = Expression('1 + x[0]* x[0] + 2*x[1]* x [1]', degree=2)


def boundary(x, on_boundary):
    """应用哪些点到边界上
    """
    return on_boundary


bc = DirichletBC(V, u_D, boundary)

u = TrialFunction(V)  # 定义实验空间
v = TestFunction(V)  # 定义测试空间
f = Constant(-6.0)  # 定义边界条件
a = dot(grad(u), grad(v)) * dx  # 定义变分问题
L = f * v * dx

# 计算解决方案
u = Function(V)
solve(a == L, u, bc)

# Plot solution and mesh
plot(u, title="Finite element solution")
plot(mesh, title="Finite element mesh")
# 将文件保存到VTK格式
vtkfile = File('poisson/solution.pvd')
vtkfile << u

# # 计算误差 in L2 norm
error_L2 = errornorm(u_D, u, 'L2')

# compute maximum error at vertices
vertex_values_u_D = u_D.compute_vertex_values(mesh)
vertex_values_u = u.compute_vertex_values(mesh)

error_max = np.max(np.abs(vertex_values_u_D - vertex_values_u))

# Print errors
print('error_L2 =', error_L2)
print('error_max =', error_max)
