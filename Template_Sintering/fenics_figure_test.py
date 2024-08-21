import numpy as np
import matplotlib.pyplot as plt
from fenics import *

# 创建一个简单的网格和函数空间
mesh = UnitSquareMesh(200, 150)  # 创建一个 200 x 150 的网格
V = FunctionSpace(mesh, 'P', 1)

# 定义边界条件
u_D = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_D, boundary)

# 定义变分问题
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(-6.0)
a = dot(grad(u), grad(v)) * dx
L = f * v * dx

# 求解变分问题
u_sol = Function(V)
solve(a == L, u_sol, bc)

# 提取位移场数据到 numpy 数组
displacement_field = u_sol.vector().get_local()

# 确定要 reshape 的形状
fig_element = (201, 151)  # 创建网格时会有更多的点数

try:
    # 尝试将一维数组 reshape 成二维数组
    displacement_field = displacement_field.reshape(fig_element)

    # 创建网格坐标 X, Y
    x = np.linspace(0, 1, fig_element[1])
    y = np.linspace(0, 1, fig_element[0])
    X, Y = np.meshgrid(x, y)

    # 绘制位移云图
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, displacement_field, cmap='coolwarm', levels=100)
    plt.colorbar(label='Displacement')
    plt.title('Displacement Map from FEniCS')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.grid(True)
    plt.show()

except ValueError as e:
    print(f"Error: {e}")
