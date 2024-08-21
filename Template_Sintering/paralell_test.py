from fenics import *
from mpi4py import MPI
import numpy as np
import logging

# 设置日志记录
logging.basicConfig(filename='fenics_solver.log', level=logging.INFO)

# 创建网格和函数空间
mesh = UnitSquareMesh(MPI.COMM_WORLD, 32, 32)


# Allen-Cahn方程
def solve_allen_cahn():
    V = FunctionSpace(mesh, 'P', 1)
    u = TrialFunction(V)
    v = TestFunction(V)

    # 初始条件
    u_n = Function(V)
    u_n.interpolate(Expression('sin(pi*x[0])*sin(pi*x[1])', degree=2, domain=mesh))

    # 时间步长
    dt = 0.01

    # 定义Allen-Cahn方程
    F = u * v * dx + dt * dot(grad(u), grad(v)) * dx - (u_n + dt * u * (1 - u * u)) * v * dx
    a, L = lhs(F), rhs(F)

    # 创建边界条件
    bc = DirichletBC(V, Constant(0.0), 'on_boundary')

    # 创建解函数
    u = Function(V)

    # 时间步进
    t = 0.0
    T = 2.0
    while t < T:
        t += dt
        solve(a == L, u, bc)
        u_n.assign(u)
    logging.info('Allen-Cahn equation solved.')


# Cahn-Hilliard方程
def solve_cahn_hilliard():
    V = FunctionSpace(mesh, 'P', 1)
    W = FunctionSpace(mesh, MixedElement([FiniteElement('P', mesh.ufl_cell(), 1),
                                          FiniteElement('P', mesh.ufl_cell(), 1)]))

    u = Function(W)
    u_n = Function(W)
    phi, mu = split(u)
    phi_n, mu_n = split(u_n)
    v_phi, v_mu = TestFunctions(W)

    # 时间步长
    dt = 0.01
    epsilon = 1e-2

    # 定义Cahn-Hilliard方程
    F1 = (phi - phi_n) / dt * v_phi * dx + dot(grad(mu), grad(v_phi)) * dx
    F2 = mu * v_mu * dx - (phi ** 3 - phi - epsilon ** 2 * div(grad(phi))) * v_mu * dx
    F = F1 + F2
    a, L = lhs(F), rhs(F)

    # 创建边界条件
    bc = [DirichletBC(W.sub(0), Constant(0.0), 'on_boundary'),
          DirichletBC(W.sub(1), Constant(0.0), 'on_boundary')]

    # 时间步进
    t = 0.0
    T = 2.0
    while t < T:
        t += dt
        solve(a == L, u, bc)
        u_n.assign(u)
    logging.info('Cahn-Hilliard equation solved.')


# Stokes方程
def solve_stokes():
    V = VectorFunctionSpace(mesh, 'P', 2)
    Q = FunctionSpace(mesh, 'P', 1)
    W = MixedFunctionSpace([V, Q])

    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)

    # 定义Stokes方程
    f = Constant((0, 0))
    a = inner(grad(u), grad(v)) * dx - div(v) * p * dx - q * div(u) * dx
    L = dot(f, v) * dx

    # 创建边界条件
    bc = [DirichletBC(W.sub(0), Constant((0, 0)), 'on_boundary')]

    # 创建解函数
    w = Function(W)

    # 求解Stokes方程
    solve(a == L, w, bc)
    logging.info('Stokes equation solved.')


# 主函数
if __name__ == "__main__":
    solve_allen_cahn()
    solve_cahn_hilliard()
    solve_stokes()