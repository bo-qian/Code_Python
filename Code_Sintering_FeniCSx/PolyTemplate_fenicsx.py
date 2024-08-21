# ************************************************************************************************************
# Description: This is a phase-field simulation for template grain growth.
# Wrote by Bo Qian at Shanghai University
# Date: 2024-06-16
# Use the new fenicsx package
# ************************************************************************************************************
import logging
import sys
import matplotlib.pyplot as plt
import os
import datetime
import time
import shutil
# from fenics import *
import logging as filehandle


import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import ufl
from basix.ufl import element, mixed_element
from dolfinx import default_real_type, log, plot, mesh, cpp
from dolfinx.fem import Function, functionspace
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType, create_rectangle, locate_entities_boundary
from dolfinx.nls.petsc import NewtonSolver
from ufl import dx, grad, inner

import ffcx

# 定义输出 **************************************************************************************
# 创建输出文件夹
start_running_time = datetime.datetime.now()
primary_directory = os.getcwd() + "/TGG_Simulation_Parallel"
if not os.path.exists(primary_directory):
    os.makedirs(primary_directory)
# 获取当前文件夹内的所有内容
all_contents = os.listdir(primary_directory)
# 过滤出所有的子文件夹
subdirectories = [content for content in all_contents if os.path.isdir(os.path.join(primary_directory, content))]
# 计算子文件夹的数量
num_subdirectories = len(subdirectories)
time_stamp = start_running_time.strftime("%B %d, %Y, %H") + ":" + datetime.datetime.now().strftime(
    "%M") + ":" + datetime.datetime.now().strftime("%S")
filename = 'Test-' + str(num_subdirectories + 1) + ' (' + datetime.datetime.now().strftime("%B %d, %Y, %H-%M-%S") + ")"
output_document_type = ["Log", "Error", "Codes", "Input", "Output", "Figures", "Data"]
for i in output_document_type:
    globals()[i + '_directory'] = primary_directory + '/' + filename + '/' + i
    if not os.path.exists(globals()[i + '_directory']):
        os.makedirs(globals()[i + '_directory'], exist_ok=True)

# 创建Readme文件
readme_file_path = primary_directory + '/' + filename + '/Readme.txt'
with open(readme_file_path, 'w', encoding='utf-8') as file:
    file.write(f"####################################################################\n"
               f"This is the computing log of phase-field simulation for template grain growth.\n"
               f"Operated by Bo Qian at Shanghai University\n"
               f"Date: {time_stamp}\n"
               f"####################################################################\n\n")

# 创建日志文件
log_file_path = os.path.join(Log_directory + '/log.log')
error_file_path = os.path.join(Error_directory + '/error.log')
# 创建日志记录器
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()
log_file_handler = logging.FileHandler(str(log_file_path))
log_file_handler.setLevel(logging.INFO)
logger.addHandler(log_file_handler)
# 创建错误记录器
error_file_handler = logging.FileHandler(str(error_file_path))
error_file_handler.setLevel(logging.ERROR)
error_formatter = logging.Formatter('%(asctime)s - %(levelname)% - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
error_file_handler.setFormatter(error_formatter)
logger.addHandler(error_file_handler)

# 定义分母正则化
denim_reg = lambda x: x if abs(x) > 1e-12 else 1e-12


# 读取参数 ****************************************************************************************************
def read_input_file(file_name):
    parameters_input = {}
    with open(file_name, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            var_name, var_value = line.split('=')
            parameters_input[var_name.strip()] = var_value.strip()
    return parameters_input


# 多晶函数 ****************************************************************************************************
class VoronoiParticle:
    # 定义当前晶粒的默认中心坐标
    center = [0, 0]
    # 定义其他晶粒的默认中心坐标
    other = [[1, 1]]
    # 定义双曲正切函数中的参数
    epsilon = 1
    # 定义一个函数接受一个参数x， 如果没有传入参数，则默认返回0
    obstacle = lambda x: 0

    # 定义返回值的形状
    def value_shape(self):
        return ()

    def eval(self, value, x):
        ufl.dx = x[0] - self.center[0]
        dy = x[1] - self.center[1]
        # 计算当前晶粒的距离
        closest_ind = np.argmin(
            [np.sqrt(pow((x[0] - self.others[k][0]), 2) + pow((x[1] - self.others[k][1]), 2)) for k in
             range(len(self.others))])
        # 获取最近的其他晶粒的中心坐标
        closest = self.others[closest_ind]
        # 最近的其他晶粒与当前晶粒之间的矢量差
        dpx = closest[0] - self.center[0]  # closest[0] - self.center[0]
        dpy = closest[1] - self.center[1]  # closest[1] - self.center[1]
        # 这个公式是计算两个晶粒之间的距离
        dp = np.sqrt(pow(dpx, 2) + pow(dpy, 2))
        value[0] = (1 - self.obstacle(x)) * (
                1 + np.tanh((0.5 * pow(dp, 2) - ufl.dx * dpx - dy * dpy) / (np.sqrt(2) * self.epsilon * dp))) / 2


# 定义一个函数，从0到1平滑的矩形过渡*******************************************************************************
def smooth_rectangle(x, y, x_center, y_center, width, height, epsilon):
    # 在x方向上进行双曲正切函数操作
    sigmoid_x = 0.5 * (1 - np.tanh((x - x_center - width / 2) / (np.sqrt(2) * epsilon))) - 0.5 * (
            1 - np.tanh((x - x_center + width / 2) / (np.sqrt(2) * epsilon)))
    # 在y方向上进行双曲正切函数操作
    sigmoid_y = 0.5 * (1 - np.tanh((y - y_center - height / 2) / (np.sqrt(2) * epsilon))) - 0.5 * (
            1 - np.tanh((y - y_center + height / 2) / (np.sqrt(2) * epsilon)))
    # 返回x和y方向上的双曲正切函数的乘积
    return sigmoid_x * sigmoid_y


# 生成模板晶粒 ************************************************************************************************
class Template:
    # 生成模板晶粒的中心坐标
    center = [0.5, 0.5]
    # 生成模板晶粒的尺寸
    dimensions = [0.1, 0.1]
    # 设置平滑参数
    epsilon = 1 / 20

    def value_shape(self):
        return ()

    def eval(self, value, x):
        value[0] = smooth_rectangle(x[0], x[1], self.center[0], self.center[1], self.dimensions[0], self.dimensions[1],
                                    self.epsilon)

    def __call__(self, x):
        return smooth_rectangle(x[0], x[1], self.center[0], self.center[1], self.dimensions[0], self.dimensions[1],
                                self.epsilon)

    def set_center(self, center):
        self.center = center

    def set_dimensions(self, dimensions):
        self.dimensions = dimensions

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    # 定义一个函数，将模板晶粒转换为字符串表达式
    def to_string_expression(self):
        return '*'.join([
            f'((1 - tanh((x[0] - {self.center[0]} - {self.dimensions[0]}/2)/(sqrt(2)*epsilon)))/2'
            f'-(1 - tanh((x[0] - {self.center[0]} + {self.dimensions[0]}/2)/(sqrt(2)*epsilon)))/2)',
            f'((1 - tanh((x[1] - {self.center[1]} - {self.dimensions[1]}/2)/(sqrt(2)*epsilon)))/2'
            f'-(1 - tanh((x[1] - {self.center[1]} + {self.dimensions[1]}/2)/(sqrt(2)*epsilon)))/2)'
        ])


def circle_string_expression(radius, center):
    return (f"(1 - tanh((sqrt(pow((x[0]-{center[0]}), 2) + pow((x[1]-{center[1]}), 2)) - {radius}) / (sqrt("
            f"2)*epsilon)))/2")


# 定义固态烧结参数 *********************************************************************************************
class SolidStateSintering:
    def __init__(self, a, b, kc, keta, Dsf, Dgb, Dvol, L, zeta, Np, Neps, muSurf, viscosity_ratio):
        self.a = a
        self.b = b
        self.kc = kc
        self.keta = keta
        self.Dsf = Dsf
        self.Dgb = Dgb
        self.Dvol = Dvol
        self.L = L
        self.Neps = Neps
        self.muSurf = muSurf
        self.zeta = zeta - (2 / 3) * muSurf
        self.Np = Np
        self.viscosity_ratio = viscosity_ratio
        self.c_stabilizer = 2 * a
        self.eta_stabilizer = 12 * b * L

    def N(self, C):
        return (C ** 2) * (1 + 2 * (1 - C) + self.Neps * (1 - C) ** 2)

    def dN(self, C):
        return 2 * C * (1 - C) * (3 + self.Neps * (1 - 2 * C))

    def f(self, C):
        return self.a * (C ** 2) * ((1 - C) ** 2)

    def df(self, C):
        return 2 * self.a * C * (1 - C) * (1 - 2 * C)

    def S(self, eta):
        return self.b * (ufl.Constant(1)
                         - 4 * sum([eta[i] ** 3 for i in range(self.Np)])
                         + 3 * sum([eta[i] ** 2 for i in range(self.Np)]) ** 2)

    def dS(self, eta, i):
        return 12 * self.b * (-eta[i] ** 2 + eta[i] * sum([eta[j] ** 2 for j in range(self.Np)]))

    def grad_eta(self, eta):
        return 0.5 * self.keta * sum([ufl.dot(ufl.grad(eta[i]), ufl.grad(eta[i])) for i in range(self.Np)])

    def M(self, C, eta):
        return pow(2 * self.a, -1) * (self.Dvol * self.N(C)
                                      + self.Dsf * (C ** 2) * ((1 - C) ** 2)
                                      + 2 * self.Dgb * sum(
                    [eta[i] ** 2 * (sum([eta[j] ** 2 for j in range(self.Np)]) - eta[i] ** 2) for i in range(self.Np)]))

    def fmu(self, C, eta):  # 其实就是表示的f(c,\eta_i)
        return self.df(C) + self.dN(C) * (self.S(eta) + self.grad_eta(eta))

    def sigma(self, u, C):  # 计算和返回应力张量
        # for this version it is important to keep viscosity_ratio < 1
        # the defined sigma here does not include contribution from Lagrangian multiplier
        return (self.viscosity_ratio + (1 - self.viscosity_ratio) * self.N(C)) \
            * self.muSurf * (ufl.grad(u) + ufl.grad(u).T)

    # 计算和返回界面应力
    def interface_stress(self, C, eta):
        return self.kc * ufl.outer(ufl.grad(C), ufl.grad(C)) + self.keta * self.N(C) * sum(
            [ufl.outer(ufl.grad(eta[i]), ufl.grad(eta[i])) for i in range(self.Np)])


def particle_centers_coord(template_particle_center, template_particle_dimensions, radius_particle, particle_layer,
                           aspect_ratio_template, particle_number_total):
    particle_radius = [radius_particle] * particle_number_total
    particle_centers_radius = []
    layer = []
    particle_sort_number = 0
    for k in range(particle_layer):
        if layer_numbers == 2:
            layer.append([aspect_ratio_template + k, 3, k * 4])
        else:
            layer.append([aspect_ratio_template + k * 3, 3])
        for m in range(layer[k][0]):
            x_coordinate = float(template_particle_center[0] + (m - (layer[k][0] - 1) / 2) * particle_size)
            y_coordinate = float(
                template_particle_center[1] + 0.5 * template_particle_dimensions[1] + (1 + k * np.sqrt(3)) *
                particle_radius[particle_sort_number + m])
            particle_centers_radius.append([x_coordinate, y_coordinate])
            y_coordinate = float(
                template_particle_center[1] - 0.5 * template_particle_dimensions[1] - (1 + k * np.sqrt(3)) *
                particle_radius[particle_sort_number + layer[k][0] + m])
            particle_centers_radius.append([x_coordinate, y_coordinate])
        for n in range(layer[k][1]):
            y_coordinate = float(template_particle_center[1] + (n - (layer[k][1] - 1) / 2) * particle_size)
            x_coordinate = float(
                template_particle_center[0] + 0.5 * template_particle.dimensions[0] + (2 * k + 1) * particle_radius[
                    particle_sort_number + layer[k][0] + layer[k][1] + n])
            particle_centers_radius.append([x_coordinate, y_coordinate])
            x_coordinate = float(
                template_particle_center[0] - 0.5 * template_particle.dimensions[0] - (2 * k + 1) * particle_radius[
                    particle_sort_number + 2 * layer[k][0] + layer[k][1] + n])
            particle_centers_radius.append([x_coordinate, y_coordinate])
        for CG in range(layer[k][2]):
            x_coordinate = float(particle_centers_radius[0] + (k + 0.5) * radius_particle * 2)
            y_coordinate = float(
                template_particle_center[1] + 0.5 * template_particle_dimensions[1] + (
                        1 + k * np.sqrt(3)) * radius_particle)
            particle_centers_radius.append([x_coordinate, y_coordinate])
            y_coordinate = float(
                template_particle_center[1] - 0.5 * template_particle_dimensions[1] - (
                        1 + k * np.sqrt(3)) * radius_particle)
            particle_centers_radius.append([x_coordinate, y_coordinate])
            x_coordinate = float(particle_centers_radius[0] + (k + 0.5) * radius_particle * 2)

            particle_sort_number += 2 * (layer[k][0] + layer[k][1])
    return particle_centers_radius, particle_radius


def particle_centers_coordinate(layer_number, aspect_ratio_template, template_particle, radius_particle,
                                particle_number_total):
    particle_radius = [radius_particle] * particle_number_total
    particle_centers_radius = []
    for i in range(layer_number):
        for j in range(aspect_ratio_template + layer_number + 1 + i):
            x_coordinate = float(
                template_particle.center[0] + (
                        j - (aspect_ratio_template + layer_number + i) / 2) * radius_particle * 2)
            y_coordinate = float(template_particle.center[1] + 0.5 * template_particle.dimensions[1] + (
                    1 + (layer_number - 1 - i) * np.sqrt(3)) * radius_particle)
            particle_centers_radius.append([x_coordinate, y_coordinate])
            y_coordinate = float(template_particle.center[1] - 0.5 * template_particle.dimensions[1] - (
                    1 + (layer_number - 1 - i) * np.sqrt(3)) * radius_particle)
            particle_centers_radius.append([x_coordinate, y_coordinate])
        y_coordinate = float(template_particle.center[1])
        x_coordinate = float(template_particle.center[0] - 0.5 * template_particle.dimensions[0] - (
                1 + (layer_number - 1 - i) * 2) * radius_particle)
        particle_centers_radius.append([x_coordinate, y_coordinate])
        x_coordinate = float(template_particle.center[0] + 0.5 * template_particle.dimensions[0] + (
                1 + (layer_number - 1 - i) * 2) * radius_particle)
        particle_centers_radius.append([x_coordinate, y_coordinate])
    return particle_centers_radius, particle_radius


# 绘图函数 ***************************************************************************************************
def plot_function(serial_number, variant, file_directory, time_current):
    if not os.path.exists(file_directory):
        os.makedirs(file_directory)
    plt.figure(serial_number)
    plot(variant, title=f"{variant} at time {time_current}s", cmap='coolwarm')
    plt.savefig(file_directory + f'/{variant}_{time_current:f}s.png', dpi=600)
    plt.close()


# 主程序 *****************************************************************************************************
# 复制此正在运行的代码
print("the code being executed:", sys.argv)
python_script_name = os.path.basename(sys.argv[0])
shutil.copy(python_script_name, os.path.join(Codes_directory, python_script_name))
# 读取输入文件
parameters_input = read_input_file("../Template_Sintering/input_template.txt")
shutil.copy("../Template_Sintering/input_template.txt", os.path.join(Input_directory, "input_template_fenicsx.txt"))

# 从输入文件中读取参数*******************************************************************************************
# 相场参数读取
a = float(parameters_input.get("Alpha"))
b = float(parameters_input.get("Beta"))
kc = float(parameters_input.get("KappaC"))
keta = float(parameters_input.get("KappaEta"))
Dsf = float(parameters_input.get("Surface Diffusion"))
Dgb = float(parameters_input.get("GB Diffusion"))
Dvol = float(parameters_input.get("Volume Diffusion"))
L = float(parameters_input.get("Mobility L"))
Neps = float(parameters_input.get("Epsilon"))
zeta = float(parameters_input.get("Zeta"))

# 力学参数
muSurf = float(parameters_input.get("Surface viscosity"))
viscosity_ratio = float(parameters_input.get("Viscosity Ratio"))

# 模型参数
radius_particle = float(parameters_input.get("Particle Radius"))
# Np = int(parameters_input.get("Particle Number"))
timesCharacteristic = float(parameters_input.get("Characteristic Time"))
# domain_size = [float(parameters_input.get("Dimension X")), float(parameters_input.get("Dimension Y"))]
ratio_mesh = float(parameters_input.get("MeshRatio"))
# ElementNumber = [int(parameters_input.get('NumberOfElement X')), int(parameters_input.get('NumberOfElement Y'))]
aspect_ratio = int(parameters_input.get("AspectRatioOfTemplate"))
layer_numbers = int(parameters_input.get("NumberOfLayer"))
dt = float(parameters_input.get("TimeStep"))
timeInitial = float(parameters_input.get("InitialTime"))
NumberOfTimeStep = int(parameters_input.get('NumberOfTimeStep'))
counterStepInitial = int(parameters_input.get('InitialStepCounter'))
frequencyOutput = int(parameters_input.get('FrequencyOutPut'))

# 读取input文件中的备注
remark = str(parameters_input.get("Remark"))

# 计算晶粒总数
particle_number_total = 2 * layer_numbers
for i in range(layer_numbers):
    particle_number_total += 2 * (aspect_ratio + layer_numbers * 2 - i)
Np = int(particle_number_total + 1)

theta = float(parameters_input.get("theta"))
pde = SolidStateSintering(a, b, kc, keta, Dsf, Dgb, Dvol, L, zeta, Np, Neps, muSurf, viscosity_ratio)

# 几何模型建立 *************************************************************************************************
particle_size = radius_particle * 2
domain_size = [float((aspect_ratio + layer_numbers * 2 + 2) * particle_size),
               float((1 + layer_numbers * 2 + 1.5) * particle_size)]
ElementNumber = [int(domain_size[0] * ratio_mesh), int(domain_size[1] * ratio_mesh)]

# 创建模板晶粒
template_particle = Template()
template_particle.set_center([int(domain_size[0] / 2), int(domain_size[1] / 2)])
template_particle.set_dimensions([aspect_ratio * particle_size, particle_size])
# 柔化晶界和表面处的固气相变
template_particle.set_epsilon(1)
# 计算模板的面积
area_template = template_particle.dimensions[0] * template_particle.dimensions[1]

particle_centers, particle_radii = particle_centers_coordinate(layer_numbers, aspect_ratio, template_particle,
                                                               radius_particle, particle_number_total)

# 计算体积分数
volume_fraction_template = aspect_ratio / (aspect_ratio + particle_number_total * np.pi)

# 输出前置参数到Readme文件中 *************************************************************************************
with open(readme_file_path, 'a', encoding='utf-8') as file:
    file.write(
        f"\n####################################################################\n"
        f"Starting time : {start_running_time.strftime('%B %d, %Y, %H:%M:%S')}"
        f"                                                                          Test-{num_subdirectories + 1}\n"
        f"**************************************************************************************************\n"
        f"Remark : {remark}\n"
        f"Time step : {dt}s\n"
        f"Number of time steps : {NumberOfTimeStep}\n"
        f"Number of mesh element : {ElementNumber[0]}*{ElementNumber[1]}={ElementNumber[0] * ElementNumber[1]}\n"
        f"Output frequency : once every {frequencyOutput} time steps\n"
        f"The aspect ratio of template : {aspect_ratio}\n"
        f"The number of particle : {pde.Np}\n"
        f"The layer number : {layer_numbers}\n"
        f"The volume fraction of template : {volume_fraction_template * 100:.2f}%\n"
    )

# 计算过程 ****************************************************************************************************
# 开启Fenics表达式优化
form_compiler_parameters = {"cpp_optimize": True, "optimize": True}

# 创建网格
mesh = create_rectangle(MPI.COMM_WORLD, [np.array([0, 0]), np.array([domain_size[0], domain_size[1]])],
                        [ElementNumber[0], ElementNumber[1]], CellType.quadrilateral)
# mesh = RectangleMesh(Point(0, 0), Point(domain_size[0], domain_size[1]), ElementNumber[0], ElementNumber[1])

# 定义函数空间
P1 = ufl.FiniteElement('CG', triangle, 1)
C_Space = FunctionSpace(mesh, MixedElement([P1, P1]))
Eta_Space = FunctionSpace(mesh, P1)

PV = VectorElement('CG', triangle, 2)
VS = FunctionSpace(mesh, MixedElement([PV, P1]))

PS = TensorElement('CG', triangle, 1)
WS = FunctionSpace(mesh, PS)

# 定义测试函数和试探函数
c_trial, mu_trial = ufl.TrialFunctions(C_Space)
c_test, mu_test = ufl.TestFunctions(C_Space)
eta_trial = ufl.TrialFunction(Eta_Space)
eta_test = ufl.TestFunction(Eta_Space)

# c_trial, mu_trial = TrialFunctions(C_Space)
# c_test, mu_test = TestFunctions(C_Space)
# eta_trial = TrialFunction(Eta_Space)
# eta_test = TestFunction(Eta_Space)

# 定义变量
u_new = Function(C_Space)
eta_new = [Function(Eta_Space) for i in range(pde.Np)]
u_prev = Function(C_Space)
eta_prev = [Function(Eta_Space) for i in range(pde.Np)]

# u_new = Function(C_Space)
# eta_new = [Function(Eta_Space) for i in range(pde.Np)]
# u_prev = Function(C_Space)
# eta_prev = [Function(Eta_Space) for i in range(pde.Np)]

# 分割混合函数
c_new, mu_new = u_new.split()
c_prev, mu_prev = u_prev.split()

# c_new, mu_new = split(u_new)
# c_prev, mu_prev = split(u_prev)

# 分割测试函数
v_trial, p_trial = ufl.TrialFunctions(VS)
v_test, p_test = ufl.TestFunctions(VS)
v_combined = Function(VS)
v, CG = v_combined.split()


# v_trial, p_trial = TrialFunctions(VS)
# v_test, p_test = TestFunctions(VS)
# v_combined = Function(VS)
# v, CG = split(v_combined)


# 定义边界条件 ************************************************************************************************
def boundary(x):
    return np.isclose(x[0], 0.0) or np.isclose(x[0], domain_size[0]) or np.isclose(x[1], 0.0) or np.isclose(x[1],
                                                                                                            domain_size[
                                                                                                                1])


# def boundary(x, on_boundary):
#     return on_boundary

bc = DirichletBC(VS.sub(0), ufl.Constant((0, 0)), boundary)

# bc = DirichletBC(VS.sub(0), Constant((0, 0)), boundary)

# 定义eta初始值 ***********************************************************************************************
eta_initial = [VoronoiParticle(degree=1) for k in range(pde.Np - 1)] + [template_particle]
for k in range(pde.Np - 1):
    # 设置eta初始值
    eta_initial[k].others = particle_centers.copy()
    # 设置eta初始值的中心坐标
    eta_initial[k].center = eta_initial[k].others.pop(k)
    # 设置eta初始值的epsilon
    eta_initial[k].epsilon = 1
    # 设置eta初始值的限制条件
    eta_initial[k].obstacle = template_particle

# 定义c的初始值 ***********************************************************************************************
# 生成一个列表，包含所有的圆形表达式
c_initial = Expression(
    '+'.join([circle_string_expression(particle_radii[k], particle_centers[k]) for k in range(pde.Np - 1)]
             + [template_particle.to_string_expression()]),
    degree=1, epsilon=template_particle.epsilon)

# 定义mu的初始值 **********************************************************************************************
mu_initial = ufl.Constant(0)

c0 = interpolate(c_initial, C_Space.sub(0).collapse())
mu0 = interpolate(mu_initial, C_Space.sub(1).collapse())
eta0 = [interpolate(eta_initial[k], Eta_Space) for k in range(pde.Np)]
print(eta0)

c_mid = (1.0 - theta) * c_prev + theta * c_trial

# 定义弱形式和变分问题 *****************************************************************************************
# 定义CH方程中表达式c的弱形式
c_term1 = c_trial * c_test * ufl.dx
c_term2 = - c_prev * c_test * ufl.dx
c_term3 = dt * ufl.dot(pde.M(c_prev, eta_prev) * ufl.grad(mu_trial), ufl.grad(c_test)) * ufl.dx
c_term4 = - dt * ufl.dot(v * c_trial, ufl.grad(c_test)) * ufl.dx
c_WeakForm = c_term1 + c_term2 + c_term3 + c_term4

# 定义CH方程中表达式mu的弱形式
mu_term1 = mu_trial * mu_test * ufl.dx
mu_term2 = - (pde.kc * ufl.dot(ufl.grad(c_mid), ufl.grad(mu_test)) * ufl.dx)
mu_term3 = - (pde.fmu(c_prev, eta_prev) + pde.c_stabilizer * (c_trial - c_prev)) * mu_test * ufl.dx
mu_WeakForm = mu_term1 + mu_term2 + mu_term3

# 组合CH方程弱形式
CH_WeakForm = c_WeakForm + mu_WeakForm
# 分割CH方程弱形式
CH_WeakForm_L, CH_WeakForm_R = ufl.lhs(CH_WeakForm), ufl.rhs(CH_WeakForm)

# 定义AC方程弱形式
eta_WeakForm = []
for k in range(pde.Np):
    # 定义eta的弱形式
    term1 = ufl.dot(eta_trial, eta_test) * ufl.dx
    term2 = - ufl.dot(eta_prev[k], eta_test) * ufl.dx
    term3 = dt * ufl.dot(v, ufl.grad(eta_trial)) * eta_test * ufl.dx
    term4 = dt * pde.L * ufl.inner(pde.keta * pde.N(c_prev) * ufl.grad((1.0 - theta) * eta_prev[k] + theta * eta_trial),
                                   ufl.grad(eta_test)) * ufl.dx
    term5 = dt * pde.L * pde.N(c_prev) * pde.dS(eta_prev, k) * eta_test * ufl.dx
    term6 = dt * pde.eta_stabilizer * (eta_trial - eta_prev[k]) * eta_test * ufl.dx
    # 组合eta的弱形式
    eta_WeakForm.append(term1 + term2 + term3 + term4 + term5 + term6)
# 分割eta的弱形式
eta_WeakForm_L = [ufl.lhs(eta_WeakForm[k]) for k in range(pde.Np)]
eta_WeakForm_R = [ufl.rhs(eta_WeakForm[k]) for k in range(pde.Np)]

# 定义自由能方程弱形式
E = (0.5 * pde.kc * ufl.dot(ufl.grad(c_new), ufl.grad(c_new)) + pde.f(c_new) + pde.N(c_new) * (
        pde.S(eta_new) + pde.grad_eta(eta_new))) * ufl.dx

# 定义斯托克斯方程弱形式
stokes_term1 = ufl.inner(pde.sigma(v_trial, c_new), ufl.grad(v_test)) * ufl.dx
stokes_term2 = ufl.div(v_test) * p_trial * ufl.dx + p_test * ufl.div(v_trial) * ufl.dx
stokes_term3 = - ufl.inner(pde.interface_stress(c_new, eta_new), ufl.grad(v_test)) * ufl.dx
stokes_WeakForm = stokes_term1 + stokes_term2 + stokes_term3
stokes_WeakForm_L, stokes_WeakForm_R = ufl.lhs(stokes_WeakForm), ufl.rhs(stokes_WeakForm)

# 定义应力方程弱形式
stress_test = ufl.TestFunction(WS)
stress_trial = ufl.TrialFunction(WS)
stress_L = ufl.inner(stress_trial, stress_test) * ufl.dx

stress_R_term1 = pde.sigma(v, c_new) * CG * ufl.Identity(2)
stress_R_term2 = - (0.5 * pde.kc * ufl.dot(ufl.grad(c_new), ufl.grad(c_new)) + pde.f(c_new) + pde.N(c_new) * (
        pde.S(eta_new) + pde.grad_eta(eta_new))) * ufl.Identity(2)
stress_R = ufl.inner(stress_R_term1 + stress_R_term2, stress_test) * ufl.dx

eta_wr_L = eta_trial * eta_test * ufl.dx
eta_wr_R = (c_new ** 2) * sum([eta_new[i] ** 2 for i in range(pde.Np)]) * eta_test * ufl.dx

# 设置求解器 **************************************************************************************************
PhaseFields_params = {'ksp_type': 'tfqmr', 'pc_type': 'ilu'}
Stokes_params = {'ksp_type': 'preonly', 'pc_type': 'lu'}

# 重置初始条件 ************************************************************************************************
u_new.assign([c0, mu0])
u_prev.assign(u_new)
for i in range(pde.Np):
    eta_new[i].assign(eta0[i])
    eta_prev[i].assign(eta_new[i])

# 初始时间处求解斯托克斯方程 ************************************************************************************
v_combined.assign([ufl.interpolate(ufl.Constant((0, 0)), VS.sub(0).collapse()),
                   ufl.interpolate(ufl.Constant(0), VS.sub(1).collapse())])
# assign(v_combined,
#        [interpolate(Constant((0, 0)), VS.sub(0).collapse()), interpolate(Constant(0), VS.sub(1).collapse())])
solve(stokes_WeakForm_L == stokes_WeakForm_R, v_combined, bc, solver_parameters=Stokes_params)

# 输出结果文件 ************************************************************************************************
xdmf_file = io.XDMFFile(MPI.COMM_WORLD, os.path.join(Output_directory, 'solution.xdmf'), "w")

# file = File(os.path.join(Output_directory, 'solution.pvd'))
# 存储c
c_wr = u_new.split()[0]
c_wr.rename('C', 'C')
# 存储纯eta
eta_pure = Function(Eta_Space)
eta_pure.rename('eta_pure', 'eta_pure')
# 存储eta
eta_wr = Function(Eta_Space)
eta_wr.rename('eta', 'eta')
# 存储v
v_wr = v_combined.split()[0]
v_wr.rename('v', 'v')
# 存储应力
stress_wr = Function(WS)
stress_wr.rename('stress', 'stress')

# 设置初始时间
timeCurrent = timeInitial
counterStep = counterStepInitial

# 输出初始结果 ************************************************************************************************
# 输出纯eta结果
eta_pure_L = eta_trial * eta_test * ufl.dx
eta_pure_R = sum([eta_new[i] ** 2 for i in range(pde.Np)]) * eta_test * ufl.dx
solve(eta_pure_L == eta_pure_R, eta_wr)
xdmf_file.write_mesh(mesh)
xdmf_file.write_function(eta_pure, timeCurrent)
# file << (eta_pure, timeCurrent)

# 输出c, eta, v, stress结果
xdmf_file.write_function(c_wr, timeCurrent)
# file << (c_wr, timeCurrent)
eta_wr_L = eta_trial * eta_test * ufl.dx
eta_wr_R = (c_new ** 2) * sum([eta_new[i] ** 2 for i in range(pde.Np)]) * eta_test * ufl.dx
solve(eta_wr_L == eta_wr_R, eta_wr)
xdmf_file.write_function(eta_wr, timeCurrent)
xdmf_file.write_function(v_wr, timeCurrent)
# file << (eta_wr, timeCurrent)
# file << (v_wr, timeCurrent)
solve(stress_L == stress_R, stress_wr)
xdmf_file.write_function(stress_wr, timeCurrent)
# file << (stress_wr, timeCurrent)

step_number = NumberOfTimeStep
Time_Simulation = step_number * dt

plot_function(1, c_wr, os.path.join(Figures_directory, 'C'), timeCurrent)
plot_function(2, eta_wr, os.path.join(Figures_directory, 'eta'), timeCurrent)
plot_function(3, v_wr, os.path.join(Figures_directory, 'v'), timeCurrent)
plot_function(4, stress_wr, os.path.join(Figures_directory, 'stress'), timeCurrent)
plot_function(5, eta_pure, os.path.join(Figures_directory, 'eta_pure'), timeCurrent)

# 计算能量
energy = np.empty(step_number + 1)
energy[counterStep] = ufl.assemble(E)
print("the energy at time", timeCurrent, "is", energy[counterStep])

# 时间循环 ****************************************************************************************************
start_time = time.time()
while timeCurrent < Time_Simulation:
    quotient = counterStep // frequencyOutput
    counterStep += 1
    timeCurrent = round(counterStep * dt, 8)
    print("Time step:", counterStep, "Time:", timeCurrent)
    u_prev.assign(u_new)
    for i in range(pde.Np):
        eta_prev[i].assign(eta_new[i])
    print("Solving CH equation")
    solve(CH_WeakForm_L == CH_WeakForm_R, u_new, solver_parameters=PhaseFields_params)
    print("Solving AC equation")
    for i in range(pde.Np):
        solve(eta_WeakForm_L[i] == eta_WeakForm_R[i], eta_new[i], solver_parameters=PhaseFields_params)
    print("Solving Stokes equation")
    solve(stokes_WeakForm_L == stokes_WeakForm_R, v_combined, bc, solver_parameters=Stokes_params)
    print("Solving stress equation")
    solve(stress_L == stress_R, stress_wr)
    energy[counterStep] = ufl.assemble(E)
    print("the energy at time", timeCurrent, "is", energy[counterStep])

    if np.mod(counterStep, frequencyOutput) == 0:
        print("Plot Figures")
        plot_function(6 * quotient + 1, c_wr, os.path.join(Figures_directory, 'C'), timeCurrent)
        plot_function(6 * quotient + 2, eta_wr, os.path.join(Figures_directory, 'eta'), timeCurrent)
        plot_function(6 * quotient + 3, v_wr, os.path.join(Figures_directory, 'v'), timeCurrent)
        plot_function(6 * quotient + 4, stress_wr, os.path.join(Figures_directory, 'stress'), timeCurrent)
        print("Output results")
        xdmf_file.write_function(c_wr, timeCurrent)
        # file << (c_wr, timeCurrent)
        solve(eta_wr_L == eta_wr_R, eta_wr)
        xdmf_file.write_function(eta_wr, timeCurrent)
        xdmf_file.write_function(v_wr, timeCurrent)
        # file << (eta_wr, timeCurrent)
        # file << (v_wr, timeCurrent)
        solve(stress_L == stress_R, stress_wr)
        xdmf_file.write_function(stress_wr, timeCurrent)
        # file << (stress_wr, timeCurrent)

# 计算运行时间 ************************************************************************************************
end_running_time = datetime.datetime.now()
total_time = end_running_time - start_running_time
total_seconds = total_time.total_seconds()
hours, remainder = divmod(total_seconds, 3600)
minutes, seconds = divmod(remainder, 60)
if hours > 0:
    output_time = f"Total running time : {int(hours)} hours {int(minutes)} minutes {int(seconds)} seconds\n"
elif minutes > 0:
    output_time = f"Total running time : {int(minutes)} minutes {int(seconds)} seconds\n"
else:
    output_time = f"Total running time : {int(seconds)} seconds\n"

# 写入相关信息至Readme中 ***************************************************************************************
with open(readme_file_path, 'a', encoding='utf-8') as file:
    file.write(
        f"**************************************************************************************************\n"
        f"Ending time : {end_running_time.strftime('%B %d, %Y, %H:%M:%S')}\n"
        + output_time +
        f"===========================================================\n"
    )
