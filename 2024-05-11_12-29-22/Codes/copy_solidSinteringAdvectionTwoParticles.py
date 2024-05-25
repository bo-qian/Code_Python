import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import random
import numpy as np
import os
import datetime
import time
import subprocess
import csv
import logging
import shutil
import sys
import math

from fenics import *
from mshr import *
from logging import FileHandler

# Define the output directory    
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_directory = os.path.join(os.getcwd(), timestamp)
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Define the log directory
log_directory = output_directory + "/Log"
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

# Define the error directory
error_directory = output_directory + "/Error"
if not os.path.exists(error_directory):
    os.makedirs(error_directory)

# Define the Codes directory
codes_directory = output_directory + "/Codes"
if not os.path.exists(codes_directory):
    os.makedirs(codes_directory)

# Define the input directory
input_directory = output_directory + "/Input"
if not os.path.exists(input_directory):
    os.makedirs(input_directory)

# Define the results directory
results_directory = output_directory + "/Output"
if not os.path.exists(results_directory):
    os.makedirs(results_directory)

# Configure logging
log_file_path = os.path.join(log_directory, 'log.log')
error_file_path = os.path.join(error_directory, 'error.log')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()

# Configure log file handler
log_file_handler = logging.FileHandler(log_file_path)
log_file_handler.setLevel(logging.INFO)
logger.addHandler(log_file_handler)

# Configure error file handler
error_file_handler = FileHandler(error_file_path)
error_file_handler.setLevel(logging.ERROR)
error_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
error_file_handler.setFormatter(error_formatter)
logger.addHandler(error_file_handler)

# Define denominator regularization
denom_reg = lambda x: x if abs(x) > 1e-12 else 1e-12


# Read the input file
def read_input_file(file_name):
    parametersInput = {}  # 定义一个空字典
    with open(file_name, 'r') as f:
        for line in f:
            line = line.strip()  # 去掉每行头尾空白
            if not line or line.startswith('#'):  # Skip empty lines and comments， 跳过空行和注释
                continue
            var_name, var_value = line.split("=")  # 等号左边是变量名，右边是变量值
            parametersInput[var_name.strip()] = var_value.strip()  # 字典的键是变量名，值是变量值
    return parametersInput


def line_integral(u, A, B, num_segments):
    """
    Compute the line integral of a scalar function over a line segment.
    计算标量函数沿线段的线积分。
    Parameters:
        u: Expression or Function
            The scalar function to be integrated. 积分的标量函数。
        A: numpy.ndarray
            Starting point of the line segment. 表示线段的起点。
        B: numpy.ndarray
            Ending point of the line segment. 表示线段的终点。
        num_segments: int
            Number of elements to partition the line segment into. 将线段分割为的元素数。

    Returns:
        float
            The value of the line integral. 线积分的值。
    """
    # Input validation 输入验证
    assert isinstance(u, (Expression, Function)), "Function u must be an Expression or Function object."
    # 函数u必须是表达式或函数对象。
    assert len(A) == len(B), "Vectors A and B must have the same dimension."
    # 向量A和B必须具有相同的维度。
    assert np.linalg.norm(A - B) > 0, "Vectors A and B must not be identical."
    # 向量A和B不能相同。
    assert num_segments > 0, "Number of elements (num_segments) must be positive."
    # 元素数（num_segments）必须为正数。

    # Generate mesh coordinates along the line segment 生成沿线段的网格坐标
    mesh_coordinates = np.array([A + t * (B - A) for t in np.linspace(0, 1, num_segments + 1)])

    # Create a mesh from the coordinates 从坐标创建网格
    mesh = Mesh()

    # Define the topological and geometric dimensions of the mesh 网格的拓扑和几何维度
    tdim = 1  # topological dimension 拓扑维度
    gdim = len(A)  # geometric dimension 几何维度

    # Create mesh vertices and cells 创建网格顶点和单元
    editor = MeshEditor()  # 创建网格编辑器
    editor.open(mesh, 'interval', tdim, gdim)  # 打开网格编辑器
    editor.init_vertices(len(mesh_coordinates))  # 初始化顶点
    editor.init_cells(len(mesh_coordinates) - 1)  # 初始化单元

    # Add vertices 添加顶点
    for i, vertex in enumerate(mesh_coordinates):
        editor.add_vertex(i, vertex)

    # Add cells 添加单元
    for i in range(len(mesh_coordinates) - 1):  # Loop over the number of segments 循环遍历段数
        editor.add_cell(i, np.array([i, i + 1], dtype='uintp'))  # Add a cell 添加单元

    # Close the mesh editor 关闭网格编辑器
    editor.close()

    # Define function space 定义函数空间
    V = FunctionSpace(mesh, u.ufl_element().family(), u.ufl_element().degree())

    # Interpolate u onto function space V 将u插值到函数空间V上
    u_interpolated = interpolate(u, V)

    # Compute line integral 计算线积分
    integral_value = assemble(u_interpolated * dx(domain=mesh))

    return integral_value


# Here, various model parameters are defined:
class solidSintering:  # 定义一个类
    def __init__(self, a, b, kc, keta, Dsf, Dgb, Dvol, L, Neps, muSurf, muGB, muBulk, viscosity_ratio, Np):
        self.a = Constant(a)
        self.b = Constant(b)
        self.kc = Constant(kc)
        self.keta = Constant(keta)
        self.Dsf = Constant(Dsf)
        self.Dgb = Constant(Dgb)
        self.Dvol = Constant(Dvol)
        self.Neps = Constant(Neps)
        self.L = Constant(L)
        self.Np = Np
        self.muSurf = muSurf
        self.muGB = muGB
        self.muBulk = muBulk
        self.viscosity_ratio = viscosity_ratio
        self.c_stabilizer = Constant(2 * a)
        self.eta_stabilizer = Constant(12 * b * L)

    def N(self, C):  # 插值函数N(C)
        return (C ** 2) * (1 + 2 * (1 - C) + (self.Neps) * (1 - C) ** 2)

    def dN(self, C):  # 插值函数N(C)的导数
        return 2 * C * (1 - C) * (3 + self.Neps * (1 - 2 * C))

    def f(self, C):  # 自由能函数f(C)的前半部分
        return self.a * (C ** 2) * ((1 - C) ** 2)

    def df(self, C):  # 自由能函数f(C)前半部分的导数
        return 2 * self.a * C * (1 - C) * (1 - 2 * C)

    def S(self, eta):  # 自由能函数f(C)的没有乘以插值函数N(C)的后半部分
        return self.b * (Constant(1)
                         - 4 * sum([eta[i] ** 3 for i in range(self.Np)])
                         + 3 * sum([eta[i] ** 2 for i in range(self.Np)]) ** 2)

    def dS(self, eta, i):  # 自由能函数f(C)的没有乘以插值函数N(C)的后半部分的导数
        return 12 * self.b * (-eta[i] ** 2 + eta[i] * sum([eta[j] ** 2 for j in range(self.Np)]))

    def grad_eta(self, eta):  # 自由能泛函中积分号内没有乘以插值函数N(C)的后半部分的梯度
        return 0.5 * self.keta * sum([dot(grad(eta[i]), grad(eta[i])) for i in range(self.Np)])

    def M(self, C, eta):  # 用于计算扩散系数M
        return pow(2 * self.a, -1) * (self.Dvol * self.N(C)
                                      + self.Dsf * (C ** 2) * ((1 - C) ** 2)
                                      + 4 * self.Dgb * self.N(C) * sum(
                    [eta[i] ** 2 * (sum([eta[j] ** 2 for j in range(self.Np)]) - eta[i] ** 2) for i in range(self.Np)]))

    def fmu(self, C, eta):  # 自由能F对C的偏导数，对应文章中的公式
        return self.df(C) + self.dN(C) * (self.S(eta) + self.grad_eta(eta))

    def sigma(self, u, C, eta):
        # for this version it is important to keep viscosity_ratio<1
        # the defined sigma here does not include contribution from Lagrangian multiplier
        return ( \
                    (self.viscosity_ratio + (1 - self.viscosity_ratio) * self.N(C)) * self.muBulk
                    + self.muSurf * (C ** 2) * ((1 - C) ** 2)
                    + 2 * self.muGB * self.N(C) * sum(
                [eta[i] ** 2 * (sum([eta[j] ** 2 for j in range(self.Np)]) - eta[i] ** 2) for i in range(self.Np)]) \
            ) * (grad(u) + grad(u).T)

    def interface_stress(self, C, eta):  # 表面张力
        return self.kc * outer(grad(C), grad(C)) + self.keta * self.N(C) * sum(
            [outer(grad(eta[i]), grad(eta[i])) for i in range(self.Np)])


# Main code starts from here 主程序从这里开始
# Copy the codes that you are running 复制您正在运行的代码
print("the code being executed:", sys.argv)
# sys.argv is the python_script_name when you use "python3 python_script_name" to run python_script_name
python_script_name = os.path.basename(sys.argv[0])  # change the name of source code to the one you are running
shutil.copy(python_script_name, os.path.join(codes_directory, 'copy_' + python_script_name))
# Read the input file 读取输入文件"input.txt"
input_file = "input.txt"  # Replace with your input file name
parametersInput = read_input_file(input_file)  # 读取输入文件
shutil.copy(input_file, os.path.join(input_directory, "input_copy.txt"))  # 复制输入文件到input_directory

# Assign values from the input file to parametersInput 从输入文件中为parametersInput赋值
# Phase-field parameters Input
a = float(parametersInput.get('Alpha'))
b = float(parametersInput.get('Beta'))
kc = float(parametersInput.get('KappaC'))
keta = float(parametersInput.get('KappaEta'))
Dsf = float(parametersInput.get('Surface Diffusion'))
Dgb = float(parametersInput.get('GB Diffusion'))
Dvol = float(parametersInput.get('Volume Diffusion'))
L = float(parametersInput.get('Mobility L'))
Neps = float(parametersInput.get('Epsilon'))

# Mechanics Parameters
muSurf = float(parametersInput.get('Surface viscosity'))
muGB = float(parametersInput.get('GB viscosity'))
muBulk = float(parametersInput.get('Bulk viscosity'))
viscosity_ratio = float(parametersInput.get('Viscosity Ratio'))

# Computational Setting
# 粒子半径
radiusParticle = float(parametersInput.get('Particle Radius'))
# 粒子数量
Np = int(parametersInput.get('Particle Number'))
# 特征时间
timeCharacteristic = float(parametersInput.get('Characteristic Time'))
# 空间区域，也就是函数空间
domain_size = [float(parametersInput.get('Dimension X')), float(parametersInput.get('Dimension Y'))]
# 网格单元数量
numberOfElement = [int(parametersInput.get('NumberOfElement X')), int(parametersInput.get('NumberOfElement Y'))]
# 时间步长
dt = float(parametersInput.get('TimeStep'))
# 初始时间
timeInitial = float(parametersInput.get('InitialTime'))
# 时间步数
NumberOfTimeStep = int(parametersInput.get('NumberOfTimeStep'))
# 初始步数计数器
counterStepInitial = int(parametersInput.get('InitialStepCounter'))
# 输出频率
frequencyOutput = int(parametersInput.get('FrequencyOutPut'))

# Algorithmic parameters
theta = float(parametersInput.get('theta'))

pde = solidSintering(a, b, kc, keta, Dsf, Dgb, Dvol, L, Neps, muSurf, muGB, muBulk, viscosity_ratio, Np)
logging.info("# Phase-field parametersInput")  # 记录参数到日志文件中
logging.info(f"Alpha: {pde.a.values()[0]}")
logging.info(f"Beta: {pde.b.values()[0]}")
logging.info(f"Kappac: {pde.kc.values()[0]}")
logging.info(f"KappaEta: {pde.keta.values()[0]}")
logging.info(f"Surface Diffusion: {pde.Dsf.values()[0]}")
logging.info(f"GB Diffusion: {pde.Dgb.values()[0]}")
logging.info(f"Volume Diffusion: {pde.Dvol.values()[0]}")
logging.info(f"Mobility L: {pde.L.values()[0]}")
logging.info(f"Epsilon: {pde.Neps.values()[0]}")

# Mechanics Parameters
logging.info("")
logging.info("# Mechanics Parameters")
logging.info(f"Surface viscosity: {muSurf}")
logging.info(f"GB viscosity: {muGB}")
logging.info(f"Bulk viscosity: {muBulk}")
logging.info(f"Viscosity Ratio: {viscosity_ratio}")

# Computational Setting
logging.info(" ")
logging.info("# Computational Setting")
logging.info(f"Particle Radius: {radiusParticle}")
logging.info(f"Particle Number: {Np}")
logging.info(f"Characteristic time: {timeCharacteristic}")
logging.info(f"Dimension X: {domain_size[0]}")
logging.info(f"Dimension Y: {domain_size[1]}")
logging.info(f"NumberOfElement X: {numberOfElement[0]}")
logging.info(f"NumberOfElement Y: {numberOfElement[1]}")
logging.info(f"TimeStep: {dt}")
logging.info(f"InitialTime: {timeInitial}")
logging.info(f"NumberOfTimeStep: {NumberOfTimeStep}")
logging.info(f"IntialStepCounter: {counterStepInitial}")
logging.info(f"FrequencyOutPut: {frequencyOutput}")

# Algorithmic parameters
logging.info(" ")
logging.info("# Algorithmic parameters")
logging.info(f"theta: {theta}")

# Normalized Materials Properties  归一化材料属性
energySurfSpeciNormalized = (sqrt(2 * pde.kc.values()[0] * pde.a.values()[0]) / 6)  # 表面能
thicknessSurfaceNormalized = sqrt(8 * pde.kc.values()[0] / pde.a.values()[0])  # 表面厚度
thicknessGbNormalized = sqrt(4 * pde.keta.values()[0] / (3 * pde.b.values()[0]))  # 晶界厚度
logging.info(" ")
logging.info("# Normalized Materials Properties")
logging.info(f"specfic surface energy: {energySurfSpeciNormalized}")
logging.info(f"surface thickness: {thicknessSurfaceNormalized}")
logging.info(f"Grain boundary thickness: {thicknessGbNormalized}")

# block with mesh, spaces and weak forms  网格、空间和弱形式的块
# Form compiler options  表单编译器选项
parameters["form_compiler"]["optimize"] = True  # 优化
parameters["form_compiler"]["cpp_optimize"] = True  # 优化
# tell the form to apply optimization strategies in the code generation
# phase and the use compiler optimization flags when compiling the
# generated C++ code. Using the option ``["optimize"] = True`` will
# generally result in faster code (sometimes orders of magnitude faster
# for certain operations, depending on the equation), but it may take
# considerably longer to generate the code and the generation phase may
# use considerably more memory.

# Create mesh and build function space  创建网格和构建函数空间
mesh = RectangleMesh(Point(0, 0), Point(domain_size[0], domain_size[1]), numberOfElement[0], numberOfElement[1],
                     'crossed')

# FunctionSpace consists of c, mu, and etas  函数空间由c、mu和etas组成
P1 = FiniteElement("CG", triangle, 1)  # first order polynomial finite element 第一阶多项式有限元
# For c and mu 对于c和mu
spaceMixedCH = FunctionSpace(mesh, MixedElement([P1, P1]))
# For each eta 对于每个eta
spaceLinear = FunctionSpace(mesh, P1)

# Taylor-Hood function space for velocity  泰勒-胡德函数空间用于速度
PV = VectorElement("CG", triangle, 2)
# Mixed space for both velocity and pressure from stokes equation  来自斯托克斯方程的速度和压力的混合空间
spaceMixedStokes = FunctionSpace(mesh, MixedElement([PV, P1]))

# Create a separate space to output the stress  创建一个单独的空间来输出应力
PS = TensorElement("CG", triangle, 1)  # first order tensor element 第一阶张量元素
spaceTensor = FunctionSpace(mesh, PS)  # function space for stress 应力的函数空间

# Trial and test functions from the mixed function space are now defined:  现在在混合函数空间内定义试验和测试函数
# Define trial and test functions
# du    = TrialFunction(spaceMixedCH)
c_trial, mu_trial = TrialFunction(spaceMixedCH)  # c和mu的试验函数
c_test, mu_test = TestFunctions(spaceMixedCH)  # c和mu的测试函数
eta_trial = TrialFunction(spaceLinear)  # eta的试验函数
eta_test = TestFunction(spaceLinear)  # eta的测试函数

# auxiliary variable trial and test function space  辅助变量试验和测试函数空间
auxiliaryVar_trial = TrialFunction(spaceLinear)  # for later line integration  用于后续线积分
auxiliaryVar_test = TestFunction(spaceLinear)  # for later line integration  用于后续线积分

# Define functions  定义函数
u_new = Function(spaceMixedCH)  # current combined solution of c_new and mu_new  c_new和mu_new的当前组合解
eta_new = [Function(spaceLinear) for i in range(pde.Np)]
u_prev = Function(spaceMixedCH)  # previous combined solution of c_prev and mu_prev  c_prev和mu_prev的先前组合解
eta_prev = [Function(spaceLinear) for i in range(pde.Np)]

# Split mixed functions  分割混合函数
# The line ``c, mu = split(u)`` permits direct access to the components
# of a mixed function. Note that ``c`` and ``mu`` are references for
# components of ``u``, and not copies.
c_new, mu_new = split(u_new)  # split the combined solution into c_new and mu_new  将组合解分割为c_new和mu_new
c_prev, mu_prev = split(u_prev)  # split the combined solution into c_prev and mu_prev  将组合解分割为c_prev和mu_prev

# All functions for velocity space and pressure space
v_trial, p_trial = TrialFunctions(spaceMixedStokes)  # trial functions for velocity and pressure  速度和压力的试验函数
v_test, p_test = TestFunctions(spaceMixedStokes)  # test functions for velocity and pressure  速度和压力的测试函数
v_p_combined = Function(spaceMixedStokes)  # combined solution of velocity and pressure  速度和压力的组合解
v, p = split(v_p_combined)  # split the combined solution into velocity and pressure  将组合解分割为速度和压力


def boundary(x, on_boundary):  # 定义边界条件
    return on_boundary


# Set up the Dirichlet boundary condition for velocity  为速度设置Dirichlet边界条件
bc = DirichletBC(spaceMixedStokes.sub(0), Constant((0, 0)), boundary)
# x and y coordinates  x和y坐标
x, y = Expression('x[0]', degree=1), Expression('x[1]', degree=1)
# Two end points of the center line to calculate horizontal shrinkage  计算水平收缩的中心线的两个端点
Point_A, Point_B = np.array([0, domain_size[1] / 2]), np.array([domain_size[0], domain_size[1] / 2])
# Two end points of the vertical line to calculate neck length  计算颈部长度的垂直线的两个端点
Point_C, Point_D = np.array([domain_size[0] / 2, 0]), np.array([domain_size[0] / 2, domain_size[1]])

# Initial conditions are created and then interpolating the initial conditions into a finite element space
# 创建初始条件，然后将初始条件插值为有限元空间

# Create initial conditions  创建初始条件

c_init = Expression('(1-tanh((sqrt(pow((x[0]-X1), 2)+pow((x[1]-Y1), 2))-R1)/(sqrt(2)*epsilon)))/2' +
                    '+(1-tanh((sqrt(pow((x[0]-X2), 2)+pow((x[1]-Y2), 2))-R2)/(sqrt(2)*epsilon)))/2',
                    degree=1, epsilon=Constant(1), R1=radiusParticle, R2=radiusParticle,
                    X1=domain_size[0] / 2 - radiusParticle, Y1=domain_size[1] / 2,
                    X2=domain_size[0] / 2 + radiusParticle, Y2=domain_size[1] / 2)
mu_init = Constant(0)

# eta_init= [Constant(1), Constant(0)]

# 这段代码定义了两个初始条件，它们是FEniCS表达式对象，用于描述两个相位场变量的初始状态。
# 这两个表达式都是双曲正切函数，它们在x[0]=50的位置从1过渡到0（对于第一个表达式）和从0过渡到1（对于第二个表达式）。
# 这两个表达式都依赖于空间坐标x[0]（系统中的x坐标）和参数epsilon。
# epsilon是一个小的正数，用于控制相位场模型中相位之间过渡区域的宽度。
# 这两个表达式都定义为随着x[0]从负无穷大增加到正无穷大，它们从1过渡到0（或从0过渡到1）。
# Expression函数中的degree参数指定了将表达式插值到的有限元函数空间的程度。
# Expression函数中的epsilon参数是一个Constant对象，它在FEniCS中表示一个不会在模拟过程中改变的常数值。
eta_init = [
    # 第一个eta变量的初始条件。
    # 随着`x[0]`的增加，表达式从1过渡到0。
    Expression('(1-tanh((x[0]-Cx)/(sqrt(2)*epsilon)))/2', degree=1, epsilon=Constant(1), Cx=domain_size[0] / 2),

    # 第二个eta变量的初始条件。
    # 随着`x[0]`的增加，表达式从0过渡到1。
    Expression('(1+tanh((x[0]-Cx)/(sqrt(2)*epsilon)))/2', degree=1, epsilon=Constant(1), Cx=domain_size[0] / 2)
]

# Interpolate  插值
c0 = interpolate(c_init, spaceMixedCH.sub(0).collapse())
mu0 = interpolate(mu_init, spaceMixedCH.sub(1).collapse())
eta0 = [interpolate(eta_init[i], spaceLinear) for i in range(pde.Np)]

# semi_implicit c for laplace  拉普拉斯的半隐式c
c_mid = (1.0 - theta) * c_prev + theta * c_trial

# Reset initial conditions  重置初始条件
assign(u_new, [c0, mu0])
u_prev.assign(u_new)
for i in range(pde.Np):
    assign(eta_new[i], eta0[i])
    eta_prev[i].assign(eta_new[i])

# Variables for line integration  线积分变量
c_lineIntegration = Function(spaceLinear)

# Define some quantities  定义一些量
timeSimulation = NumberOfTimeStep * dt  # total simulation time  总模拟时间
energy = np.empty(NumberOfTimeStep + 1)  # total free energy  总自由能
dR = np.empty(NumberOfTimeStep + 1)  # horizontal shrinkage  水平收缩
dL = np.empty(NumberOfTimeStep + 1)  # vertical shrinkage  垂直收缩
dR_adv = np.empty(NumberOfTimeStep + 1)  # horizontal shrinkage  水平收缩
radiusNeckIntegrate = np.empty(NumberOfTimeStep + 1)  # neck radius  颈部半径
radiusNeckFormula = np.empty(NumberOfTimeStep + 1)  # neck radius  颈部半径
velocityMechanicalAveragedX = np.empty(NumberOfTimeStep + 1)  # average velocity  平均速度
velocityShiftAveragedX = np.empty(NumberOfTimeStep + 1)  # average velocity  平均速度

dR[counterStepInitial] = 0  # 初始时间时水平收缩为0
dL[counterStepInitial] = 0  # 初始时间时垂直收缩为0
dR_adv[counterStepInitial] = 0  # 初始时间时水平收缩为0

velocityMechanicalAveragedX[counterStepInitial] = 0  # 初始时间时平均速度为0
velocityShiftAveragedX[counterStepInitial] = 0  # 初始时间时平均速度为0

# calculate the total free energy  计算总自由能
E = (0.5 * pde.kc * dot(grad(c_new), grad(c_new)) + pde.f(c_new) + pde.N(c_new) * (
        pde.S(eta_new) + pde.grad_eta(eta_new))) * dx
energy[counterStepInitial] = assemble(E)

# R0 is the distance between the centers of particles  R0是粒子中心之间的距离
R0 = (assemble(c_new * eta_new[1] * x * dx) / denom_reg(assemble(c_new * eta_new[1] * dx))) \
     - (assemble(c_new * eta_new[0] * x * dx) / denom_reg(assemble(c_new * eta_new[0] * dx)))
print('')
print('Initial distance between the centers of the two particles: R0=', R0)
print('')

# solve for c_new and store it in c_lineIntegration: why have to solve it?
# 解决c_new并将其存储在c_lineIntegration中：为什么必须解决它？
# because the line integration needs a direct Fenics object to perform line integration
# 因为线积分需要一个直接的Fenics对象来执行线积分
auxiliary_L = auxiliaryVar_trial * auxiliaryVar_test * dx
# define the auxiliary problem  定义辅助问题
solve(auxiliary_L == c_new * auxiliaryVar_test * dx, c_lineIntegration)
# do line integration of c_new over the line defined by Point_A and Point_B
L0 = line_integral(c_lineIntegration, Point_A, Point_B, numberOfElement[0])
print('')
print('Initial distance between the boundary of the two particles: L0=', L0)
print('')

# calculate neck radius  计算颈部半径
# radiusNeck[0]=0.5*line_integral(c_lineIntegration,Point_C, Point_D,numberOfElement[1])
radiusNeckIntegrate[0] = 0.5 * line_integral(c_lineIntegration, Point_C, Point_D, numberOfElement[1])
radiusNeckFormula[0] = 0.5 * 6 * assemble(c_new * eta_new[0] * c_new * eta_new[1] * dx) / thicknessSurfaceNormalized

# The linear form  线性形式
# weak form of C  C的弱形式
cWeakForm = c_trial * c_test * dx - c_prev * c_test * dx \
            + dt * dot(pde.M(c_prev, eta_prev) * grad(mu_trial), grad(c_test)) * dx \
            - dt * dot(v * c_trial, grad(c_test)) * dx
# weak form of mu  mu的弱形式
muWeakForm = mu_trial * mu_test * dx - pde.kc * dot(grad(c_mid), grad(mu_test)) * dx \
             - (pde.fmu(c_prev, eta_prev) + pde.c_stabilizer * (c_trial - c_prev)) * mu_test * dx
# combined weak form of c and mu; c and mu are solved together  c和mu的组合弱形式；c和mu一起解决
CH_WeakFormCombined = cWeakForm + muWeakForm
CH_WeakFormCombined_L, CH_WeakFormCombined_R = lhs(CH_WeakFormCombined), rhs(CH_WeakFormCombined)

# weak form of each eta  每个eta的弱形式
eta_WeakForm = [dot(eta_trial, eta_test) * dx - dot(eta_prev[i], eta_test) * dx \
                + dt * dot(v, grad(eta_trial)) * eta_test * dx \
                + dt * pde.L * inner(pde.keta * pde.N(c_prev) * grad((1.0 - theta) * eta_prev[i] + theta * eta_trial),
                                     grad(eta_test)) * dx \
                + dt * pde.L * pde.N(c_prev) * pde.dS(eta_prev, i) * eta_test * dx + dt * pde.eta_stabilizer * (
                        eta_trial - eta_prev[i]) * eta_test * dx
                for i in range(pde.Np)]
eta_WeakForm_L = [lhs(eta_WeakForm[i]) for i in range(pde.Np)]
eta_WeakForm_R = [rhs(eta_WeakForm[i]) for i in range(pde.Np)]

# The weak form of velocity and pressure  速度和压力的弱形式
Stokes_WeakFormCombined = inner(pde.sigma(v_trial, c_new, eta_new), grad(v_test)) * dx + div(
    v_test) * p_trial * dx + p_test * div(v_trial) * dx \
                          - inner(pde.interface_stress(c_new, eta_new), grad(v_test)) * dx
Stokes_WeakFormCombined_L, Stokes_WeakFormCombined_R = lhs(Stokes_WeakFormCombined), rhs(Stokes_WeakFormCombined)

# define equations for physical stress  为物理应力定义方程
stress_test = TestFunction(spaceTensor)
stress_trial = TrialFunction(spaceTensor)
stress_L = inner(stress_trial, stress_test) * dx
stress_R = inner(pde.sigma(v, c_new, eta_new) + p * Identity(2)
                 - (0.5 * pde.kc * dot(grad(c_new), grad(c_new)) + pde.f(c_new)
                    + pde.N(c_new) * (pde.S(eta_new) + pde.grad_eta(eta_new))) * Identity(2), stress_test) * dx

# block with solver and solution  带有求解器和解的块
# solver settings  求解器设置
PhaseField_params = {'linear_solver': 'tfqmr', 'preconditioner': 'ilu'}  # 'tfqmr' is an iterative solver

Stokes_params = {'linear_solver': 'mumps'}  # 'mumps' is a direct solver

# Solving Stokes at initial time and store the solutions in v_p_combined  在初始时间解决斯托克斯方程并将解存储在v_p_combined中
assign(v_p_combined, [interpolate(Constant((0, 0)), spaceMixedStokes.sub(0).collapse()),
                      interpolate(Constant(0), spaceMixedStokes.sub(1).collapse())])
solve(Stokes_WeakFormCombined_L == Stokes_WeakFormCombined_R, v_p_combined, bc, solver_parameters=Stokes_params)

# To run the solver and save the output to a VTK file for later visualization,
# the solver is advanced in time from t_n to t_{n+1} until
# a terminal time timeSimulation is reached.

# Output file
file = File(os.path.join(results_directory, "solution.pvd"))
# variable for storing c  存储c的变量
cwr = u_new.split()[0]
cwr.rename('C', 'C')
# variable for storing c^2*eta^2  存储c^2*eta^2的变量
etawr = Function(spaceLinear)
etawr.rename('eta', 'eta')
# store velocity  存储速度
vwr = v_p_combined.split()[0]
vwr.rename('v', 'v')

# Variables for writing stress and strainRate  用于写应力和应变率的变量
stress = Function(spaceTensor)
stress.rename('stress', 'stress')

# Step in time  时间步长
timeCurrent = timeInitial
counterStep = counterStepInitial
# store c  存储c
file << (cwr, timeCurrent)
# solve for c^2*eta^2 and store it in etawr  解决c^2*eta^2并将其存储在etawr中
etawr_L = eta_trial * eta_test * dx
etawr_R = (c_new ** 2) * sum([eta_new[j] ** 2 for j in range(pde.Np)]) * eta_test * dx
solve(etawr_L == etawr_R, etawr)
file << (etawr, timeCurrent)
# store velocity  存储速度
file << (vwr, timeCurrent)
# store stress  存储应力
solve(stress_L == stress_R, stress)
file << (stress, timeCurrent)

print('')
print('initial particle distance L0 is: ', L0)
print('')
print('the initial neck radius from line intergration is: ', radiusNeckIntegrate[0])
print('the initial neck radius from the formula is: ', radiusNeckFormula[0])
print('')
print("the energy at time", timeCurrent, "is", energy[counterStep])
print('')

start_time = time.time()
while timeCurrent < timeSimulation:
    counterStep += 1
    timeCurrent = round(counterStep * dt, 8)
    print("*******************************************")
    print('starting computations at time', timeCurrent)
    u_prev.assign(u_new)
    for i in range(pde.Np):
        eta_prev[i].assign(eta_new[i])
    print('')
    print('--solving for C and mu')
    # solve for c and mu and store them in u_new  解决c和mu并将它们存储在u_new中
    solve(CH_WeakFormCombined_L == CH_WeakFormCombined_R, u_new, solver_parameters=PhaseField_params)
    print(" ")
    print('--solving for eta')
    for i in range(pde.Np):
        solve(eta_WeakForm_L[i] == eta_WeakForm_R[i], eta_new[i], solver_parameters=PhaseField_params)
    print(" ")
    print('--solving Stokes')
    # solving for v and pressure and store them in v_p_combined  解决v和压力并将它们存储在v_p_combined中
    solve(Stokes_WeakFormCombined_L == Stokes_WeakFormCombined_R, v_p_combined, bc, solver_parameters=Stokes_params)
    # calculate the system energy  计算系统能量
    energy[counterStep] = assemble(E)
    print('')
    print("the energy at time", timeCurrent, "is ", energy[counterStep])
    # dR is the shrinkage between the centers of the two-particle model  dR是两粒子模型中心之间的收缩
    dR[counterStep] = R0 - (assemble(c_new * eta_new[1] * x * dx) / denom_reg(assemble(c_new * eta_new[1] * dx))) \
                      + (assemble(c_new * eta_new[0] * x * dx) / denom_reg(assemble(c_new * eta_new[0] * dx)))

    # solve for c_new and store it in c_lineIntegration  解决c_new并将其存储在c_lineIntegration中
    # etawr_L and eta_test are defined from the line space; so they can be re-used here.
    # etawr_L和eta_test是从线空间定义的；因此它们可以在这里重复使用。
    solve(etawr_L == c_new * eta_test * dx, c_lineIntegration)

    # dL is the length change of the two particle model  dL是两粒子模型的长度变化
    dL[counterStep] = L0 - line_integral(c_lineIntegration, Point_A, Point_B, numberOfElement[0])

    # calculate the neck radius  计算颈部半径
    radiusNeckIntegrate[counterStep] = 0.5 * line_integral(c_lineIntegration, Point_C, Point_D, numberOfElement[1])
    radiusNeckFormula[counterStep] = 0.5 * 6 * assemble(
        c_new * eta_new[0] * c_new * eta_new[1] * dx) / thicknessSurfaceNormalized

    # Advection contribution for dR  dR的平流贡献
    velocityMechanicalAveragedX[counterStep] = (assemble(c_new * eta_new[0] * (v[0]) * dx) / denom_reg(
        assemble(c_new * eta_new[0] * dx))) \
                                               - (assemble(c_new * eta_new[1] * (v[0]) * dx) / denom_reg(
        assemble(c_new * eta_new[1] * dx)))
    dR_adv[counterStep] = dR_adv[counterStep - 1] + dt * velocityMechanicalAveragedX[counterStep]

    velocityShiftAveragedX[counterStep] = (assemble(
        eta_new[0] * (-pde.M(c_new, eta_new) * (grad(mu_new)[0])) * dx) / denom_reg(
        assemble(c_new * eta_new[0] * dx))) - (assemble(
        eta_new[1] * (-pde.M(c_new, eta_new) * (grad(mu_new)[0])) * dx) / denom_reg(assemble(c_new * eta_new[1] * dx)))

    # velocityShiftAveragedX[counterStep] = velocityShiftAveraged[0]

    print('')
    print("the dR at time ", timeCurrent, "is ", dR[counterStep])
    print("the dR_adv at time ", timeCurrent, "is ", dR_adv[counterStep])
    print("the dL at time ", timeCurrent, "is ", dL[counterStep])
    print("the radiusNeckIntegrate at time ", timeCurrent, "is ", radiusNeckIntegrate[counterStep])
    print("the radiusNeckFormula at time ", timeCurrent, "is ", radiusNeckFormula[counterStep])

    if np.mod(counterStep, frequencyOutput) == 0:
        quotient = counterStep // frequencyOutput
        print('')
        print('--writing the data to file--')
        # write C to file
        file << (cwr, timeCurrent)
        # solve c^2*eta^2
        solve(etawr_L == etawr_R, etawr)
        # write c^2*eta^2 to file
        file << (etawr, timeCurrent)
        # write velocity to file
        file << (vwr, timeCurrent)
        # solve for stress tensor
        solve(stress_L == stress_R, stress)
        # write stress to file
        file << (stress, timeCurrent)

        timeValues = np.linspace(0, counterStep * dt, counterStep + 1);
        # use surface viscosity and specific surface energy to normalize time
        # timeValuesNormalized = energySurfSpeciNormalized*timeValues/(1.5*radiusParticle*pde.muSurf)+1e-15
        # timeValuesNormalized = energySurfSpeciNormalized*timeValues/(radiusParticle*pde.muBulk)+1e-15
        timeValuesNormalized = timeValues / timeCharacteristic + 1e-15  # 1e-15 is to avoid 0 in log plot

        nameFileShrinkage = f"/dataShrinkageNormalized_{counterStep:06d}.txt"
        nameFileLenghNeck = f"/dataLengthNeckNormalized_{counterStep:06d}.txt"
        nameFileEnergy = f"/dataEnergy_{counterStep:06d}.txt"
        nameFileVelocity = f"/dataVelocity_{counterStep:06d}.txt"

        dR_Normalized = dR[:counterStep + 1] / radiusParticle
        dR_adv_Normalized = dR_adv[:counterStep + 1] / radiusParticle
        dL_Normalized = dL[:counterStep + 1] / radiusParticle

        neckNormalizedIntegrate = radiusNeckIntegrate[:counterStep + 1] / radiusParticle
        neckNormalizedFormula = radiusNeckFormula[:counterStep + 1] / radiusParticle

        dataShrinkageNormalized = np.column_stack(
            (timeValuesNormalized, dR_Normalized, dR_adv_Normalized, dL_Normalized))
        np.savetxt(output_directory + nameFileShrinkage, dataShrinkageNormalized, fmt='%.6f',
                   header='Time,          dR,        dR_adv         dL', delimiter=',    ')

        dataLengthNeckNormalized = np.column_stack(
            (timeValuesNormalized, neckNormalizedIntegrate, neckNormalizedFormula))
        np.savetxt(output_directory + nameFileLenghNeck, dataLengthNeckNormalized, fmt='%.6f',
                   header='Time,    neckLengthIntegrate,    neckLengthFormula', delimiter=',    ')

        dataEnergy = np.column_stack((timeValuesNormalized, energy[:counterStep + 1]))
        np.savetxt(output_directory + nameFileEnergy, dataEnergy, fmt='%.6f', header='Time, systemEnergy',
                   delimiter=',  ')

        dataVelocityAveraged = np.column_stack((timeValuesNormalized, velocityMechanicalAveragedX[:counterStep + 1],
                                                velocityShiftAveragedX[:counterStep + 1]))
        np.savetxt(output_directory + nameFileVelocity, dataVelocityAveraged, fmt='%.6f',
                   header='Time,       mechanicalVelocity,           shiftVelocity', delimiter=',  ')



        # Plot the results  绘制结果
        # set 3 subplots with equal size  设置3个子图的大小相等
        gs = GridSpec(1, 3, width_ratios=[1, 1, 1], height_ratios=[1])

        fig = plt.figure(6 * (quotient - 1) + 1, figsize=(15, 5))
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(timeValuesNormalized[1:], dL_Normalized[1:], label='dL')
        ax1.plot(timeValuesNormalized[1:], dR_Normalized[1:], label='dR')
        ax1.plot(timeValuesNormalized[1:], dR_adv_Normalized[1:], label='dR_adv')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Shrinkage (Linear Scales)')
        ax1.set_title('Linear Scales')
        ax1.set_box_aspect(1)  # Set equal aspect ratio
        ax1.legend()

        ax2 = fig.add_subplot(gs[1])
        ax2.plot(timeValuesNormalized[1:], dL_Normalized[1:], label='dL')
        ax2.plot(timeValuesNormalized[1:], dR_Normalized[1:], label='dR')
        ax2.plot(timeValuesNormalized[1:], dR_adv_Normalized[1:], label='dR_adv')
        ax2.set_xscale('log')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Shrinkage (Linear Scales)')
        ax2.set_title('Log Scales X-axis')
        ax2.set_box_aspect(1)  # Set equal aspect ratio
        ax2.legend()

        ax3 = fig.add_subplot(gs[2])
        ax3.plot(timeValuesNormalized[1:], dL_Normalized[1:], label='dL')
        ax3.plot(timeValuesNormalized[1:], dR_Normalized[1:], label='dR')
        ax3.plot(timeValuesNormalized[1:], dR_adv_Normalized[1:], label='dR_adv')
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Shrinkage (Log Scales)')
        ax3.set_title('Log Scales')
        ax3.set_box_aspect(1)  # Set equal aspect ratio
        ax3.legend()
        plt.savefig(output_directory + f'/shrinkage_{counterStep:06d}.png')

        fig = plt.figure(6 * (quotient - 1) + 2, figsize=(15, 5))
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(timeValuesNormalized[1:], neckNormalizedIntegrate[1:], label='radiusNeckIntegrate')
        ax1.plot(timeValuesNormalized[1:], neckNormalizedFormula[1:], label='radiusNeckFormula')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Neck (Linear Scales)')
        ax1.set_title('Linear Scales')
        ax1.set_box_aspect(1)  # Set equal aspect ratio
        ax1.legend()

        ax2 = fig.add_subplot(gs[1])
        ax2.plot(timeValuesNormalized[1:], neckNormalizedIntegrate[1:], label='radiusNeckIntegrate')
        ax2.plot(timeValuesNormalized[1:], neckNormalizedFormula[1:], label='radiusNeckFormula')
        ax2.set_xscale('log')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Neck (Linear Scales)')
        ax2.set_title('Log Scales X-axis')
        ax2.set_box_aspect(1)  # Set equal aspect ratio
        ax2.legend()

        ax3 = fig.add_subplot(gs[2])
        ax3.plot(timeValuesNormalized[1:], neckNormalizedIntegrate[1:], label='radiusNeckIntegrate')
        ax3.plot(timeValuesNormalized[1:], neckNormalizedFormula[1:], label='radiusNeckFormula')
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.set_xlabel('Time(Log Scales)')
        ax3.set_ylabel('Neck (Log Scales)')
        ax3.set_title('Log Scales')
        ax3.set_box_aspect(1)  # Set equal aspect ratio
        ax3.legend()
        fig.savefig(output_directory + f'/neckRadius_{counterStep:06d}.png')

        plt.figure(6 * (quotient - 1) + 3)
        plt.plot(timeValuesNormalized, energy[:counterStep + 1], label='energy')
        plt.xlabel('Time')
        plt.ylabel('Energy')
        plt.legend()
        plt.savefig(output_directory + f'/energy_{counterStep:06d}.png')

        plt.figure(6 * (quotient - 1) + 4)
        plt.plot(timeValuesNormalized, velocityMechanicalAveragedX[:counterStep + 1], label='mechanical Velocity')
        plt.plot(timeValuesNormalized, velocityShiftAveragedX[:counterStep + 1], label='shift Velocity')
        plt.xlabel('Time')
        plt.ylabel('Averaged Velocity')
        plt.legend()
        plt.savefig(output_directory + f'/velocity_{counterStep:06d}.png')

        plt.figure(6 * (quotient - 1) + 5)
        plot(c_new, cmap='coolwarm')
        plt.savefig(output_directory + f'/c_new_{counterStep:06d}.png')

        plt.figure(6 * (quotient - 1) + 6)
        plot(etawr, cmap='coolwarm')
        plt.savefig(output_directory + f'/gb_new_{counterStep:06d}.png')

    # 对vtu文件进行分类并移动到对应的文件夹中
    # 定义源文件夹和目标文件夹
    source_folder = "Output"
    target_folder_c = "Output/c"
    target_folder_eta = "Output/eta"
    target_folder_v = "Output/v"
    target_folder_stress = "Output/stress"

    # 创建目标文件夹
    os.makedirs(target_folder_c, exist_ok=True)
    os.makedirs(target_folder_eta, exist_ok=True)
    os.makedirs(target_folder_v, exist_ok=True)
    os.makedirs(target_folder_stress, exist_ok=True)

    # 遍历源文件夹中的所有文件
    for filename in os.listdir(source_folder):
        # 检查文件是否为vtu文件
        if filename.endswith(".vtu"):
            # 提取文件名中的数字
            file_number = int(''.join(filter(str.isdigit, filename)))
            # 根据文件名中的数字将文件移动到相应的文件夹中
            if file_number % 4 == 0:
                shutil.move(os.path.join(source_folder, filename), target_folder_c)
            elif file_number % 4 == 1:
                shutil.move(os.path.join(source_folder, filename), target_folder_eta)
            elif file_number % 4 == 2:
                shutil.move(os.path.join(source_folder, filename), target_folder_v)
            else:
                shutil.move(os.path.join(source_folder, filename), target_folder_stress)

end_time = time.time()
execution_time = end_time - start_time
print('')
print(f"Execution time: {execution_time} seconds")
print('')
