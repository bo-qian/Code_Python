# ************************************************************************************************************
# Description: This is a phase-field simulation for poly particle solid-state sintering.
# Wrote by Bo Qian at Shanghai University
# Date: 2024-06-06
# 1st update: 2024-06-25
# 2nd update: 2024-07-01
# 3rd update: 2024-07-02 (Add the function of reading rest results)
# 4th update: 2024-07-08 (Finished to add the function of reading rest results)
# ************************************************************************************************************
import logging
import sys
import csv
import signal

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.tri import Triangulation
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
import numpy as np
import os
import datetime
import time
import shutil
from fenics import *


#############################################################################################################
# Function Part
#############################################################################################################

# Function of reading input file ******************************************************************************
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


# Function of setting space of eta ****************************************************************************
class VoronoiParticle(UserExpression):
    center = [0, 0]
    other = [[1, 1]]
    epsilon = 1

    def value_shape(self):
        return ()

    def eval(self, value, x):
        dx = x[0] - self.center[0]
        dy = x[1] - self.center[1]
        # Calculate the distance between the current grain and the closest other grain
        closest_ind = np.argmin([sqrt(pow((x[0] - self.others[k][0]), 2) + pow((x[1] - self.others[k][1]), 2)) for k in
                                 range(len(self.others))])
        # Obtain the coordinates of the closest other grain
        closest = self.others[closest_ind]
        # Calculate the distance between the current grain and the closest other grain
        dpx = closest[0] - self.center[0]
        dpy = closest[1] - self.center[1]
        # Calculate the distance between the current grain and the closest other grain
        dp = sqrt(pow(dpx, 2) + pow(dpy, 2))
        value[0] = (1 + np.tanh((0.5 * pow(dp, 2) - dx * dpx - dy * dpy) / (sqrt(2) * self.epsilon * dp))) / 2


# Function of smooth rectangle ********************************************************************************
def smooth_rectangle(x, y, x_center, y_center, width, height, epsilon):
    # At x direction, perform hyperbolic tangent function operation
    sigmoid_x = 0.5 * (1 - np.tanh((x - x_center - width / 2) / (np.sqrt(2) * epsilon))) - 0.5 * (
            1 - np.tanh((x - x_center + width / 2) / (np.sqrt(2) * epsilon)))
    # At y direction, perform hyperbolic tangent function operation
    sigmoid_y = 0.5 * (1 - np.tanh((y - y_center - height / 2) / (np.sqrt(2) * epsilon))) - 0.5 * (
            1 - np.tanh((y - y_center + height / 2) / (np.sqrt(2) * epsilon)))
    # Return the product of the two sigmoid functions
    return sigmoid_x * sigmoid_y


# Function of creat template grain ****************************************************************************
class Template(UserExpression):
    center = [0.5, 0.5]
    dimensions = [0.1, 0.1]
    epsilon = 1

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

    def to_string_expression(self):
        return '*'.join([
            f'((1 - tanh((x[0] - {self.center[0]} - {self.dimensions[0]}/2)/(sqrt(2)*{self.epsilon})))/2'
            f'-(1 - tanh((x[0] - {self.center[0]} + {self.dimensions[0]}/2)/(sqrt(2)*{self.epsilon})))/2)',
            f'((1 - tanh((x[1] - {self.center[1]} - {self.dimensions[1]}/2)/(sqrt(2)*{self.epsilon})))/2'
            f'-(1 - tanh((x[1] - {self.center[1]} + {self.dimensions[1]}/2)/(sqrt(2)*{self.epsilon})))/2)'
        ])


# Function of creat particle expression ***********************************************************************
def circle_string_expression(radius, center):
    return (f"(1 - tanh((sqrt(pow((x[0]-{center[0]}), 2) + pow((x[1]-{center[1]}), 2)) - {radius}) / (sqrt("
            f"2)*{epsilon})))/2")


# Function of set governing equation **************************************************************************
class SolidStateSintering:
    def __init__(self, a, b, kc, keta, Dsf, Dgb, Dvol, L, zeta, Np, Neps, muSurf, muGB, muBulk, viscosity_ratio):
        self.a = Constant(a)
        self.b = Constant(b)
        self.kc = Constant(kc)
        self.keta = Constant(keta)
        self.Dsf = Constant(Dsf)
        self.Dgb = Constant(Dgb)
        self.Dvol = Constant(Dvol)
        self.L = Constant(L)
        self.Neps = Constant(Neps)
        self.muSurf = muSurf
        self.muGB = muGB
        self.muBulk = muBulk
        self.zeta = zeta - (2 / 3) * muSurf
        self.Np = Np
        self.viscosity_ratio = viscosity_ratio
        self.c_stabilizer = Constant(2 * a)
        self.eta_stabilizer = Constant(12 * b * L)

    def N(self, C):
        return (C ** 2) * (1 + 2 * (1 - C) + self.Neps * (1 - C) ** 2)

    def dN(self, C):
        return 2 * C * (1 - C) * (3 + self.Neps * (1 - 2 * C))

    def f(self, C):
        return self.a * (C ** 2) * ((1 - C) ** 2)

    def df(self, C):
        return 2 * self.a * C * (1 - C) * (1 - 2 * C)

    def S(self, eta):
        return self.b * (Constant(1)
                         - 4 * sum([eta[i] ** 3 for i in range(self.Np)])
                         + 3 * sum([eta[i] ** 2 for i in range(self.Np)]) ** 2)

    def dS(self, eta, i):
        return 12 * self.b * (-eta[i] ** 2 + eta[i] * sum([eta[j] ** 2 for j in range(self.Np)]))

    def grad_eta(self, eta):
        return 0.5 * self.keta * sum([dot(grad(eta[i]), grad(eta[i])) for i in range(self.Np)])

    def M(self, C, eta):
        return pow(2 * self.a, -1) * (self.Dvol * self.N(C)
                                      + self.Dsf * (C ** 2) * ((1 - C) ** 2)
                                      + 4 * self.Dgb * self.N(C) * sum(
                    [eta[i] ** 2 * (sum([eta[j] ** 2 for j in range(self.Np)]) - eta[i] ** 2) for i in range(self.Np)]))

    def fmu(self, C, eta):
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

    # Calculate the interface stress
    def interface_stress(self, C, eta):
        return self.kc * outer(grad(C), grad(C)) + self.keta * self.N(C) * sum(
            [outer(grad(eta[i]), grad(eta[i])) for i in range(self.Np)])


def particle_centers_without_template(radius_particle, particle_number_total, number_x, number_y, domain):
    particle_radius = [radius_particle] * particle_number_total
    particle_centers_coordinate = []
    for j in range(number_y):
        for i in range(number_x):
            x_coordinate = int(domain[0] / 2 + (i + (1 - number_x) / 2) * radius_particle * 2)
            y_coordinate = int(domain[1] / 2 + (j + (1 - number_y) / 2) * radius_particle * 2)
            particle_centers_coordinate.append([x_coordinate, y_coordinate])
    return particle_centers_coordinate, particle_radius


def particle_centers_with_template(layer_number, aspect_ratio_template, template_particle, radius_particle,
                                   particle_number_total):
    particle_radius = [radius_particle] * particle_number_total
    particle_centers_radius = []
    for i in range(layer_number):
        for j in range(aspect_ratio_template + layer_number + 1 + i):
            x_coordinate = float(
                template_particle.center[0] + (
                        j - (aspect_ratio_template + layer_number + i) / 2) * radius_particle * 2)
            y_coordinate = float(template_particle.center[1] + 0.5 * template_particle.dimensions[1] + (
                    1 + (layer_number - 1 - i) * sqrt(3)) * radius_particle)
            particle_centers_radius.append([x_coordinate, y_coordinate])
            y_coordinate = float(template_particle.center[1] - 0.5 * template_particle.dimensions[1] - (
                    1 + (layer_number - 1 - i) * sqrt(3)) * radius_particle)
            particle_centers_radius.append([x_coordinate, y_coordinate])
        y_coordinate = float(template_particle.center[1])
        x_coordinate = float(template_particle.center[0] - 0.5 * template_particle.dimensions[0] - (
                1 + (layer_number - 1 - i) * 2) * radius_particle)
        particle_centers_radius.append([x_coordinate, y_coordinate])
        x_coordinate = float(template_particle.center[0] + 0.5 * template_particle.dimensions[0] + (
                1 + (layer_number - 1 - i) * 2) * radius_particle)
        particle_centers_radius.append([x_coordinate, y_coordinate])
    return particle_centers_radius, particle_radius


def signal_handler(sig, frame):
    global interrupted
    print("===== Received interrupt signal. Stopping after current iteration. =====")
    interrupted = True

# Function to plot figures of phase-field variants *************************************************************
def plot_function(variant, file_directory, time_current):
    if not os.path.exists(file_directory):
        os.makedirs(file_directory)
    with plt.rc_context({'font.family': 'serif', 'font.serif': ['Arial'], 'font.weight': 'regular', 'font.size': 24}):
        plt.subplots(figsize=(12, 9), dpi=100)
        plot(variant, title=f"{variant} at time {time_current}s", cmap='coolwarm')
        plt.savefig(file_directory + f'/{variant}_{time_current:g}s.png', dpi=100, bbox_inches='tight')
        plt.close()


# Function to plot curves of data results **********************************************************************
def plot_curve(X, Y, unit_y, name, time_current):
    file_directory = os.path.join(Figures_directory, name)
    if not os.path.exists(file_directory):
        os.makedirs(file_directory)
    with plt.rc_context(
            {'font.family': 'serif', 'font.serif': ['Times New Roman'], 'font.weight': 'bold', 'font.size': 32}):
        fig, ax = plt.subplots(figsize=(12, 9), dpi=100)
        ax.spines['top'].set_linewidth(3)
        ax.spines['bottom'].set_linewidth(3)
        ax.spines['left'].set_linewidth(3)
        ax.spines['right'].set_linewidth(3)
        plt.tick_params(axis='both', direction='in', width=3, which='both', pad=10)  # 设置刻度朝内，边框厚度为 2
        plt.plot(X, Y, label=fr'In {time_current:g} seconds', linewidth=3, color='black')
        offset = ax.yaxis.get_offset_text()
        transform = offset.get_transform()
        offset.set_transform(transform + plt.matplotlib.transforms.ScaledTranslation(0, 5 / 72., fig.dpi_scale_trans))
        plt.title(name, pad=20, fontweight='bold')
        plt.xlabel(f'Time (s)', fontweight='bold')
        plt.ylabel(f'{name} ({unit_y})', fontweight='bold')
        plt.tight_layout()
        plt.legend(fontsize='small')
        plt.savefig(file_directory + f'/{name}_{time_current:g}s.png', dpi=100, bbox_inches='tight')
        plt.close()


# Function to plot curve group ********************************************************************************
def plot_curve_group(data, name, counter_step_current, time_current):
    file_directory = os.path.join(Figures_directory, "Curve Group")
    line_width = 4
    if not os.path.exists(file_directory):
        os.makedirs(file_directory)
    with plt.rc_context(
            {'font.family': 'serif', 'font.serif': ['Times New Roman'], 'font.weight': 'bold', 'font.size': 32}):
        fig = plt.figure(figsize=(24, 18), dpi=100)
        gs = GridSpec(2, 2, figure=fig)
        axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
        for i, ax in enumerate(axes):
            ax.spines['top'].set_linewidth(line_width)
            ax.spines['bottom'].set_linewidth(line_width)
            ax.spines['left'].set_linewidth(line_width)
            ax.spines['right'].set_linewidth(line_width)
            ax.tick_params(axis='both', direction='in', length=10, width=line_width, which='both', pad=10)
            ax.plot(data[:counter_step_current + 1, 0], data[:counter_step_current + 1, i + 1], label=name[0, i],
                    linewidth=line_width, color='black')
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

            ax.set_title(name[0, i + 1], pad=20, fontweight='bold')
            ax.set_xlabel(rf'{name[0, 0]} ($\mathrm{{{name[1, 0]}}}$)', fontweight='bold')
            ax.set_ylabel(rf'{name[0, i + 1]} ($\mathrm{{{name[1, i + 1]}}}$)', fontweight='bold')

            offset = ax.yaxis.get_offset_text()
            transform = offset.get_transform()
            offset.set_transform(
                transform + plt.matplotlib.transforms.ScaledTranslation(0, 5 / 72., fig.dpi_scale_trans))
        plt.tight_layout()
        plt.savefig(file_directory + f'/curve group at {time_current:g}s.png', dpi=100, bbox_inches='tight')
        plt.close()


def initial_schematic(coordinates, radii, domain):
    file_directory = os.path.join(Figures_directory, "Initial Schematic" )
    if not os.path.exists(file_directory):
        os.makedirs(file_directory)
    with plt.rc_context(
            {'font.family': 'serif', 'font.serif': ['Times New Roman'], 'font.weight': 'bold', 'font.size': 32}):
        fig, ax = plt.subplots(figsize=(12, 9), dpi=100)
        ax.spines['top'].set_linewidth(3)
        ax.spines['bottom'].set_linewidth(3)
        ax.spines['left'].set_linewidth(3)
        ax.spines['right'].set_linewidth(3)

        ax.set_aspect('equal', 'box')
        ax.set_xlim(0, domain[0])
        ax.set_ylim(0, domain[1])


        circle = []
        for i in range(len(coordinates)):
            circle.append(plt.Circle((coordinates[i][0], coordinates[i][1]), radii[i], edgecolor='black',fill=True, facecolor='white', linewidth=3, zorder=2))
            plt.text(coordinates[i][0], coordinates[i][1], rf"$\eta_{{{i + 1}}}$", fontsize=32, ha='center', va='center', zorder=3)
            ax.add_artist(circle[i])

        ax.grid(True, linestyle='--', linewidth=3, zorder=1)
        plt.tick_params(axis='both', direction='in', width=3, which='both', pad=10)  # 设置刻度朝内，边框厚度为 2

        plt.xlabel('X-axis', fontweight='bold')
        plt.ylabel('Y-axis', fontweight='bold')
        plt.title('Multi-particle Sintering', pad=20, fontweight='bold')
        plt.savefig(file_directory + f'/Initial Schematic.png', dpi=100, bbox_inches='tight')
        plt.close()


def plot_function_on_mesh(fenics_function, mesh, time_current, title):
    file_directory = os.path.join(Figures_directory, title)
    if not os.path.exists(file_directory):
        os.makedirs(file_directory)
    """
    Visualize a FEniCS function on a triangular mesh with continuous color distribution.

    Parameters:
    fenics_function (Function): The FEniCS function to visualize.
    mesh (Mesh): The FEniCS mesh associated with the function.
    title (str): Title of the plot.
    cmap (str): Color map to use for visualization.
    levels (int): Number of levels for contour smoothing.
    """
    # 获取网格的顶点和三角形元素
    coordinates = mesh.coordinates()
    triangles = mesh.cells()

    # 提取函数在顶点上的值
    values = fenics_function.compute_vertex_values(mesh)
    # 获取值的最小值和最大值
    vmin = values.min()
    vmax = values.max()

    # 使用matplotlib的Triangulation来绘制三角形网格
    triang = Triangulation(coordinates[:, 0], coordinates[:, 1], triangles)

    with plt.rc_context({'font.family': 'serif', 'font.serif': ['Arial'], 'font.weight': 'regular', 'font.size': 32}):
        fig, ax = plt.subplots(figsize=(12, 9), dpi=100)

        ax.set_aspect('equal', 'box')
        # tric = plt.tricontourf(triang, values, levels=500, cmap='coolwarm')  # 通过增加levels来细化插值
        tric = plot(fenics_function, title=f"{title} at {time_current:g}s", levels=500, cmap='coolwarm')  # 通过增加levels来细化插值

        # 隐藏坐标轴刻度
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        divider = make_axes_locatable(plt.gca())
        width = axes_size.AxesY(ax, aspect=1. / 20)
        pad = axes_size.Fraction(0.5, width)
        cax = divider.append_axes("right", size=width, pad=pad)
        cbar = plt.colorbar(tric, cax=cax)
        # 设置颜色条的刻度
        ticks = np.linspace(vmin, vmax, 6)  # 在vmin和vmax之间生成6个均匀分布的值
        cbar.set_ticks(ticks)

        plt.tight_layout()
        plt.savefig(file_directory + f'/{title} at {time_current:g}s.png', dpi=100, bbox_inches='tight')
        plt.close()



#############################################################################################################
# Main Part
#############################################################################################################

# Read the input file ****************************************************************************************
parameters_input = read_input_file("input_polyparticle_reboot.txt")
# Phase-field parameters
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
# Mechanics parameters
muSurf = float(parameters_input.get("Surface viscosity"))
muGB = float(parameters_input.get('GB viscosity'))
muBulk = float(parameters_input.get('Bulk viscosity'))
viscosity_ratio = float(parameters_input.get("Viscosity Ratio"))
# Model parameters
radius_particle = float(parameters_input.get("Particle Radius"))
timesCharacteristic = float(parameters_input.get("Characteristic Time"))
ratio_mesh = float(parameters_input.get("MeshRatio"))
particle_number_x = int(parameters_input.get("ParticleNumberOfX"))
particle_number_y = int(parameters_input.get("ParticleNumberOfY"))
dt = float(parameters_input.get("TimeStep"))
# timeInitial = float(parameters_input.get("InitialTime"))
NumberOfTimeStep = int(parameters_input.get('NumberOfTimeStep'))
counterStepInitial = int(parameters_input.get('InitialStepCounter'))
frequencyOutput = int(parameters_input.get('FrequencyOutPut'))
# Algorithmic parameters
theta = float(parameters_input.get("theta"))
# Remark
remark = str(parameters_input.get("Remark"))

energySurfSpeciNormalized = (sqrt(2 * kc * a) / 6)
thicknessSurfaceNormalized = sqrt(8 * kc / a)
thicknessGbNormalized = sqrt(4 * keta / (3 * b))
# The parameter to control the smoothness
omega = 0.05  # The parameter to control the smoothness of the particle
epsilon = thicknessSurfaceNormalized / (2 * np.sqrt(2) * np.arctanh(1 - 2 * omega))
# epsilon = 1

# Define the output directory *********************************************************************************
# Creat the output folder
start_running_time = datetime.datetime.now()
primary_directory = os.getcwd() + "/MPSS_reboot"
if not os.path.exists(primary_directory):
    os.makedirs(primary_directory)
all_contents = os.listdir(primary_directory)
subdirectories = [content for content in all_contents if os.path.isdir(os.path.join(primary_directory, content))]
num_subdirectories = len(subdirectories)
time_stamp = start_running_time.strftime("%B %d, %Y, %H") + ":" + datetime.datetime.now().strftime(
    "%M") + ":" + datetime.datetime.now().strftime("%S")

timeInitial = counterStepInitial * dt
if timeInitial == 0:
    filename = 'MPSS-reboot-' + str(num_subdirectories + 1) + ' (' + datetime.datetime.now().strftime(
        "%B %d, %Y, %H-%M-%S") + ")"
else:
    filename = str(parameters_input.get("Job For Continue"))

output_document_type = ["Log", "Error", "Codes", "Input", "Output", "Figures", "Data"]
for i in output_document_type:
    globals()[i + '_directory'] = primary_directory + '/' + filename + '/' + i
    if not os.path.exists(globals()[i + '_directory']):
        os.makedirs(globals()[i + '_directory'], exist_ok=True)

# Creat the readme file ***************************************************************************************
readme_file_path = primary_directory + '/' + filename + '/Readme.txt'

# Configure the log file *************************************************************************************
log_file_path = os.path.join(Log_directory + '/log.log')
error_file_path = os.path.join(Error_directory + '/error.log')
# Configure log file handler
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()
log_file_handler = logging.FileHandler(str(log_file_path))
log_file_handler.setLevel(logging.INFO)
logger.addHandler(log_file_handler)
# Configure error file handler
error_file_handler = logging.FileHandler(str(error_file_path))
error_file_handler.setLevel(logging.ERROR)
error_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
error_file_handler.setFormatter(error_formatter)
logger.addHandler(error_file_handler)

# Set the drawing font ****************************************************************************************
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic',
    'mathtext.bf': 'Times New Roman:bold',
    'axes.unicode_minus': False
})

# Define denominator regularization ***************************************************************************
denim_reg = lambda x: x if abs(x) > 1e-12 else 1e-12

# Copy the code file to the output directory ******************************************************************
print("the code being executed:", sys.argv)
python_script_name = os.path.basename(sys.argv[0])
shutil.copy(python_script_name, os.path.join(Codes_directory, python_script_name))
shutil.copy("input_polyparticle_reboot.txt", os.path.join(Input_directory, "input_polyparticle_reboot.txt"))

# Input parameter processing **********************************************************************************
# Calculate the total number of particles and diameter of particles
Np = particle_number_y * particle_number_x
particle_size = radius_particle * 2
# Calculate the domain size, element number and specific grain boundary data_curve
domain_size = [float((particle_number_x + 2) * particle_size), float((particle_number_y + 2) * particle_size)]
ElementNumber = [int(domain_size[0] * ratio_mesh), int(domain_size[1] * ratio_mesh)]
gamma_gb = np.sqrt(4 * b * keta / 3)
# Create the phase-field model
pde = SolidStateSintering(a, b, kc, keta, Dsf, Dgb, Dvol, L, zeta, Np, Neps, muSurf, muGB, muBulk, viscosity_ratio)

# Obtain the particle centers and radii **********************************************************************
particle_centers, particle_radii = particle_centers_without_template(radius_particle, Np, particle_number_x,
                                                                     particle_number_y, domain_size)

# Calculate process ******************************************************************************************
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True

# Creat mesh
mesh = RectangleMesh(Point(0, 0), Point(domain_size[0], domain_size[1]), ElementNumber[0], ElementNumber[1], 'crossed')

# Define the function space
P1 = FiniteElement('CG', triangle, 1)
C_Space = FunctionSpace(mesh, MixedElement([P1, P1]))
Eta_Space = FunctionSpace(mesh, P1)

PV = VectorElement('CG', triangle, 2)
VS = FunctionSpace(mesh, MixedElement([PV, P1]))

PS = TensorElement('CG', triangle, 1)
WS = FunctionSpace(mesh, PS)

# Define the test and trial functions
c_trial, mu_trial = TrialFunctions(C_Space)
c_test, mu_test = TestFunctions(C_Space)
eta_trial = TrialFunction(Eta_Space)
eta_test = TestFunction(Eta_Space)

# Define variables
u_new = Function(C_Space)
eta_new = [Function(Eta_Space) for i in range(pde.Np)]
u_prev = Function(C_Space)
eta_prev = [Function(Eta_Space) for i in range(pde.Np)]

# Split the mixed functions
c_new, mu_new = split(u_new)
c_prev, mu_prev = split(u_prev)

# Split the test and trial functions
v_trial, p_trial = TrialFunctions(VS)
v_test, p_test = TestFunctions(VS)
v_combined = Function(VS)
v, p = split(v_combined)


# Define boundary condition ***********************************************************************************
def boundary(x, on_boundary):
    return on_boundary


bc = DirichletBC(VS.sub(0), Constant((0, 0)), boundary)

# Define initial value ***************************************************************************************
# Define initial value of eta
eta_initial = [VoronoiParticle(degree=1) for k in range(pde.Np)]
for k in range(pde.Np):
    eta_initial[k].others = particle_centers.copy()
    eta_initial[k].center = eta_initial[k].others.pop(k)
    eta_initial[k].epsilon = epsilon

# Define initial value of C
c_initial = Expression(
    '+'.join([circle_string_expression(particle_radii[k], particle_centers[k]) for k in range(pde.Np)]),
    degree=1, epsilon=Constant(epsilon))

# Define initial value of mu
mu_initial = Constant(0)

# Interpolate the initial value to function space
c0 = interpolate(c_initial, C_Space.sub(0).collapse())
mu0 = interpolate(mu_initial, C_Space.sub(1).collapse())
eta0 = [interpolate(eta_initial[k], Eta_Space) for k in range(pde.Np)]
c_mid = (1.0 - theta) * c_prev + theta * c_trial

# Define the weak form ****************************************************************************************
# The weak form of c in CH equation
c_term1 = c_trial * c_test * dx
c_term2 = - c_prev * c_test * dx
c_term3 = dt * dot(pde.M(c_prev, eta_prev) * grad(mu_trial), grad(c_test)) * dx
c_term4 = - dt * dot(v * c_trial, grad(c_test)) * dx
c_WeakForm = c_term1 + c_term2 + c_term3 + c_term4
# The weak form of mu in CH equation
mu_term1 = mu_trial * mu_test * dx
mu_term2 = - (pde.kc * dot(grad(c_mid), grad(mu_test)) * dx)
mu_term3 = - (pde.fmu(c_prev, eta_prev) + pde.c_stabilizer * (c_trial - c_prev)) * mu_test * dx
mu_WeakForm = mu_term1 + mu_term2 + mu_term3
# Combine the weak form of CH equation
CH_WeakForm = c_WeakForm + mu_WeakForm
# Split the weak form of CH equation
CH_WeakForm_L, CH_WeakForm_R = lhs(CH_WeakForm), rhs(CH_WeakForm)

# Define the weak form of eta
eta_WeakForm = []
for k in range(pde.Np):
    term1 = dot(eta_trial, eta_test) * dx
    term2 = - dot(eta_prev[k], eta_test) * dx
    term3 = dt * dot(v, grad(eta_trial)) * eta_test * dx
    term4 = dt * pde.L * inner(pde.keta * pde.N(c_prev) * grad((1.0 - theta) * eta_prev[k] + theta * eta_trial),
                               grad(eta_test)) * dx
    term5 = dt * pde.L * pde.N(c_prev) * pde.dS(eta_prev, k) * eta_test * dx
    term6 = dt * pde.eta_stabilizer * (eta_trial - eta_prev[k]) * eta_test * dx
    # Combine the weak form of eta
    eta_WeakForm.append(term1 + term2 + term3 + term4 + term5 + term6)
# Split the weak form of eta
eta_WeakForm_L = [lhs(eta_WeakForm[k]) for k in range(pde.Np)]
eta_WeakForm_R = [rhs(eta_WeakForm[k]) for k in range(pde.Np)]

# Define the weak form of Stokes equation
stokes_term1 = inner(pde.sigma(v_trial, c_new, eta_new), grad(v_test)) * dx
stokes_term2 = div(v_test) * p_trial * dx + p_test * div(v_trial) * dx
stokes_term3 = inner(pde.interface_stress(c_new, eta_new), grad(v_test)) * dX
stokes_WeakForm = stokes_term1 + stokes_term2 - stokes_term3
stokes_WeakForm_L, stokes_WeakForm_R = lhs(stokes_WeakForm), rhs(stokes_WeakForm)

# Define the weak form of stress equation
stress_test = TestFunction(WS)
stress_trial = TrialFunction(WS)
stress_L = inner(stress_trial, stress_test) * dx
stress_R_term1 = pde.sigma(v, c_new, eta_new) * p * Identity(2)
stress_R_term2 = - (0.5 * pde.kc * dot(grad(c_new), grad(c_new)) + pde.f(c_new) + pde.N(c_new) * (
        pde.S(eta_new) + pde.grad_eta(eta_new))) * Identity(2)
stress_R = inner(stress_R_term1 + stress_R_term2, stress_test) * dx

# Define the weak form of data_curve
""" This weak form just used to calculate the data_curve of the system """
surface_energy = (0.5 * pde.kc * dot(grad(c_new), grad(c_new)) + pde.f(c_new)) * dx
grain_boundary_energy = (pde.N(c_new) * (pde.S(eta_new) + pde.grad_eta(eta_new))) * dx
E = surface_energy + grain_boundary_energy

# Define the initial expression form of eta
eta_pure_L = eta_trial * eta_test * dx
eta_pure_R = sum([eta_new[i] ** 2 for i in range(pde.Np)]) * eta_test * dx

# Define the final expression form of eta
eta_wr_L = eta_trial * eta_test * dx
eta_wr_R = (c_new ** 2) * sum([eta_new[i] ** 2 for i in range(pde.Np)]) * eta_test * dx

# Reset initial conditions ************************************************************************************
PhaseFields_params = {'linear_solver': 'tfqmr', 'preconditioner': 'ilu'}
Stokes_params = {'linear_solver': 'mumps'}

timeCurrent = timeInitial
counterStep = counterStepInitial
Time_Simulation = NumberOfTimeStep * dt

# Set the output file for reprocessing and output results of initial timestep ***********************************
xdmf_file = XDMFFile(MPI.comm_world, Output_directory + f"/solution_{counterStep:06}.xdmf")
xdmf_file.parameters["flush_output"] = True
xdmf_file.parameters["functions_share_mesh"] = True

# Set output data for curve ***********************************************************************************
data_curve = np.empty((NumberOfTimeStep + 1, 9))
# header and unit of the data_curve
name_unit = np.array(
    [['Time', 'Surface Energy', 'Grain Boundary Energy', 'Total Free Energy', 'Area of Grain Boundary',
      'Normalized '
      'Time',
      'Normalized Surface Energy', 'Normalized Grain Boundary Energy', 'Normalized Total Free Energy'],
     ['s', 'J', 'J', 'J', 'm^2', ' ', ' ', ' ', ' ']])

# 标志位，指示是否收到中断信号
interrupted = False
# 注册信号处理程序
signal.signal(signal.SIGINT, signal_handler)

try:
    if timeInitial == 0:
        # Write the input parameters to the log file *******************************************************************
        logging.info("======================================")
        logging.info(f"Remark: {remark}")
        logging.info("#################################################")
        logging.info(f"TimeStep: {dt}")
        logging.info(f"InitialTime: {timeInitial}")
        logging.info(f"NumberOfTimeStep: {NumberOfTimeStep}")
        logging.info(f"InitialStepCounter: {counterStepInitial}")
        logging.info(f"FrequencyOutPut: {frequencyOutput}")
        logging.info("#################################################")
        # Phase-field parameters
        logging.info("# Phase-field parametersInput")
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
        logging.info("#################################################")
        logging.info("# Mechanics Parameters")
        logging.info(f"Surface viscosity: {muSurf}")
        logging.info(f"GB viscosity: {muGB}")
        logging.info(f"Bulk viscosity: {muBulk}")
        logging.info(f"Viscosity Ratio: {viscosity_ratio}")
        # Computational Setting
        logging.info("#################################################")
        logging.info("# Computational Setting")
        logging.info(f"Particle Radius: {radius_particle}")
        logging.info(f"Particle Number: {Np}")
        logging.info(f"Characteristic time: {timesCharacteristic}")
        logging.info(f"Dimension X: {domain_size[0]}")
        logging.info(f"Dimension Y: {domain_size[1]}")
        logging.info(f"NumberOfElement X: {ElementNumber[0]}")
        logging.info(f"NumberOfElement Y: {ElementNumber[1]}")
        logging.info(f"MeshRatio: {ratio_mesh}")
        logging.info(f"The total number of elements: {ElementNumber[0] * ElementNumber[1]}")
        logging.info(f"Number of particles in the x-direction is {particle_number_x}")
        logging.info(f"Number of particles in the y-direction is {particle_number_y}")
        # Normalized Materials Properties
        logging.info("#################################################")
        logging.info("# Normalized Materials Properties")
        logging.info(f"specfic surface data_curve: {energySurfSpeciNormalized}")
        logging.info(f"surface thickness: {thicknessSurfaceNormalized}")
        logging.info(f"Grain boundary thickness: {thicknessGbNormalized}")
        # Algorithmic parameters
        logging.info("#################################################")
        logging.info("# Algorithmic parameters")
        logging.info(f"theta: {theta}")
        logging.info(f"epsilon: {epsilon}")
        logging.info("#################################################")

        # Set the initial time, counter step and total simulation time *************************************************
        with open(readme_file_path, 'w', encoding='utf-8') as file:
            file.write(
                f"####################################################################\n"
                f"This is the computing log of phase-field simulation for poly particle sintering.\n"
                f"Operated by Bo Qian at Shanghai University\n"
                f"Date: {time_stamp}\n"
                f"####################################################################\n\n"
                f"\n####################################################################\n"
                f"Starting time : {start_running_time.strftime('%B %d, %Y, %H:%M:%S')}"
                f"                                                                          Test-{num_subdirectories + 1}\n"
                f"**************************************************************************************************\n"
                f"Remark : {remark}\n"
                f"Time step : {dt}s\n"
                f"Number of time steps : {NumberOfTimeStep}\n"
                f"Number of mesh element : {ElementNumber[0]}*{ElementNumber[1]}={ElementNumber[0] * ElementNumber[1]}\n"
                f"Output frequency : once every {frequencyOutput} time steps\n"
                f"Number of particles in the x-directio : {particle_number_x}\n"
                f"Number of particles in the y-directio : {particle_number_y}\n"
                f"Number of particle : {pde.Np}\n"
            )

        # 重置初始条件 ************************************************************************************************
        print(f"\nTime step: {counterStep}, Time: {timeCurrent:g}s, Time of Simulation: {Time_Simulation:g}s")
        assign(u_new, [c0, mu0])
        u_prev.assign(u_new)
        for i in range(pde.Np):
            assign(eta_new[i], eta0[i])
            eta_prev[i].assign(eta_new[i])

        print("----- Solving Stokes equation in of initial time step -----")
        assign(v_combined,
               [interpolate(Constant((0, 0)), VS.sub(0).collapse()), interpolate(Constant(0), VS.sub(1).collapse())])
        solve(stokes_WeakForm_L == stokes_WeakForm_R, v_combined, bc, solver_parameters=Stokes_params)

        # Define the output parameters ********************************************************************************

        c_wr = u_new.split()[0]
        c_wr.rename('C', 'C')
        mu_wr = u_new.split()[1]
        mu_wr.rename('mu', 'mu')
        eta_pure = Function(Eta_Space)
        eta_pure.rename('eta_pure', 'eta_pure')
        for i in range(pde.Np):
            eta_new[i].rename(f'eta_new[{i}]', f'eta_new[{i}]')
        eta_wr = Function(Eta_Space)
        eta_wr.rename('eta', 'eta')
        v_wr = v_combined.split()[0]
        v_wr.rename('v', 'v')
        p_wr = v_combined.split()[1]
        p_wr.rename('p', 'p')
        stress_wr = Function(WS)
        stress_wr.rename('stress', 'stress')

        # Output the header and unit to csv file
        data_curve_path = os.path.join(Data_directory, 'various_data.csv')
        with open(data_curve_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            for row in name_unit:
                writer.writerow(row)

        print("----- Outputting results -----")
        solve(eta_pure_L == eta_pure_R, eta_pure)
        xdmf_file.write(eta_pure, timeCurrent)
        xdmf_file.write(c_wr, timeCurrent)
        solve(eta_wr_L == eta_wr_R, eta_wr)
        xdmf_file.write(eta_wr, timeCurrent)
        xdmf_file.write(v_wr, timeCurrent)
        solve(stress_L == stress_R, stress_wr)
        xdmf_file.write(stress_wr, timeCurrent)

        # Save various data or initial timestep to csv file ************************************************************
        normalized_time = timeCurrent / timesCharacteristic + 1e-15
        data_curve[counterStep, 0] = timeCurrent
        data_curve[counterStep, 1] = assemble(surface_energy) * a * 1e-16 / 12
        data_curve[counterStep, 2] = assemble(grain_boundary_energy) * a * 1e-16 / 12
        data_curve[counterStep, 3] = assemble(E) * a * 1e-16 / 12
        data_curve[counterStep, 4] = data_curve[counterStep, 2] * a * 1e-16 / (gamma_gb * 12)
        data_curve[counterStep, 5] = normalized_time
        data_curve[counterStep, 6] = assemble(surface_energy)
        data_curve[counterStep, 7] = assemble(grain_boundary_energy)
        data_curve[counterStep, 8] = assemble(E)

        with open(data_curve_path, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [timeCurrent, data_curve[counterStep, 1], data_curve[counterStep, 2], data_curve[counterStep, 3],
                 data_curve[counterStep, 4], data_curve[counterStep, 5], data_curve[counterStep, 6],
                 data_curve[counterStep, 7], data_curve[counterStep, 8]])

        print(f"the total free energy at time {timeCurrent:g}s is {data_curve[counterStep, 3]:.4e}J\n"
              f"the surface energy at time {timeCurrent:g}s is {data_curve[counterStep, 1]:.4e}J\n"
              f"the grain boundary energy at time {timeCurrent:g}s is {data_curve[counterStep, 2]:.4e}J\n"
              f"the area of grain boundary at time {timeCurrent:g}s is {data_curve[counterStep, 4]:.4e}m^2\n"
              f"the surface energy at time {timeCurrent:g}s is {data_curve[counterStep, 6]:.4f}\n"
              f"the grain boundary energy at time {timeCurrent:g}s is {data_curve[counterStep, 7]:.4f}\n"
              f"the total free energy at time {timeCurrent:g}s is {data_curve[counterStep, 8]:.4f}"
              )

        print("----- Plotting figures of C, eta, eta_pure -----")
        plot_function_on_mesh(c_wr, mesh, timeCurrent, "C")
        plot_function_on_mesh(eta_wr, mesh, timeCurrent, "eta")
        plot_function_on_mesh(eta_pure, mesh, timeCurrent, "eta_pure")
        print("----- Plotting figures of v and stress -----")
        initial_schematic(particle_centers, particle_radii, domain_size)
        plot_function(v_wr, os.path.join(Figures_directory, 'v'), timeCurrent)
        plot_function(stress_wr, os.path.join(Figures_directory, 'stress'), timeCurrent)

    else:
        # Write the input parameters to the log file *******************************************************************
        logging.info("======================================")
        logging.info(f"Reboot Stage")
        logging.info(f"Remark: {remark}")
        logging.info("#################################################")
        logging.info(f"TimeStep: {dt}")
        logging.info(f"InitialTime: {timeInitial}")
        logging.info("Reboot Continue: {}".format(parameters_input.get("Job For Continue")))
        logging.info(f"NumberOfTimeStep: {NumberOfTimeStep}")
        logging.info(f"InitialStepCounter: {counterStepInitial}")
        logging.info(f"FrequencyOutPut: {frequencyOutput}")
        logging.info("#################################################")

        # Output the pre-processing parameters to readme file **********************************************************
        with open(readme_file_path, 'a', encoding='utf-8') as file:
            file.write(
                f"\nReboot Stage\n"
                f"####################################################################\n"
                f"Starting time : {start_running_time.strftime('%B %d, %Y, %H:%M:%S')}"
                f"                                                                          Test-{num_subdirectories}\n"
                f"**************************************************************************************************\n"
                f"Start time of simulation : {timeCurrent:g}s\n"
                f"Start time step of simulation : {counterStep}\n"
                f"End time of simulation : {Time_Simulation:g}s\n"
                f"End time step of simulation : {NumberOfTimeStep}\n"
                f"Remark : {remark}\n"
                f"Time step : {dt}s\n"
                f"Number of time steps : {NumberOfTimeStep}\n"
                f"Number of mesh element : {ElementNumber[0]}*{ElementNumber[1]}={ElementNumber[0] * ElementNumber[1]}\n"
                f"Output frequency : once every {frequencyOutput} time steps\n"
                f"Number of particles in the x-directio : {particle_number_x}\n"
                f"Number of particles in the y-directio : {particle_number_y}\n"
                f"Number of particle : {pde.Np}\n"
            )



        data_curve_path = Data_directory + '/various_data.csv'

        origin_data = []
        with open(data_curve_path, mode='r', newline='') as csv_file:
            reader = csv.reader(csv_file)
            next(reader)
            next(reader)
            for row in reader:
                origin_data.append(row)
        data_curve_origin = np.array(origin_data)

        data_curve[:counterStep + 1] = data_curve_origin[:counterStep + 1]
        with open(data_curve_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            for row in name_unit:
                writer.writerow(row)
            for row in data_curve_origin[:counterStep + 1]:
                writer.writerow(row)

        checkpoint_file = HDF5File(MPI.comm_world, Output_directory + f"/checkpoint_{counterStep}.h5", "r")

        u_input = Function(C_Space)
        eta_input = [Function(Eta_Space) for i in range(pde.Np)]
        v_combined_input = Function(VS)

        checkpoint_file.read(u_input, "CheckPoint/u")
        for i in range(pde.Np):
            checkpoint_file.read(eta_input[i], f"CheckPoint/eta{i}")
        checkpoint_file.read(v_combined_input, "CheckPoint/v_combined")
        checkpoint_file.close()

        assign(u_new, u_input)
        for i in range(pde.Np):
            assign(eta_new[i], eta_input[i])

        assign(v_combined, v_combined_input)

        c_wr = u_new.split()[0]
        c_wr.rename('C', 'C')
        mu_wr = u_new.split()[1]
        mu_wr.rename('mu', 'mu')
        for i in range(pde.Np):
            eta_new[i].rename(f'eta_new[{i}]', f'eta_new[{i}]')
        eta_wr = Function(Eta_Space)
        eta_wr.rename('eta', 'eta')
        v_wr = v_combined.split()[0]
        v_wr.rename('v', 'v')
        p_wr = v_combined.split()[1]
        p_wr.rename('p', 'p')
        stress_wr = Function(WS)
        stress_wr.rename('stress', 'stress')

    # Time stepping *******************************************************************************************
    while timeCurrent < Time_Simulation:
        if interrupted:
            print(
                f"===== Exiting loop after current iteration at time step {counterStep}. =====\n"
                f"===== You can continue the simulation at time step {np.floor_divide(counterStep, frequencyOutput) * frequencyOutput} =====\n"
                f"===== You can continue the simulation of job {filename} ====="
            )
            break

        counterStep += 1
        quotient = counterStep // frequencyOutput
        timeCurrent = counterStep * dt
        normalized_time = timeCurrent / timesCharacteristic + 1e-15
        print(f"\nTime step: {counterStep}, Time: {timeCurrent:g}s, Time of Simulation: {Time_Simulation:g}s")

        u_prev.assign(u_new)
        for i in range(pde.Np):
            eta_prev[i].assign(eta_new[i])
        print("----- Solving CH equation -----")
        solve(CH_WeakForm_L == CH_WeakForm_R, u_new, solver_parameters=PhaseFields_params)
        print("----- Solving AC equation -----")
        for i in range(pde.Np):
            solve(eta_WeakForm_L[i] == eta_WeakForm_R[i], eta_new[i], solver_parameters=PhaseFields_params)
        print("----- Solving Stokes equation -----")
        solve(stokes_WeakForm_L == stokes_WeakForm_R, v_combined, bc, solver_parameters=Stokes_params)

        data_curve[counterStep, 0] = timeCurrent
        data_curve[counterStep, 1] = assemble(surface_energy) * a * 1e-16 / 12
        data_curve[counterStep, 2] = assemble(grain_boundary_energy) * a * 1e-16 / 12
        data_curve[counterStep, 3] = assemble(E) * a * 1e-16 / 12
        data_curve[counterStep, 4] = data_curve[counterStep, 2] * a * 1e-16 / (gamma_gb * 12)
        data_curve[counterStep, 5] = normalized_time
        data_curve[counterStep, 6] = assemble(surface_energy)
        data_curve[counterStep, 7] = assemble(grain_boundary_energy)
        data_curve[counterStep, 8] = assemble(E)

        print(f"the total free energy at time {timeCurrent:g}s is {data_curve[counterStep, 3]:.4e}J\n"
              f"the surface energy at time {timeCurrent:g}s is {data_curve[counterStep, 1]:.4e}J\n"
              f"the grain boundary energy at time {timeCurrent:g}s is {data_curve[counterStep, 2]:.4e}J\n"
              f"the area of grain boundary at time {timeCurrent:g}s is {data_curve[counterStep, 4]:.4e}m^2\n"
              f"the surface energy at time {timeCurrent:g}s is {data_curve[counterStep, 6]:.4f}\n"
              f"the grain boundary energy at time {timeCurrent:g}s is {data_curve[counterStep, 7]:.4f}\n"
              f"the total free energy at time {timeCurrent:g}s is {data_curve[counterStep, 8]:.4f}"
              )
        # Write the data for curve to csv file
        with open(data_curve_path, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [timeCurrent, data_curve[counterStep, 1], data_curve[counterStep, 2], data_curve[counterStep, 3],
                 data_curve[counterStep, 4], data_curve[counterStep, 5], data_curve[counterStep, 6],
                 data_curve[counterStep, 7], data_curve[counterStep, 8]])



        if np.mod(counterStep, frequencyOutput) == 0:

            if os.path.exists(Output_directory + f"/checkpoint_{counterStep - frequencyOutput}.h5") and counterStep != counterStepInitial:
                os.remove(Output_directory + f"/checkpoint_{counterStep - frequencyOutput}.h5")

            print("----- Outputting checkpoint -----")
            checkpoint_file = HDF5File(MPI.comm_world, Output_directory + f"/checkpoint_{counterStep}.h5", "w")
            checkpoint_file.write(u_new, "CheckPoint/u")
            for i in range(pde.Np):
                checkpoint_file.write(eta_new[i], f"CheckPoint/eta{i}")
            checkpoint_file.write(v_combined, "CheckPoint/v_combined")

            print("----- Outputting results -----")
            xdmf_file.write(c_wr, timeCurrent)
            solve(eta_wr_L == eta_wr_R, eta_wr)
            xdmf_file.write(eta_wr, timeCurrent)
            xdmf_file.write(v_wr, timeCurrent)
            solve(stress_L == stress_R, stress_wr)
            xdmf_file.write(stress_wr, timeCurrent)

            print("----- Plotting figures of C, eta, eta_pure -----")
            plot_function_on_mesh(c_wr, mesh, timeCurrent, "C")
            plot_function_on_mesh(eta_wr, mesh, timeCurrent, "eta")

            print("----- Plotting figures of v and stress -----")
            initial_schematic(particle_centers, particle_radii, domain_size)
            plot_function(v_wr, os.path.join(Figures_directory, 'v'), timeCurrent)
            plot_function(stress_wr, os.path.join(Figures_directory, 'stress'), timeCurrent)

            print("----- Plotting curve figures of different kind of data -----")
            plot_curve_group(data_curve, name_unit, counterStep, timeCurrent)
            plot_curve(data_curve[:counterStep + 1, 0], data_curve[:counterStep + 1, 1], rf"{name_unit[1, 1]}",
                       name_unit[0, 1], timeCurrent)
            plot_curve(data_curve[:counterStep + 1, 0], data_curve[:counterStep + 1, 2], rf"{name_unit[1, 2]}",
                       name_unit[0, 2], timeCurrent)
            plot_curve(data_curve[:counterStep + 1, 0], data_curve[:counterStep + 1, 3], rf"{name_unit[1, 3]}",
                       name_unit[0, 3], timeCurrent)
            plot_curve(data_curve[:counterStep + 1, 0], data_curve[:counterStep + 1, 4], rf"{name_unit[1, 4]}",
                       name_unit[0, 4], timeCurrent)
            print("Plot finished.")

except KeyboardInterrupt as e:
    logger.error(f"\n# Error\nThe simulation is interrupted by the user (KeyboardInterrupt).", exc_info=True)
    error_type = "KeyboardInterrupt"
    # Output the ending time to the readme file
    with open(readme_file_path, 'a', encoding='utf-8') as file:
        file.write(f"Error type: {error_type}\n")

except Exception as e:
    logger.error(f"\n# Error\nThe simulation is interrupted by an error.", exc_info=True)
    error_type = str(type(e).__name__)
    # Output the ending time to the readme file
    with open(readme_file_path, 'a', encoding='utf-8') as file:
        file.write(f"Error type: {error_type}\n")

finally:
    xdmf_file.close()
    checkpoint_file.close()

    # Calculate the running time **********************************************************************************
    end_running_time = datetime.datetime.now()
    total_time = end_running_time - start_running_time

    logging.info("#################################################")
    logging.info("# Computational time")
    logging.info(f"Starting time : {start_running_time.strftime('%B %d, %Y, %H:%M:%S')}")
    logging.info(f"Ending time : {end_running_time.strftime('%B %d, %Y, %H:%M:%S')}")
    logging.info(f"Total running time: {str(total_time)}")
    logging.info("#################################################")
    logging.info(" ")

    # Output the ending time to the readme file
    with open(readme_file_path, 'a', encoding='utf-8') as file:
        file.write(
            f"**************************************************************************************************\n"
            f"Ending time : {end_running_time.strftime('%B %d, %Y, %H:%M:%S')}\n"
            f"Total running time: {str(total_time)}\n"
            f"===========================================================\n"
        )
