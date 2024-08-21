# ************************************************************************************************************
# Description: This is a phase-field simulation for poly particle viscosity sintering.
# Wrote by Bo Qian at Shanghai University
# Date: 2024-06-23
# 1st update: 2024-07-01
# 2nd update: 2024-07-15
# ************************************************************************************************************
import logging
import csv
import sys
import signal

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import datetime
import numpy as np
import time
import shutil
from fenics import *

# Read the input file
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


def line_integral(u, A, B, num_segments):
    """
    Compute the line integral of a scalar function over a line segment.

    Parameters:
        u: Expression or Function
            The scalar function to be integrated.
        A: numpy.ndarray
            Starting point of the line segment.
        B: numpy.ndarray
            Ending point of the line segment.
        num_segments: int
            Number of elements to partition the line segment into.

    Returns:
        float
            The value of the line integral.
    """
    # Input validation
    assert isinstance(u, (Expression, Function)), "Function u must be an Expression or Function object."
    assert len(A) == len(B), "Vectors A and B must have the same dimension."
    assert np.linalg.norm(A - B) > 0, "Vectors A and B must not be identical."
    assert num_segments > 0, "Number of elements (num_segments) must be positive."

    # Generate mesh coordinates along the line segment
    mesh_coordinates = np.array([A + t * (B - A) for t in np.linspace(0, 1, num_segments + 1)])

    # Create a mesh
    mesh = Mesh()

    # Define the topological and geometric dimensions
    tdim = 1
    gdim = len(A)

    # Create mesh vertices and cells
    editor = MeshEditor()
    editor.open(mesh, 'interval', tdim, gdim)
    editor.init_vertices(len(mesh_coordinates))
    editor.init_cells(len(mesh_coordinates) - 1)

    # Add vertices
    for i, vertex in enumerate(mesh_coordinates):
        editor.add_vertex(i, vertex)

    # Add cells
    for i in range(len(mesh_coordinates) - 1):
        editor.add_cell(i, np.array([i, i + 1], dtype='uintp'))

    # Close the mesh editor
    editor.close()

    # Define function space
    V = FunctionSpace(mesh, u.ufl_element().family(), u.ufl_element().degree())

    # Interpolate u onto function space
    u_interpolated = interpolate(u, V)

    # Compute line integral
    integral_value = assemble(u_interpolated * dx(domain=mesh))

    return integral_value


def circle_string_expression(radius, center):
    return (f"(1 - tanh((sqrt(pow((x[0]-{center[0]}), 2) + pow((x[1]-{center[1]}), 2)) - {radius}) / (sqrt("
            f"2)*epsilon)))/2")


# Here, various model parameters are defined::
class viscousSintering:
    def __init__(self, a, kc, D, Neps, muBulk, viscosity_ratio, Np):
        self.a = Constant(a)
        self.kc = Constant(kc)
        self.D = Constant(D)
        self.Neps = Constant(Neps)
        self.Np = Np
        self.muBulk = muBulk
        self.viscosity_ratio = viscosity_ratio
        self.c_stabilizer = Constant(2 * a)

    def N(self, C):
        return (C ** 2) * (1 + 2 * (1 - C) + (self.Neps) * (1 - C) ** 2)

    def f(self, C):
        return self.a * (C ** 2) * ((1 - C) ** 2)

    def df(self, C):
        return 2 * self.a * C * (1 - C) * (1 - 2 * C)

    def M(self, C):
        return self.D

    def fmu(self, C):
        return self.df(C)

    def sigma(self, u, C):
        # for this version it is important to keep viscosity_ratio<1
        # the defined sigma here does not include contribution from Lagrangian multiplier
        return ((self.viscosity_ratio + (1 - self.viscosity_ratio) * self.N(C)) * self.muBulk) * (grad(u) + grad(u).T)

    def interface_stress(self, C):
        return self.kc * outer(grad(C), grad(C))


def particle_centers_without_template(radius_particle, particle_number_total, number_x, number_y, domain):
    particle_radius = [radius_particle] * particle_number_total
    particle_centers_coordinate = []
    for j in range(number_y):
        for i in range(number_x):
            x_coordinate = int(domain[0] / 2 + (i + (1 - number_x) / 2) * radius_particle * 2)
            y_coordinate = int(domain[1] / 2 + (j + (1 - number_y) / 2) * radius_particle * 2)
            particle_centers_coordinate.append([x_coordinate, y_coordinate])
    return particle_centers_coordinate, particle_radius

def signal_handler(sig, frame):
    global interrupted
    print("===== Received interrupt signal. Stopping after current iteration. =====")
    interrupted = True

# 绘图函数 ***************************************************************************************************
def plot_function(serial_number, variant, file_directory, time_current):
    if not os.path.exists(file_directory):
        os.makedirs(file_directory)
    with plt.rc_context({'font.family': 'serif', 'font.weight': 'regular', 'font.size': 24}):
        #'font.serif': ['Arial'],
        plt.figure(serial_number, figsize=(12, 9), dpi=100)
        plot(variant, title=f"{variant} at time {time_current}s", cmap='coolwarm')
        plt.savefig(file_directory + f'/{variant}_{time_current:g}s.png', dpi=100)
        plt.close()


def plot_curve(X, Y, title, name_x, name_y, time_current):
    file_directory = os.path.join(Figures_directory, title)
    if not os.path.exists(file_directory):
        os.makedirs(file_directory)

    with plt.rc_context(
            {'font.family': 'serif', 'font.weight': 'bold', 'font.size': 32}):
        # 'font.serif': ['Times New Roman'],
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
        plt.title(title, pad=20, fontweight='bold')
        plt.xlabel(rf'{name_x}', fontweight='bold')
        plt.ylabel(rf'{name_y}', fontweight='bold')
        plt.tight_layout()
        # plt.legend(fontsize='small')
        plt.savefig(file_directory + f'/{title}_{time_current:g}s.png', dpi=100, bbox_inches='tight')
        plt.close()


def plot_curve_group(data, name, counter_step_current, time_current):
    file_directory = os.path.join(Figures_directory, "Curve Group")
    line_width = 4
    if not os.path.exists(file_directory):
        os.makedirs(file_directory)
    with plt.rc_context(
            {'font.family': 'serif', 'font.weight': 'bold', 'font.size': 32}):
        #'font.serif': ['Times New Roman'],
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

#############################################################################################################
# Main Part
#############################################################################################################

# Read the input file ****************************************************************************************
# Copy the codes that you are running
parameters_input = read_input_file("input_polyparticle_viscosity_v1.txt")
# Phase-field parameters
a = float(parameters_input.get('Alpha'))
kc = float(parameters_input.get('KappaC'))
D = float(parameters_input.get('Relaxation Parameter'))
Neps = float(parameters_input.get('Epsilon'))
# Mechanics Parameters
muBulk = float(parameters_input.get('Bulk viscosity'))
viscosity_ratio = float(parameters_input.get('Viscosity Ratio'))
timeCharacteristic = float(parameters_input.get('Characteristic Time'))
# Computational Setting
radius_particle = float(parameters_input.get('Particle Radius'))
ratio_mesh = float(parameters_input.get("MeshRatio"))
particle_number_x = int(parameters_input.get("ParticleNumberOfX"))
particle_number_y = int(parameters_input.get("ParticleNumberOfY"))
dt = float(parameters_input.get('TimeStep'))
# timeInitial = float(parameters_input.get('InitialTime'))
NumberOfTimeStep = int(parameters_input.get('NumberOfTimeStep'))
counterStepInitial = int(parameters_input.get('InitialStepCounter'))
frequencyOutput = int(parameters_input.get('FrequencyOutPut'))
# Remark Information
remark = str(parameters_input.get('Remark'))
# Algorithmic parameters
theta = float(parameters_input.get('theta'))

# Define the output directory *********************************************************************************
# Define Output Folder
start_running_time = datetime.datetime.now()
primary_directory = os.getcwd() + "/PolyParticle_Viscosity_Simulation_v1"
if not os.path.exists(primary_directory):
    os.makedirs(primary_directory)
all_contents = os.listdir(primary_directory)
subdirectories = [content for content in all_contents if os.path.isdir(os.path.join(primary_directory, content))]
num_subdirectories = len(subdirectories)
time_stamp = start_running_time.strftime("%B %d, %Y, %H") + ":" + datetime.datetime.now().strftime(
    "%M") + ":" + datetime.datetime.now().strftime("%S")

timeInitial = counterStepInitial * dt
if timeInitial == 0:
    filename = 'PPVS-V1-' + str(num_subdirectories + 1) + ' (' + datetime.datetime.now().strftime(
        "%B %d, %Y, %H-%M-%S") + ")"
else:
    filename = str(parameters_input.get("Job For Continue"))

output_document_type = ["Log", "Error", "Codes", "Input", "Output", "Figures", "Data"]
for i in output_document_type:
    globals()[i + '_directory'] = primary_directory + '/' + filename + '/' + i
    if not os.path.exists(globals()[i + '_directory']):
        os.makedirs(globals()[i + '_directory'], exist_ok=True)

# Creat Readme File *******************************************************************************************
readme_file_path = primary_directory + '/' + filename + '/Readme.txt'

# Configure logging *******************************************************************************************
log_file_path = os.path.join(Log_directory + '/log.log')
error_file_path = os.path.join(Error_directory + '/error.log')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()
# Configure log file handler
log_file_handler = logging.FileHandler(str(log_file_path))
log_file_handler.setLevel(logging.INFO)
logger.addHandler(log_file_handler)
# Configure error file handler
error_file_handler = logging.FileHandler(str(error_file_path))
error_file_handler.setLevel(logging.ERROR)
error_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
error_file_handler.setFormatter(error_formatter)
logger.addHandler(error_file_handler)

# Setting the drawing font ***********************************************************************************
'''
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic',
    'mathtext.bf': 'Times New Roman:bold',
    'axes.unicode_minus': False  # 确保负号使用正确的字体
})
'''

# Copy the code file to the output directory ******************************************************************
print("the code being executed:", sys.argv)
python_script_name = os.path.basename(sys.argv[0])
shutil.copy(python_script_name, os.path.join(Codes_directory, python_script_name))
shutil.copy("input_polyparticle_viscosity_v1.txt", os.path.join(Input_directory, "input_polyparticle_viscosity_v1.txt"))

# Input parameter processing **********************************************************************************
# Calculate the total number of particles and diameter of particles
Np = particle_number_y * particle_number_x
particle_size = radius_particle * 2
domain_size = [float((particle_number_x + 2) * particle_size), float((particle_number_y + 2) * particle_size)]
ElementNumber = [int(domain_size[0] * ratio_mesh), int(domain_size[1] * ratio_mesh)]
pde = viscousSintering(a, kc, D, Neps, muBulk, viscosity_ratio, Np)
# Computing radius and location of particles *****************************************************************
particle_centers, particle_radii = particle_centers_without_template(radius_particle, Np, particle_number_x,
                                                                     particle_number_y, domain_size)

# Computing Process ******************************************************************************************
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
# tell the form to apply optimization strategies in the code generation
# phase and then use compiler optimization flags when compiling the
# generated C++ code. Using the option ``["optimize"] = True`` will
# generally result in faster code (sometimes orders of magnitude faster
# for certain operations, depending on the equation), but it may take
# considerably longer to generate the code and the generation phase may
# use considerably more memory.

# Create mesh and build function space
mesh = RectangleMesh(Point(0, 0), Point(domain_size[0], domain_size[1]), ElementNumber[0], ElementNumber[1],
                     'crossed')

# FunctionSpace consists of c, mu
P1 = FiniteElement("CG", triangle, 1)
# Mixed function space for c and mu from Cahn-Hilliard equation (CH)
SpaceMixedCH = FunctionSpace(mesh, MixedElement([P1, P1]))
# Linear function space
SpaceLinear = FunctionSpace(mesh, P1)

# Taylor-Hood function space for velocity
PV = VectorElement("CG", triangle, 2)
# For pressure
PP = FiniteElement("CG", triangle, 1)
# Mixed space for both velocity and pressure from stokes equation
SpaceMixedStokes = FunctionSpace(mesh, MixedElement([PV, PP]))

# Create a separate space to output the stress
PS = TensorElement("CG", triangle, 1)
SpaceTensor = FunctionSpace(mesh, PS)

# Trial and test functions of the mixed function space are now defined::

# Define trial and test functions
c_trial, mu_trial = TrialFunction(SpaceMixedCH)  # u_trail consists of c_trial and mu_trial
c_test, mu_test = TestFunctions(SpaceMixedCH)
c_auxiliary = TrialFunction(SpaceLinear)  # for later line integration
c_auxiliary_test = TestFunction(SpaceLinear)  # for later lin integration

# Define functions
u_new = Function(SpaceMixedCH)  # current solution of c and mu
u_prev = Function(SpaceMixedCH)  # solution from previous converged step

# Split mixed functions of c and mu
c_new, mu_new = split(u_new)
c_prev, mu_prev = split(u_prev)

# All functions for velocity space
v_trial, p_trial = TrialFunctions(SpaceMixedStokes)
v_test, p_test = TestFunctions(SpaceMixedStokes)
v_p_combined = Function(SpaceMixedStokes)
v_prev, p_prev = split(v_p_combined)


def boundary(x, on_boundary):
    return on_boundary


# Set up the Dirichlet boundary condition for velocity
bc = DirichletBC(SpaceMixedStokes.sub(0), Constant((0, 0)), boundary)

x, y = Expression('x[0]', degree=1), Expression('x[1]', degree=1)
# Two end points of the center line to calculate horizontal shrinkage
Point_A, Point_B = np.array([0, domain_size[1] / 2]), np.array([domain_size[0], domain_size[1] / 2])
# Two end points of the verticle line to calculate neck length
Point_C, Point_D = np.array([domain_size[0] / 2, 0]), np.array([domain_size[0] / 2, domain_size[1]])

# Initial conditions are created by using the class defined at the *********************************************
# beginning and then interpolating the initial conditions
# into a finite element space::
c_initial = Expression(
    '+'.join([circle_string_expression(particle_radii[k], particle_centers[k]) for k in range(pde.Np)]),
    degree=1, epsilon=Constant(1))

mu_initial = Constant(0)
c0 = interpolate(c_initial, SpaceMixedCH.sub(0).collapse())
mu0 = interpolate(mu_initial, SpaceMixedCH.sub(1).collapse())
c_mid = (1.0 - theta) * c_prev + theta * c_trial

# Define the weak forms **************************************************************************************
# The linear weak form of c and mu
cWeakForm = c_trial * c_test * dx - c_prev * c_test * dx \
            + dt * dot(pde.M(c_prev) * grad(mu_trial), grad(c_test)) * dx \
            - dt * dot(v_prev * c_trial, grad(c_test)) * dx
muWeakForm = mu_trial * mu_test * dx - pde.kc * dot(grad(c_mid), grad(mu_test)) * dx \
             - (pde.fmu(c_prev) + pde.c_stabilizer * (c_trial - c_prev)) * mu_test * dx
CH_WeakFormCombined = cWeakForm + muWeakForm
CH_WeakFormCombined_L, CH_WeakFormCombined_R = lhs(CH_WeakFormCombined), rhs(CH_WeakFormCombined)

# The weak form of velocity and pressure
Stokes_WeakFormCombined = inner(pde.sigma(v_trial, c_new), grad(v_test)) * dx + div(
    v_test) * p_trial * dx + p_test * div(v_trial) * dx \
                          - inner(pde.interface_stress(c_new), grad(v_test)) * dx
Stokes_WeakFormCombined_L, Stokes_WeakFormCombined_R = lhs(Stokes_WeakFormCombined), rhs(Stokes_WeakFormCombined)

c_auxiliary_L = c_auxiliary * c_auxiliary_test * dx

# define equations for physical stress
stress_test = TestFunction(SpaceTensor)
stress_trial = TrialFunction(SpaceTensor)
stress_L = inner(stress_trial, stress_test) * dx
stress_R = inner(pde.sigma(v_prev, c_new) + p_prev * Identity(2)
                 - (0.5 * pde.kc * dot(grad(c_new), grad(c_new)) + pde.f(c_new)) * Identity(2), stress_test) * dx

# Define the weak form of data_curve
""" This weak form just used to calculate the data_curve of the system """
E = (0.5 * pde.kc * dot(grad(c_new), grad(c_new)) + pde.f(c_new)) * dx

# block with solver and solution ******************************************************************************
# solver settings
CH_params = {'linear_solver': 'tfqmr', 'preconditioner': 'ilu'}
Stokes_params = {'linear_solver': 'mumps'}

# Variables for line integration
c_lineIntegration = Function(SpaceLinear)
timeCurrent = timeInitial
counterStep = counterStepInitial
timeSimulation = NumberOfTimeStep * dt

# Define result file ******************************************************************************************
xdmf_file = XDMFFile(MPI.comm_world, Output_directory + f"/solution_{counterStep:06}.xdmf")
xdmf_file.parameters["flush_output"] = True
xdmf_file.parameters["functions_share_mesh"] = True

# Set output data for curve ***********************************************************************************
various_data = np.empty((NumberOfTimeStep + 1, 8))
# header and unit of the various_data
name_unit = np.array([['Time', 'Total Free Energy', 'Shrinkage', 'Radius Neck', 'Normalized Time',
                       'Normalized Shrinkage', 'Normalized Neck Length', 'Neck Growth Rate']])

assign(u_new, [c0, mu0])
u_prev.assign(u_new)

# solve for c_new and store it in c_lineIntegration in order to use line_integral
solve(c_auxiliary_L == c_new * c_auxiliary_test * dx, c_lineIntegration,
      solver_parameters={'linear_solver': 'mumps'})

# do line integration of c_new over the line defined by Point_A and Point_B
L0 = line_integral(c_lineIntegration, Point_A, Point_B, ElementNumber[0])
various_data[counterStep, 0] = timeCurrent
various_data[counterStep, 1] = assemble(E)
various_data[counterStep, 2] = L0 - line_integral(c_lineIntegration, Point_A, Point_B, ElementNumber[0])
various_data[counterStep, 3] = 0.5 * line_integral(c_lineIntegration, Point_C, Point_D, ElementNumber[1])
timeValuesNormalized = timeCurrent / timeCharacteristic + 1e-15
various_data[counterStep, 4] = timeValuesNormalized
dL_Normalized = various_data[counterStep, 2] / L0
various_data[counterStep, 5] = dL_Normalized
neckNormalized = (various_data[counterStep, 3] / radius_particle) * 0.7071
neck_prev = various_data[counterStep, 3]
neck_current = various_data[counterStep, 3]
various_data[counterStep, 6] = neckNormalized
growthrate_Of_neck = (neck_current - neck_prev) / dt
various_data[counterStep, 7] = growthrate_Of_neck

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
        logging.info("# Phase-field parameters_input")
        logging.info(f"Alpha: {pde.a.values()[0]}")
        logging.info(f"Kappac: {pde.kc.values()[0]}")
        logging.info(f"Relaxation Parameter: {pde.D.values()[0]}")
        logging.info(f"Epsilon: {pde.Neps.values()[0]}")
        # Mechanics Parameters
        logging.info("#################################################")
        logging.info("# Mechanics Parameters")
        logging.info(f"Bulk viscosity: {muBulk}")
        logging.info(f"Viscosity Ratio: {viscosity_ratio}")
        # Computational Setting
        logging.info("#################################################")
        logging.info("# Computational Setting")
        logging.info(f"Particle Radius: {radius_particle}")
        logging.info(f"Particle Number: {Np}")
        logging.info(f"Characteristic Time : {timeCharacteristic}")
        logging.info(f"Dimension X: {domain_size[0]}")
        logging.info(f"Dimension Y: {domain_size[1]}")
        logging.info(f"NumberOfElement X: {ElementNumber[0]}")
        logging.info(f"NumberOfElement Y: {ElementNumber[1]}")
        logging.info(f"MeshRatio: {ratio_mesh}")
        logging.info(f"The total number of elements: {ElementNumber[0] * ElementNumber[1]}")
        logging.info(f"Number of particles in the x-direction is {particle_number_x}")
        logging.info(f"Number of particles in the y-direction is {particle_number_y}")
        # Algorithmic parameters
        logging.info("#################################################")
        logging.info("# Algorithmic parameters")
        logging.info(f"theta: {theta}")
        # Normalized Materials Properties *****************************************************************************
        energySurfSpeciNormalized = (sqrt(2 * pde.kc.values()[0] * pde.a.values()[0]) / 6)
        thicknessSurfaceNormalized = sqrt(8 * pde.kc.values()[0] / pde.a.values()[0])
        logging.info("#################################################")
        logging.info("# Normalized Materials Properties")
        logging.info(f"specfic surface energy: {energySurfSpeciNormalized}")
        logging.info(f"surface thickness: {thicknessSurfaceNormalized}")
        logging.info("#################################################")

        with open(readme_file_path, 'w', encoding='utf-8') as file:
            file.write(
                f"####################################################################\n"
                f"This is the computing log of phase-field simulation for poly particle viscosity sintering.\n"
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


        # Reset initial conditions ************************************************************************************
        print(f"\nTime step: {counterStep}, Time: {timeCurrent:.3f}, Time of Simulation : {timeSimulation}")

        various_data_path = os.path.join(Data_directory, 'various_data.csv')

        with open(various_data_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            for row in name_unit:
                writer.writerow(row)
            writer.writerow(
                [timeCurrent, various_data[counterStep, 1], various_data[counterStep, 2], various_data[counterStep, 3],
                 various_data[counterStep, 4], various_data[counterStep, 5], various_data[counterStep, 6],
                 various_data[counterStep, 7]], )

        print(f"The total free energy at time {timeCurrent:g}s is {various_data[counterStep, 1]:.4e}")

        # Reset initial conditions ************************************************************************************
        assign(v_p_combined, [interpolate(Constant((0, 0)), SpaceMixedStokes.sub(0).collapse()),
                              interpolate(Constant(0), SpaceMixedStokes.sub(1).collapse())])
        solve(Stokes_WeakFormCombined_L == Stokes_WeakFormCombined_R, v_p_combined, bc, solver_parameters=Stokes_params)

        # Output file
        cwr = u_new.split()[0]
        cwr.rename('C', 'C')
        vwr = v_p_combined.split()[0]
        vwr.rename('v', 'v')
        stress = Function(SpaceTensor)
        stress.rename('stress', 'stress')

        print('----- writing the data of initial time step to file -----')
        xdmf_file.write(cwr, timeCurrent)
        xdmf_file.write(vwr, timeCurrent)
        solve(stress_L == stress_R, stress, solver_parameters={'linear_solver': 'mumps'})
        xdmf_file.write(stress, timeCurrent)

        # Plot Figures of initial time step
        print("----- Plotting figures of C,  v and stress -----")
        initial_schematic(particle_centers, particle_radii, domain_size)
        plot_function(1, cwr, os.path.join(Figures_directory, 'C'), timeCurrent)
        plot_function(3, vwr, os.path.join(Figures_directory, 'v'), timeCurrent)
        plot_function(4, stress, os.path.join(Figures_directory, 'stress'), timeCurrent)

    else:
        # Write the input parameters to the log file *******************************************************************
        logging.info("======================================")
        logging.info(f"Reboot Stage")
        logging.info(f"Remark: {remark}")
        logging.info("#################################################")
        logging.info(f"TimeStep: {dt}")
        logging.info(f"InitialTime: {timeInitial}")
        logging.info(f"Reboot Continue: {parameters_input.get("Job For Continue")}")
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
                f"End time of simulation : {timeSimulation:g}s\n"
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

        various_data_path = os.path.join(Data_directory, 'various_data.csv')
        origin_data = []
        with open(various_data_path, mode='r', newline='') as csv_file:
            reader = csv.reader(csv_file)
            next(reader)
            for row in reader:
                origin_data.append(row)
        data_curve_origin = np.array(origin_data)

        various_data[:counterStep + 1] = data_curve_origin[:counterStep + 1]
        with open(various_data_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            for row in name_unit:
                writer.writerow(row)
            for row in data_curve_origin[:counterStep + 1]:
                writer.writerow(row)

        checkpoint_file = HDF5File(MPI.comm_world, Output_directory + f"/checkpoint_{counterStep}.h5", "r")

        u_input = Function(SpaceMixedCH)
        v_p_combined_input = Function(SpaceMixedStokes)

        checkpoint_file.read(u_input, "CheckPoint/u")
        checkpoint_file.read(v_p_combined_input, "CheckPoint/v_p_combined")
        checkpoint_file.close()

        assign(u_new, u_input)
        assign(v_p_combined, v_p_combined_input)

        # Output file
        cwr = u_new.split()[0]
        cwr.rename('C', 'C')
        vwr = v_p_combined.split()[0]
        vwr.rename('v', 'v')
        stress = Function(SpaceTensor)
        stress.rename('stress', 'stress')

    # Time stepping *******************************************************************************************
    while timeCurrent < timeSimulation:
        if interrupted:
            print(
                f"===== Exiting loop after current iteration at time step {counterStep}. =====\n"
                f"===== You can continue the simulation at time step {np.floor_divide(counterStep, frequencyOutput) * frequencyOutput} =====\n"
                f"===== You can continue the simulation of job {filename} ====="
            )
            break

        counterStep += 1
        timeCurrent = counterStep * dt
        print(f"\nTime step: {counterStep}, Time: {timeCurrent:g}s, Time of Simulation : {timeSimulation:g}s")

        u_prev.assign(u_new)
        print('--solving for C and mu')
        solve(CH_WeakFormCombined_L == CH_WeakFormCombined_R, u_new, solver_parameters=CH_params)
        print('--solving Stokes')
        solve(Stokes_WeakFormCombined_L == Stokes_WeakFormCombined_R, v_p_combined, bc, solver_parameters=Stokes_params)
        various_data[counterStep, 0] = timeCurrent
        various_data[counterStep, 1] = assemble(E)
        print(f"The total free energy at time {timeCurrent:g}s is {various_data[counterStep, 1]}")
        # solve for c_new and store it in c_lineIntegration
        solve(c_auxiliary_L == c_new * c_auxiliary_test * dx, c_lineIntegration,
              solver_parameters={'linear_solver': 'mumps'})

        # Calculate the shrinkage
        various_data[counterStep, 2] = L0 - line_integral(c_lineIntegration, Point_A, Point_B, ElementNumber[0])
        # calculate neck radius
        various_data[counterStep, 3] = 0.5 * line_integral(c_lineIntegration, Point_C, Point_D, ElementNumber[1])

        timeValuesNormalized = timeCurrent / timeCharacteristic + 1e-15
        dL_Normalized = various_data[counterStep, 2] / L0
        neckNormalized = (various_data[counterStep, 3] / radius_particle) * 0.7071
        neck_current = various_data[counterStep, 3]
        growthrate_Of_neck = (neck_current - neck_prev) / dt
        neck_prev = neck_current

        various_data[counterStep, 4] = timeValuesNormalized
        various_data[counterStep, 5] = dL_Normalized
        various_data[counterStep, 6] = neckNormalized
        various_data[counterStep, 7] = growthrate_Of_neck

        print(f"The dL at time {timeCurrent:g}s is {various_data[counterStep, 2]}")
        print(f"The radiusNeck at time {timeCurrent:g}s is {various_data[counterStep, 3]}")

        with open(various_data_path, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [timeCurrent, various_data[counterStep, 1], various_data[counterStep, 2], various_data[counterStep, 3],
                 various_data[counterStep, 4], various_data[counterStep, 5], various_data[counterStep, 6],
                 various_data[counterStep, 7]])

        if np.mod(counterStep, frequencyOutput) == 0:
            if os.path.exists(Output_directory + f"/checkpoint_{counterStep - frequencyOutput}.h5") and counterStep != counterStepInitial:
                os.remove(Output_directory + f"/checkpoint_{counterStep - frequencyOutput}.h5")

            print("----- Outputting checkpoint -----")
            checkpoint_file = HDF5File(MPI.comm_world, Output_directory + f"/checkpoint_{counterStep}.h5", "w")
            checkpoint_file.write(u_new, "CheckPoint/u")
            checkpoint_file.write(v_p_combined, "CheckPoint/v_p_combined")

            quotient = counterStep // frequencyOutput
            print('--writing the data to file--')
            xdmf_file.write(cwr, timeCurrent)
            xdmf_file.write(vwr, timeCurrent)
            solve(stress_L == stress_R, stress, solver_parameters={'linear_solver': 'mumps'})
            xdmf_file.write(stress, timeCurrent)

            # Plot Figures
            print("Plotting figures of C, eta, eta_pure, v and stress.")
            plot_function(6 * quotient + 1, cwr, os.path.join(Figures_directory, 'C'), timeCurrent)
            plot_function(6 * quotient + 3, vwr, os.path.join(Figures_directory, 'v'), timeCurrent)
            plot_function(6 * quotient + 4, stress, os.path.join(Figures_directory, 'stress'), timeCurrent)

            plot_curve(various_data[:counterStep + 1, 0], various_data[:counterStep + 1, 1], "Total Free Energy",
                       "Time (s)", "Total Free Energy (J)", timeCurrent)
            plot_curve(various_data[:counterStep + 1, 0], various_data[:counterStep + 1, 2], "Shrinkage", "Time (s)",
                       "Shrinkage", timeCurrent)
            plot_curve(various_data[:counterStep + 1, 0], various_data[:counterStep + 1, 3], "Radius of Neck",
                       "Time (s)",
                       "Radius of Neck", timeCurrent)
            plot_curve(various_data[:counterStep + 1, 4], various_data[:counterStep + 1, 5], "Normalized Shrinkage",
                       "Normalized Time",
                       "Normalized Sharinkage", timeCurrent)
            plot_curve(various_data[:counterStep + 1, 4], various_data[:counterStep + 1, 6], "Normalized Neck Length",
                       "Normalized Time", "Normalized Neck Length", timeCurrent)

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
