import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import random
import os
import datetime
import numpy as np
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


# Read the input file
def read_input_file(file_name):
    parametersInput = {}
    with open(file_name, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):  # Skip empty lines and comments
                continue
            var_name, var_value = line.split("=")
            parametersInput[var_name.strip()] = var_value.strip()
    return parametersInput


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
        # return self.D*(C**2)*((1-C)**2)
        # return pow(2*self.a,-1)*self.D*(C**2)*((1-C)**2)

    def fmu(self, C):
        return self.df(C)

    def sigma(self, u, C):
        # for this version it is important to keep viscosity_ratio<1
        # the defined sigma here does not include contribution from Lagrangian multiplier
        return ((self.viscosity_ratio + (1 - self.viscosity_ratio) * self.N(C)) * self.muBulk) * (grad(u) + grad(u).T)

    def interface_stress(self, C):
        return self.kc * outer(grad(C), grad(C))


# Main code starts from here
# Copy the codes that you are running
print("the code being executed:",
      sys.argv)  # sys.argv is the python_script_name when you use "python3 python_script_name" to run python_script_name
python_script_name = os.path.basename(sys.argv[0])  # change the name of source code to the one you are running
shutil.copy(python_script_name, os.path.join(codes_directory, 'copy_' + python_script_name))
# Read the input file
input_file = "input.txt"  # Replace with your input file name
parametersInput = read_input_file(input_file)
shutil.copy(input_file, os.path.join(input_directory, "input.txt"))

# Assign values from the input file to parametersInput
# Phase-field parameters Input
a = float(parametersInput.get('Alpha'))
kc = float(parametersInput.get('KappaC'))
D = float(parametersInput.get('Relaxation Parameter'))
Neps = float(parametersInput.get('Epsilon'))

# Mechanics Parameters
muBulk = float(parametersInput.get('Bulk viscosity'))
viscosity_ratio = float(parametersInput.get('Viscosity Ratio'))
timeCharacteristic = float(parametersInput.get('Characteristic Time'))

# Computational Setting
radiusParticle = float(parametersInput.get('Particle Radius'))
Np = int(parametersInput.get('Particle Number'))
domain_size = [float(parametersInput.get('Dimension X')), float(parametersInput.get('Dimension Y'))]
numberOfElement = [int(parametersInput.get('NumberOfElement X')), int(parametersInput.get('NumberOfElement Y'))]
dt = float(parametersInput.get('TimeStep'))
timeInitial = float(parametersInput.get('InitialTime'))
NumberOfTimeStep = int(parametersInput.get('NumberOfTimeStep'))
counterStepInitial = int(parametersInput.get('InitialStepCounter'))
frequencyOutput = int(parametersInput.get('FrequencyOutPut'))

# Algorithmic parameters
theta = float(parametersInput.get('theta'))

pde = viscousSintering(a, kc, D, Neps, muBulk, viscosity_ratio, Np)
logging.info("# Phase-field parametersInput")
logging.info(f"Alpha: {pde.a.values()[0]}")
logging.info(f"Kappac: {pde.kc.values()[0]}")
logging.info(f"Relaxation Parameter: {pde.D.values()[0]}")
logging.info(f"Epsilon: {pde.Neps.values()[0]}")

# Mechanics Parameters
logging.info("")
logging.info("# Mechanics Parameters")
logging.info(f"Bulk viscosity: {muBulk}")
logging.info(f"Viscosity Ratio: {viscosity_ratio}")
logging.info(f"Characteristic Time : {timeCharacteristic}")

# Computational Setting
logging.info(" ")
logging.info("# Computational Setting")
logging.info(f"Particle Radius: {radiusParticle}")
logging.info(f"Particle Number: {Np}")
logging.info(f"Dimension X: {domain_size[0]}")
logging.info(f"Dimension Y: {domain_size[1]}")
logging.info(f"NumberOfElement X: {numberOfElement[0]}")
logging.info(f"NumberOfElement Y: {numberOfElement[1]}")
logging.info(f"TimeStep: {dt}")
logging.info(f"InitialTime: {timeInitial}")
logging.info(f"NumberOfTimeStep: {NumberOfTimeStep}")
logging.info(f"InitialStepCounter: {counterStepInitial}")
logging.info(f"FrequencyOutPut: {frequencyOutput}")

# Algorithmic parameters
logging.info(" ")
logging.info("# Algorithmic parameters")
logging.info(f"theta: {theta}")

# Normalized Materials Properties
energySurfSpeciNormalized = (sqrt(2 * pde.kc.values()[0] * pde.a.values()[0]) / 6)
thicknessSurfaceNormalized = sqrt(8 * pde.kc.values()[0] / pde.a.values()[0])

logging.info(" ")
logging.info("# Normalized Materials Properties")
logging.info(f"specfic surface energy: {energySurfSpeciNormalized}")
logging.info(f"surface thickness: {thicknessSurfaceNormalized}")

# block with mesh, spaces and weak forms
# Form compiler options
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
# tell the form to apply optimization strategies in the code generation
# phase and then use compiler optimization flags when compiling the
# generated C++ code. Using the option ``["optimize"] = True`` will
# generally result in faster code (sometimes orders of magnitude faster
# for certain operations, depending on the equation), but it may take
# considerably longer to generate the code and the generation phase may
# use considerably more memory).
#

# Create mesh and build function space
mesh = RectangleMesh(Point(0, 0), Point(domain_size[0], domain_size[1]), numberOfElement[0], numberOfElement[1],
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
v, p = split(v_p_combined)


def boundary(x, on_boundary):
    return on_boundary


# Set up the Dirichlet boundary condition for velocity
bc = DirichletBC(SpaceMixedStokes.sub(0), Constant((0, 0)), boundary)

x, y = Expression('x[0]', degree=1), Expression('x[1]', degree=1)
# Two end points of the center line to calculate horizontal shrinkage
Point_A, Point_B = np.array([0, domain_size[1] / 2]), np.array([domain_size[0], domain_size[1] / 2])
# Two end points of the verticle line to calculate neck length
Point_C, Point_D = np.array([domain_size[0] / 2, 0]), np.array([domain_size[0] / 2, domain_size[1]])

# Initial conditions are created by using the class defined at the
# beginning and then interpolating the initial conditions
# into a finite element space::

c_init = Expression('(1-tanh((sqrt(pow((x[0]-75), 2)+pow((x[1]-75), 2))-25)/(sqrt(2)*epsilon)))/2' + \
                    '+(1-tanh((sqrt(pow((x[0]-125), 2)+pow((x[1]-75), 2))-25)/(sqrt(2)*epsilon)))/2',
                    degree=1, epsilon=Constant(1))

mu_init = Constant(0)
c0 = interpolate(c_init, SpaceMixedCH.sub(0).collapse())
mu0 = interpolate(mu_init, SpaceMixedCH.sub(1).collapse())

# semi_implicit c for laplace
c_mid = (1.0 - theta) * c_prev + theta * c_trial

# Reset initial conditions
assign(u_new, [c0, mu0])
u_prev.assign(u_new)

# Variables for line integration
c_lineIntegration = Function(SpaceLinear)

timeSimulation = round(NumberOfTimeStep * dt, 8)
energy = np.empty(NumberOfTimeStep + 1)
dL = np.empty(NumberOfTimeStep + 1)
radiusNeck = np.empty(NumberOfTimeStep + 1)

E = (0.5 * pde.kc * dot(grad(c_new), grad(c_new)) + pde.f(c_new)) * dx
energy[counterStepInitial] = assemble(E)
dL[counterStepInitial] = 0

# solve for c_new and store it in c_lineIntegration in order to use line_integral
c_auxiliary_L = c_auxiliary * c_auxiliary_test * dx
solve(c_auxiliary_L == c_new * c_auxiliary_test * dx, c_lineIntegration, solver_parameters={'linear_solver': 'mumps'})

# do line integration of c_new over the line defined by Point_A and Point_B
# L0 左右两端的初始长度
L0 = line_integral(c_lineIntegration, Point_A, Point_B, numberOfElement[0])
# L0=2*Np*radiusParticle


# calculate neck radius
radiusNeck[0] = 0.5 * line_integral(c_lineIntegration, Point_C, Point_D, numberOfElement[1])

# The linear weak form of c and mu
cWeakForm = c_trial * c_test * dx - c_prev * c_test * dx \
            + dt * dot(pde.M(c_prev) * grad(mu_trial), grad(c_test)) * dx \
            - dt * dot(v * c_trial, grad(c_test)) * dx
muWeakForm = mu_trial * mu_test * dx - pde.kc * dot(grad(c_mid), grad(mu_test)) * dx \
             - (pde.fmu(c_prev) + pde.c_stabilizer * (c_trial - c_prev)) * mu_test * dx
CH_WeakFormCombined = cWeakForm + muWeakForm
CH_WeakFormCombined_L, CH_WeakFormCombined_R = lhs(CH_WeakFormCombined), rhs(CH_WeakFormCombined)

# The weak form of velocity and pressure
Stokes_WeakFormCombined = inner(pde.sigma(v_trial, c_new), grad(v_test)) * dx \
                          + div(v_test) * p_trial * dx\
                          + p_test * div(v_trial) * dx \
                          - inner(pde.interface_stress(c_new), grad(v_test)) * dx
Stokes_WeakFormCombined_L, Stokes_WeakFormCombined_R = lhs(Stokes_WeakFormCombined), rhs(Stokes_WeakFormCombined)

# define equations for physical stress
stress_test = TestFunction(SpaceTensor)
stress_trial = TrialFunction(SpaceTensor)
stress_L = inner(stress_trial, stress_test) * dx
stress_R = inner(pde.sigma(v, c_new) + p * Identity(2) - (0.5 * pde.kc * dot(grad(c_new), grad(c_new)) + pde.f(c_new)) * Identity(2), stress_test) * dx

# block with solver and solution
# solver settings
CH_params = {'linear_solver': 'tfqmr',
             'preconditioner': 'ilu'}
Stokes_params = {'linear_solver': 'mumps'}

# Solving Stokes at initial time
assign(v_p_combined, [interpolate(Constant((0, 0)), SpaceMixedStokes.sub(0).collapse()),
                      interpolate(Constant(0), SpaceMixedStokes.sub(1).collapse())])
solve(Stokes_WeakFormCombined_L == Stokes_WeakFormCombined_R, v_p_combined, bc, solver_parameters=Stokes_params)

# To run the solver and save the output to a VTK file for later visualization,
# the solver is advanced in time until a terminal time is reached.

# Output file
file = File(os.path.join(results_directory, "solution.pvd"))
cwr = u_new.split()[0]
cwr.rename('C', 'C')
vwr = v_p_combined.split()[0]
vwr.rename('v', 'v')

# Variables for writing physical stress
stress = Function(SpaceTensor)
stress.rename('stress', 'stress')

# Step in time
timeCurrent = timeInitial
counterStep = counterStepInitial
file << (cwr, timeCurrent)
file << (vwr, timeCurrent)
solve(stress_L == stress_R, stress, solver_parameters={'linear_solver': 'mumps'})
# ,solver_parameters={'linear_solver':'mumps'}
file << (stress, timeCurrent)

print('')
print('initial particle distance L0 is: ', L0)
print('')
print('initial neck radius  is: ', radiusNeck[0])
print('')
print("the energy at time", timeCurrent, "is", energy[counterStep])
print('')

start_time = time.time()
while (timeCurrent < timeSimulation):
    counterStep += 1
    print('counterStep', counterStep)
    print('timeCurrent', timeCurrent)
    print('timeSimulation', timeSimulation)

    timeCurrent = round(counterStep * dt, 8)
    print('')
    print('starting computations at time', timeCurrent)
    u_prev.assign(u_new)
    print('')
    print('--solving for C and mu')
    solve(CH_WeakFormCombined_L == CH_WeakFormCombined_R, u_new, solver_parameters=CH_params)
    print('')
    print('--solving Stokes')
    solve(Stokes_WeakFormCombined_L == Stokes_WeakFormCombined_R, v_p_combined, bc, solver_parameters=Stokes_params)
    # calculate system energy
    energy[counterStep] = assemble(E)
    print('')
    print("the energy at time", timeCurrent, "is", energy[counterStep])
    print('')

    # solve for c_new and store it in c_lineIntegration
    solve(c_auxiliary_L == c_new * c_auxiliary_test * dx, c_lineIntegration,
          solver_parameters={'linear_solver': 'mumps'})
    # dL 两端的长度变化量dL=L0-L(t)
    dL[counterStep] = L0 - line_integral(c_lineIntegration, Point_A, Point_B, numberOfElement[0])

    # calculate neck radius
    radiusNeck[counterStep] = 0.5 * line_integral(c_lineIntegration, Point_C, Point_D, numberOfElement[1])

    print('')
    print('the particle distance L0 at time is: ', L0)
    print('')
    print("the dL at time", timeCurrent, "is", dL[counterStep])
    print('')
    print("the radiusNeck at time", timeCurrent, "is", radiusNeck[counterStep])

    if np.mod(counterStep, frequencyOutput) == 0:
        quotient = counterStep // frequencyOutput
        print('')
        print('--writing the data to file--')
        print('')
        file << (cwr, timeCurrent)
        file << (vwr, timeCurrent)
        solve(stress_L == stress_R, stress, solver_parameters={'linear_solver': 'mumps'})
        file << (stress, timeCurrent)

        timeValues = np.linspace(0, counterStep * dt, counterStep + 1)
        timeValuesNormalized = timeValues / timeCharacteristic + 1e-15

        nameFileShrinkage = f"/dataShrinkageNormalized_{counterStep:06d}.txt"
        nameFileLenghNeck = f"/dataLengthNeckNormalized_{counterStep:06d}.txt"
        nameFileEnergy = f"/dataEnergy_{counterStep:06d}.txt"

        dL_Normalized = dL[:counterStep + 1] / L0
        neckNormalized = (radiusNeck[:counterStep + 1] / radiusParticle) * 0.7071

        dataShrinkageNormalized = np.column_stack((timeValuesNormalized, dL_Normalized))
        np.savetxt(output_directory + nameFileShrinkage, dataShrinkageNormalized, fmt='%.6f',
                   header='Time,         strain', delimiter=',    ')

        dataLengthNeckNormalized = np.column_stack((timeValuesNormalized, neckNormalized))
        np.savetxt(output_directory + nameFileLenghNeck, dataLengthNeckNormalized, fmt='%.6f',
                   header='Time, neckLength', delimiter=',    ')

        dataEnergy = np.column_stack((timeValues, energy[:counterStep + 1]))
        np.savetxt(output_directory + nameFileEnergy, dataEnergy, fmt='%.6f', header='Time, systemEnergy',
                   delimiter=',  ')

        # Plot the results
        gs = GridSpec(1, 3, width_ratios=[1, 1, 1], height_ratios=[1])

        fig = plt.figure(4 * (quotient - 1) + 1, figsize=(15, 5))

        fig.savefig(output_directory + f'/neckRadius_{counterStep:06d}.png')

        plt.figure(4 * (quotient - 1) + 3)
        plt.plot(timeValues, energy[:counterStep + 1], label='energy')
        plt.xlabel('Time')
        plt.ylabel('Energy')
        plt.legend()
        plt.savefig(output_directory + f'/energy_{counterStep:06d}.png')

        plt.figure(4 * (quotient - 1) + 4)
        plot(c_new, cmap='coolwarm')
        plt.savefig(output_directory + f'/c_new_{counterStep:06d}.png')

end_time = time.time()

execution_time = end_time - start_time

print('')
print(f"Execution time: {execution_time} seconds")
