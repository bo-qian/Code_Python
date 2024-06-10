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
class solidSintering:
    def __init__(self, a, b, kc, keta, Dsf, Dgb, Dvol, L, Neps, muSurf, muGB, muBulk,
                 viscosity_ratio, Np):
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

    def N(self, C):
        return (C ** 2) * (1 + 2 * (1 - C) + (self.Neps) * (1 - C) ** 2)

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
                    (self.viscosity_ratio + (1 - self.viscosity_ratio) * self.N(C)) * self.muBulk \
                    + self.muSurf * (C ** 2) * ((1 - C) ** 2) \
                    + 2 * self.muGB * self.N(C) * sum(
                [eta[i] ** 2 * (sum([eta[j] ** 2 for j in range(self.Np)]) - eta[i] ** 2) for i in range(self.Np)]) \
            ) * (grad(u) + grad(u).T)

    def interface_stress(self, C, eta):
        return self.kc * outer(grad(C), grad(C)) + self.keta * self.N(C) * sum(
            [outer(grad(eta[i]), grad(eta[i])) for i in range(self.Np)])


# Main code starts from here
# Copy the codes that you are running
print("the code being executed:", sys.argv)
# sys.argv is the python_script_name when you use "python3 python_script_name" to run python_script_name
python_script_name = os.path.basename(sys.argv[0])  # change the name of source code to the one you are running
shutil.copy(python_script_name, os.path.join(codes_directory, 'copy_' + python_script_name))
# Read the input file
input_file = "input.txt"  # Replace with your input file name
parametersInput = read_input_file(input_file)
shutil.copy(input_file, os.path.join(input_directory, "input_copy.txt"))

# Assign values from the input file to parametersInput
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
radiusParticle = float(parametersInput.get('Particle Radius'))
Np = int(parametersInput.get('Particle Number'))
timeCharacteristic = float(parametersInput.get('Characteristic Time'))
domain_size = [float(parametersInput.get('Dimension X')), float(parametersInput.get('Dimension Y'))]
numberOfElement = [int(parametersInput.get('NumberOfElement X')), int(parametersInput.get('NumberOfElement Y'))]
dt = float(parametersInput.get('TimeStep'))
timeInitial = float(parametersInput.get('InitialTime'))
NumberOfTimeStep = int(parametersInput.get('NumberOfTimeStep'))
counterStepInitial = int(parametersInput.get('InitialStepCounter'))
frequencyOutput = int(parametersInput.get('FrequencyOutPut'))

# Algorithmic parameters
theta = float(parametersInput.get('theta'))

pde = solidSintering(a, b, kc, keta, Dsf, Dgb, Dvol, L, Neps, muSurf, muGB, muBulk, viscosity_ratio, Np)
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

# Normalized Materials Properties
energySurfSpeciNormalized = (sqrt(2 * pde.kc.values()[0] * pde.a.values()[0]) / 6)
thicknessSurfaceNormalized = sqrt(8 * pde.kc.values()[0] / pde.a.values()[0])
thicknessGbNormalized = sqrt(4 * pde.keta.values()[0] / (3 * pde.b.values()[0]))
logging.info(" ")
logging.info("# Normalized Materials Properties")
logging.info(f"specfic surface energy: {energySurfSpeciNormalized}")
logging.info(f"surface thickness: {thicknessSurfaceNormalized}")
logging.info(f"Grain boundary thickness: {thicknessGbNormalized}")

# block with mesh, spaces and weak forms
# Form compiler options
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
# tell the form to apply optimization strategies in the code generation
# phase and the use compiler optimization flags when compiling the
# generated C++ code. Using the option ``["optimize"] = True`` will
# generally result in faster code (sometimes orders of magnitude faster
# for certain operations, depending on the equation), but it may take
# considerably longer to generate the code and the generation phase may
# use considerably more memory).

# Create mesh and build function space
mesh = RectangleMesh(Point(0, 0), Point(domain_size[0], domain_size[1]), numberOfElement[0], numberOfElement[1],
                     'crossed')

# FunctionSpace consists of c, mu, and etas
P1 = FiniteElement("CG", triangle, 1)  # first order polynomial finite element
# For c and mu
spaceMixedCH = FunctionSpace(mesh, MixedElement([P1, P1]))
# For each eta
spaceLinear = FunctionSpace(mesh, P1)

# Taylor-Hood function space for velocity
PV = VectorElement("CG", triangle, 2)
# Mixed space for both velocity and pressure from stokes equation
spaceMixedStokes = FunctionSpace(mesh, MixedElement([PV, P1]))

# Create a separate space to output the stress
PS = TensorElement("CG", triangle, 1)
spaceTensor = FunctionSpace(mesh, PS)

# Trial and test functions from the mixed function space are now defined::
# Define trial and test functions
# du    = TrialFunction(spaceMixedCH)
c_trial, mu_trial = TrialFunction(spaceMixedCH)
c_test, mu_test = TestFunctions(spaceMixedCH)
eta_trial = TrialFunction(spaceLinear)
eta_test = TestFunction(spaceLinear)

# auxiliary variable trial and test function space
auxiliaryVar_trial = TrialFunction(spaceLinear)  # for later line integration
auxiliaryVar_test = TestFunction(spaceLinear)  # for later line integration

# Define functions
u_new = Function(spaceMixedCH)  # current combined solution of c_new and mu_new
eta_new = [Function(spaceLinear) for i in range(pde.Np)]
u_prev = Function(spaceMixedCH)  # previous combined solution of c_prev and mu_prev
eta_prev = [Function(spaceLinear) for i in range(pde.Np)]

# Split mixed functions
# The line ``c, mu = split(u)`` permits direct access to the components
# of a mixed function. Note that ``c`` and ``mu`` are references for
# components of ``u``, and not copies.
c_new, mu_new = split(u_new)
c_prev, mu_prev = split(u_prev)

# All functions for velocity space and pressure space
v_trial, p_trial = TrialFunctions(spaceMixedStokes)
v_test, p_test = TestFunctions(spaceMixedStokes)
v_p_combined = Function(spaceMixedStokes)
v, p = split(v_p_combined)


def boundary(x, on_boundary):
    return on_boundary


# Set up the Dirichlet boundary condition for velocity
bc = DirichletBC(spaceMixedStokes.sub(0), Constant((0, 0)), boundary)

x, y = Expression('x[0]', degree=1), Expression('x[1]', degree=1)
# Two end points of the center line to calculate horizontal shrinkage
Point_A, Point_B = np.array([0, domain_size[1] / 2]), np.array([domain_size[0], domain_size[1] / 2])
# Two end points of the vertical line to calculate neck length
Point_C, Point_D = np.array([domain_size[0] / 2, 0]), np.array([domain_size[0] / 2, domain_size[1]])

# Initial conditions are created and then interpolating the initial conditions
# into a finite element space::
# Create initial conditions
c_init = Expression('(1-tanh((sqrt(pow((x[0]-30), 2)+pow((x[1]-40), 2))-20)/(sqrt(2)*epsilon)))/2' + \
                    '+(1-tanh((sqrt(pow((x[0]-70), 2)+pow((x[1]-40), 2))-20)/(sqrt(2)*epsilon)))/2',
                    degree=1, epsilon=Constant(1))
mu_init = Constant(0)

# eta_init= [Constant(1), Constant(0)]                     

eta_init = [Expression('(1-tanh((x[0]-50)/(sqrt(2)*epsilon)))/2',
                       degree=1, epsilon=Constant(1)),
            Expression('(1+tanh((x[0]-50)/(sqrt(2)*epsilon)))/2',
                       degree=1, epsilon=Constant(1))]

# Interpolate
c0 = interpolate(c_init, spaceMixedCH.sub(0).collapse())
mu0 = interpolate(mu_init, spaceMixedCH.sub(1).collapse())
eta0 = [interpolate(eta_init[i], spaceLinear) for i in range(pde.Np)]

# semi_implicit c for laplace
c_mid = (1.0 - theta) * c_prev + theta * c_trial

# Reset initial conditions
assign(u_new, [c0, mu0])
u_prev.assign(u_new)
for i in range(pde.Np):
    assign(eta_new[i], eta0[i])
    eta_prev[i].assign(eta_new[i])

# Variables for line integration
c_lineIntegration = Function(spaceLinear)

# Define some quantities
timeSimulation = NumberOfTimeStep * dt
energy = np.empty(NumberOfTimeStep + 1)
dR = np.empty(NumberOfTimeStep + 1)
dL = np.empty(NumberOfTimeStep + 1)
dR_adv = np.empty(NumberOfTimeStep + 1)
radiusNeckIntegrate = np.empty(NumberOfTimeStep + 1)
radiusNeckFormula = np.empty(NumberOfTimeStep + 1)
velocityMechanicalAveragedX = np.empty(NumberOfTimeStep + 1)
velocityShiftAveragedX = np.empty(NumberOfTimeStep + 1)

dR[counterStepInitial] = 0
dL[counterStepInitial] = 0
dR_adv[counterStepInitial] = 0

velocityMechanicalAveragedX[counterStepInitial] = 0
velocityShiftAveragedX[counterStepInitial] = 0

# calculate the total free energy
E = (0.5 * pde.kc * dot(grad(c_new), grad(c_new)) + pde.f(c_new) + pde.N(c_new) * (
        pde.S(eta_new) + pde.grad_eta(eta_new))) * dx
energy[counterStepInitial] = assemble(E)

# R0 is the distance between the centers of particles
R0 = (assemble(c_new * eta_new[1] * x * dx) / denom_reg(assemble(c_new * eta_new[1] * dx))) \
     - (assemble(c_new * eta_new[0] * x * dx) / denom_reg(assemble(c_new * eta_new[0] * dx)))
print('')
print('Initial distance between the centers of the two particles: R0=', R0)
print('')

# solve for c_new and store it in c_lineIntegration: why have to solve it?
# because the line integration needs a direct Fenics object to perform line integration
auxiliary_L = auxiliaryVar_trial * auxiliaryVar_test * dx
solve(auxiliary_L == c_new * auxiliaryVar_test * dx, c_lineIntegration)
# do line integration of c_new over the line defined by Point_A and Point_B
L0 = line_integral(c_lineIntegration, Point_A, Point_B, numberOfElement[0])
print('')
print('Initial distance between the boundary of the two particles: L0=', L0)
print('')

# calculate neck radius
# radiusNeck[0]=0.5*line_integral(c_lineIntegration,Point_C, Point_D,numberOfElement[1])
radiusNeckIntegrate[0] = 0.5 * line_integral(c_lineIntegration, Point_C, Point_D, numberOfElement[1])
radiusNeckFormula[0] = 0.5 * 6 * assemble(c_new * eta_new[0] * c_new * eta_new[1] * dx) / thicknessSurfaceNormalized

# The linear form
# weak form of C
cWeakForm = c_trial * c_test * dx - c_prev * c_test * dx \
            + dt * dot(pde.M(c_prev, eta_prev) * grad(mu_trial), grad(c_test)) * dx \
            - dt * dot(v * c_trial, grad(c_test)) * dx
# weak form of mu
muWeakForm = mu_trial * mu_test * dx - pde.kc * dot(grad(c_mid), grad(mu_test)) * dx \
             - (pde.fmu(c_prev, eta_prev) + pde.c_stabilizer * (c_trial - c_prev)) * mu_test * dx
# combined weak form of c and mu; c and mu are solved together
CH_WeakFormCombined = cWeakForm + muWeakForm
CH_WeakFormCombined_L, CH_WeakFormCombined_R = lhs(CH_WeakFormCombined), rhs(CH_WeakFormCombined)

# weak form of each eta
eta_WeakForm = [dot(eta_trial, eta_test) * dx - dot(eta_prev[i], eta_test) * dx \
                + dt * dot(v, grad(eta_trial)) * eta_test * dx \
                + dt * pde.L * inner(pde.keta * pde.N(c_prev) * grad((1.0 - theta) * eta_prev[i] + theta * eta_trial),
                                     grad(eta_test)) * dx \
                + dt * pde.L * pde.N(c_prev) * pde.dS(eta_prev, i) * eta_test * dx + dt * pde.eta_stabilizer * (
                        eta_trial - eta_prev[i]) * eta_test * dx
                for i in range(pde.Np)]
eta_WeakForm_L = [lhs(eta_WeakForm[i]) for i in range(pde.Np)]
eta_WeakForm_R = [rhs(eta_WeakForm[i]) for i in range(pde.Np)]

# The weak form of velocity and pressure
Stokes_WeakFormCombined = inner(pde.sigma(v_trial, c_new, eta_new), grad(v_test)) * dx + div(
    v_test) * p_trial * dx + p_test * div(v_trial) * dx \
                          - inner(pde.interface_stress(c_new, eta_new), grad(v_test)) * dx
Stokes_WeakFormCombined_L, Stokes_WeakFormCombined_R = lhs(Stokes_WeakFormCombined), rhs(Stokes_WeakFormCombined)

# define equations for physcial stress
stress_test = TestFunction(spaceTensor)
stress_trial = TrialFunction(spaceTensor)
stress_L = inner(stress_trial, stress_test) * dx
stress_R = inner(pde.sigma(v, c_new, eta_new) + p * Identity(2)
                 - (0.5 * pde.kc * dot(grad(c_new), grad(c_new)) + pde.f(c_new)
                    + pde.N(c_new) * (pde.S(eta_new) + pde.grad_eta(eta_new))) * Identity(2), stress_test) * dx

# block with solver and solution
# solver settings
PhaseField_params = {'linear_solver': 'tfqmr',
                     'preconditioner': 'ilu'}

Stokes_params = {'linear_solver': 'mumps'}

# Solving Stokes at initial time and store the solutions in v_p_combined
assign(v_p_combined, [interpolate(Constant((0, 0)), spaceMixedStokes.sub(0).collapse()),
                      interpolate(Constant(0), spaceMixedStokes.sub(1).collapse())])
solve(Stokes_WeakFormCombined_L == Stokes_WeakFormCombined_R, v_p_combined, bc, solver_parameters=Stokes_params)

# To run the solver and save the output to a VTK file for later visualization,
# the solver is advanced in time from t_n to t_{n+1} until
# a terminal time timeSimulation is reached.

# Output file
file = File(os.path.join(results_directory, "solution.pvd"))
# variable for storing c
cwr = u_new.split()[0]
cwr.rename('C', 'C')
# variable for storing c^2*eta^2
etawr = Function(spaceLinear)
etawr.rename('eta', 'eta')
# store velocity
vwr = v_p_combined.split()[0]
vwr.rename('v', 'v')

# Variables for writing stress and strainRate
stress = Function(spaceTensor)
stress.rename('stress', 'stress')

# Step in time
timeCurrent = timeInitial
counterStep = counterStepInitial
# store c
file << (cwr, timeCurrent)
# solve for c^2*eta^2 and store it in etawr
etawr_L = eta_trial * eta_test * dx
etawr_R = (c_new ** 2) * sum([eta_new[j] ** 2 for j in range(pde.Np)]) * eta_test * dx
solve(etawr_L == etawr_R, etawr)
file << (etawr, timeCurrent)
# store velocity
file << (vwr, timeCurrent)
# store stress
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
    for i in range(pde.Np): eta_prev[i].assign(eta_new[i])
    print('')
    print('--solving for C and mu')
    # solve for c and mu and store them in u_new
    solve(CH_WeakFormCombined_L == CH_WeakFormCombined_R, u_new, solver_parameters=PhaseField_params)
    print(" ")
    print('--solving for eta')
    for i in range(pde.Np):
        solve(eta_WeakForm_L[i] == eta_WeakForm_R[i], eta_new[i], solver_parameters=PhaseField_params)
    print(" ")
    print('--solving Stokes')
    # solving for v and pressure and store them in v_p_combined
    solve(Stokes_WeakFormCombined_L == Stokes_WeakFormCombined_R, v_p_combined, bc, solver_parameters=Stokes_params)
    # calculate the system energy
    energy[counterStep] = assemble(E)
    print('')
    print("the energy at time", timeCurrent, "is ", energy[counterStep])
    # dR is the shrinkage between the centers of the two-particle model
    dR[counterStep] = R0 - (assemble(c_new * eta_new[1] * x * dx) / denom_reg(assemble(c_new * eta_new[1] * dx))) \
                      + (assemble(c_new * eta_new[0] * x * dx) / denom_reg(assemble(c_new * eta_new[0] * dx)))

    # solve for c_new and store it in c_lineIntegration
    # etawr_L and eta_test are defined from the line space; so they can be re-used here.
    solve(etawr_L == c_new * eta_test * dx, c_lineIntegration)

    # dL is the length change of the two particle model
    dL[counterStep] = L0 - line_integral(c_lineIntegration, Point_A, Point_B, numberOfElement[0])

    # calculate the neck radius
    radiusNeckIntegrate[counterStep] = 0.5 * line_integral(c_lineIntegration, Point_C, Point_D, numberOfElement[1])
    radiusNeckFormula[counterStep] = 0.5 * 6 * assemble(
        c_new * eta_new[0] * c_new * eta_new[1] * dx) / thicknessSurfaceNormalized

    # Advection contribution for dR
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

        # Plot the results
        # set 3 subplots with equal size
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

end_time = time.time()
execution_time = end_time - start_time
print('')
print(f"Execution time: {execution_time} seconds")
print('')
