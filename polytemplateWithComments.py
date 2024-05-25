from fenics import *
from mshr import *
# from ufl_legacy import transpose, tanh
from ufl import transpose, tanh

# parameters block
import random
import numpy as np


class VoronoiParticle(UserExpression):
    # default center coordinates of current grain 定义当前晶粒的默认中心坐标
    center = [0, 0]
    # default center coordinates of other grains 定义其他晶粒的默认中心坐标
    others = [[1, 1]]
    # parameter in the hyperbolic tangent function 定义双曲正切函数中的参数
    epsilon = 1
    # defines a lambda function named obstacle 定义名为obstacle的lambda函数 This defines a lambda function that takes one
    # argument x and always returns 0 by default. 这个函数接受一个参数x，如果没有传入参数，则默认返回0
    obstacle = lambda x: 0

    # defines the shape of the returned value. In this case, it returns an empty tuple () indicating that the
    # expression evaluates to a scalar value. 定义返回值的形状。在这种情况下，它返回一个空元组()，表示表达式计算为标量值。
    def value_shape(self):  # 定义返回值的形状
        return ()  # 返回一个空元组

    def eval(self, value, x):
        # vector difference between point X and the current grain center 计算点X与当前晶粒中心的矢量差
        dx = x[0] - self.center[0]  # x[0] - self.center[0]
        dy = x[1] - self.center[1]  # x[1] - self.center[1]
        # find the index closest_ind of other grain whose cener is closest to point X 找到离点X最近的其他晶粒的中心的索引closest_ind
        closest_ind = np.argmin([sqrt(pow((x[0] - self.others[k][0]), 2) + pow((x[1] - self.others[k][1]), 2)) for k in
                                 range(len(self.others))])
        # get the center coordinates of the closest other grain closest = self.others[closest_ind] 获取最近的其他晶粒的中心坐标
        closest = self.others[closest_ind]  # 获取最近的其他晶粒的中心坐标
        # vector difference between the closest other grain and the current grain  最近的其他晶粒与当前晶粒之间的矢量差
        dpx = closest[0] - self.center[0]  # closest[0] - self.center[0]
        dpy = closest[1] - self.center[1]  # closest[1] - self.center[1]
        # the distance between the the closest other grain and the current grain
        dp = sqrt(pow(dpx, 2) + pow(dpy, 2))  # 这个公式是计算两个晶粒之间的距离
        # (dx*dpx + dy*dpy) = |dx|*|dp|*cos(theta)   计算dx在dp上的投影 (dx*dpx + dy*dpy)/|dp| = |dx|*cos(theta) calculates
        # the projection of dx on to dp  计算dx在dp上的投影 use tanh to initialize the value of a eta and make sure the
        # transition between neighboring grains is smooth.  使用tanh函数初始化eta的值，并确保相邻晶粒之间的过渡是平滑的 (0.5*pow(dp,2)-dx*dpx -
        # dy*dpy)/dp calculate the distance between the midpoint of dp and the projection  计算dp的中点与投影之间的距离
        value[0] = (1 - self.obstacle(x)) * (
                1 + np.tanh((0.5 * pow(dp, 2) - dx * dpx - dy * dpy) / (sqrt(2) * self.epsilon * dp))) / 2


def smooth_rectangle(x, y, x_center, y_center, width, height, epsilon):  # 定义一个函数，从0到1平滑的矩形过渡
    # (x - x_center - width/2)：calculates the distance from the current x-coordinate (x) to the right edge of the
    # rectangle (x - x_center + width/2)：calculates the distance from the current x-coordinate (x) to the left edge
    # of the rectangle np.tanh((x - x_center - width/2)/(np.sqrt(2)*epsilon)): zero at the right edge; 1 far to the
    # right edge from right side note that tanh(width/2)/(np.sqrt(2)*epsilon) is 1 as width is much larger than epsilon
    sigmoid_x = 0.5 * (1 - np.tanh((x - x_center - width / 2) / (np.sqrt(2) * epsilon))) - 0.5 * (
            1 - np.tanh((x - x_center + width / 2) / (np.sqrt(2) * epsilon)))
    # similar operation in y direction  在y方向上进行类似的操作
    sigmoid_y = 0.5 * (1 - np.tanh((y - y_center - height / 2) / (np.sqrt(2) * epsilon))) - 0.5 * (
            1 - np.tanh((y - y_center + height / 2) / (np.sqrt(2) * epsilon)))
    # the function returns the product of sigmoid_x and sigmoid_y, which effectively combines the horizontal and
    # vertical components to generate the smooth rectangular mask
    return sigmoid_x * sigmoid_y


class obstacle(UserExpression):  # 定义一个类obstacle，继承自UserExpression，其实就是模板创建的函数
    # Default values of the attributes  属性的默认值
    center = [0.5, 0.5]  # 矩形模板的中心坐标
    dimensions = [0.1, 0.1]  # 矩形模板的尺寸
    epsilon = 1 / 20  # 平滑参数

    # Returns the shape of the values returned by the expression. In this case, it returns an empty tuple,
    # indicating a scalar value.
    def value_shape(self):  # 返回一个空元组，表示表达式计算为标量值
        return ()

    # Evaluates the expression at a given point x. It calls the smooth_rectangle function
    def eval(self, value, x):  # 在给定点x处计算表达式，调用smooth_rectangle函数来计算该点的值
        value[0] = smooth_rectangle(x[0], x[1], self.center[0], self.center[1], self.dimensions[0], self.dimensions[1],
                                    self.epsilon)

    # This method allows instances of the class to be called as functions. It's essentially equivalent to the eval
    # method but provides a more convenient way to use instances of the class as functions
    def __call__(self, x):  # 允许类的实例被调用为函数，提供了一个更方便的方法来使用类的实例作为函数。基本上等同于eval方法
        return smooth_rectangle(x[0], x[1], self.center[0], self.center[1], self.dimensions[0], self.dimensions[1],
                                self.epsilon)

    # These methods allow the user to set the center coordinates, dimensions, and smoothing parameter of the
    # obstacle, respectively
    def set_center(self, center):  # 允许用户设置障碍物的中心坐标
        self.center = center

    def set_dimensions(self, dimensions):  # 允许用户设置障碍物的尺寸
        self.dimensions = dimensions

    def set_epsilon(self, epsilon):  # 允许用户设置平滑参数
        self.epsilon = epsilon

    # This method returns a string representation of the expression defining the obstacle
    def to_string_expression(self):  # 返回定义障碍物的表达式的字符串表示形式
        return '*'.join(['((1 - tanh((x[0] - ' + str(self.center[0]) + ' - ' + str(
            self.dimensions[0]) + '/2)/(sqrt(2)*epsilon)))/2-(1 - tanh((x[0] - ' + str(self.center[0]) + ' + ' + str(
            self.dimensions[0]) + '/2)/(sqrt(2)*epsilon)))/2)',
                         '((1 - tanh((x[1] - ' + str(self.center[1]) + ' - ' + str(
                             self.dimensions[1]) + '/2)/(sqrt(2)*epsilon)))/2-(1 - tanh((x[1] - ' + str(
                             self.center[1]) + ' + ' + str(self.dimensions[1]) + '/2)/(sqrt(2)*epsilon)))/2)'])


# generates a string expression representing a circle with a given radius and center coordinates
def circle_string_expression(radius, center):  # 生成一个表示具有给定半径和中心坐标的圆的字符串表达式
    return '(1-tanh((sqrt(pow((x[0]-' + str(center[0]) + '), 2)+pow((x[1]-' + str(center[1]) + '), 2))-' + str(
        radius) + ')/(sqrt(2)*epsilon)))/2'


# Here, various model parameters are defined:


class solidSintering:  # 定义一个类solidSintering
    def __init__(self, a, b, kc, keta, Dsf, Dgb, Dvol, L, mu, zeta,
                 Np=2, Neps=3.01, viscosity_ratio=1):
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
        self.mu = mu
        self.zeta = zeta - (2 / 3) * mu
        self.visc_ratio = viscosity_ratio

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
                                      + 2 * self.Dgb * sum(
                    [eta[i] ** 2 * (sum([eta[j] ** 2 for j in range(self.Np)]) - eta[i] ** 2) for i in range(self.Np)]))

    def fmu(self, C, eta):
        return self.df(C) + self.dN(C) * (self.S(eta) + self.grad_eta(eta))

    def sigma(self, u, C):
        # for this version it is important to keep viscosity_ratio<1
        # the defined sigma here does not include contribution from Lagrangian multiplier
        return (self.visc_ratio + (1 - self.visc_ratio) * self.N(C)) \
            * self.mu * (grad(u) + transpose(grad(u)))

    # Version 1
    # def interface_stress(self,C,mu_c,eta,mu_eta):
    # return mu_c*grad(C)+sum([mu_eta[i]*grad(eta[i]) for i in range(self.Np)])
    # Version 2
    def interface_stress(self, C, eta):
        return self.kc * outer(grad(C), grad(C)) + self.keta * self.N(C) * sum(
            [outer(grad(eta[i]), grad(eta[i])) for i in range(self.Np)])


# parameter Normalization


pde = solidSintering(a=17.32, b=1.00, kc=19.48, keta=6.75,
                     Dsf=22.88, Dgb=2.29, Dvol=0.14, L=26.33, mu=10, zeta=0.00,
                     viscosity_ratio=0.01, Np=9)

dt = 5e-03

# block with mesh, spaces and weak forms

# Form compiler options
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True

# Create mesh and build function space
# mesh = UnitSquareMesh(64, 64)
domain_size = [128, 64]
mesh = RectangleMesh(Point(0, 0), Point(domain_size[0], domain_size[1]), 192, 96)

# FunctionSpace consists of c, mu, and etas
P1 = FiniteElement("CG", triangle, 1)
C_Space = FunctionSpace(mesh, MixedElement([P1, P1]))
Eta_Space = FunctionSpace(mesh, P1)

# Taylor-Hood function space for velocity
PV = VectorElement("CG", triangle, 2)
VS = FunctionSpace(mesh, MixedElement([PV, P1]))

# Create a separate space to output the stress
PS = TensorElement("CG", triangle, 1)
WS = FunctionSpace(mesh, PS)

# Trial and test functions of the space ``ME`` are now defined::

# Define trial and test functions
du = TrialFunction(C_Space)
c_test, mu_test = TestFunctions(C_Space)
deta = TrialFunction(Eta_Space)
eta_test = TestFunction(Eta_Space)

# Define functions
u_new = Function(C_Space)  # current solution
eta_new = [Function(Eta_Space) for i in range(pde.Np)]
u_prev = Function(C_Space)  # solution from previous converged step
eta_prev = [Function(Eta_Space) for i in range(pde.Np)]

# Split mixed functions
dc, dmu = split(du)
c_new, mu_new = split(u_new)
c_prev, mu_prev = split(u_prev)

# All functions for velocity space
dv, dp = TrialFunctions(VS)
v_test, p_test = TestFunctions(VS)
v_combined = Function(VS)
v, p = split(v_combined)


def boundary(x, on_boundary):
    return on_boundary


bc = DirichletBC(VS.sub(0), Constant((0, 0)), boundary)

# Create initial conditions and interpolate

particle_size = int(domain_size[0] / 8)
particle_numbers = [3, 3]

# particle_centers=[[34,32],[44,22],[44,42],[54,32],
#                   [74,32],[84,22],[84,42],[94,32]]
# particle_centers=[[48,48],[64,48],[80,48],
#                   [48,16],[64,16],[80,16],
#                   [32,32],[96,32]]
template_particle = obstacle(degree=1)
template_particle.set_center([int(domain_size[0] / 2), int(domain_size[1] / 2)])
# no corners
# template_particle.set_dimensions([particle_numbers[0]*particle_size,(particle_numbers[1])*particle_size])
# with corners
template_particle.set_dimensions([particle_numbers[0] * particle_size, (particle_numbers[1] - 2) * particle_size])
template_particle.set_epsilon(1)
particle_radii = [int(particle_size / 2)] * (pde.Np - 1)

particle_centers = [[int(template_particle.center[0] + (i - (particle_numbers[0] - 1) / 2) * particle_size),
                     int(template_particle.center[1] + 0.5 * template_particle.dimensions[1] + particle_radii[i])] for i
                    in range(particle_numbers[0])] \
                   + [[int(template_particle.center[0] + (i - (particle_numbers[0] - 1) / 2) * particle_size), int(
    template_particle.center[1] - 0.5 * template_particle.dimensions[1] - particle_radii[particle_numbers[0] + i])] for
                      i in range(particle_numbers[0])] \
                   + [[int(
    template_particle.center[0] + 0.5 * template_particle.dimensions[0] + particle_radii[2 * particle_numbers[0] + i]),
    int(template_particle.center[1] + (i - (particle_numbers[1] - 1) / 2) * particle_size)] for i in
                       range(particle_numbers[1])] \
                   + [[int(template_particle.center[0] - 0.5 * template_particle.dimensions[0] - particle_radii[
    2 * particle_numbers[0] + particle_numbers[1] + i]),
                       int(template_particle.center[1] + (i - (particle_numbers[1] - 1) / 2) * particle_size)] for i in
                      range(particle_numbers[1])]

eta_init = [VoronoiParticle(degree=1) for k in range(pde.Np - 1)] \
           + [template_particle]

for k in range(pde.Np - 1):
    #  Sets the others attribute of the eta_init[k] object to a copy of the particle_centers list
    eta_init[k].others = particle_centers.copy()
    # Sets the center attribute of the eta_init[k] object to the k-th element removed from the others attribute list
    eta_init[k].center = eta_init[k].others.pop(k)
    eta_init[k].epsilon = 1
    # Sets the obstacle attribute of the eta_init[k] object to the value of template_particle
    eta_init[k].obstacle = template_particle

c_init = Expression(
    '+'.join([circle_string_expression(particle_radii[k], particle_centers[k]) for k in range(pde.Np - 1)] \
             + [template_particle.to_string_expression()]),
    degree=1, epsilon=Constant(template_particle.epsilon))
mu_init = Constant(0)

c0 = interpolate(c_init, C_Space.sub(0).collapse())
mu0 = interpolate(mu_init, C_Space.sub(1).collapse())
eta0 = [interpolate(eta_init[i], Eta_Space) for i in range(pde.Np)]

# .. index:: semi_implicit c for laplace
theta = 0.5
c_mid = (1.0 - theta) * c_prev + theta * dc

# The linear form
FC1 = dc * c_test * dx - c_prev * c_test * dx \
      + dt * dot(pde.M(c_prev, eta_prev) * grad(dmu), grad(c_test)) * dx \
      - dt * dot(v * dc, grad(c_test)) * dx
FC2 = dmu * mu_test * dx - pde.kc * dot(grad(c_mid), grad(mu_test)) * dx \
      - pde.fmu(c_prev, eta_prev) * mu_test * dx
FC = FC1 + FC2
FCL, FCR = lhs(FC), rhs(FC)

Feta = [dot(deta, eta_test) * dx - dot(eta_prev[i], eta_test) * dx \
        + dt * dot(v, grad(deta)) * eta_test * dx \
        + dt * pde.L * inner(pde.keta * pde.N(c_prev) * grad((1.0 - theta) * eta_prev[i] + theta * deta),
                             grad(eta_test)) * dx \
        + dt * pde.L * pde.N(c_prev) * pde.dS(eta_prev, i) * eta_test * dx
        for i in range(pde.Np)]

FetaL = [lhs(Feta[i]) for i in range(pde.Np)]
FetaR = [rhs(Feta[i]) for i in range(pde.Np)]

E = (0.5 * pde.kc * dot(grad(c_new), grad(c_new)) + pde.f(c_new) + pde.N(c_new) * (
        pde.S(eta_new) + pde.grad_eta(eta_new))) * dx

# Linear Stokes equation for velocity
VF = inner(pde.sigma(dv, c_new), grad(v_test)) * dx + div(v_test) * dp * dx + p_test * div(dv) * dx \
     - inner(pde.interface_stress(c_new, eta_new), grad(v_test)) * dx
VL, VR = lhs(VF), rhs(VF)

# define equations for stress and eta
stress_test = TestFunction(WS)
dstress = TrialFunction(WS)
stress_L = inner(dstress, stress_test) * dx
stress_R = inner(pde.sigma(v, c_new) + p * Identity(2)
                 - (0.5 * pde.kc * dot(grad(c_new), grad(c_new)) + pde.f(c_new)
                    + pde.N(c_new) * (pde.S(eta_new) + pde.grad_eta(eta_new))) * Identity(2), stress_test) * dx

etawr_L = deta * eta_test * dx
etawr_R = (c_new ** 2) * sum([eta_new[j] ** 2 for j in range(pde.Np)]) * eta_test * dx

# block with solver and solution
params = {'linear_solver': 'tfqmr',
          'preconditioner': 'ilu'}

Stokes_params = {'linear_solver': 'mumps'}
# Stokes_params={'linear_solver': 'gmres',
#        'preconditioner': 'hypre_amg'}

# Reset initial conditions
assign(u_new, [c0, mu0])
u_prev.assign(u_new)
for i in range(pde.Np):
    assign(eta_new[i], eta0[i])
    eta_prev[i].assign(eta_new[i])

# Solving Stokes at initial time
assign(v_combined, [interpolate(Constant((0, 0)), VS.sub(0).collapse()),
                    interpolate(Constant(0), VS.sub(1).collapse())])
solve(VL == VR, v_combined, bc, solver_parameters=Stokes_params)

# Output file
file = File(rootpath + "FreeSintering/test/output.pvd", "compressed")
cwr = u_new.split()[0]
cwr.rename('C', 'C')
etawr = Function(Eta_Space)
etawr.rename('eta', 'eta')
vwr = v_combined.split()[0]
vwr.rename('v', 'v')

# Variables for writing stress and strainRate
stress = Function(WS)
stress.rename('stress', 'stress')

# Step in time
t = 0.0
n = 0
file << (cwr, t)
# solve for c^2*eta^2 and store it in etawr
solve(etawr_L == etawr_R, etawr)
file << (etawr, t)
file << (vwr, t)
solve(stress_L == stress_R, stress)
file << (stress, t)

numsteps = 0
T = numsteps * dt
energy = np.empty(numsteps + 1)
energy[n] = assemble(E)

print("the energy at time", t, "is", energy[n])

while (t < T):
    n += 1
    t = round(n * dt, 8)
    print('starting computations at time', t)
    u_prev.assign(u_new)
    for i in range(pde.Np): eta_prev[i].assign(eta_new[i])
    print('--solving for C')
    solve(FCL == FCR, u_new, solver_parameters=params)
    print('--solving for eta')
    for i in range(pde.Np):
        solve(FetaL[i] == FetaR[i], eta_new[i], solver_parameters=params)

    print('--solving Stokes')
    solve(VL == VR, v_combined, bc, solver_parameters=Stokes_params)

    energy[n] = assemble(E)
    print("-the energy at time", t, "is", energy[n])

    if np.mod(n, 1) == 0:
        print('--writing the data to file--')
        file << (cwr, t)
        solve(etawr_L == etawr_R, etawr)
        file << (etawr, t)
        file << (vwr, t)
        solve(stress_L == stress_R, stress)
        file << (stress, t)
