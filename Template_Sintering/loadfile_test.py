from dolfin import *

# 设置文件名
filename = "checkpoint.h5"

# 定义网格和函数空间
mesh = UnitSquareMesh(32, 32)
V = FunctionSpace(mesh, "CG", 1)

# 定义两个函数（假设这是两个不同的变量）
u = Function(V)
v = Function(V)

# 在 u 和 v 上进行一些计算，这里仅作为示例
u.interpolate(Expression("x[0]*x[0] + x[1]*x[1]", degree=2))
v.interpolate(Expression("sin(pi*x[0])*cos(pi*x[1])", degree=2))

# 创建 HDF5File 对象来写入数据
checkpoint_file = HDF5File(MPI.comm_world, filename, "w")

# 将函数写入检查点文件中，分别存储到不同的组
checkpoint_file.write(u, "/group1/u")
checkpoint_file.write(v, "/group2/v")

# 关闭检查点文件
checkpoint_file.close()

# 现在可以在同一个程序中读取数据并进行进一步的计算
# 创建 HDF5File 对象来读取数据
checkpoint_file_read = HDF5File(MPI.comm_world, filename, "r")

# 定义新的函数变量来存储读取的数据
u_read = Function(V)
v_read = Function(V)

# 从检查点文件中读取数据到函数变量中，注意读取时路径要对应好
checkpoint_file_read.read(u_read, "/group1/u")
checkpoint_file_read.read(v_read, "/group2/v")

# 关闭检查点文件
checkpoint_file_read.close()

# 现在可以在 u_read 和 v_read 上进行进一步的计算
# 例如，计算 u_read 和 v_read 的某些操作
# 这里仅作为示例，可以根据具体需求进行修改
u_squared = u_read**2
v_plus_u = v_read + u_read

# 打印或者进行其他操作
print("u_read squared:", u_squared)
print("v_read + u_read:", v_plus_u)
