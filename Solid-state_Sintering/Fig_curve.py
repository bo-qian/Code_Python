import numpy as np
import matplotlib.pyplot as plt

# 定义变量
r = np.linspace(20, 30, 400)  # 在 -10 到 10 的范围内生成 400 个点
r0 = 25  # 可以根据需要调整 r0 的值
epsilon_1 = 1  # 可以根据需要调整 epsilon 的值
epsilon_2 = 1.5  # 可以根据需要调整 epsilon 的值
eosilon_3 = 2  # 可以根据需要调整 epsilon 的值

# 定义表达式
mu_1 = (1 - np.tanh((r - r0) / (np.sqrt(2) * epsilon_1))) / 2
mu_2 = (1 - np.tanh((r - r0) / (np.sqrt(2) * epsilon_2))) / 2
mu_3 = (1 - np.tanh((r - r0) / (np.sqrt(2) * eosilon_3))) / 2

# 设置全局字体为 Arial
plt.rcParams['font.sans-serif'] = ['Arial']
# 设置全局字体大小
plt.rcParams['font.size'] = 32
# 设置图像大小和分辨率，并设置纵横比相同
fig, ax = plt.subplots(figsize=(12, 9), dpi=100)
# 设置四周边框的宽度
ax.spines['top'].set_linewidth(3)
ax.spines['bottom'].set_linewidth(3)
ax.spines['left'].set_linewidth(3)
ax.spines['right'].set_linewidth(3)
# 设置刻度标签朝内和厚度, 通过调整 pad 参数来设置刻度标签与轴的距离
plt.tick_params(axis='both', direction='in', width=3, which='both', pad=10)  # 设置刻度朝内，边框厚度为 2

# 绘制曲线
plt.plot(r, mu_1, label=r'$\varepsilon=$' + str(epsilon_1), linewidth=3)
plt.plot(r, mu_2, label=r'$\varepsilon=$' + str(epsilon_2), linewidth=3)
plt.plot(r, mu_3, label=r'$\varepsilon=$' + str(eosilon_3), linewidth=3)
plt.xlabel(r'$r$', fontweight='normal')
plt.ylabel(r'$\mu$', fontweight='normal')
plt.title(r'Plot of $\mu$ as a function of $r$', pad=20)
plt.legend()
plt.grid(True)
plt.show()




