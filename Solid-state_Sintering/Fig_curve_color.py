import numpy as np
import matplotlib.pyplot as plt

# 设置全局字体为 Arial
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif'],
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'DejaVu Serif',
    'mathtext.it': 'DejaVu Serif:italic',
    'mathtext.bf': 'DejaVu Serif:bold',
    'axes.unicode_minus': False  # 确保负号使用正确的字体
})
# 设置全局字体大小
plt.rcParams['font.size'] = 32

t = np.linspace(0, 2 * np.pi, 512)
data2d = np.sin(t)[:, np.newaxis] * np.cos(t)[np.newaxis, :]

fig, ax = plt.subplots(figsize=(12, 9), dpi=100)
im = ax.imshow(data2d, aspect=0.5)
ax.set_title('Pan on the colorbar to shift the color mapping\n'
             'Zoom on the colorbar to scale the color mapping')

fig.colorbar(im, ax=ax, label='Interactive colorbar')
plt.show()
