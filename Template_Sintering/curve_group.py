import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# 定义数据路径
data_1_path = os.getcwd() + "PolyParticle_Viscosity_Simulation/Test-1 (June 30, 2024, 19-33-25)/Data/various_data.csv"
data_2_path = os.getcwd() + "PolyParticle_Viscosity_Simulation_GBDF2/Test-1 (July 02, 2024, 20-32-57)/Data/various_data.csv"

# 检查文件路径是否存在
if not os.path.exists(data_1_path):
    print(f"Error: The file '{data_1_path}' does not exist.")
if not os.path.exists(data_2_path):
    print(f"Error: The file '{data_2_path}' does not exist.")

# 读取数据
data_1 = pd.read_csv(data_1_path)
data_2 = pd.read_csv(data_2_path)

# 获取标题行
title_1 = data_1.columns
title_2 = data_2.columns

print("Data 1 titles:", title_1)
print("Data 2 titles:", title_2)

# 提取数据列
x_1 = data_1.iloc[:, 4]
y_1 = data_1.iloc[:, 1]
x_2 = data_2.iloc[:, 4]
y_2 = data_2.iloc[:, 1]

# 检查数据列是否正确读取
print("x_1 head:", x_1.head())
print("y_1 head:", y_1.head())
print("x_2 head:", x_2.head())
print("y_2 head:", y_2.head())

# Function to plot curve group ********************************************************************************
file_directory = os.getcwd() + "/Curve Group"
line_width = 4
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

    plt.plot(x_1, y_1, label=fr'Without GBDF2', linewidth=3, color='black')
    plt.plot(x_2, y_2, label=fr'With GBDF2', linewidth=3, color='red')

    offset = ax.yaxis.get_offset_text()
    transform = offset.get_transform()
    offset.set_transform(transform + plt.matplotlib.transforms.ScaledTranslation(0, 5 / 72., fig.dpi_scale_trans))
    plt.title("Total Free Energy", pad=20, fontweight='bold')
    plt.xlabel(f'{title_1[4]}', fontweight='bold')
    plt.ylabel(f'{title_1[1]}', fontweight='bold')
    plt.tight_layout()
    plt.legend(fontsize='small')
    plt.savefig(file_directory + f'/energy with gbdf2 and none gbdf2.png', dpi=100, bbox_inches='tight')
    plt.close()
