import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import os
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 32,
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic',
    'mathtext.bf': 'Times New Roman:bold',
    'axes.unicode_minus': False
})

# plt.rcParams.update({
#     'text.usetex': True,  # 启用 LaTeX
#     'font.family': 'serif',  # 使用衬线字体
#     'font.serif': 'Times New Roman',  # 指定 Times New Roman
#     'font.size': 32,  # 全局字体大小
#     'legend.fontsize': 28,  # 图例字体大小
#     'legend.fontfamily': 'serif',  # 图例字体族
#     'legend.fontstyle': 'normal',  # 图例字体样式
#     'axes.linewidth': 3,  # 坐标轴线宽
#     'xtick.major.width': 3,  # X轴主刻度线宽
#     'ytick.major.width': 3  # Y轴主刻度线宽
# })

base_path = "/mnt/d/OneDrive/Science_Research/2.Secondstage_CodeAndData_Python/Code_Python"
base_directory = "/mnt/d/OneDrive/Science_Research"


def curve_comparison_two(path, label, x_1, y_1):
    # Function to plot curve group ********************************************************************************
    file_directory = os.getcwd() + "/Comparison of Viscosity Sintering"
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

        data_csv = [None for _ in path]
        for i in range(len(path)):
            data_csv[i] = pd.read_csv(path[i])
            # 获取标题行
            title = data_csv[i].columns
            # 提取数据列
            data_x_1 = data_csv[i].iloc[:, x_1]
            data_y_1 = data_csv[i].iloc[:, y_1]
            ax.plot(data_x_1, data_y_1, label=fr'{label[i]}', linewidth=3)

        offset = ax.yaxis.get_offset_text()
        transform = offset.get_transform()
        offset.set_transform(transform + plt.matplotlib.transforms.ScaledTranslation(0, 5 / 72., fig.dpi_scale_trans))
        plt.title(f'Comparison of {title[y_1]}', pad=20, fontweight='bold')
        plt.xlabel(f'{title[x_1]}', fontweight='bold')
        plt.ylabel(f'{title[y_1]}', fontweight='bold')
        plt.tight_layout()
        plt.legend(fontsize='small')
        plt.savefig(file_directory + f'/Comparison of {title[y_1]}.png', dpi=100, bbox_inches='tight')
        plt.close()


# data_path = [
#     base_path + "/Viscosity_Sintering/PolyParticle_Viscosity_Simulation/Test-1 (June 30, 2024, 19-33-25)/Data/various_data.csv",
#     base_path + "/Viscosity_Sintering/PolyParticle_Viscosity_Simulation_GBDF2/Test-1 (July 02, 2024, 20-32-57)/Data/various_data.csv",
#     base_path + "/Viscosity_Sintering/PolyParticle_Viscosity_Simulation_GBDF2_cluster/Test-3 (July 03, 2024, 21-49-16)/Data/various_data.csv",
#     base_path + "/Viscosity_Sintering/PolyParticle_Viscosity_Simulation_GBDF2_cluster/Test-4 (July 03, 2024, 21-49-45)/Data/various_data.csv",
#     base_path + "/Viscosity_Sintering/PolyParticle_Viscosity_Simulation_GBDF2_cluster/Test-5 (July 03, 2024, 21-50-01)/Data/various_data.csv"]
# label_data = ["Without GBDF2", r"With GBDF2 ($\beta = 1$)", r"With GBDF2 ($\beta = 3$)", r"With GBDF2 ($\beta = 5$)",
#               r"With GBDF2 ($\beta = 7$)"]

# curve_comparison_two(
#     [
#         base_path + "/Viscosity_Sintering/PolyParticle_Viscosity_Simulation_testing/ (July 15, 2024, 16-30-07)/Data/various_data.csv",
#         base_path + "/Viscosity_Sintering/PolyParticle_Viscosity_Simulation_v1/PPSS-V1-4 (July 15, 2024, 16-27-11)/Data/various_data.csv"
#     ],
#     ["Viscosity Sintering with Nothing Changed", "Viscosity Sintering with Reboot Function"],
#     4,
#     1
# )

def plot_curve_group_comparison(end_curve, path, label, name):
    file_directory = os.getcwd() + "/Curve Group Comparison of Solid-state Sintering"
    line_width = 4
    if not os.path.exists(file_directory):
        os.makedirs(file_directory)

    markers = ['o', 's', '^', 'D', 'v']
    data_csv = [None for _ in path]
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

            for j in range(len(path)):
                data_csv[j] = pd.read_csv(path[j], skiprows=2)
                header = pd.read_csv(path[j], nrows=2, header=None)
                data_curve = data_csv[j].values
                name_unit = header.values

                ax.plot(data_curve[:end_curve, 0], data_curve[:end_curve, i + 1], label=rf"{label[j]}",
                        linewidth=3)  # """marker=markers[j % len(markers)], markevery=45 + 15 * j, markersize=8"""
                ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

            ax.set_title(f'{name_unit[0, i + 1]}', pad=20, fontweight='bold')
            ax.set_xlabel(rf'{name_unit[0, 0]} ($\mathrm{{{name_unit[1, 0]}}}$)', fontweight='bold')
            ax.set_ylabel(rf'{name_unit[0, i + 1]} ($\mathrm{{{name_unit[1, i + 1]}}}$)', fontweight='bold')

            offset = ax.yaxis.get_offset_text()
            transform = offset.get_transform()
            offset.set_transform(
                transform + plt.matplotlib.transforms.ScaledTranslation(0, 5 / 72., fig.dpi_scale_trans))
            ax.legend(fontsize='small')

        plt.tight_layout()
        plt.savefig(file_directory + f'/{name}.png', dpi=100, bbox_inches='tight')
        plt.close()


# plot_curve_group_comparison(
#     6840,
#     [
#         base_path + "/Template_Sintering/PolyParticle_Simulation_cluster/Test-1 (July 01, 2024, 20-26-04)/Data/energy_data.csv",
#         base_path + "/Template_Sintering/TGG_Simulation_Cluster/TGGSS-43 (July 06, 2024, 16-49-18)/Data/various_data.csv"
#     ],
#     ["Multi-particle Sintering", "Template Sintering"],
#     "Multi-particle Sintering vs Template Sintering"
# )

# plot_curve_group_comparison(
#     None,
#     [
#         base_path + "/Template_Sintering/PolyParticle_Simulation_v1/PPSS-V1-12 (July 22, 2024, 15-33-02)/Data/various_data.csv",
#         base_path + "/Template_Sintering/PolyParticle_Simulation_v1/PPSS-V1-13 (July 22, 2024, 16-03-23)/Data/various_data.csv"
#     ],
#     [r"$\epsilon=1.0$", r"$\epsilon=0.72$"],
#     "epsilon=1.0 vs epsilon=0.72"
# )

# plot_curve_group_comparison(
#     200,
#     [
#         base_path + "/Template_Sintering/PolyParticle_Simulation_v1/PPSS-V1-1 (July 11, 2024, 14-58-23)/Data/various_data.csv",
#         base_path + "/Template_Sintering/PolyParticle_Simulation/PPSS-25 (July 11, 2024, 14-57-43)/Data/various_data.csv"
#     ],
#     ["With Interruption", "Without Interruption"],
#     "With Interruption vs Without Interruption"
# )


# Function to plot curve group ********************************************************************************
def plot_curve_group(name, Data_directory):
    file_directory = os.getcwd() + "/Curve Group of Solid-state Sintering"
    line_width = 4
    if not os.path.exists(file_directory):
        os.makedirs(file_directory)

    data = pd.read_csv(Data_directory, skiprows=2)
    header = pd.read_csv(Data_directory, nrows=2, header=None)
    data_curve = data.iloc
    name_unit = header.iloc

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
            ax.plot(data_curve[:, 0], data_curve[:, i + 1], label=f"{name_unit[0, i + 1]}", linewidth=line_width,
                    color='black')
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

            ax.set_title(name_unit[0, i + 1], pad=20, fontweight='bold')
            ax.set_xlabel(rf'{name_unit[0, 0]} ($\mathrm{{{name_unit[1, 0]}}}$)', fontweight='bold')
            ax.set_ylabel(rf'{name_unit[0, i + 1]} ($\mathrm{{{name_unit[1, i + 1]}}}$)', fontweight='bold')

            offset = ax.yaxis.get_offset_text()
            transform = offset.get_transform()
            offset.set_transform(
                transform + plt.matplotlib.transforms.ScaledTranslation(0, 5 / 72., fig.dpi_scale_trans))

        plt.tight_layout()
        plt.savefig(file_directory + f'/{name}.png', dpi=100, bbox_inches='tight')
        plt.close()


# # plot_curve_group("Poly Particle Sintering", base_path + "/Template_Sintering/PolyParticle_Simulation_cluster/Test-6 (June 30, 2024, 20-32-34)/Data/energy_data.csv")
# plot_curve_group("TGG have not fixed - same number of data", base_directory + "/Process/240711第十八次汇报/data_without_fixed_TGG/various_data - 副本.csv")
# plot_curve_group("TGG have fixed", base_directory + "/Process/240711第十八次汇报/data_with_fixed_TGG/various_data.csv")
plot_curve_group("Four particles Sintering", base_path + "/Template_Sintering/PolyParticle_Simulation_cluster/PPSS-V1-12 (July 24, 2024, 14-49-39)/Data/various_data.csv")


def plot_curve(x, y, name, Data_directory):
    file_directory = os.getcwd() + "/Curve Group of Solid-state Sintering"
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

        data = pd.read_csv(Data_directory, skiprows=2)
        header = pd.read_csv(Data_directory, nrows=2, header=None)
        data_curve = data.iloc
        name_unit = header.iloc

        # 提取数据列

        plt.plot(data_curve[:, x], data_curve[:, y], label=fr'{name_unit[0, y]}', linewidth=3, color='black')

        offset = ax.yaxis.get_offset_text()
        transform = offset.get_transform()
        offset.set_transform(transform + plt.matplotlib.transforms.ScaledTranslation(0, 5 / 72., fig.dpi_scale_trans))
        plt.title(f"{name_unit[0, y]}", pad=20, fontweight='bold')
        plt.xlabel(f'{name_unit[0, x]} ({name_unit[1, x]})', fontweight='bold')
        plt.ylabel(f'{name_unit[0, y]} ({name_unit[1, y]})', fontweight='bold')
        plt.tight_layout()
        # plt.legend(fontsize='small')
        plt.savefig(file_directory + f'/{name} - {name_unit[0, y]}.png', dpi=100, bbox_inches='tight')
        plt.close()


plot_curve(0, 3, "Four Particles Sintering", base_path + "/Template_Sintering/PolyParticle_Simulation_cluster/PPSS-V1-12 (July 24, 2024, 14-49-39)/Data/various_data.csv")


def plot_three_dimension(r, epsilon, center, name):
    file_directory = os.getcwd() + "/Figures of Three Dimension"
    if not os.path.exists(file_directory):
        os.makedirs(file_directory)

    # 定义函数 c_0
    def c_0(x, y):
        term1 = 1 / 2 * (1 - np.tanh(
            (np.sqrt((x - 75) ** 2 + (y - 75) ** 2) - 25) * (2 * np.arctanh(0.9) / 3)))
        term2 = 1 / 2 * (1 - np.tanh(
            (np.sqrt((x - 125) ** 2 + (y - 75) ** 2) - 25) * (2 * np.arctanh(0.9) / 3)))
        return term1 + term2

    with plt.rc_context(
            {'font.family': 'serif', 'font.serif': ['Times New Roman'], 'font.weight': 'bold', 'font.size': 32}):
        fig, ax = plt.subplots(figsize=(12, 9), dpi=100)
        x = np.linspace(0, 200, 2000)
        y = np.linspace(0, 150, 1500)
        x, y = np.meshgrid(x, y)

        # 计算 z 值
        # z = (1 - np.tanh((np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2) - r) / (np.sqrt(2) * epsilon))) / 2
        z = c_0(x, y)
        # 创建 3D 图

        # 使用 imshow 函数绘制二维颜色图
        cax = ax.imshow(z, extent=[0, 200, 0, 150], origin='lower', cmap='coolwarm', aspect='auto')

        # 添加颜色条
        cbar = fig.colorbar(cax, ax=ax, orientation='vertical', shrink=0.76)
        cbar.set_label(r'$c_0$')
        ax.set_aspect('equal', adjustable='box')
        # 设置轴标签
        ax.set_xlabel('X-axis', fontweight='bold')
        ax.set_ylabel('Y-axis', fontweight='bold')

        # 调整边距
        plt.tight_layout()

        # 保存图形

        plt.savefig(file_directory + f'/{name}_3D.png', dpi=100, bbox_inches='tight')
        plt.close()


# plot_three_dimension(1, 0.1, [0, 0], "two particle")

def plot_expression(r_epsilon, name):
    file_directory = os.getcwd() + "/Figures of Two Dimension"
    if not os.path.exists(file_directory):
        os.makedirs(file_directory)

    with plt.rc_context(
            {'font.family': 'serif', 'font.serif': ['Times New Roman'], 'font.weight': 'bold', 'font.size': 32}):
        fig, ax = plt.subplots(figsize=(12, 9), dpi=100)
        ax.spines['top'].set_linewidth(3)
        ax.spines['bottom'].set_linewidth(3)
        ax.spines['left'].set_linewidth(3)
        ax.spines['right'].set_linewidth(3)

        plt.tick_params(axis='both', direction='in', width=3, which='both', pad=10)

        print(f"当d=3，w=0.05时，\ne={3 / (2 * np.sqrt(2) * np.arctanh(1 - 0.1))}\n")

        print(f"当d=3，e=1时，\nw={(1 - np.tanh(3 / (2 * np.sqrt(2)))) / 2}\n")

        print(f"当e=1，w=1e-3时，\nd={2 * np.sqrt(2) * np.arctanh(1 - 2e-3)}\n")

        for i in range(len(r_epsilon)):
            r = r_epsilon[i][0]
            e = r_epsilon[i][1]
            d = r_epsilon[i][2]
            w = r_epsilon[i][3]
            w1 = 1e-3

            x = np.linspace(r - d / 2 - 2, r + d / 2 + 2, 400)
            x_1 = np.linspace(-15, 15, 4000)
            y = (1 - np.tanh((x - r) / (np.sqrt(2) * e))) / 2
            y_1 = (1 - np.tanh(x_1 * (2 * np.arctanh(1 - 2 * w1))/ d)) / 2
            y_2 = (1 - np.tanh(x_1 / (np.sqrt(2) * e))) / 2



            # plt.plot(x, y, linewidth=3,
            #          label=rf"$y=\frac{{1}}{{2}}\left[1 - \tan\left(\frac{{x - {r}}}{{\sqrt{{2}} \cdot {e}}}\right)\right]$")
            # plt.plot(x_1, y_1, linewidth=3, label=f"$r_0={r}$   $\\epsilon={e}$   $\\delta={d}$")
            plt.plot(x_1, y_2, linewidth=3, label=f"$r_0={r}$   $\\epsilon={e}$   $\\delta={d}$")
            plt.fill_between([np.sqrt(2) * e * np.arctanh(1 - 2 * (1 - w1)), np.sqrt(2) * e * np.arctanh(1 - 2 * w1)], 0, 1, color='lightgreen', alpha=0.5)
            plt.axvline(x=np.sqrt(2) * e * np.arctanh(1 - 2 * (1 - w1)), color='b', linestyle='--', linewidth=3)
            plt.text(np.sqrt(2) * e * np.arctanh(1 - 2 * (1 - w1)) - 9, 0.5, f"$x_1=\\sqrt{{2}}\\epsilon\\cdot\\mathrm{{arctanh}}(1-2(1-\\omega))$", fontsize=20, color='b', ha='left', va='center')
            plt.axvline(x=np.sqrt(2) * e * np.arctanh(1 - 2 * w1), color='r', linestyle='--', linewidth=3)
            plt.text(np.sqrt(2) * e * np.arctanh(1 - 2 * w1) + 0.5, 0.5, f"$x_2=\\sqrt{{2}}\\epsilon\\cdot\\mathrm{{arctanh}}(1-2\\omega)$", fontsize=20,
                     color='r', ha='left', va='center')

        plt.legend(fontsize='small')
        plt.xlabel(r'$x$', fontweight='bold')
        plt.ylabel(r'$y$', fontweight='bold')
        plt.grid(True)
        # plt.title(rf"$y=\frac{{1}}{{2}}\left[1 - \tanh\left(\frac{{r - r_0}}{{\sqrt{{2}} \cdot \epsilon}}\right)\right]$", pad=40, fontweight='normal')
        plt.tight_layout()
    plt.savefig(file_directory + f'/{name}.png', dpi=100, bbox_inches='tight')
    plt.close()


# plot_expression([(25, 0.7, 3),(25, 0.5, 2),(25, 0.2, 1)], "expression_2D")
# plot_expression([ (25, 1, 3, 1e-3)], "expression_2D")
