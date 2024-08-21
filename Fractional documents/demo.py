import pandas as pd
import matplotlib.pyplot as plt
import os


# 定义函数
def plot_temperature_by_month(data, month, temp_type='TMAX'):
    """
    根据月份和气温类型（最大值或最小值）绘制日期与气温的图表。

    参数：
    data (str): 数据文件路径。
    month (int): 要查询的月份（1-12）。
    temp_type (str): 'TMAX' 或 'TMIN'。
    """
    # 读取数据
    df = pd.read_csv(data)

    # 解析日期
    df['DATE'] = pd.to_datetime(df['DATE'], format='%Y/%m/%d')

    # 筛选指定月份的数据
    df = df[df['DATE'].dt.month == month]

    # 筛选气温类型
    if temp_type not in ['TMAX', 'TMIN']:
        raise ValueError("temp_type 参数必须是 'TMAX' 或 'TMIN'")

    # 绘图
    plt.figure(figsize=(10, 5))
    plt.plot(df['DATE'], df[temp_type], marker='o')
    plt.title(f'{temp_type} for Month {month}')
    plt.xlabel('Date')
    plt.ylabel(temp_type)
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# 使用示例
data_file = os.getcwd() + '/weather_simple.csv'  # 替换为你的数据文件路径
plot_temperature_by_month(data_file, 7, 'TMAX')
plot_temperature_by_month(data_file, 10, 'TMIN')
