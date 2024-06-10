import os
import shutil

# 对vtu文件进行分类并移动到对应的文件夹中
# 定义源文件夹和目标文件夹
source_folder = os.getcwd()
target_folder_c = os.getcwd() + "/c"
target_folder_eta = os.getcwd() + "/eta"
target_folder_v = os.getcwd() + "/v"
target_folder_stress = os.getcwd() + "/stress"
# 创建目标文件夹
os.makedirs(target_folder_c, exist_ok=True)
os.makedirs(target_folder_eta, exist_ok=True)
os.makedirs(target_folder_v, exist_ok=True)
os.makedirs(target_folder_stress, exist_ok=True)
# 遍历源文件夹中的文件
for filename in os.listdir(source_folder):
    # 检查文件是否为vtu文件
    if filename.endswith(".vtu"):
        # 提取文件名中的数字
        file_number = int(''.join(filter(str.isdigit, filename)))
        # 根据文件名中的数字将文件移动到相应的文件夹中
        if file_number % 4 == 0:
            target_folder = target_folder_c
        elif file_number % 4 == 1:
            target_folder = target_folder_eta
        elif file_number % 4 == 2:
            target_folder = target_folder_v
        else:
            target_folder = target_folder_stress
        # 移动文件
        shutil.move(os.path.join(source_folder, filename), target_folder)
        # 获取目标文件夹中当前文件数量
        num_files = len(os.listdir(target_folder))
        # 构建新文件名
        new_filename = f"{os.path.basename(target_folder)}_{num_files - 1}.vtu"
        # 获取移动后的文件路径
        moved_file_path = os.path.join(target_folder, filename)
        # 获取重命名后的文件路径
        renamed_file_path = os.path.join(target_folder, new_filename)
        # 重命名文件
        os.rename(moved_file_path, renamed_file_path)

print("文件分类完成。")
