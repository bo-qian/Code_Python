import os
import shutil


source_folder = "E:\Output"
target_folder1 = "E:\Solution\C"
target_folder2 = "E:\Solution\V"
target_folder3 = "E:\Solution\S"
os.remove(source_folder + "\solution.pvd")
os.makedirs(target_folder1, exist_ok=True)
os.makedirs(target_folder2, exist_ok=True)
os.makedirs(target_folder3, exist_ok=True)
C = 0
V = 1
S = 2
for filename in os.listdir(source_folder):
    file_number = int(''.join(filter(str.isdigit, filename)))
    if file_number == C:
        source_file = os.path.join(source_folder, filename)
        target_file1 = os.path.join(target_folder1, filename)
        shutil.move(source_file, target_file1)
        C += 3
    elif file_number == V:
        source_file = os.path.join(source_folder, filename)
        target_file2 = os.path.join(target_folder2, filename)
        shutil.move(source_file, target_file2)
        V += 3
    elif file_number == S:
        source_file = os.path.join(source_folder, filename)
        target_file3 = os.path.join(target_folder3, filename)
        shutil.move(source_file, target_file3)
        S += 3








