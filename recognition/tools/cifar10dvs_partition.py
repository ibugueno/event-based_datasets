import os
import random
import shutil

# Rutas de las carpetas de entrada y salida
input_folder = '../CIFAR10DVS/CIFAR10DVS_aedat/train/'
output_folder = '../CIFAR10DVS/CIFAR10DVS_aedat_partition/'

# Nombres de las subcarpetas para entrenamiento, validación y prueba
subfolders = ['train', 'validation', 'test']

# Crear las subcarpetas de salida
for folder in subfolders:
    for class_folder in os.listdir(input_folder):
        os.makedirs(os.path.join(output_folder, folder, class_folder), exist_ok=True)

# Realizar la partición para cada clase
for class_folder in os.listdir(input_folder):
    class_path = os.path.join(input_folder, class_folder)
    all_files = os.listdir(class_path)
    sorted(all_files)

    for i, file_name in enumerate(all_files):
        source_path = os.path.join(class_path, file_name)
        if i < 600:
            destination_path = os.path.join(output_folder, 'train', class_folder, file_name)
        elif i < 800:
            destination_path = os.path.join(output_folder, 'validation', class_folder, file_name)
        else:
            destination_path = os.path.join(output_folder, 'test', class_folder, file_name)
        
        shutil.copyfile(source_path, destination_path)

print("Partición completada.")
