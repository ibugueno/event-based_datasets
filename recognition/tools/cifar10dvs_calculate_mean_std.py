import os
import numpy as np
from PIL import Image
from torchvision import transforms

# Función para calcular media y desviación estándar
def calculate_mean_std(data_path):
    image_paths = [os.path.join(root, filename)
                   for root, _, filenames in os.walk(data_path)
                   for filename in filenames if filename.endswith('.jpg')]

    num_images = len(image_paths)
    mean = np.zeros(3)
    std = np.zeros(3)

    for image_path in image_paths:
        img = Image.open(image_path)
        img_array = np.array(img) / 255.0
        mean += np.mean(img_array, axis=(0, 1))
        std += np.std(img_array, axis=(0, 1))

    mean /= num_images
    std /= num_images
    return mean, std

if __name__=='__main__':

    # Rutas a las carpetas de entrenamiento y prueba
    train_data_path = '../CIFAR10DVS/event_frames_ts/train/'
    test_data_path  = '../CIFAR10DVS/event_frames_ts/test/'
    val_data_path   = '../CIFAR10DVS/event_frames_ts/validation/'

    # Calcular media y desviación estándar para entrenamiento y prueba
    train_mean, train_std = calculate_mean_std(train_data_path)
    test_mean, test_std = calculate_mean_std(test_data_path)
    val_mean, val_std = calculate_mean_std(val_data_path)

    # Calcular media y desviación estándar para toda la base de datos
    total_mean = (train_mean + test_mean + val_mean) / 2
    total_std = (train_std + test_std + val_std) / 2

    print("Total Mean:", total_mean)
    print("Total Std:", total_std)
