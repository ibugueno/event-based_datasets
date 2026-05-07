import numpy as np
import os
from glob import glob

# Parámetros de la cámara
width, height = 240,180

# Directorios de entrada y salida
input_root = '/media/ignacio/KINGSTON/event-cameras/redat/npy'
output_root = f'/media/ignacio/KINGSTON/event-cameras/redat/output'


#Descargar archivos, extraer y solo dejar npy