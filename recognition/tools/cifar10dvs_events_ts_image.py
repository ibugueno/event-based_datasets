import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import cv2
import shutil
from dv import LegacyAedatFile
from dv import AedatFile
from PIL import Image

from scipy.stats import rankdata


class TimestampImage:

    def __init__(self, sensor_size):
        self.sensor_size = sensor_size
        self.num_pixels = sensor_size[0]*sensor_size[1]
        self.image = np.ones(sensor_size)

    def set_init(self, value):
        self.image = np.ones_like(self.image)*value

    def add_event(self, x, y, t, p):
        self.image[int(y), int(x)] = t

    def add_events(self, xs, ys, ts, ps):
        for x, y, t in zip(xs, ys, ts):
            self.add_event(x, y, t, 0)

    def get_image(self):
        sort_args = rankdata(self.image, method='dense')
        sort_args = sort_args-1
        sort_args = sort_args.reshape(self.sensor_size)
        sort_args = sort_args/np.max(sort_args)
        return sort_args


class TimeStampImageVisualizer():

    def __init__(self, sensor_size):
        self.ts_img = TimestampImage(sensor_size)
        self.sensor_size = sensor_size

    def plot_events(self, data, save_path):
        xs, ys, ts, ps = data['x'], data['y'], data['timestamp']*1e-6, data['polarity'] *2-1
        self.ts_img.set_init(ts[0])
        self.ts_img.add_events(xs, ys, ts, ps)
        timestamp_image = self.ts_img.get_image()

        timestamp_image = np.rot90(timestamp_image, k=1) #rotar
        timestamp_image = np.flip(timestamp_image, axis=1) #reflejar

        # Aplicar el mapa de colores "viridis"
        cmap = plt.get_cmap('viridis')
        img_color = cmap(timestamp_image, bytes=True)

        # Convertir la imagen a modo RGB
        img_rgb = img_color[:, :, :3]  # Tomar solo los canales RGB, omitiendo el canal alfa

        # Crear una imagen PIL desde el arreglo RGB
        img_pil = Image.fromarray(img_rgb, 'RGB')

        # Guardar la imagen sin bordes blancos
        img_pil.save(save_path)
        

if __name__ == "__main__":

    #'''
    path_aedat = '../CIFAR10DVS/CIFAR10DVS_aedat/'
    path_frame = '../CIFAR10DVS/event_frames_ts/'

    try:
        shutil.rmtree(path_frame)
    except:
        pass

    os.mkdir(path_frame)

    for path_1 in sorted( os.listdir(path_aedat) ):        
        os.mkdir(path_frame + path_1 + '/')
        
        for path_2 in sorted( os.listdir(path_aedat + path_1) ):
            os.mkdir(path_frame + path_1 + '/' + path_2 + '/')

            for path_3 in sorted( os.listdir(path_aedat + path_1 + '/' + path_2) ):
                
                input_path  = path_aedat + path_1 + '/' + path_2 + '/' + path_3
                output_path = path_frame + path_1 + '/' + path_2 + '/' + path_3[:-7] + '.jpg'

                print(input_path)
                print(output_path, '\n')

                with AedatFile(input_path) as f:
                    np_array = np.hstack([packet for packet in f['events'].numpy()])

                #'''
                #np_array = np.load(input_path)
                
                #print(np_array)

                sensor_size = (128, 128)
                visualizer = TimeStampImageVisualizer(sensor_size)
                visualizer.plot_events(np_array, output_path)

                #'''

                #break
            
            #break
            
        #break

                
                

