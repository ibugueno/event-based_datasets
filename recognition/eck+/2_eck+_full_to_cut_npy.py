import os
import numpy as np
from glob import glob
from tqdm import tqdm

# Configuración
width, height = 346, 260
frame_duration = 1.0 / 30  # porque los índices de los frame_XXXX.png provienen de un video a 30fps
bbox_padding = 0
extra_time = 0.1  # agregar 100ms al final
output_root = '../../../input/recognition/eck+/ddbb_cropped/e-ck+_346_full_cropped_events_100ms'

# Rutas
frames_root = '../../../input/recognition/eck+/data/e-ck+_frames_10fps'
bboxes_root = '../../../input/recognition/eck+/ddbb_cropped/ck+_frames_process_30fps/bboxes'
events_root = '../../../input/recognition/eck+/data/e-ck+_346_full_ms'

# Utilidad para extraer índice de frame
def extract_frame_index(filename):
    name = os.path.splitext(os.path.basename(filename))[0]
    return int(name.replace("frame_", ""))

# Extraer bounding box para un frame dado
def get_bbox(txt_path):
    with open(txt_path, 'r') as f:
        line = f.readline().strip()
    x_norm, y_norm, w_norm, h_norm = map(float, line.split())
    xmin = x_norm
    ymin = y_norm
    xmax = x_norm + w_norm
    ymax = y_norm + h_norm
    return [xmin, ymin, xmax, ymax]


# Crear carpeta de salida
os.makedirs(output_root, exist_ok=True)

# Recorremos Train y Test Set
for split in ['Train_Set', 'Test_Set']:
    print(f"\nProcesando {split}...")
    split_frame_path = os.path.join(frames_root, split)
    split_event_path = os.path.join(events_root, split)
    for class_id in sorted(os.listdir(split_frame_path)):
        class_frame_path = os.path.join(split_frame_path, class_id)
        class_event_path = os.path.join(split_event_path, class_id)
        for seq_id in sorted(os.listdir(class_frame_path)):
            frame_dir = os.path.join(class_frame_path, seq_id)
            frame_files = sorted(glob(os.path.join(frame_dir, 'frame_*.png')))
            if not frame_files:
                continue

            # Primer y último frame (en índices del video a 30fps)
            first_frame_idx = extract_frame_index(frame_files[0])
            last_frame_idx = extract_frame_index(frame_files[-1])
            raw_start = first_frame_idx * frame_duration
            t_start = np.floor(raw_start * 10) / 10

            t_end = t_end = last_frame_idx * frame_duration + extra_time

            # Bounding box del primer frame
            bbox_path = os.path.join(bboxes_root, f"{seq_id}.txt")
            if not os.path.exists(bbox_path):
                print(f"⚠️  No bbox para {seq_id}")
                continue
            bbox = get_bbox(bbox_path)
            xmin = int((bbox[0] - bbox_padding) * width)
            ymin = int((bbox[1] - bbox_padding) * height)
            xmax = int((bbox[2] + bbox_padding) * width)
            ymax = int((bbox[3] + bbox_padding) * height)
            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(width, xmax), min(height, ymax)

            # Cargar eventos originales
            events_file = os.path.join(class_event_path, f"{seq_id}.npy")
            if not os.path.exists(events_file):
                print(f"⚠️  No eventos para {seq_id}")
                continue
            events = np.load(events_file)
            x, y, t, p = events[:, 0], events[:, 1], events[:, 2], events[:, 3]

            # Recorte temporal
            mask_time = (t >= t_start) & (t < t_end)
            events = events[mask_time]

            # Recorte espacial
            x, y = events[:, 0], events[:, 1]
            mask_space = (x >= xmin) & (x < xmax) & (y >= ymin) & (y < ymax)
            events = events[mask_space]

            # Ajustar coordenadas relativas al bbox
            events[:, 0] = events[:, 0] - xmin  # x ← x - xmin
            events[:, 1] = events[:, 1] - ymin  # y ← y - ymin

            # Guardar con nombre extendido (incluyendo tamaño del recorte)
            crop_w = xmax - xmin + 1
            crop_h = ymax - ymin + 1
            save_dir = os.path.join(output_root, split, class_id)
            os.makedirs(save_dir, exist_ok=True)
            output_name = f"{seq_id}_w{crop_w}_h{crop_h}.npy"
            np.save(os.path.join(save_dir, output_name), events)


            if (seq_id == 'S114_001'):
                print(f"[✓] {seq_id} → eventos: {len(events)}")

                # Duración temporal
                if len(events) > 0:
                    t_min, t_max = events[:, 2].min(), events[:, 2].max()
                    duration = t_max - t_min
                    print(f"    ↳ duración: {duration:.6f} s (desde {t_min:.6f} hasta {t_max:.6f})")
                    print(f"    ↳ duración: {duration:.6f} s (desde {t_start:.6f} hasta {t_end:.6f})")
                else:
                    print("    ↳ sin eventos")
