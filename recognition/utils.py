import os
import numpy as np

def make_event_accumulate(events, img_size):
    # Inicializa en 127 (gris neutro)
    img = np.full(img_size, 127, dtype=np.uint8)

    # Extrae coordenadas y polaridad
    x, y, p = events[:, 0], events[:, 1], events[:, 3]
    polarity = 2 * p - 1  # → -1 o +1

    # Imagen temporal para acumulación
    acc = np.zeros(img_size, dtype=np.int32)
    np.add.at(acc, (y, x), polarity)

    # Eventos positivos → blanco
    img[acc > 0] = 255
    # Eventos negativos → negro
    img[acc < 0] = 0

    return img

def make_sae(events, img_size):
    sae = np.zeros((2, *img_size), dtype=np.float32)
    if len(events) == 0:
        return np.full(img_size, 127, dtype=np.uint8)

    t_start = events[:, 2].min()
    t_range = events[:, 2].ptp() + 1  # ptp = max - min

    for idx, pol in enumerate([1, 0]):
        evs = events[events[:, 3] == pol]
        flat_idx = evs[:, 1] * img_size[1] + evs[:, 0]
        _, unique_idx = np.unique(flat_idx[::-1], return_index=True)
        selected = evs[::-1][unique_idx]
        t_norm = ((selected[:, 2] - t_start) / t_range) * 255.0
        sae[idx, selected[:, 1], selected[:, 0]] = np.clip(t_norm, 0, 255)

    return (0.5 * sae[0] + 0.5 * sae[1]).astype(np.uint8)


def make_tbr(events, img_size, num_bins):
    t0, t1 = events[:, 2][0], events[:, 2][-1]
    bins = np.linspace(t0, t1, num_bins + 1)
    volume = np.zeros((num_bins, *img_size), dtype=np.uint8)
    x, y, t = events[:, 0], events[:, 1], events[:, 2]
    bin_idx = np.searchsorted(bins, t, side='right') - 1
    bin_idx = np.clip(bin_idx, 0, num_bins - 1)
    for b, i, j in zip(bin_idx, y, x):
        volume[b, i, j] = 1
    weights = 2 ** np.arange(num_bins)[::-1].reshape((num_bins, 1, 1))
    img = np.tensordot(volume, weights, axes=(0, 0))
    return img.astype(np.uint8)

def make_tbr_tensor(events, original_size, final_size, num_bins, rescale=True):
    """
    Crea un tensor TBR (H, W, BINS) a partir de eventos.
    - Si rescale=True, los eventos se escalan a final_size.
    - Si rescale=False, se asume que los eventos ya están en final_size.
    """
    if rescale:
        h_orig, w_orig = original_size
        h_target, w_target = final_size

        scale_y = h_target / h_orig
        scale_x = w_target / w_orig

        events = events.copy()
        events[:, 0] = np.clip((events[:, 0] * scale_x).astype(np.int32), 0, w_target - 1)
        events[:, 1] = np.clip((events[:, 1] * scale_y).astype(np.int32), 0, h_target - 1)
    else:
        h_target, w_target = original_size  # usar directamente la resolución original

    # === Construir volumen TBR ===
    t0, t1 = events[:, 2][0], events[:, 2][-1]
    bins = np.linspace(t0, t1, num_bins + 1)
    volume = np.zeros((num_bins, h_target, w_target), dtype=np.uint8)

    x, y, t = events[:, 0], events[:, 1], events[:, 2]
    bin_idx = np.searchsorted(bins, t, side='right') - 1
    bin_idx = np.clip(bin_idx, 0, num_bins - 1)

    for b, i, j in zip(bin_idx, y, x):
        volume[b, i, j] = 1

    return np.transpose(volume, (1, 2, 0))  # (H, W, BINS)


def make_tqr_tensor(events, original_size, final_size, num_bins, rescale=True):
    """
    Crea un tensor TQR (H, W, BINS, 2) a partir de eventos.
    - Si rescale=True, los eventos se reescalan desde original_size a final_size.
    - Si rescale=False, se asume que los eventos ya están en final_size.
    - Canal 0: evento negativo, Canal 1: evento positivo.
    """
    if rescale:
        h_orig, w_orig = original_size
        h_target, w_target = final_size

        scale_y = h_target / h_orig
        scale_x = w_target / w_orig

        events = events.copy()
        events[:, 0] = np.clip((events[:, 0] * scale_x).astype(np.int32), 0, w_target - 1)
        events[:, 1] = np.clip((events[:, 1] * scale_y).astype(np.int32), 0, h_target - 1)
    else:
        h_target, w_target = original_size  # usar original_size como final

    # === Inicializar tensor (num_bins, h_target, w_target, 2) ===
    volume = np.zeros((num_bins, h_target, w_target, 2), dtype=np.uint8)

    # === Discretizar en bins temporales ===
    t0, t1 = events[:, 2][0], events[:, 2][-1]
    bins = np.linspace(t0, t1, num_bins + 1)
    x, y, t, p = events[:, 0], events[:, 1], events[:, 2], events[:, 3]
    bin_idx = np.searchsorted(bins, t, side='right') - 1
    bin_idx = np.clip(bin_idx, 0, num_bins - 1)

    for b, i, j, polarity in zip(bin_idx, y, x, p):
        volume[b, i, j, polarity] = 1

    return np.transpose(volume, (1, 2, 0, 3))  # (H, W, BINS, 2)



def make_tencode(events, img_size):
    img = np.full((*img_size, 3), 255, dtype=np.uint8)
    if len(events) == 0:
        return img

    t_max = np.uint64(events[:, 2].max())
    t_min = np.uint64(events[:, 2].min())
    delta = np.maximum(t_max - t_min, np.uint64(1))

    x, y, t, p = events[:, 0], events[:, 1], events[:, 2], events[:, 3]

    t_float = t.astype(np.float64)
    delta_float = float(delta)

    g = ((t_max - t_float) / delta_float * 255).astype(np.uint8)

    img[y[p == 1], x[p == 1]] = np.stack([
        255 * np.ones_like(g[p == 1]), g[p == 1], np.zeros_like(g[p == 1])
    ], axis=-1)
    img[y[p == 0], x[p == 0]] = np.stack([
        np.zeros_like(g[p == 0]), g[p == 0], 255 * np.ones_like(g[p == 0])
    ], axis=-1)

    return img

def make_behi(events, img_size):
    img = np.zeros(img_size, dtype=np.uint8)
    if len(events) == 0:
        return img
    img[events[:, 1], events[:, 0]] = 255
    return img