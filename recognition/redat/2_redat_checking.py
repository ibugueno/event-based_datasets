import numpy as np
import pandas as pd

path = "/media/ignacio/KINGSTON/event-cameras/redat/npy/idle/idle_10_events.npy"
events = np.load(path, allow_pickle=True)

print("Shape:", events.shape)
print("Dtype:", events.dtype)

# REDAT viene como (4, N): [x, y, t, p]
if events.shape[0] == 4:
    names = ["x", "y", "t", "p"]
    data = events
elif events.shape[1] == 4:
    names = ["x", "y", "t", "p"]
    data = events.T
else:
    raise ValueError(f"Formato desconocido: {events.shape}")

summary = pd.DataFrame({
    "field": names,
    "min": data.min(axis=1),
    "max": data.max(axis=1),
    "first": data[:, 0],
    "last": data[:, -1],
    "n_unique_sample": [len(np.unique(data[i, :1000])) for i in range(4)]
})

print("\nFirst 10 events as [x, y, t, p]:")
print(data[:, :10].T)

print("\nSummary:")
print(summary)

print("\nTimestamp checks:")
t = data[2]
print("t first:", t[0])
print("t last:", t[-1])
print("t monotonic:", np.all(np.diff(t) >= 0))
print("duration us:", t[-1] - t[0])
print("duration s:", (t[-1] - t[0]) / 1e6)

print("\nPolarity values:")
print(np.unique(data[3]))