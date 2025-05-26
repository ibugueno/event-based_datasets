import tonic

# Ruta donde se descarga el dataset
DATA_DIR = "./data/dvs_gesture"

# Cargar split de entrenamiento
train_dataset = tonic.datasets.DVSGesture(save_to=DATA_DIR, train=True)
print(f"Train samples: {len(train_dataset)}")

# Cargar split de prueba
test_dataset = tonic.datasets.DVSGesture(save_to=DATA_DIR, train=False)
print(f"Test samples: {len(test_dataset)}")
