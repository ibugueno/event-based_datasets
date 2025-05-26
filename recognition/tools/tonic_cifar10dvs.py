import matplotlib.pyplot as plt

import tonic

cifar10dvs = tonic.datasets.CIFAR10DVS("")

'''
events, label = cifar10dvs[300]

transform = tonic.transforms.ToImage(
    sensor_size=cifar10dvs.sensor_size,
)

image = transform(events)

print(events, label)

print(image.shape)

plt.imshow(image[1] - image[0])
#plt.axis(False)
plt.show()
'''