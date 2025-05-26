import matplotlib.pyplot as plt

import tonic

dataset = tonic.datasets.NCALTECH101("")

'''
events, labels = cifar10dvs[0]

transform = tonic.transforms.ToImage(
    sensor_size=cifar10dvs.sensor_size,
)

image = transform(events[2000:3000])

print(events, labels)
'''

for events, label in dataset:
    print(label)