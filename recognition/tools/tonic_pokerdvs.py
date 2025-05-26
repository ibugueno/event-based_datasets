import matplotlib.pyplot as plt

import tonic

train_dataset = tonic.datasets.POKERDVS(save_to='../', train=True)
test_dataset = tonic.datasets.POKERDVS(save_to='../', train=False)


'''
for events, label in dataset:
    print(label)
'''