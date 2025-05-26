import matplotlib.pyplot as plt

import tonic

train_dataset = tonic.prototype.datasets.NCARS(root='../', train=True)
test_dataset = tonic.prototype.datasets.NCARS(root='../', train=False)


'''
for events, label in dataset:
    print(label)
'''