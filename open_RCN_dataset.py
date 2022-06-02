# -*- coding: utf-8 -*-

import os, pickle
import numpy as np

dataset = pickle.load(open('RCN_dataset.pickle', 'rb'))

print('feature vector axis: ')
for i in dataset['features']:
	print(i)

print('\nX and Y:')
print(dataset['X'].shape)
print(dataset['Y'].shape)

print('\nfirst data point/target value: {}/{}'.format(dataset['X'][0], dataset['Y'][0]))