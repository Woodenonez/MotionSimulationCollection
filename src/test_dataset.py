import os, sys
import random
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from _data_handle.data_handler import DataHandler
from _data_handle.dataset import ImageStackDataset

a = []
for _ in range(100):
    a.append(random.gauss(5,1))
plt.hist(a)
plt.show()
sys.exit(0)

root_dir = Path(__file__).resolve().parents[1]
data_dir = os.path.join(root_dir, 'Data', 'GCD_1t20_train')
csv_path = os.path.join(data_dir, 'all_data.csv')

myDS = ImageStackDataset(csv_path, data_dir, transform=None, pred_offset_range=(1,10), ref_image_name=None, image_ext='png')
sample = myDS[0]
print('='*30)
for key, value in sample.items():
    if isinstance(value, np.ndarray):
        print(f'{key}: {value.shape}')
    else:
        print(f'{key}: {value}')
print('='*30)

myDH = DataHandler(myDS, batch_size=1)


