import os, sys
import random
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from _data_handle_mmp.data_handler import DataHandler
from _data_handle_mmp.dataset import ImageStackDataset


root_dir = Path(__file__).resolve().parents[1]
data_dir = os.path.join(root_dir, 'Data', 'ALD_1t20_train')
csv_path = os.path.join(data_dir, 'all_data.csv')

myDS = ImageStackDataset(csv_path, data_dir, transform=None, pred_offset_range=(1,20), ref_image_name='label.png', image_ext='png')

_, ax = plt.subplots()
for i in range(0, 20):
    ax.cla()

    sample = myDS[i]
    print('='*30)
    for key, value in sample.items():
        if isinstance(value, np.ndarray):
            print(f'{key}: {value.shape}')
        else:
            print(f'{key}: {value}')
    print('='*30)

    myDH = DataHandler(myDS, batch_size=1)

    ax.imshow(sample['input'][:,:,-1], cmap='gray')
    plt.plot(np.array(sample['traj'])[:,0], np.array(sample['traj'])[:,1], 'k.', label='past')
    plt.plot(sample['target'][:,0], sample['target'][:,1], 'r.', label='future')
    plt.legend()
    plt.pause(0.1)
    while not plt.waitforbuttonpress():
        pass

plt.show()

