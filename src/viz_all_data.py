import os
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = os.path.join(ROOT_DIR, 'Data', 'WSD_1t20_train')
IMG_PATH = os.path.join(DATA_DIR, 'background.png')


plt.figure()
plt.imshow(plt.imread(IMG_PATH), cmap='spring')
for folder in os.listdir(DATA_DIR):
    folder_path = os.path.join(DATA_DIR, folder)
    if os.path.isdir(folder_path):
        csv_path = os.path.join(folder_path, 'data.csv')
        data = pd.read_csv(csv_path)
        for i in data['id'].unique():
            trajectory = data[data['id'] == i]
            plt.plot(trajectory['x'], trajectory['y'], 'b.', alpha=0.1)
plt.title(f'Trajectory for dataset {DATA_DIR}')
plt.xlabel('x')
plt.ylabel('y')

plt.show()