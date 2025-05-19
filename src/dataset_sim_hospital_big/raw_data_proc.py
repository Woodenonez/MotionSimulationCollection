import os
import pathlib

import numpy as np

import matplotlib.pyplot as plt


root_dir = pathlib.Path(__file__).resolve().parents[2]

map_path = os.path.join(root_dir, 'src', 'dataset_sim_hospital_big/hospital_big_sim_original', 'mymap.pgm')
the_map = plt.imread(map_path)

the_map = np.array(the_map)
the_map[the_map>210] = 255
the_map[the_map<=210] = 0
the_map[:,[0,-1]] = 0
the_map[[0,-1],:] = 0

fig, ax = plt.subplots(figsize=(10,10))
ax.imshow(the_map, cmap='gray')
ax.set_aspect('equal')
ax.axis('off')
fig.set_size_inches(2.81, 5.91) # XXX depends on your dpi!
fig.tight_layout(pad=0)
# fig.savefig(os.path.join(root_dir, 'src', 'dataset_sim_hospital_big/hospital_big_sim_original', 'label.png'))
plt.show()
