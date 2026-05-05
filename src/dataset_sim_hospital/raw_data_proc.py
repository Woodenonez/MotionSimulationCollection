import os
import pathlib

import numpy as np

import matplotlib.pyplot as plt


root_dir = pathlib.Path(__file__).resolve().parents[2]

def process_map(map_name='zospital'):
    if map_name is 'zospital':
        size = (3.21, 3.21)
    elif map_name is 'hospital':
        size = (4.01, 4.01)
    else:
        raise ValueError(f"Unknown map name: {map_name}")
    
    map_path = os.path.join(root_dir, 'src', f'dataset_sim_{map_name}/{map_name}_sim_original', 'map.pgm')
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
    fig.set_size_inches(*size) # XXX depends on your dpi!
    fig.tight_layout(pad=0)
    fig.savefig(os.path.join(root_dir, 'src', f'dataset_sim_{map_name}/{map_name}_sim_original', 'label.png'))
    plt.show()


if __name__ == "__main__":
    process_map()
