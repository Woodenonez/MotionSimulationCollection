import os
import pathlib

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def read_pgm(pgmf, bit_depth:int=16, one_line_head:bool=False, skip_second_line:bool=True):
    """Return a raster of integers from a PGM as a list of lists."""
    """The head is normally [P5 Width Height Depth]"""
    header = pgmf.readline()  # the 1st line
    if one_line_head:
        magic_num = header[:2]
        (width, height) = [int(i) for i in header.split()[1:3]]
        depth = int(header.split()[3])
    else:
        magic_num = header
        if skip_second_line:
            comment = pgmf.readline() # the 2nd line if there is
            print(f'Comment: [{comment}]')
        (width, height) = [int(i) for i in pgmf.readline().split()]
        depth = int(pgmf.readline())

    if bit_depth == 8:
        assert magic_num[:2] == 'P5'
        assert depth <= 255
    elif bit_depth == 16:
        assert magic_num[:2] == b'P5'
        assert depth <= 65535

    raster = []
    for _ in range(height):
        row = []
        for _ in range(width):
            row.append(ord(pgmf.read(1)))
        raster.append(row)
    return raster

root_dir = pathlib.Path(__file__).resolve().parents[2]

map_path = os.path.join(root_dir, 'src', 'warehouse_sim_dataset/warehouse_sim_original', 'mymap.pgm')
with open (map_path, 'rb') as pgmf:
    the_map = read_pgm(pgmf)

the_map = np.array(the_map)
the_map[the_map>210] = 255
the_map[the_map<=210] = 0
the_map[:,[0,-1]] = 0
the_map[[0,-1],:] = 0

fig, ax = plt.subplots(figsize=(10,10))
ax.imshow(the_map, cmap='gray')
ax.set_aspect('equal')
ax.axis('off')
fig.set_size_inches(5, 5) # XXX depends on your dpi!
fig.tight_layout(pad=0)
plt.show()
