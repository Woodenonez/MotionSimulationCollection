import os
import pathlib

import numpy as np
import pandas as pd
from PIL import Image

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
csv_path = os.path.join(root_dir, 'src', 'bookstore_sim_dataset/bookstore_sim_original', 'pose_list_2.csv')
raw_csv = pd.read_csv(csv_path)
print(raw_csv)

x = raw_csv['actor_1_x']
y = raw_csv['actor_1_y']
x_px = (x - (-7.5)) / 0.01        # resolution: 0.01m
y_px = 1500 - (y - (-7.5)) / 0.01 # size[px]: 1500*1500

map_path = os.path.join(root_dir, 'src', 'bookstore_sim_dataset/bookstore_sim_original', 'mymap2.pgm')
with open (map_path, 'rb') as pgmf:
    the_map = read_pgm(pgmf)
# map_path = os.path.join(root_dir, 'Data', 'bookstore_sim', 'sim_env_top-down_view.jpg')
# the_map = Image.open(map_path)

the_map = np.array(the_map)
the_map[the_map>210] = 255
the_map[the_map<=210] = 0
the_map[:,[0,-1]] = 0
the_map[[0,-1],:] = 0

fig, ax = plt.subplots(figsize=(10,10))
ax.imshow(the_map, cmap='gray')
ax.set_aspect('equal')
ax.axis('off')
fig.set_size_inches(4, 4) # XXX depends on your dpi!
fig.tight_layout(pad=0)
plt.plot(x_px, y_px, 'r.')
plt.show()

time_ms   = (raw_csv['time_stamp [milli-sec]']).to_list()
time_list = list(range(len(time_ms)))
x_list    = list(x_px/1500*400)
y_list    = list(y_px/1500*400)
id_list   = [1]*len(time_list)
idx_list  = ['bookstore_sim']*len(time_list)
save_dict = {'t':time_list, 'id':id_list, 'index':idx_list, 'x':x_list, 'y':y_list}
save_csv = pd.DataFrame(data=save_dict)
print(save_csv)

save_csv.to_csv(os.path.join(root_dir, 'Data', 'BSD', 'data.csv'))