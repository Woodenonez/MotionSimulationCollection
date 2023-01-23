import os
import pathlib
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.axes import Axes

def plot_node(ax:Axes, node:list, style:str):
    ax.plot(np.array(node)[0], np.array(node)[1], style)

def plot_path(ax:Axes, nodes:list, style:str):
    ax.plot(np.array(nodes)[:,0], np.array(nodes)[:,1], style)

def plot_edges(ax:Axes, vertices:list, style:str):
    vis_vertices = np.array(vertices + [vertices[0]])
    ax.plot(vis_vertices[:,0], vis_vertices[:,1], style)

def plot_polygon(ax:Axes, vertices:list, color:str, alpha:float=0.2):
    vertices_np = np.array(vertices)[:,:2]
    polygon = patches.Polygon(vertices_np, fc=color, alpha=alpha)
    ax.add_patch(polygon)

root_dir = pathlib.Path(__file__).resolve().parents[0]
map_file_name = 'drawing_map.json'
node_file_name = 'drawing_node.json'
with open(os.path.join(root_dir, map_file_name)) as jf:
    map_dict = json.load(jf)
with open(os.path.join(root_dir, node_file_name)) as jf:
    node_dict = json.load(jf)

map_objs = list(map_dict)
node_objs = list(node_dict)
print(map_objs, node_objs)

helper_lines = map_dict['helper']
obstacle_list = map_dict['obstacle']
lane_area = map_dict['lane']
crosswalk_area = map_dict['crosswalk']

path_nodes = node_dict['node_lane']
ped_nodes = node_dict['node_ped']

_, ax = plt.subplots()

for obs in obstacle_list:
    plot_edges(ax, obs, 'k')
for al in helper_lines:
    plot_edges(ax, al, 'r--')

for ln in lane_area:
    plot_polygon(ax, ln, 'gray', alpha=1)
for cw in crosswalk_area:
    plot_polygon(ax, cw, 'y')

for pn in path_nodes:
    plot_node(ax, pn['position'], 'ko')
    plt.text(pn['position'][0], pn['position'][1], str(pn['id']))
    print(f"{pn['id']}: {tuple([int(x*10) for x in pn['position'][:2]])}")
for pn in ped_nodes:
    plot_node(ax, pn['position'], 'yo')
    plt.text(pn['position'][0], pn['position'][1], str(pn['id']))
    # print(f"{pn['id']}: {tuple([int(x*10) for x in pn['position'][:2]])}")


ax.axis('equal')
plt.show()

