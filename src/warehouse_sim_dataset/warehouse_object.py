import sys

import math, random
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image
import networkx as nx

# WSD - Warehouse Simulation Dataset
'''
File info:
    Name    - [wsd_object]
    Author  - [Ze]
    Date    - [Aug. 2022] -> [Sep. 2022]
    Exe     - [Yes]
File description:
    Single moving object interaction in warehouse dataset (BSD) simulation.
File content:
    Graph               <class> - Define a map object.
    Moving_Object       <class> - Define a moving object.
    return_map          <func>  - Load the map.
    return_path         <func>  - Load a path.
Comments:
    500px*500px (20m*20m) square area: Everything is proportional to the real size.
    For example, the coordinates (100,100) in the 400x400 real scene would be (80,80).
'''

def return_map(map_path):
    '''500X * 500X area'''
    the_map = Image.open(map_path)
    return the_map

def return_path(start_node_index, num_traversed_nodes):
    assert(0<=start_node_index<=39)
    assert(1<=num_traversed_nodes<=15)
    graph = return_netgraph()

    path_idx = [start_node_index]
    path = [graph.nodes[start_node_index]['pos']]
    for _ in range(num_traversed_nodes):
        connected_nodes_idx = list(graph.adj[path_idx[-1]])
        connected_nodes_idx = [x for x in connected_nodes_idx if x not in path_idx]
        if not connected_nodes_idx:
            return path
        next_idx = random.choice(connected_nodes_idx) # NOTE: Change this to get desired path pattern
        path_idx.append(next_idx)
        path.append(graph.nodes[next_idx]['pos'])
    return path

def return_netgraph():
    nodes = {0:(137, 12), 1:(137, 80), 2:(137,150), 3:(137,237), 4:(137,305), 5:(137,380), 6:(137,480), 
             7:(180, 12), 8:(180, 80), 9:(180,150), 10:(180,237), 11:(185,305), 12:(185,380), 13:(185,480), 
             14:(240,80), 15:(240,150), 16:(240,237), 17:(240,305), 18:(240,380), 19:(240,480), 
             20:(300, 8), 21:(300, 65), 22:(300,126), 23:(300, 200), 
             24:(305,260), 25:(305,305), 26:(305,350), 27:(305,395), 28:(305,442), 29:(305,487),
             30:(425, 8), 31:(425, 65), 32:(425,126), 33:(425, 200), 
             34:(425,260), 35:(425,305), 36:(425,350), 37:(425,395), 38:(425,442), 39:(425,487)
             }

    # nodes = {0:(137, 80), 1:(137,150), 2:(137,237), 3:(137,305), 4:(137,380), 5:(137,480), 6:(475,262), 
    #          7:(475,200), 8:(137, 12), 9:(162,480), 10:(162,225), 11:(237,395), 12:(237,350), 13:(237, 300),
    #          14:(237,262), 15:(237,200), 16:(305,80), 17:(305,150), 18:(315,237), 19:(315,305), 20:(315,380),
    #          21:(315,480), 22:(450,497), 23:(450,12), 24:(475,487), 25:(475, 395), 26:(475,350),
    #          }

    edges = [(0,1), (0,7), (1,2), (1,8), (2,3), (2,9), (3,4), (3,10), (4,5), (4,11), (5,6), (5,12), (6,13), 
             (7,8), (8,9), (8,14), (9,10), (9,15), (10,11), (10,16), (11,12), (11,17), (12,13), (12,18), (13,19), 
             (14,15), (14,21), (15,16), (15,22), (16,17), (16,23), (16,24), (17,18), (17,25), (18,19), (18,26), (18,27), (19,28), (19,29),
             (20,21), (20,30), (21,22), (21,31), (22,23), (22,32), (23,24), (23,33), (24,25), (24,34),
             (25,26), (25,35), (26,27), (26,36), (27,28), (27,37), (28,29), (28,38), (29,39), 
             (30,31), (31,32), (32,33), (33,34), (34,35), (35,36), (36,37), (37,38), (38,39),
             ]

    G = nx.Graph()
    for p in nodes:
        G.add_node(p, pos=nodes[p])
    G.add_edges_from(edges)
    return G

# (65, 0), (65, 110)
# (60, 115), (60, 195)

# (55, 168), (122,168)
# (55, 140), (122, 140)
# (55, 105), (126,105)
# (55, 78), (126, 78)
# (55, 48), (126,48)
# (55, 8), (126,8)
# (55, 5), (190,5)

# (95, 120), (190, 120)
# (95, 95), (190, 95)
# (95, 80), (190, 95)
# (95, 60), (190, 60)
# (95, 42), (190, 42)

# (180, 1), (180, 195)

class Graph:
    def __init__(self, map_path):
        if map_path is not None:
            self.the_map = return_map(map_path)
        else:
            self.the_map = None

    def get_path(self, start_node_index, num_traversed_nodes):
        self.path = return_path(start_node_index, num_traversed_nodes)
        return self.path

    def plot_map(self, ax):
        ax.imshow(self.the_map)

    def plot_path(self, ax, color='k--', path=None):
        this_path = self.path
        if path is not None:
            this_path = path
        for i in range(len(this_path)-1):
            ax.plot([this_path[i][0], this_path[i+1][0]], [this_path[i][1], this_path[i+1][1]], color)

class MovingObject():
    def __init__(self, current_position, stagger=0):
        self.stagger = stagger
        self.traj = [current_position]

    @staticmethod
    def motion_model(ts, state, action):
        x,y = state[0], state[1]
        vx, vy = action[0], action[1]
        x += ts*vx
        y += ts*vy
        return (x,y)

    def one_step(self, ts, action):
        self.traj.append(self.motion_model(ts, self.traj[-1], action))

    def run(self, path, ts=.2, vmax=0.5):
        coming_path = path[1:]
        cnt = 0
        while(len(coming_path)>0):
            cnt += 1

            stagger = random.choice([1,-1]) * random.randint(0,10)/10*self.stagger
            x, y = self.traj[-1][0], self.traj[-1][1]
            dist_to_next_goal = math.hypot(coming_path[0][0]-x, coming_path[0][1]-y)
            if dist_to_next_goal < (vmax*ts):
                coming_path.pop(0)
                continue
            else:
                dire = ((coming_path[0][0]-x)/dist_to_next_goal, (coming_path[0][1]-y)/dist_to_next_goal)
                action = (dire[0]*vmax+stagger, dire[1]*vmax+stagger)
                self.one_step(ts, action)


if __name__ == '__main__':
    import os, sys
    import pathlib

    SIM_TIMES = 20
    root_dir = pathlib.Path(__file__).resolve().parents[2]
    map_path = os.path.join(root_dir, 'Data_V1', 'WSD/source', 'label.png')
    the_map = return_map(map_path=map_path)
    netgraph = return_netgraph()

    _, [ax1,ax2,ax3] = plt.subplots(1,3)
    [ax.set_aspect('equal', 'box') for ax in [ax1,ax2,ax3]]

    ax1.imshow(the_map)
    [ax1.plot(netgraph.nodes[n]['pos'][0], netgraph.nodes[n]['pos'][1], 'x') for n in list(netgraph.nodes)]
    [ax1.text(netgraph.nodes[n]['pos'][0], netgraph.nodes[n]['pos'][1], n) for n in list(netgraph.nodes)]
    ax2.imshow(np.ones_like(the_map))
    [ax2.plot(netgraph.nodes[n]['pos'][0], netgraph.nodes[n]['pos'][1], 'x') for n in list(netgraph.nodes)]
    [ax2.text(netgraph.nodes[n]['pos'][0], netgraph.nodes[n]['pos'][1], n) for n in list(netgraph.nodes)]
    nx.draw_networkx_edges(netgraph, nx.get_node_attributes(netgraph,'pos'), ax=ax1, edge_color='r')
    nx.draw_networkx_edges(netgraph, nx.get_node_attributes(netgraph,'pos'), ax=ax2, edge_color='r')

    ts = 0.2
    stagger = 8 + random.randint(1,5)
    vmax = 40 + random.randint(1,30) # 1m = 40px, reasonable speed is 40~70px/s
    graph = Graph(map_path=map_path)
    for _ in range(SIM_TIMES):
        ax3.cla()
        start_node_index = random.choice(range(39))
        num_traversed_nodes = random.choice(range(5,10))

        path = graph.get_path(start_node_index=start_node_index, num_traversed_nodes=num_traversed_nodes)
        obj = MovingObject(path[0], stagger)
        obj.run(path, ts, vmax)

        graph.plot_map(ax3)
        graph.plot_path(ax3, color='go--')
        ax3.plot(np.array(obj.traj)[:,0],np.array(obj.traj)[:,1],'.')

        plt.pause(0.5)

    plt.tight_layout()
    plt.show()