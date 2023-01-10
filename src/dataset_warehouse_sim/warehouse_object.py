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
    Single moving object interaction in warehouse dataset (WSD) simulation.
File content:
    Graph               <class> - Define a map object.
    Moving_Object       <class> - Define a moving object.
    return_map          <func>  - Load the map.
    return_path         <func>  - Load a path.
Comments:
    330*293 (33m*29.3m) square area: Everything is proportional to the real size.
'''

def return_map(map_path):
    '''330 * 293 area'''
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
    nodes = {1:(110, 20), 2:(110,75), 3:(110,103), 4:(110,138), 5:(110,165), 6:(110,195), 7:(110, 250), 
             8:(160, 20), 9:(160,75), 10:(160,103), 32:(160,120), 11:(160,138), 12:(160,165), 13:(160,210), 14:(160,250), 
             15:(235,20), 16:(235,120), 17:(235,175), 18:(235,210), 19:(235,250), 
             20:(255, 20), 21:(255, 145), 22:(255,175), 23:(255,200), 24:(255,220), 25:(255,250), 
             26:(300,20), 27:(300,115), 28:(310,145), 29:(310,175), 30:(310,200), 31:(310,250), 
             }

    edges = [(1,2), (1,8), (2,3), (2,9), (3,4), (3,10), (4,5), (4,11), (5,6), (5,12), (6,7), (6,13), (7,14),
             (8,9), (8,15), (9,10), (10,32), (32,16), (11,12), (11,32), (12,13), (12,17), (13,14), (13,18), (14,19),
             (15,16), (15,20), (16,17), (16,21), (16,27), (17,18), (17,22), (18,19), (18,23), (18,24), (19,25), 
             (20,21), (20,26), (21,22), (21,28), (22,23), (22,29), (23,24), (23,30), (24,25), (25, 31),
             (26,27), (27,28), (28,29), (29,30), (30,31),
             (23,31), (25,30), (24,30), (24,31)
             ]

    G = nx.Graph()
    for p in nodes:
        G.add_node(p, pos=nodes[p])
    G.add_edges_from(edges)
    return G

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
    map_path = os.path.join(root_dir, 'Data', 'WSD/source', 'label.png')
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
    stagger = 3 + random.randint(1,5)
    vmax = 10 + random.randint(1,5) # 1m = 10px, reasonable speed is 10~20px/s || old:# 1m = 40px, reasonable speed is 40~70px/s
    graph = Graph(map_path=map_path)
    for _ in range(SIM_TIMES):
        ax3.cla()
        # start_node_index = random.choice(range(39))
        start_node_index = random.choice(range(1,31))
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