import sys

import math, random
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image
import networkx as nx

# BSD - Bookstore Simulation Dataset
'''
File info:
    Name    - [bsd_object]
    Author  - [Ze]
    Date    - [Jul. 2022] -> [Aug. 2022]
    Exe     - [Yes]
File description:
    Single moving object interaction in bookstore dataset (BSD) simulation.
File content:
    Graph               <class> - Define a map object.
    Moving_Object       <class> - Define a moving object.
    return_map          <func>  - Load the map.
    return_path         <func>  - Load a path.
    return_dyn_obs_path <func>  - Load the path for a dynamic obstacle.
Comments:
    500px*500px (15m*15m) square area: Everything is proportional to the real size.
    For example, the coordinates (100,100) in the 400x400 real scene would be (80,80).
'''

def return_map(map_path):
    '''500X * 500X area'''
    the_map = Image.open(map_path)
    return the_map

def return_path(start_node_index, num_traversed_nodes):
    assert(0<=start_node_index<=25)
    assert(1<=num_traversed_nodes<=10)
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
    nodes1 = {0:(20, 45),  1:(20, 285), 2:(20, 450)}
    nodes2 = {3:(70, 450), 4:(130,450), 5:(190,450), 6:(260,450), 7:(320,450), 8:(380,450), 9:(485,450)}
    nodes3 = {10:(70, 355), 11:(130,355), 12:(190,355), 13:(260,355), 14:(320,355), 15:(380,355)}
    nodes4 = {16:(130,260), 17:(190,260), 18:(250,260), 19:(320,260), 20:(380,260)}
    nodes5 = {21:(130, 45), 22:(240, 85), 23:(130,180), 24:(250,180), 25:(410,180)}
    nodes = {**nodes1, **nodes2, **nodes3, **nodes4, **nodes5}

    edges = [(0,1), (1,2), (1,10), (1,16),
             (2,3), (3,4), (3,10), (4,5), (4,11), (5,6), (5,12), (6,7), (6,13), (7,8), (7,14), (8,9), (8,15),
             (10,11), (11,12), (11,16), (12,18), (12,17), (13,14), (13,17), (13,18), (14,15), (14,19), (15,20),
             (16,17), (17,18), (18,19), (18,24), (19,20), (20,25),
             (22,21), (24,22), (24,23), (24,25), (21,0),
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

    CHECK_NETGRAPH = True
    root_dir = pathlib.Path(__file__).resolve().parents[2]
    map_path = os.path.join(root_dir, 'Data', 'BSD/source', 'label.png')
    the_map = return_map(map_path=map_path)
    netgraph = return_netgraph()

    if CHECK_NETGRAPH:
        _, [ax1,ax2] = plt.subplots(1,2)
        ax1.imshow(the_map)
        [ax1.plot(netgraph.nodes[n]['pos'][0], netgraph.nodes[n]['pos'][1], 'x') for n in list(netgraph.nodes)]
        [ax1.text(netgraph.nodes[n]['pos'][0], netgraph.nodes[n]['pos'][1], n) for n in list(netgraph.nodes)]
        ax2.imshow(np.ones_like(the_map))
        [ax2.plot(netgraph.nodes[n]['pos'][0], netgraph.nodes[n]['pos'][1], 'x') for n in list(netgraph.nodes)]
        [ax2.text(netgraph.nodes[n]['pos'][0], netgraph.nodes[n]['pos'][1], n) for n in list(netgraph.nodes)]
        nx.draw_networkx_edges(netgraph, nx.get_node_attributes(netgraph,'pos'), ax=ax1, edge_color='r')
        nx.draw_networkx_edges(netgraph, nx.get_node_attributes(netgraph,'pos'), ax=ax2, edge_color='r')
        plt.show()

    graph = Graph(map_path=map_path)
    path = graph.get_path(start_node_index=0, num_traversed_nodes=10)

    ts = 0.1

    stagger = 10
    vmax = 70 # 1m = 33.3px, reasonable speed is 34~70px/s

    obj = MovingObject(path[0], stagger)
    obj.run(path, ts, vmax)

    fig, ax = plt.subplots()
    # ------------------------
    # ax.axis('off')
    graph.plot_map(ax)
    graph.plot_path(ax, color='go--')
    ax.plot(np.array(obj.traj)[:,0],np.array(obj.traj)[:,1],'.')
    ax.set_aspect('equal', 'box')
    plt.tight_layout()
    # ------------------------
    plt.show()