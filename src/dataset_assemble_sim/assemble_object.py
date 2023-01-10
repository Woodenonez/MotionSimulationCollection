import sys

import math, random
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image
import networkx as nx

'''
File info:
    Name    - [assemble_object]
    Author  - [Ze]
    Date    - [Jan. 2023] -> [X]
    Exe     - [Yes]
File description:
    Assemble line dataset (ALD) simulation.
File content:
    Graph               <class> - Define a map object.
    Moving_Object       <class> - Define a moving object.
    return_map          <func>  - Load the map.
    return_path         <func>  - Load a path.
Comments:
    640*400 (64m*40m) rectangular area: Everything is proportional to the real size.
'''

def return_map(map_path):
    '''640*400 area'''
    the_map = Image.open(map_path)
    return the_map

def return_path(netgraph:nx.Graph, start_node_index, num_traversed_nodes):
    # assert(0<=start_node_index<=39)
    # assert(1<=num_traversed_nodes<=15)

    path_idx = [start_node_index]
    path = [netgraph.nodes[start_node_index]['pos']]
    for _ in range(num_traversed_nodes):
        connected_nodes_idx = list(netgraph.adj[path_idx[-1]])
        connected_nodes_idx = [x for x in connected_nodes_idx if x not in path_idx]
        if not connected_nodes_idx:
            return path
        next_idx = random.choice(connected_nodes_idx) # NOTE: Change this to get desired path pattern
        path_idx.append(next_idx)
        path.append(netgraph.nodes[next_idx]['pos'])
    return path

def return_netgraph_ped(inversed_y=False, y_max=None):
    if inversed_y & y_max is None:
        raise ValueError('Maximal Y cannot be None.')

    nodes = {1: (120, 110), 3: (120, 190), 5: (360, 190), 6: (600, 190), 11: (600, 310), 
             12: (40, 310), 7: (360, 110), 8: (408, 110), 9: (408, 70), 10: (600, 70), 
             4: (40, 190), 2: (40, 110), 13: (0, 110), 14: (0, 190), 15: (40, 0), 
             16: (120, 0), 17: (40, 388), 18: (600, 388)
             }

    edges = [(1,2), (1,3), (1,7), (1,16), (2,4), (2,13), (2,15), (3,4), (3,5), (4,12), (4,14), 
             (5,6), (5,7), (6,10), (6,11), (7,8), (8,9), (9,10), (11,12), (11,18), 
             (12,17)
             ]

    if inversed_y:
        for id in list(nodes):
            nodes[id] = (nodes[id][0], y_max-nodes[id][1])

    G = nx.Graph()
    for p in nodes:
        G.add_node(p, pos=nodes[p])
    G.add_edges_from(edges)
    return G

class Graph:
    def __init__(self, netgraph:nx.Graph, map_path):
        self.netgraph = netgraph
        if map_path is not None:
            self.the_map = return_map(map_path)
        else:
            self.the_map = None

    def get_path(self, start_node_index, num_traversed_nodes):
        self.path = return_path(self.netgraph, start_node_index, num_traversed_nodes)
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
    current_dir = pathlib.Path(__file__).resolve().parents[0]
    map_path = os.path.join(current_dir, 'label.png')
    the_map = return_map(map_path=map_path)
    netgraph = return_netgraph_ped(inversed_y=True, y_max=400)

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
    vmax = 10 + random.randint(1,5) # 1m = 10px, reasonable speed is 10~20px/s 
    graph = Graph(netgraph, map_path=map_path)
    for _ in range(SIM_TIMES):
        ax3.cla()
        # start_node_index = random.choice(range(39))
        start_node_index = random.choice(range(1,18))
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