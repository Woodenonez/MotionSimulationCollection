import sys

import math, random
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image
import networkx as nx # type: ignore

# HPD - Hospital Dataset
'''
File info:
    Name    - [hpd_object]
    Author  - [Ze]
    Date    - [Dec. 2024] -> [Dec. 2024]
    Exe     - [Yes]
File description:
    Single moving object interaction in hospital dataset (HPD) simulation.
File content:
    Graph               <class> - Define a map object.
    Moving_Object       <class> - Define a moving object.
    return_map          <func>  - Load the map.
    return_path         <func>  - Load a path.
Comments:
    401*401 (16.04m * 16.04m) square area: Everything is proportional to the real size.
'''

def return_map(map_path):
    '''401 * 401 area'''
    the_map = Image.open(map_path)
    return the_map

def return_path(start_node_index, num_traversed_nodes):
    assert(1<=start_node_index<=22)
    assert(2<=num_traversed_nodes<=10)
    graph = return_netgraph()
    noise = 10

    path_idx = [start_node_index]
    path = [graph.nodes[start_node_index]['pos']]
    for _ in range(num_traversed_nodes):
        connected_nodes_idx = list(graph.adj[path_idx[-1]])
        connected_nodes_idx = [x for x in connected_nodes_idx if x not in path_idx]
        if not connected_nodes_idx:
            break
        next_idx = random.choice(connected_nodes_idx) # NOTE: Change this to get desired path pattern
        path_idx.append(next_idx)
        path.append(graph.nodes[next_idx]['pos'])
    path = [(int(x[0])+random.randint(-noise, noise), 
             int(x[1])+random.randint(-noise, noise)) for x in path]
    return path

def return_netgraph():
    nodes = {1: [40, 60], 2: [50, 390], 3: [70, 60], 4: [70, 10], 5: [40, 335], 6: [400, 335], 7: [200, 335],
             8: [200, 390], 9: [185, 335], 10: [185, 60], 11: [330, 200], 12: [50, 335], 13: [300, 60], 14: [300, 10],
             15: [400, 60], 16: [400, 200], 17: [185, 185], 18: [330, 185], 19: [185, 250], 20: [330, 250],
             21: [330, 335], 22: [330, 60]
            }
    edges = [(1, 3), (1, 5), (2, 12), (3, 4), (3, 10), (5, 12), (6, 21), 
             (7, 8), (7, 9), (7, 21), (9, 19), (9, 12), (10, 13), (10, 17), 
             (11, 16), (11, 18), (11, 20), (13, 14), (13, 15), (13, 22), 
             (17, 18), (17, 19), (18, 22), (19, 20), (20, 21)
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

    SIM_TIMES = 10
    root_dir = pathlib.Path(__file__).resolve().parents[2]
    map_path = os.path.join(root_dir, 'src', 'dataset_sim_hospital/hospital_sim_original', 'label.png')
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
    stagger = 1 + random.randint(1, 5)
    vmax = 11 + random.randint(1, 12) # 1m = 25px, reasonable speed is 12~25px/s 
    graph = Graph(map_path=map_path)
    for i in range(SIM_TIMES): # type: ignore
        print(i)
        ax3.cla()
        [ax3.plot(netgraph.nodes[n]['pos'][0], netgraph.nodes[n]['pos'][1], 'x') for n in list(netgraph.nodes)]
        [ax3.text(netgraph.nodes[n]['pos'][0], netgraph.nodes[n]['pos'][1], n) for n in list(netgraph.nodes)]
        # start_node_index = random.choice(range(39))
        start_node_index = random.choice(range(1,22))
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