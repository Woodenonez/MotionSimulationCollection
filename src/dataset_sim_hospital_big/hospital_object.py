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
    Exe     - [Yes]
File description:
    Single moving object interaction in hospital dataset (HPD) simulation.
File content:
    Graph               <class> - Define a map object.
    Moving_Object       <class> - Define a moving object.
    return_map          <func>  - Load the map.
    return_path         <func>  - Load a path.
Comments:
    281*591 (28.1m * 59.1m) square area: Everything is proportional to the real size.
'''

def return_map(map_path):
    '''281 * 591 area'''
    the_map = Image.open(map_path)
    return the_map

def return_path(start_node_index, num_traversed_nodes):
    assert(1<=start_node_index<=15)
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
    nodes = {1: [67, 564], 2: [92, 564], 3: [115, 564], 
             4: [92, 474], 5: [190, 474],
             6: [92, 375], 7: [190, 375],
             8: [92, 315], 9: [190, 315],
             10: [92, 214], 11: [190, 214],
             12: [92, 145], 13: [190, 145],
             14: [141, 145], 15: [141, 88],
            }
    edges = [(1, 2), (2, 3), (2, 4),
             (4, 5), (4, 6), (5, 7), (6, 7),
             (6, 8), (7, 9), (8, 10), (9, 11), (8, 9), (10, 11),
             (10, 12), (11, 13), (12, 14), (13, 14), (14, 15)
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

    def run(self, path, ts=.2, vmax=0.5, random_stop=False):
        coming_path = path[1:]
        cnt = 0
        stop_cnt = random.randint(5, 15)
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
                if random_stop and stop_cnt > 0:
                    action = (random.randint(-1,1)*1, random.randint(-1,1)*1)
                    stop_cnt -= 1
                self.one_step(ts, action)


if __name__ == '__main__':
    import os, sys
    import pathlib

    SIM_TIMES = 1
    root_dir = pathlib.Path(__file__).resolve().parents[2]
    map_path = os.path.join(root_dir, 'src', 'dataset_sim_hospital_big/hospital_big_sim_original', 'label.png')
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
    vmax = 7 + random.randint(1, 5) # 1m = 10px, reasonable speed is 8~12 px/s 
    graph = Graph(map_path=map_path)
    for i in range(SIM_TIMES): # type: ignore
        print(i)
        ax3.cla()
        [ax3.plot(netgraph.nodes[n]['pos'][0], netgraph.nodes[n]['pos'][1], 'x') for n in list(netgraph.nodes)]
        [ax3.text(netgraph.nodes[n]['pos'][0], netgraph.nodes[n]['pos'][1], n) for n in list(netgraph.nodes)]
        # start_node_index = random.choice(range(39))
        start_node_index = random.choice(range(1,15))
        num_traversed_nodes = random.choice(range(5,10))

        path = graph.get_path(start_node_index=start_node_index, num_traversed_nodes=num_traversed_nodes)
        obj = MovingObject(path[0], stagger)
        obj.run(path, ts, vmax, random_stop=True)

        graph.plot_map(ax3)
        graph.plot_path(ax3, color='go--')
        ax3.plot(np.array(obj.traj)[:,0],np.array(obj.traj)[:,1],'.')

        plt.pause(0.5)

    plt.tight_layout()
    plt.show()