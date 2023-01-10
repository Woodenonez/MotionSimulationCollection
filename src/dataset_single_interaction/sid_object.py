import sys

import math, random
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# SID - Single-target Interaction Dataset
'''
File info:
    Name    - [sid_object]
    Author  - [Ze]
    Date    - [Dec. 2021] -> [Mar. 2022]
    Exe     - [Yes]
File description:
    Single moving object interaction dataset (SID) simulation.
File content:
    Graph               <class> - Define a map object.
    Moving_Object       <class> - Define a moving object.
    return_map          <func>  - Load the map.
    return_path         <func>  - Load a path.
    return_dyn_obs_path <func>  - Load the path for a dynamic obstacle.
Comments:
    10x10 meters area: a crossing scene with two options:
    1. Is there a dynamic obstacle? (interact)
    2. Is one of the roads blocked? (block)
    I'm too tired to make more comments...
'''

def return_map(map_index):
    '''10m x 10m area'''
    boundary_coords = [(0, 0), (10, 0), (10, 10), (0, 10)]
    obstacle_list = [ 
        [(0.0, 0.0), (0.0, 4.0), (4.0, 4.0), (4.0, 0.0)],
        [(0.0, 6.0), (0.0, 10.0), (4.0, 10.0), (4.0, 6.0)],
        [(6.0, 6.0), (6.0, 10.0), (10.0, 10.0), (10.0, 6.0)],
        [(6.0, 0.0), (6.0, 4.0), (10.0, 4.0), (10.0, 0.0)],
    ]
    if map_index == 2:
        obstacle_list.append([(4.0, 4.1), (4.0, 5.9), (3.0, 5.9), (3.0, 4.1)])

    return boundary_coords, obstacle_list

def return_path(path):
    if path == 1: # forward
        start = (5.5, 0)
        turning = (5.5, 5)
        end = (5.5, 10)
    elif path == 2: # left
        start = (5.5, 0)
        turning = (5.5, 5.5)
        end = (0, 5.5)
    elif path == 3: # right
        start = (5.5, 0)
        turning = (5.5, 4.5)
        end = (10, 4.5)
    else:
        raise ModuleNotFoundError("Path not found!")
    return [start, turning, end]

def return_dyn_obs_path(ts, start=10):
    return [(4.5, start-x*ts) for x in range(int(10/ts)+1)]

class Graph:
    def __init__(self, map_index):
        self.boundary_coords, self.obstacle_list = return_map(map_index)

    def get_obs_path(self, ts, start=10):
        return return_dyn_obs_path(ts, start=start)

    def get_path(self, path):
        self.path = return_path(path)
        return self.path

    def plot_map(self, ax, clean=False, empty=False):
        boundary = self.boundary_coords + [self.boundary_coords[0]]
        boundary = np.array(boundary)
        if empty:
            ax.plot(boundary[:,0], boundary[:,1], 'white')
        else:
            for obs in self.obstacle_list:
                obs = np.array(obs)
                poly = patches.Polygon(obs, color='gray')
                ax.add_patch(poly)
            if not clean:
                ax.plot(boundary[:,0], boundary[:,1], 'k')
        ax.set_xlim([min(boundary[:,0]), max(boundary[:,0])])
        ax.set_ylim([min(boundary[:,1]), max(boundary[:,1])])

    def plot_path(self, ax, color='k--', path=None):
        if path is None:
            this_path = self.path
        else:
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

    def run(self, path, ts=.2, vmax=0.5, dyn_obs_path=[(0,0)]):
        coming_path = path[1:]
        cnt = 0
        while(len(coming_path)>0):
            cnt += 1
            try:
                dyn_obs_pos = dyn_obs_path[cnt]
            except:
                dyn_obs_pos = dyn_obs_path[-1]

            stagger = random.choice([1,-1]) * random.randint(0,10)/10*self.stagger
            x, y = self.traj[-1][0], self.traj[-1][1]
            if ((y>4) & (dyn_obs_pos[1]>4.5)): # hard constraint from the dynamic obstacle
                self.one_step(ts, (0,0))
                continue
            dist_to_next_goal = math.hypot(coming_path[0][0]-x, coming_path[0][1]-y)
            if dist_to_next_goal < (vmax*ts):
                coming_path.pop(0)
                continue
            else:
                dire = ((coming_path[0][0]-x)/dist_to_next_goal, (coming_path[0][1]-y)/dist_to_next_goal)
                action = (dire[0]*math.sqrt(vmax)+stagger, dire[1]*math.sqrt(vmax)+stagger)
                self.one_step(ts, action)


if __name__ == '__main__':

    ts = 0.2

    interact = False
    plot_dyn = False

    stagger = 0.2
    vmax = 1

    graph = Graph(1)
    path_list = [graph.get_path(x) for x in [1,2,3]]
    if interact:
        dyn_obs_path = graph.get_obs_path(ts)
    else:
        dyn_obs_path = [(-1,-1)]

    obj_list = [MovingObject(path[0], stagger) for path in path_list]
    for obj, path in zip(obj_list, path_list):
        obj.run(path, ts, vmax, dyn_obs_path=dyn_obs_path)
    traj_list = [obj.traj for obj in obj_list]

    if plot_dyn:
        fig, ax = plt.subplots()
        for i, pos in enumerate(traj_list[0]):
            ax.cla()
            graph.plot_map(ax, clean=False, empty=False)
            plt.plot(pos[0], pos[1], 'rx')
            try:
                plt.plot(dyn_obs_path[i][0], dyn_obs_path[i][1], 'go')
            except:
                plt.plot(dyn_obs_path[-1][0], dyn_obs_path[-1][1], 'go')
            # ax.axis('off')
            ax.set_aspect('equal', 'box')
            ax.set_ylim([0,10])
            plt.pause(0.1)
        plt.show()

    else:
        fig, ax = plt.subplots()
        # ------------------------
        # ax.axis('off')
        graph.plot_map(ax, clean=False, empty=False)
        graph.plot_path(ax, color='go--', path=dyn_obs_path)
        [graph.plot_path(ax, color='rx--', path=path) for path in path_list]
        [ax.plot(np.array(traj)[:,0],np.array(traj)[:,1],'.') for traj in traj_list]
        ax.set_aspect('equal', 'box')
        plt.tight_layout()
        # ------------------------
        plt.show()