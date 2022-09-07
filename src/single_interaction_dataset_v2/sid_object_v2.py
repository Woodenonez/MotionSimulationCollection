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
    Date    - [Sep. 2022]
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
    10x10 meters area: a crossing scene with different cases.
    Case 1: 3 dirs, l:s:r=1:1:1, right track sigma=0
    Case 2: 3 dirs, l:s:r=2:1:1, right track sigma=0
    Case 3: 3 dirs, l:s:r=8:1:1, right track sigma=0
    Case 4: 3 dirs, l:s:r=1:1:1, right track sigma=0.2
    Case 5: 3 dirs, l:s:r=1:1:1, right track sigma=1
    I'm too tired to make more comments...
'''

def return_map():
    '''10m x 10m area'''
    boundary_coords = [(0, 0), (10, 0), (10, 10), (0, 10)]
    obstacle_list = [ 
        [(0.0, 0.0), (0.0, 4.0), (4.0, 4.0), (4.0, 0.0)],
        [(0.0, 6.0), (0.0, 10.0), (4.0, 10.0), (4.0, 6.0)],
        [(6.0, 6.0), (6.0, 10.0), (10.0, 10.0)],
        [(6.0, 0.0), (6.0, 4.0),  (10.0, 0.0)],
    ]
    return boundary_coords, obstacle_list

def return_path(path, right_track_end=5):
    if path == 1: # forward
        start = (5, 0)
        turning = (5, 5)
        end = (5, 10)
    elif path == 2: # left
        start = (5, 0)
        turning = (5, 5)
        end = (0, 5)
    elif path == 3: # right
        start = (5, 0)
        turning = (5, 5)
        end = (10, right_track_end)
    else:
        raise ModuleNotFoundError("Path not found!")
    return [start, turning, end]


class Graph:
    def __init__(self, case_index):
        self.boundary_coords, self.obstacle_list = return_map()
        self.case_parameter(case_index)

    def case_parameter(self, case_index):
        if case_index == 1:
            self.proportion = (1,1,1)
            self.right_track_end = 5
        elif case_index == 2:
            self.proportion = (2,1,1)
            self.right_track_end = 5
        elif case_index == 3:
            self.proportion = (8,1,1)
            self.right_track_end = 5
        elif case_index == 4:
            self.proportion = (1,1,1)
            self.right_track_end = random.gauss(5, 0.2)
        elif case_index == 5:
            self.proportion = (1,1,1)
            self.right_track_end = random.gauss(5, 1)
        else:
            raise ModuleNotFoundError

    def get_path(self, path):
        self.path = return_path(path, self.right_track_end)
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
                action = (dire[0]*math.sqrt(vmax)+stagger, dire[1]*math.sqrt(vmax)+stagger)
                self.one_step(ts, action)


if __name__ == '__main__':

    PLOT_DYN = False
    CASE_IDX = 5

    ts = 0.2
    stagger = 0.2
    vmax = 1

    graph = Graph(CASE_IDX)
    path_list = [graph.get_path(x) for x in [1,2,3]]

    obj_list = [MovingObject(path[0], stagger) for path in path_list]
    for obj, path in zip(obj_list, path_list):
        obj.run(path, ts, vmax)
    traj_list = [obj.traj for obj in obj_list]

    if PLOT_DYN:
        fig, ax = plt.subplots()
        for i, pos in enumerate(traj_list[0]):
            ax.cla()
            graph.plot_map(ax, clean=False, empty=False)
            plt.plot(pos[0], pos[1], 'rx')
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
        [graph.plot_path(ax, color='rx--', path=path) for path in path_list]
        [ax.plot(np.array(traj)[:,0],np.array(traj)[:,1],'.') for traj in traj_list]
        ax.set_aspect('equal', 'box')
        plt.tight_layout()
        # ------------------------
        plt.show()