import math
import random
from typing import List, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import pyclipper

try:
    from general_crossing_dataset import path_dict
except:
    pass

def get_ref_path(object_type:str, start_dir:str, action:str) -> List[tuple]:
    object_type = object_type.lower()
    if object_type == 'car':
        path = path_dict.car_path(start_dir, action)
    elif object_type == 'human':
        path = path_dict.human_path(start_dir, action)
    else:
        raise ModuleNotFoundError
    return path

class MovingObject():
    def __init__(self, current_position:tuple, stagger:float=0):
        self.stagger = stagger
        self.traj = [current_position]

    @staticmethod
    def motion_model(ts:int, state:tuple, action:tuple) -> tuple:
        x,y = state[0], state[1]
        vx, vy = action[0], action[1]
        x += ts*vx
        y += ts*vy
        return (x,y)

    def one_step(self, ts, action):
        self.traj.append(self.motion_model(ts, self.traj[-1], action))

    def run(self, path, ts=.2, vmax=0.5):
        coming_path = path[1:]
        while(len(coming_path)>0):
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

class Graph:
    '''
    Description:
        Define a map.
    Arguments:
        None
    Attributes:
        boundary_coordinates <list of tuples> - Each tuple is a vectex of the boundary polygon.
                                              - Defined in counter-clockwise ordering.
        obstacle_list        <list of lists>  - Each sub-list represents an obstacle in the form of a list of tuples.
                                              - Defined in clockwise ordering.
    Functions
        plot_map <vis> - Visualization of the map. Plot directly.
    '''
    def __init__(self, inflate_margin=0):
        self.boundary_coords = [(0, 0), (12, 0), (12, 12), (0, 12)]  # in counter-clockwise ordering
        self.obstacle_list = [[(0, 0), (0, 3), (3, 3), (3, 0)],
                              [(0, 9), (0, 12), (3, 12), (3, 9)],
                              [(9, 9), (9, 12), (12, 12), (12, 9)],
                              [(9, 0), (9, 3), (12, 3), (12, 0)]] # in clock-wise ordering
        self.sidewalk_list = [[(0, 3), (0, 4), (4, 4), (4, 0), (3, 0), (3, 3)],
                              [(0, 8), (0, 9), (3, 9), (3, 12), (4, 12), (4, 8)],
                              [(8, 8), (8, 12), (9, 12), (9, 9), (12, 9), (12, 8)],
                              [(8, 0), (8, 4), (12, 4), (12, 3), (9, 3), (9, 0)]]
        self.crossing_area = [[(4, 3), (4, 4), (8, 4), (8, 3)],
                              [(3, 4), (3, 8), (4, 8), (4, 4)],
                              [(4, 8), (4, 9), (8, 9), (8, 8)],
                              [(8, 4), (8, 8), (9, 8), (9, 4)]]

        self.inflation(inflate_margin=inflate_margin)

    def plot_map(self, ax, clean=False, start=None, end=None):
        boundary = np.array(self.boundary_coords + [self.boundary_coords[0]])
        if not clean:
            # Boundary
            ax.plot(boundary[:,0], boundary[:,1], 'k')
            # Lane
            ax.plot([0, 12], [6, 6], c='orange', linestyle='--')
            ax.plot([6, 6], [0, 12], c='orange', linestyle='--')
            ax.fill_between([0, 12], [4, 4], [8, 8], color='lightgray')
            ax.fill_between([4, 8], [0, 0], [12, 12], color='lightgray')
            # Area
            for cs in self.crossing_area:
                cs_edge = cs + [cs[0]]
                xs, ys = zip(*cs_edge)
                plt.plot(xs,ys,'gray')
                poly = patches.Polygon(np.array(cs), hatch='-', color='white')
                ax.add_patch(poly)
        # Area
        for sw in self.sidewalk_list:
            poly = patches.Polygon(np.array(sw), color='gray')
            ax.add_patch(poly)
        for obs in self.obstacle_list:
            obs_edge = obs + [obs[0]]
            xs, ys = zip(*obs_edge)
            plt.plot(xs,ys,'k')
            poly = patches.Polygon(np.array(obs), color='k')
            ax.add_patch(poly)
        # Start and end
        if start is not None:
            ax.plot(self.start[0], self.start[1], 'b*')
        if end is not None:
            ax.plot(self.end[0], self.end[1], 'r*')
        ax.axis('equal')

    def inflation(self, inflate_margin):
        self.inflator = pyclipper.PyclipperOffset()
        self.processed_obstacle_list   = self.__preprocess_obstacles(self.obstacle_list, 
                                                                     pyclipper.scale_to_clipper(inflate_margin))
        self.processed_boundary_coords = self.__preprocess_obstacle( pyclipper.scale_to_clipper(self.boundary_coords), 
                                                                     pyclipper.scale_to_clipper(-inflate_margin))

    def __preprocess_obstacle(self, obstacle, inflation):
        self.inflator.Clear()
        self.inflator.AddPath(obstacle, pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)
        inflated_obstacle = pyclipper.scale_from_clipper(self.inflator.Execute(inflation))[0]
        return inflated_obstacle    
    
    def __preprocess_obstacles(self, obstacle_list, inflation):
        inflated_obstacles = []
        for obs in obstacle_list:
            obstacle = pyclipper.scale_to_clipper(obs)
            inflated_obstacle = self.__preprocess_obstacle(obstacle, inflation)
            inflated_obstacle.reverse() # obstacles are ordered clockwise
            inflated_obstacles.append(inflated_obstacle)
        return inflated_obstacles


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from path_dict import car_path, human_path

    check_cars = False

    car_dir = 'nswe'
    car_act = 'lsr'
    human_dir = ['nl','nr','sl','sr','wu','wd','eu','ed']
    human_act = list(range(1,10))

    if check_cars:
        this_dir, this_act = car_dir, car_act
    else:
        this_dir, this_act = human_dir, human_act

    _, ax = plt.subplots()
    graph = Graph(0)
    for d in this_dir:
        ax.cla()
        graph.plot_map(ax, clean=True)
        for a in this_act:
            if check_cars:
                path = car_path(d, a)
            else:
                path = human_path(d, a)

            obj = MovingObject(path[0], stagger=0.4)
            obj.run(path, vmax=2)
            obj_traj = np.array(obj.traj)

            ax.plot(obj_traj[:,0], obj_traj[:,1], '.')
            ax.plot(path[0][0], path[0][1], 'r*')

            plt.draw()
            plt.pause(0.01)
            while not plt.waitforbuttonpress():  # XXX press a button to continue
                pass

    plt.show()