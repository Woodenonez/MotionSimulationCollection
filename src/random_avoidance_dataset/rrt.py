"""

Path planning Sample Code with Randomized Rapidly-Exploring Random Trees (RRT)

author: AtsushiSakai(@Atsushi_twi)
modified: Ze

"""

import math
import random

import matplotlib.pyplot as plt
import numpy as np

import shapely.geometry

import rad_dungeon
import rad_obstacle

class RRT:

    class Node:
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.path_x = []
            self.path_y = []
            self.parent = None

    def __init__(self, start:list, goal:list, map_information:dict,
                 expand_dis=5.0, path_resolution=0.5, max_iter=1000, robot_radius=0.0):
        """
        Setting Parameter

        start:          Start position [x,y]
        goal:           Goal position [x,y]
        obstacleList:   Circular obstacle positions [[x,y,size],...]
        map_boundary:   Random sampling area [xmin,xmax,ymin,ymax] or [min, max]
        act_boundary:   Stay inside this area [xmin,xmax,ymin,ymax]
        robot_radius:   Robot body modeled as circle with given radius

        """
        self.start = self.Node(start[0], start[1])
        self.end   = self.Node(goal[0], goal[1])
        self.map_info = map_information

        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.max_iter = max_iter
        self.robot_radius = robot_radius

        self.node_list = []

    def _planning(self, animation=True):
        """
        rrt path planning

        animation: flag for animation on or off
        """

        self.node_list = [self.start]
        for i in range(self.max_iter):
            rnd_range = self.get_random_range()
            rnd_node = self.get_random_node(rnd_range)
            nearest_idx = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_idx] # nearest node to the random node

            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            if self.check_within_area(new_node) and self.check_no_collision(new_node):
                self.node_list.append(new_node)

            if animation and i % 5 == 0:
                self.draw_graph(rnd_node)

            if self.calc_distance_and_angle(self.node_list[-1], self.end)[0] <= self.expand_dis:
                final_node = self.steer(self.node_list[-1], self.end, self.expand_dis)
                if self.check_no_collision(final_node):
                    return self.generate_final_course(len(self.node_list)-1), i

            if animation and i % 5:
                self.draw_graph(rnd_node)

        return None, i  # cannot find path

    def steer(self, from_node, to_node, extend_length=float("inf")):

        new_node = self.Node(from_node.x, from_node.y)
        dist, angle = self.calc_distance_and_angle(new_node, to_node)

        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]

        if extend_length > dist:
            extend_length = dist

        n_expand = math.floor(extend_length / self.path_resolution)

        for _ in range(n_expand):
            new_node.x += self.path_resolution * math.cos(angle)
            new_node.y += self.path_resolution * math.sin(angle)
            new_node.path_x.append(new_node.x)
            new_node.path_y.append(new_node.y)

        dist, _ = self.calc_distance_and_angle(new_node, to_node)
        if dist <= self.path_resolution:
            new_node.path_x.append(to_node.x)
            new_node.path_y.append(to_node.y)
            new_node.x = to_node.x
            new_node.y = to_node.y

        new_node.parent = from_node

        return new_node

    def generate_final_course(self, goal_idx):
        path = [[self.end.x, self.end.y]]
        node = self.node_list[goal_idx]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])
        return path

    def get_random_range(self) -> list: pass

    def get_random_node(self, range): pass

    def check_no_collision(self, node) -> bool: pass

    def check_within_area(self, node) -> bool: pass

    def draw_graph(self, new_node) -> None: pass

    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        dlist = [math.hypot(node.x-rnd_node.x, node.y-rnd_node.y) for node in node_list]
        return dlist.index(min(dlist))

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta

class GeometricMapRRT(RRT):

    def extract_map_info(self):
        self.map_boundary = self.map_info['map_boundary']
        self.act_boundary = self.map_info['act_boundary'] # if this is not None, it should be a sub-area of the map
        self.obstacle_list = self.map_info['obstacle_list']

        if len(self.map_boundary) == 2:
            self.map_boundary = [self.map_boundary[0], self.map_boundary[1], self.map_boundary[0], self.map_boundary[1]]

    def planning(self, animation=True):
        """
        animation: flag for animation on or off
        """
        self.extract_map_info()
        path, iter = self._planning(animation=animation)
        return path, iter

    def get_random_range(self) -> list:
        ext_range = self.expand_dis*3
        base = self.map_boundary
        xmin = max(min([n.x for n in self.node_list]), base[0])
        xmax = min(max([n.x for n in self.node_list]), base[1])
        ymin = max(min([n.y for n in self.node_list]), base[2])
        ymax = min(max([n.y for n in self.node_list]), base[3])
        return [xmin-ext_range, xmax+ext_range, ymin-ext_range, ymax+ext_range]

    def get_random_node(self, rnd_range:list=None, goal_sample_rate=0.05):
        if rnd_range is None:
            rnd_range = self.map_boundary
        if random.randint(0, 100) > goal_sample_rate*100:
            rnd = self.Node(
                random.uniform(rnd_range[0], rnd_range[1]),
                random.uniform(rnd_range[2], rnd_range[3]))
        else:  # goal point sampling
            rnd = self.Node(self.end.x, self.end.y)
        return rnd

    def check_no_collision(self, node):
        if (node is None) or (len(self.obstacle_list)==0):
            return True
        obstacleList_circle    = [x for x in self.obstacle_list if len(x)==3]
        obstacleList_rectangle = [x for x in self.obstacle_list if len(x)==4]
        check_circle    = self.check_collision_with_circle(node, obstacleList_circle, self.robot_radius)
        check_rectangle = self.check_collision_with_rectangle(node, obstacleList_rectangle, self.robot_radius)
        return check_circle & check_rectangle

    def check_within_area(self, node):
        area = self.act_boundary
        if area is None:
            return True
        flag = (area[0] <= node.x <= area[1]) and (area[2] <= node.y <= area[3])
        return flag

    def draw_graph(self, rnd=None):
        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")
            if self.robot_radius > 0.0:
                self.plot_circle(rnd.x, rnd.y, self.robot_radius, resolution=5, color='-r')
        for node in self.node_list:
            if node.parent:
                plt.plot(node.path_x, node.path_y, "-g")

        for obs_param in self.obstacle_list:
            if len(obs_param) == 3:
                resolution = 5
            else:
                resolution = self.path_resolution
            self.plot_obstacle(obs_param, resolution=resolution)

        if self.act_boundary is not None:
            plt.plot([self.act_boundary[0], self.act_boundary[1],
                      self.act_boundary[1], self.act_boundary[0], self.act_boundary[0]],
                     [self.act_boundary[2], self.act_boundary[2], 
                      self.act_boundary[3], self.act_boundary[3], self.act_boundary[2]],
                     "-k")

        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.end.x, self.end.y, "xr")
        plt.axis("equal")
        plt.axis([-2, 15, -2, 15])
        plt.grid(True)
        plt.pause(0.01)


    def plot_obstacle(self, param:list, resolution:float, color:str=None):
        if len(param) == 3:
            self.plot_circle(param[0], param[1], param[2], resolution, color)
        elif len(param) == 4:
            self.plot_rectangle(param[0], param[1], param[2], param[3], resolution, color)
        else:
            raise ModuleNotFoundError

    @staticmethod
    def plot_circle(x, y, size, resolution, color):
        if color is None:
            color = "-b"
        deg = list(range(0, 360, resolution)) + [0]
        xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
        yl = [y + size * math.sin(np.deg2rad(d)) for d in deg]
        plt.plot(xl, yl, color)

    @staticmethod
    def plot_rectangle(xmin, xmax, ymin, ymax, resolution, color):
        if color is None:
            color = "-b"
        x_ud = list(np.arange(xmin, xmax+resolution, resolution))
        if len(x_ud) < 2:
            x_ud = [xmin, xmax]
        y_up   = [ymax] * len(x_ud)
        y_down = [ymin] * len(x_ud)

        y_lr = list(np.arange(ymin, ymax+resolution, resolution))
        if len(y_lr) < 2:
            y_lr = [ymin, ymax]
        x_right = [xmax] * len(y_lr)
        x_left  = [xmin] * len(y_lr)
        plt.plot(x_ud+x_right+x_ud+x_left, y_up+y_lr+y_down+y_lr, color)

    @staticmethod
    def check_collision_with_circle(node, obstacleList, robot_radius):
        for (ox, oy, size) in obstacleList:
            dx_list = [ox - x for x in node.path_x]
            dy_list = [oy - y for y in node.path_y]
            d_list = [dx * dx + dy * dy for (dx, dy) in zip(dx_list, dy_list)]
            if min(d_list) <= (size+robot_radius)**2:
                return False  # collision
        return True  # safe

    @staticmethod
    def check_collision_with_rectangle(node, obstacleList, robot_radius):
        r = robot_radius
        for (xmin, xmax, ymin, ymax) in obstacleList:
            dx_list = [(xmin-r)<=x<=(xmax+r) for x in node.path_x]
            dy_list = [(ymin-r)<=y<=(ymax+r) for y in node.path_y]
            d_list = [x & y for (x, y) in zip(dx_list, dy_list)]
            if any(d_list):
                return False  # collision
        return True  # safe

class GridMapRRT(RRT):

    def extract_map_info(self):
        self.map_boundary = self.map_info['map_boundary']
        self.act_boundary = self.map_info['act_boundary'] # if this is not None, it should be a sub-area of the map
        self.grid_map = self.map_info['grid_map']

        if len(self.map_boundary) == 2:
            self.map_boundary = [self.map_boundary[0], self.map_boundary[1], self.map_boundary[0], self.map_boundary[1]]

    def planning(self, animation=True):
        """
        animation: flag for animation on or off
        """
        self.extract_map_info()
        path, iter = self._planning(animation=animation)
        return path, iter

    def get_random_range(self) -> list:
        ext_range = self.expand_dis*3
        base = self.map_boundary
        xmin = max(min([n.x for n in self.node_list]), base[0])
        xmax = min(max([n.x for n in self.node_list]), base[1])
        ymin = max(min([n.y for n in self.node_list]), base[2])
        ymax = min(max([n.y for n in self.node_list]), base[3])
        return_list = [xmin-ext_range, xmax+ext_range, ymin-ext_range, ymax+ext_range]
        return [int(x) for x in return_list]

    def get_random_node(self, rnd_range:list=None, goal_sample_rate=0.05):
        if rnd_range is None:
            rnd_range = self.map_boundary
        if random.randint(0, 100) > goal_sample_rate*100:
            rnd = self.Node(
                int(random.uniform(rnd_range[0], rnd_range[1])),
                int(random.uniform(rnd_range[2], rnd_range[3])))
        else:  # goal point sampling
            rnd = self.Node(self.end.x, self.end.y)
        return rnd

    def check_no_collision(self, node):
        if (node is None):
            return True
        flag = self.check_collision_with_occupancy_grid(node, self.grid_map, robot_radius=self.robot_radius)
        return flag

    def check_within_area(self, node):
        area = self.act_boundary
        if area is None:
            area = self.map_boundary
        flag = (area[0] <= node.x <= area[1]) and (area[2] <= node.y <= area[3])
        return flag

    def draw_graph(self, rnd=None):
        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")
            if self.robot_radius > 0.0:
                self.plot_circle(rnd.x, rnd.y, self.robot_radius, resolution=5, color='-r')
        for node in self.node_list:
            if node.parent:
                plt.plot(node.path_x, node.path_y, "-g")

        plt.imshow(self.grid_map, cmap='gray')

        if self.act_boundary is not None:
            plt.plot([self.act_boundary[0], self.act_boundary[1],
                      self.act_boundary[1], self.act_boundary[0], self.act_boundary[0]],
                     [self.act_boundary[2], self.act_boundary[2], 
                      self.act_boundary[3], self.act_boundary[3], self.act_boundary[2]],
                     "-k")

        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.end.x, self.end.y, "xr")
        plt.axis("equal")
        plt.axis(self.map_boundary)
        plt.grid(True)
        plt.pause(0.01)

    @staticmethod
    def check_collision_with_occupancy_grid(node, grid_map, robot_radius):
        index = [int(round(node.x)), int(round(node.y))]
        index = [min(max(0,index[0]), grid_map.shape[0]-1), min(max(0,index[1]), grid_map.shape[1]-1)]
        pixel = int(grid_map[index[0], index[1]])
        if pixel == 0:
            return False # collision
        else:
            return True # no collision

    @staticmethod
    def plot_circle(x, y, size, resolution, color):
        if color is None:
            color = "-b"
        deg = list(range(0, 360, resolution)) + [0]
        xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
        yl = [y + size * math.sin(np.deg2rad(d)) for d in deg]
        plt.plot(xl, yl, color)


def main_geo_map(maxIter=1000, smooth=True, animation=False):
    print("start " + __file__)

    # ====Search Path with RRT====
    obstacleList1 = [(5, 5, 1), (3, 6, 2), (3, 8, 2), (3, 10, 2), (7, 5, 2),
                     (9, 5, 2), (8, 10, 1)]  # [x, y, radius]
    obstacleList2 = [(4, 6, 0, 2), (-2, 0, 6, 8)]  # [xmin, xmax, ymin, ymax]
    obstacleList = obstacleList1 + obstacleList2
    # Set Initial parameters
    rrt = GeometricMapRRT(
        start=[0, 0],
        goal=[6, 10],
        map_information={'map_boundary':[-2, 15], 'act_boundary':[-3, 15, 0, 14], 'obstacle_list':obstacleList},
        robot_radius=0.5,
        expand_dis=2,
        max_iter=maxIter
        )
    path, iter = rrt.planning(animation=animation)

    if path is None:
        print("Cannot find path.")
    else:
        print(f"Found path ({iter} iterations).")
        if smooth:
            smoothedPath = PathSmoothing(path, maxIter, obstacleList).path

        # Draw final path
        rrt.draw_graph()
        plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
        if smooth:
            plt.plot([x for (x, y) in smoothedPath], [y for (x, y) in smoothedPath], '-c')
        plt.grid(True)
        plt.pause(0.01)  # Need for Mac
        plt.show()

def main_grid_map(maxIter=1000, smooth=True, animation=False):
    print("start " + __file__)

    # ====Search Path with RRT====
    # dungeon_generator = rad_dungeon.RandomDungeon(50)
    dungeon_generator = rad_obstacle.RandomObstacle(50)
    grid_map = dungeon_generator.generate_dungeon((6,10), (5,10), overlap=False)
    # Set Initial parameters
    rrt = GridMapRRT(
        start=[5, 5],
        goal=[45, 45],
        map_information={'map_boundary':[0, 50], 'act_boundary':None, 'grid_map':grid_map},
        robot_radius=0.5,
        expand_dis=1,
        max_iter=maxIter
        )
    path, iter = rrt.planning(animation=animation)

    if path is None:
        print("Cannot find path.")
    else:
        print(f"Found path ({iter} iterations).")
        if smooth:
            smoothedPath = PathSmoothing(path, maxIter, obstacleList).path

        # Draw final path
        rrt.draw_graph()
        plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
        if smooth:
            plt.plot([x for (x, y) in smoothedPath], [y for (x, y) in smoothedPath], '-c')
        plt.grid(True)
        plt.pause(0.01)  # Need for Mac
        plt.show()


if __name__ == '__main__':
    main_geo_map(maxIter=int(1e4), smooth=False, animation=True)