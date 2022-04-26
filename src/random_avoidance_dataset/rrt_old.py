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


class RRT:

    class Node:
        def __init__(self, x:float, y:float):
            self.x = x
            self.y = y
            self.path_x = []
            self.path_y = []
            self.parent = None

    def __init__(self, start:list, goal:list, obstacle_list:list, map_boundary:list, act_boundary=None,
                 expand_dis=3.0, path_resolution=0.5, goal_sample_rate=5, max_iter=10000, robot_radius=0.0):
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
        self.map_boundary = map_boundary
        self.act_boundary = act_boundary # if this is not Nont, it should be a sub-area of the map
        self.obstacle_list = obstacle_list

        if len(map_boundary) == 2:
            self.map_boundary = [map_boundary[0], map_boundary[1], map_boundary[0], map_boundary[1]]

        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.robot_radius = robot_radius

        self.node_list = []

    def planning(self, animation=True):
        """
        rrt path planning

        animation: flag for animation on or off
        """

        self.node_list = [self.start]
        for i in range(self.max_iter):
            rnd_range = self.get_rnd_range(ext_range=self.expand_dis*3)
            rnd_node = self.get_random_node(rnd_range)
            nearest_idx = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_idx] # nearest node to the random node

            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            if self.check_within_area(new_node, self.act_boundary) and \
               self.check_collision(new_node, self.obstacle_list, self.robot_radius):
                self.node_list.append(new_node)

            if animation and i % 5 == 0:
                self.draw_graph(rnd_node)

            if self.calc_distance_and_angle(self.node_list[-1], self.end)[0] <= self.expand_dis:
                final_node = self.steer(self.node_list[-1], self.end, self.expand_dis)
                if self.check_collision(final_node, self.obstacle_list, self.robot_radius):
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

    def get_rnd_range(self, ext_range) -> list:
        base = self.map_boundary
        xmin = max(min([n.x for n in self.node_list]), base[0])
        xmax = min(max([n.x for n in self.node_list]), base[1])
        ymin = max(min([n.y for n in self.node_list]), base[2])
        ymax = min(max([n.y for n in self.node_list]), base[3])
        return [xmin-ext_range, xmax+ext_range, ymin-ext_range, ymax+ext_range]


    def get_random_node(self, rnd_range:list=None):
        if rnd_range is None:
            rnd_range = self.map_boundary
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = self.Node(
                random.uniform(rnd_range[0], rnd_range[1]),
                random.uniform(rnd_range[2], rnd_range[3]))
        else:  # goal point sampling
            rnd = self.Node(self.end.x, self.end.y)
        return rnd

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

    def check_collision(self, node, obstacleList, robot_radius):
        if (node is None) or (len(obstacleList)==0):
            return False
        obstacleList_circle    = [x for x in obstacleList if len(x)==3]
        obstacleList_rectangle = [x for x in obstacleList if len(x)==4]
        check_circle    = self.check_collision_with_circle(node, obstacleList_circle, robot_radius)
        check_rectangle = self.check_collision_with_rectangle(node, obstacleList_rectangle, robot_radius)
        return check_circle & check_rectangle

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
    def get_nearest_node_index(node_list, rnd_node):
        dlist = [math.hypot(node.x-rnd_node.x, node.y-rnd_node.y) for node in node_list]
        return dlist.index(min(dlist))

    @staticmethod
    def check_within_area(node, area):
        if area is None:
            return True
        flag = (area[0] <= node.x <= area[1]) and (area[2] <= node.y <= area[3])
        return flag

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

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta

class PathSmoothing():
    def __init__(self, path:list, max_iter:int, obstacle_list:list) -> None:
        self.path = path
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.path_smoothing()

    def path_smoothing(self):
        le = self.get_path_length(self.path)

        for _ in range(self.max_iter):
            # Sample two points
            pickPoints = [random.uniform(0, le), random.uniform(0, le)]
            pickPoints.sort()
            first  = self.get_target_point(self.path, pickPoints[0])
            second = self.get_target_point(self.path, pickPoints[1])

            if first[2] <= 0 or second[2] <= 0:
                continue
            if (second[2] + 1) > len(self.path):
                continue
            if second[2] == first[2]:
                continue
            # collision check
            if not self.check_line_collision(first, second, self.obstacle_list):
                continue

            # Create New path
            newPath = []
            newPath.extend(self.path[:first[2] + 1])
            newPath.append([first[0], first[1]])
            newPath.append([second[0], second[1]])
            newPath.extend(self.path[second[2] + 1:])
            self.path = newPath
            le = self.get_path_length(self.path)

        return self.path

    def check_line_collision(self, first, second, obstacleList):
        obstacleList_circle    = [x for x in obstacleList if len(x)==3]
        obstacleList_rectangle = [x for x in obstacleList if len(x)==4]
        check_circle    = self.check_line_collision_with_circle(first, second, obstacleList_circle)
        check_rectangle = self.check_line_collision_with_rectangle(first, second, obstacleList_rectangle)
        return check_circle & check_rectangle

    @staticmethod
    def check_line_collision_with_circle(first, second, obstacleList):
        x1, y1 = first[0], first[1]
        x2, y2 = second[0], second[1]
        try:
            a = y2 - y1
            b = -(x2 - x1)
            c = y2 * (x2 - x1) - x2 * (y2 - y1)
        except ZeroDivisionError:
            return False
        for (ox, oy, size) in obstacleList:
            d = abs(a * ox + b * oy + c) / (math.sqrt(a * a + b * b))
            if d <= size:
                return False
        return True  # OK

    @staticmethod
    def check_line_collision_with_rectangle(first, second, obstacleList):
        line = shapely.geometry.LineString([first, second])
        for (xmin, xmax, ymin, ymax) in obstacleList:
            obstacle = shapely.geometry.Polygon([[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]])
            if line.intersects(obstacle):
                return False
        return True  # OK

    @staticmethod
    def get_path_length(path:list) -> float:
        le = 0
        for i in range(len(path) - 1):
            dx = path[i + 1][0] - path[i][0]
            dy = path[i + 1][1] - path[i][1]
            le += math.hypot(dx, dy)
        return le
        
    @staticmethod
    def get_target_point(path, targetL):
        le = 0
        ti = 0
        lastPairLen = 0
        for i in range(len(path) - 1):
            dx = path[i + 1][0] - path[i][0]
            dy = path[i + 1][1] - path[i][1]
            d = math.sqrt(dx * dx + dy * dy)
            le += d
            if le >= targetL:
                ti = i - 1
                lastPairLen = d
                break
        partRatio = (le - targetL) / lastPairLen
        x = path[ti][0] + (path[ti + 1][0] - path[ti][0]) * partRatio
        y = path[ti][1] + (path[ti + 1][1] - path[ti][1]) * partRatio
        return [x, y, ti]


def main(gx=6.0, gy=10.0, maxIter=1000, animation=False):
    print("start " + __file__)

    # ====Search Path with RRT====
    obstacleList1 = [(5, 5, 1), (3, 6, 2), (3, 8, 2), (3, 10, 2), (7, 5, 2),
                     (9, 5, 2), (8, 10, 1)]  # [x, y, radius]
    obstacleList2 = [(4, 6, 0, 2), (-2, 0, 6, 8)]  # [xmin, xmax, ymin, ymax]
    obstacleList = obstacleList1 + obstacleList2
    # Set Initial parameters
    rrt = RRT(
        start=[0, 0],
        goal=[gx, gy],
        map_boundary=[-2, 15],
        obstacle_list=obstacleList,
        act_boundary=[-3, 15, 0, 14],
        robot_radius=0.5,
        expand_dis=2,
        max_iter=maxIter
        )
    path, iter = rrt.planning(animation=animation)

    if path is None:
        print("Cannot find path.")
    else:
        print(f"Found path ({iter} iterations).")
        smoothedPath = PathSmoothing(path, maxIter, obstacleList).path

        # Draw final path
        rrt.draw_graph()
        plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
        plt.plot([x for (x, y) in smoothedPath], [y for (x, y) in smoothedPath], '-c')
        plt.grid(True)
        plt.pause(0.01)  # Need for Mac
        plt.show()


if __name__ == '__main__':
    main(maxIter=int(1e4), animation=False)