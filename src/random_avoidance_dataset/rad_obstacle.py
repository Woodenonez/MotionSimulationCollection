import random
from typing import overload

import numpy as np


class RandomObstacle:

    class Obstacle:
        """Defines an Obstacle."""
        def __init__(self, x, y, width, height):
            self.x = x
            self.y = y
            self.width = width
            self.height = height
        def __str__(self):
            return f"A {self.width}x{self.height} obstacle at ({self.x},{self.y})"

    def __init__(self, map_size) -> None:
        # map_size: width x height
        if map_size is not (list or tuple):
            self.map_size = [map_size, map_size]
        else:
            self.map_size = map_size

        self.dungeon = np.ones(self.map_size)
        self.obstacles = []

    def init_obstacles(self, num_obstacles_range, obstacle_size_range, max_iters=5, overlap=True):
        """Initializes the rooms in the dungeon."""
        (min_obs, max_obs) = num_obstacles_range
        (min_obs_size, max_obs_size) = obstacle_size_range
        total_rooms = random.randrange(min_obs, max_obs)

        for _ in range(max_iters):
            if len(self.obstacles) >= max_obs:
                break
            for _ in range(total_rooms):
                x = random.randrange(0, self.map_size[0]-min_obs_size)
                y = random.randrange(0, self.map_size[1]-min_obs_size)
                width  = random.randrange(min_obs_size, max_obs_size)
                height = random.randrange(min_obs_size, max_obs_size)
                obs = self.Obstacle(x, y, width, height)
                if not overlap:
                    if not self.check_for_overlap(obs):
                        self.obstacles.append(obs)
                else:
                    self.obstacles.append(obs)

    def update_map(self):
        for obs in self.obstacles:
            xmin = obs.x
            xmax = min(obs.x+obs.width, self.map_size[0]-1)
            ymin = obs.y
            ymax = min(obs.y+obs.height, self.map_size[1]-1)
            self.dungeon[ymin:ymax, xmin:xmax] = 0

    def check_for_overlap(self, obs):
        """Return false if the obstacle overlaps any other obstacle."""
        for current_obs in self.obstacles:
            xmin1 = obs.x
            xmax1 = obs.x + obs.width
            xmin2 = current_obs.x
            xmax2 = current_obs.x + current_obs.width
            ymin1 = obs.y
            ymax1 = obs.y + obs.height
            ymin2 = current_obs.y
            ymax2 = current_obs.y + current_obs.height
            if (xmin1 <= xmax2 and xmax1 >= xmin2) and \
            (ymin1 <= ymax2 and ymax1 >= ymin2):
                return True
        return False

    def check_for_occupy(self, coord):
        return self.dungeon[coord[1], coord[0]]

    def generate_dungeon(self, num_rooms_range, room_size_range, overlap:bool) -> np.ndarray:
        self.init_obstacles(num_rooms_range, room_size_range, overlap=overlap)
        self.update_map()
        print(f'{len(self.obstacles)} obstacles are created.')
        return self.dungeon

    def show_dungeon(self, ax):
        ax.imshow(self.dungeon, cmap='gray')

    def show_coord(self, ax, coord, style='x'):
        ax.plot(coord[0], coord[1], style)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    _, ax = plt.subplots()
    dungeon_generator = RandomObstacle(50)
    dungeon_generator.generate_dungeon((6,10), (5,10), overlap=False)
    dungeon_generator.show_dungeon(ax)
    dungeon_generator.show_coord(ax, [10,20])
    print(dungeon_generator.check_for_occupy([10,20]))
    plt.show()
