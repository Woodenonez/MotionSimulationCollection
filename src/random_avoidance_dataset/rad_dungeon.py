import random

import numpy as np


class RandomDungeon:

    class Room:
        """Defines a room of the dungeon."""
        def __init__(self, x, y, width, height):
            self.x = x
            self.y = y
            self.width = width
            self.height = height
        def __str__(self):
            return f"A {self.width}x{self.height} room at ({self.x},{self.y})"

    def __init__(self, map_size) -> None:
        # map_size: width x height
        if map_size is not (list or tuple):
            self.map_size = [map_size, map_size]
        else:
            self.map_size = map_size

        self.dungeon = np.zeros(self.map_size)
        self.rooms = []

    def init_rooms(self, num_rooms_range, room_size_range, max_iters=5):
        """Initializes the rooms in the dungeon."""
        (min_rooms, max_rooms) = num_rooms_range
        (min_room_size, max_room_size) = room_size_range
        total_rooms = random.randrange(min_rooms, max_rooms)

        self.rooms.append(self.Room(0, 0, 8, 8))
        self.rooms.append(self.Room(40, 40, 48, 48))

        for _ in range(max_iters):
            if len(self.rooms) >= max_rooms:
                break
            for _ in range(total_rooms):
                x = random.randrange(0, self.map_size[0]-min_room_size)
                y = random.randrange(0, self.map_size[1]-min_room_size)
                width  = random.randrange(min_room_size, max_room_size)
                height = random.randrange(min_room_size, max_room_size)
                room = self.Room(x, y, width, height)
                if not self.check_for_overlap(room):
                    self.rooms.append(room)

    def update_map(self):
        for room in self.rooms:
            xmin = room.x
            xmax = min(room.x+room.width, self.map_size[0]-1)
            ymin = room.y
            ymax = min(room.y+room.height, self.map_size[1]-1)
            self.dungeon[ymin:ymax, xmin:xmax] = 1

    def check_for_overlap(self, room):
        """Return false if the room overlaps any other room."""
        for current_room in self.rooms:
            xmin1 = room.x
            xmax1 = room.x + room.width
            xmin2 = current_room.x
            xmax2 = current_room.x + current_room.width
            ymin1 = room.y
            ymax1 = room.y + room.height
            ymin2 = current_room.y
            ymax2 = current_room.y + current_room.height
            if (xmin1 <= xmax2 and xmax1 >= xmin2) and \
            (ymin1 <= ymax2 and ymax1 >= ymin2):
                return True
        return False

    def check_for_occupy(self, coord):
        return self.dungeon[coord[1], coord[0]]

    def connect_rooms(self, channel_width:int, repeat=2, ax=None):
        """Draws passages randomly between the rooms."""
        for _ in range(repeat):
            random.shuffle(self.rooms)
            for i in range(len(self.rooms)-1):
                roomA = self.rooms[i]
                roomB = self.rooms[i+1]
                x1, y1 = roomA.x, roomA.y
                x2, y2 = roomB.x, roomB.y
                if random.random() > 0.5:
                    self.dungeon[y1:(y1+channel_width), min(x1,x2):max(x1,x2)+1] = 1
                    self.dungeon[min(y1,y2):max(y1,y2), x2:(x2+channel_width)] = 1
                else:
                    self.dungeon[y2:(y2+channel_width), min(x1,x2):max(x1,x2)+1] = 1
                    self.dungeon[min(y1,y2):max(y1,y2), x1:(x1+channel_width)] = 1
                
                if ax is not None:
                    ax.imshow(dungeon_generator.dungeon, cmap='gray')
                    ax.plot(x1,y1,'x', x2,y2,'x')
                    plt.pause(0.1)
                    input()

    def generate_dungeon(self, num_rooms_range, room_size_range, channel_width:int, ax=None) -> np.ndarray:
        self.init_rooms(num_rooms_range, room_size_range)
        self.update_map()
        self.connect_rooms(channel_width, ax=ax)
        print(f'{len(self.rooms)} rooms are created.')
        return self.dungeon

    def show_dungeon(self, ax):
        ax.imshow(self.dungeon, cmap='gray')

    def show_coord(self, ax, coord, style='x'):
        ax.plot(coord[0], coord[1], style)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    _, ax = plt.subplots()
    dungeon_generator = RandomDungeon(50)
    dungeon_generator.generate_dungeon((5,10), (8,15), channel_width=1, ax=None)
    dungeon_generator.show_dungeon(ax)
    dungeon_generator.show_coord(ax, [10,20])
    print(dungeon_generator.check_for_occupy([10,20]))
    plt.show()
