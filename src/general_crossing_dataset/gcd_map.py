import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

import pyclipper

'''
File info:
    Name    - [mmc_graph]
    Exe     - [Yes]
File description:
    This contains the map of the synthetic multimodal crossing (mmc) data.
File content:
    Graph     <class> - Define the map.
Comments:
    None
'''

class GraphTemplate:
    def __init__(self, external_boundary:list=None, external_obstacles:list=None) -> None:
        self.boundary_coordinates = external_boundary
        self.obstacle_list = external_obstacles

    def plot_map(self, ax, start=None, end=None):
        pass


if __name__ == '__main__':
    graph = Graph(0)
    _, ax = plt.subplots()
    graph.plot_map(ax)
    plt.show()