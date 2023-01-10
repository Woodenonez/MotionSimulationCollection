import os
import json
import pathlib
from typing import List, Tuple

import numpy as np
from xml.dom import minidom

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.axes import Axes


class RawMapSVG():
    def __init__(self, svg_fpath:str, rescale:float=1.0, verbose=False) -> None:
        self.__prtname = '[SVG]'
        self.svg_fpath = svg_fpath
        self.rescale = rescale
        self.vb = verbose
        self.working = False
        self.start()

    def __len__(self):
        if self.vb:
            print(f'{self.__prtname} {len(self.svg_dict)} layers in the SVG.')
        return len(self.svg_dict)

    def __read_svg(self, svg_fpath:str) -> Tuple[dict, List[float]]:
        '''
        Description:
            Return SVG layers and info, where the objects, such as obstacles, are defined.
        Return:
            :svg_layers - Dictionary of {'layer_id': minidom.Element}.
            :view_box   - [xmin, ymin, xmax, ymax].
        '''
        self.working = True
        self.doc:minidom.Document = minidom.parse(svg_fpath)  # parseString also exists
        svg_doc = self.doc.getElementsByTagName('svg')[0]
        view_box = svg_doc.getAttribute('viewBox').split(' ')
        view_box = [float(x) for x in view_box] # [xmin, ymin, xmax, ymax]
        svg_layers:List[minidom.Element] = self.doc.getElementsByTagName('g')
        svg_layer_ids = [layer.getAttribute('id') for layer in svg_layers]
        svg_dict = dict(zip(svg_layer_ids, svg_layers))
        return svg_dict, view_box

    def start(self):
        self.svg_dict, self.view_box = self.__read_svg(self.svg_fpath)
        if self.vb:
            print(f'{self.__prtname} Loading SVG from: {self.svg_fpath}. Run "exit" after reading.')

    def exit(self):
        self.doc.unlink()
        if self.vb:
            print(f'{self.__prtname} SVG reader exits safely.')

    def get_all_polys_from_layer(self, layer_id:str, layer_tags:list=['path', 'rect'], x_axis_flip=False, y_axis_flip=True, rescale:float=None) -> List[List[list]]:
        '''
        Description:
            Get coordinates of all polygons from a SVG layer.
        Return:
            :obj_list - [[[x,y], [...], ...], ...] (all-polygons-coords)
        '''
        if rescale is None:
            rescale = self.rescale
        layer:minidom.Element = self.svg_dict[layer_id]
        obj_list = []
        for tag in layer_tags:
            obj_list += layer.getElementsByTagName(tag)
        translate = layer.getAttribute('transform')
        translate = translate.split('(')[1].split(')')[0]
        translate = (int(translate.split(',')[0]), int(translate.split(',')[1]))
        
        xmax, ymax = self.view_box[2:4]
        for i in range(len(obj_list)):
            try:
                obj_list[i] = self.get_coords_from_svgpath(obj_list[i], translate)
            except:
                obj_list[i] = self.get_coords_from_svgrect(obj_list[i], translate)
            if x_axis_flip | y_axis_flip: # normally the y axis should be flipped
                for j in range(len(obj_list[i])):
                    if x_axis_flip:
                        obj_list[i][j][0] = xmax - obj_list[i][j][0]
                    if y_axis_flip:
                        obj_list[i][j][1] = ymax - obj_list[i][j][1]
            if rescale != 1:
                obj_list[i] = [[x*rescale for x in y] for y in obj_list[i]]
                    
        return obj_list

    def get_all_nodes_from_layer(self, layer_id:str, layer_tags:list=['circle'], x_axis_flip=False, y_axis_flip=True, rescale:float=None) -> List[list]:
        '''
        Description:
            Get coordinates of all path nodes from a SVG layer.
        Return:
            :node_list - [[x,y,z,id], [...], ...] (all-pathnode-coords)
        '''
        if rescale is None:
            rescale = self.rescale
        layer:minidom.Element = self.svg_dict[layer_id]
        node_list = []
        for tag in layer_tags:
            node_list += layer.getElementsByTagName(tag)
        translate = layer.getAttribute('transform')
        translate = translate.split('(')[1].split(')')[0]
        translate = (int(translate.split(',')[0]), int(translate.split(',')[1]))
        
        xmax, ymax = self.view_box[2:4]
        for i in range(len(node_list)):
            node_list[i] = self.get_coords_from_svgcircle(node_list[i], translate)
            if x_axis_flip | y_axis_flip: # normally the y axis should be flipped
                if x_axis_flip:
                    node_list[i][0] = xmax - node_list[i][0]
                if y_axis_flip:
                    node_list[i][1] = ymax - node_list[i][1]
            if rescale != 1:
                node_list[i][0], node_list[i][1] = node_list[i][0]*rescale, node_list[i][1]*rescale
                    
        return node_list

    def return_all_layer_ids(self) -> List[str]:
        return list(self.svg_dict)

    def save_nodes_to_json(self, svg_layer_names:List[str], object_names:List[str]=None, save_path:str=None):
        if save_path is None:
            save_path = self.svg_fpath.split('.')[0] + '_node.json'
        if not self.working:
            self.start()
        if object_names is None:
            object_names = svg_layer_names

        out_dict = {}
        for objn, layer in zip(object_names, svg_layer_names):
            list_node = self.get_all_nodes_from_layer(layer_id=layer)
            for i in range(len(list_node)):
                list_node[i] = {'id':list_node[i][-1], 'position':list_node[i][:3]}
            out_dict[objn] = list_node

        with open(save_path, 'w') as outf:
            json.dump(out_dict, outf)
        if self.vb:
            print(f'{self.__prtname} Save nodes to {save_path}.')
        
    def save_mapdata_to_json(self, svg_layer_names:List[str], object_names:List[str]=None, save_path:str=None):
        if save_path is None:
            save_path = self.svg_fpath.split('.')[0] + '_map.json'
        if not self.working:
            self.start()
        if object_names is None:
            object_names = svg_layer_names

        out_dict = {}
        for objn, layer in zip(object_names, svg_layer_names):
            list_obj = self.get_all_polys_from_layer(layer_id=layer)
            out_dict[objn] = list_obj

        with open(save_path, 'w') as outf:
            json.dump(out_dict, outf)
        if self.vb:
            print(f'{self.__prtname} Save map to {save_path}.')
        

    @staticmethod
    def get_coords_from_svgpath(path_element:minidom.Element, translate:tuple) -> List[list]:
        # M-moveto, L-lineto, H-horizotal lineto, V-vertical lineto, z-closepath (capital means absolute, otherwise relative).
        # E.x. d="m 225,83 -40,-50 h 30 l 40,50 z"
        dx, dy = translate[0], translate[1]
        info_flat_list = path_element.getAttribute('d').split(' ')
        info_nest_list = []
        coords = []
        for info in info_flat_list:
            if info[0].isalpha():
                info_nest_list.append([info])
            else:
                info_nest_list[-1].append(info)
        mflag = False
        for info_list in info_nest_list:
            if info_list[0].lower() == 'm':
                for ij in info_list[1:]:
                    if not mflag:
                        mflag = True
                        coords.append([float(ij.split(',')[0])+dx, float(ij.split(',')[1])+dy])
                    else:
                        coords.append([float(ij.split(',')[0])+coords[-1][0], float(ij.split(',')[1])+coords[-1][1]])
            elif info_list[0].lower() == 'l':
                for ij in info_list[1:]:
                    coords.append([float(ij.split(',')[0])+coords[-1][0], float(ij.split(',')[1])+coords[-1][1]])
            elif info_list[0].lower() == 'v':
                for ij in info_list[1:]:
                    coords.append([coords[-1][0], float(ij)+coords[-1][1]])
            elif info_list[0].lower() == 'h':
                for ij in info_list[1:]:
                    coords.append([float(ij)+coords[-1][0], coords[-1][1]])
        return coords

    @staticmethod
    def get_coords_from_svgrect(rect_element:minidom.Element, translate:tuple) -> List[list]:
        dx, dy = translate[0], translate[1]
        x = float(rect_element.getAttribute('x')) + dx
        y = float(rect_element.getAttribute('y')) + dy
        w = float(rect_element.getAttribute('width'))
        h = float(rect_element.getAttribute('height'))
        coords = [[x,y], [x+w,y], [x+w,y+h], [x,y+h]]
        return coords

    @staticmethod
    def get_coords_from_svgcircle(circle_element:minidom.Element, translate:tuple) -> list:
        '''
        Description:
            Get coordinates of a path node from a SVG Element.
        Return:
            :node - [x,y,z,id]
        '''
        dx, dy = translate[0], translate[1]
        x = float(circle_element.getAttribute('cx')) + dx
        y = float(circle_element.getAttribute('cy')) + dy
        z = 0.0
        id = int(circle_element.getAttribute('id').split('_')[1])
        return [x,y,z,id]


if __name__ == '__main__':
    tag_list = ['path', 'rect', 'circle'] # not used
    current_dir = pathlib.Path(__file__).parents[0]
    svg_fname = 'drawing.svg'
    svg_fpath = os.path.join(current_dir, svg_fname)

    svg_reader = RawMapSVG(svg_fpath, rescale=0.2, verbose=True)
    print(svg_reader.return_all_layer_ids())
    map_layer_ids = ['map_helper', 'map_lane', 'map_crosswalk', 'map_obstacle']
    map_save_names = ['helper', 'lane', 'crosswalk', 'obstacle']
    node_layer_ids = ['node_lane', 'node_ped']
    node_save_names = ['node_lane', 'node_ped']

    view_box = [x*svg_reader.rescale for x in svg_reader.view_box]
    list_ext = svg_reader.get_all_polys_from_layer('map_helper')
    list_lan = svg_reader.get_all_polys_from_layer('map_lane')
    list_csw = svg_reader.get_all_polys_from_layer('map_crosswalk')
    list_obs = svg_reader.get_all_polys_from_layer('map_obstacle')

    list_node_lane = svg_reader.get_all_nodes_from_layer('node_lane')
    list_node_ped  = svg_reader.get_all_nodes_from_layer('node_ped')

    svg_reader.save_nodes_to_json(node_layer_ids, node_save_names)
    svg_reader.save_mapdata_to_json(map_layer_ids, map_save_names)

    svg_reader.exit()

    def plot_polygon(ax:Axes, vertices:list, color:str, alpha:float=0.2):
        vertices_np = np.array(vertices)[:,:2]
        polygon = patches.Polygon(vertices_np, fc=color, alpha=alpha)
        ax.add_patch(polygon)
    ### Save to label
    fig, ax1 = plt.subplots(figsize=(10,10))
    ax1.set_xlim([0, 64])
    ax1.set_ylim([0, 40])
    for obs in list_obs:
        plot_polygon(ax1, obs, 'k', alpha=1)
    ax1.set_aspect('equal')
    ax1.axis('off')
    fig.set_size_inches(6.4, 4) # XXX depends on your dpi!
    fig.tight_layout(pad=0)
    fig.savefig('label.png')
    plt.show()

    ### Vis
    fig, ax = plt.subplots()
    ax.axis('equal')
    ax.set_xlim([view_box[0], view_box[2]])
    ax.set_ylim([view_box[1], view_box[3]])
    for obj in list_ext:
        plot_polygon(ax, obj, 'r')
    for obj in list_lan:
        plot_polygon(ax, obj, 'gray')
    for obj in list_csw:
        plot_polygon(ax, obj, 'y')
    for obj in list_obs:
        plot_polygon(ax, obj, 'k')
        
    plt.plot(np.array(list_node_lane)[:,0], np.array(list_node_lane)[:,1], 'x')
    for node in list_node_lane:
        plt.text(node[0], node[1], str(node[-1]))

    plt.plot(np.array(list_node_ped)[:,0], np.array(list_node_ped)[:,1], 'yx')
    for node in list_node_ped:
        plt.text(node[0], node[1], str(node[-1]))

    plt.show()
