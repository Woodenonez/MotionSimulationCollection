import os
from pathlib import Path

import random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.utils import deprecated

from dataset_single_interaction        import sid_object       # this is dynamic env
from dataset_single_interaction_v2     import sid_object_v2    # this is static env
from dataset_multiple_scene_multimodal import msmd_object      # this is static env
from dataset_general_crossing          import gcd_object       # this is static env
from dataset_bookstore_sim             import bookstore_object # this is static env
from dataset_sim_warehouse             import warehouse_object # this is static env
from dataset_sim_hospital              import hospital_object  # this is static env
from dataset_sim_zospital              import zospital_object
from dataset_assemble_sim              import assemble_object  # this is static env

@deprecated # This was used for WTA single position prediction, should not be used in the future
def gather_all_data_position(data_dir:str, past:int, maxT:int, minT:int=1, period:int=1, save_dir:str=None):
    # data_dir - index - img&csv
    if save_dir is None:
        save_dir = data_dir

    column_name = [f'p{i}' for i in range(0,past+1)] + ['t', 'id', 'index', 'T']
    df_all = pd.DataFrame(columns=column_name)
    obj_folders = os.listdir(data_dir)
    cnt = 0
    for objf in obj_folders:
        cnt += 1
        print(f'\rProcess {cnt}/{len(obj_folders)}', end='    ')
        df_scene = pd.read_csv(os.path.join(data_dir, objf, 'data.csv'))
        all_obj_id = df_scene['id'].unique()
        for i in range(len(all_obj_id)):
            obj_id = all_obj_id[i]
            df_obj = df_scene[df_scene['id'] == obj_id]
            
            if minT == maxT:
                T_list = [maxT]
            else:
                T_list = random.sample(list(range(minT,maxT+1)), k=3) # XXX
            for T in T_list:
                sample_list = []
                for i in range(len(df_obj)-past*period-T): # each sample
                    sample = []
                    ################## Sample START ##################
                    for j in range(past+1):
                        obj_past = f'{df_obj.iloc[i+j*period]["x"]}_{df_obj.iloc[i+j*period]["y"]}_{df_obj.iloc[i+j*period]["t"]}'
                        sample.append(obj_past)
                    sample.append(df_obj.iloc[i+past]['t'])
                    sample.append(df_obj.iloc[i+past+T]['id'])
                    sample.append(df_obj.iloc[i+past+T]['index'])
                    if minT == maxT:
                        sample.append(f'{df_obj.iloc[i+past+T]["x"]}_{df_obj.iloc[i+past+T]["y"]}')
                    else:
                        sample.append(f'{df_obj.iloc[i+past+T]["x"]}_{df_obj.iloc[i+past+T]["y"]}_{T}')
                    ################## Sample E N D ##################
                    sample_list.append(sample)
                df_T = pd.DataFrame(sample_list, columns=df_all.columns)
                df_all = pd.concat([df_all, df_T], ignore_index=True)
    df_all.to_csv(os.path.join(save_dir, 'all_data.csv'), index=False)
@deprecated
def gather_all_data_trajectory(data_dir:str, past:int, maxT:int, minT:int=1, period:int=1, save_dir:str=None):
    if save_dir is None:
        save_dir = data_dir

    column_name = [f'p{i}' for i in range(0,(past+1))] + ['t', 'id', 'index'] + [f'T{i}' for i in range(minT, maxT+1)]
    df_all = pd.DataFrame(columns=column_name)
    obj_folders = os.listdir(data_dir)
    cnt = 0
    for objf in obj_folders:
        cnt += 1
        print(f'\rProcess {cnt}/{len(obj_folders)}', end='    ')
        df_scene = pd.read_csv(os.path.join(data_dir, objf, 'data.csv'))
        all_obj_id = df_scene['id'].unique()
        for i in range(len(all_obj_id)):
            obj_id = all_obj_id[i]
            df_obj = df_scene[df_scene['id'] == obj_id]
            
            sample_list = []
            for i in range(len(df_obj)-past*period-maxT): # each sample
                sample = []
                ################## Sample START ##################
                for j in range(past+1):
                    obj_past = f'{df_obj.iloc[i+j*period]["x"]}_{df_obj.iloc[i+j*period]["y"]}_{df_obj.iloc[i+j*period]["t"]}'
                    sample.append(obj_past)
                sample.append(df_obj.iloc[i+past]['t'])
                sample.append(df_obj.iloc[i+past+maxT]['id'])
                sample.append(df_obj.iloc[i+past+maxT]['index'])
                for T in range(minT, maxT+1):
                    sample.append(f'{df_obj.iloc[i+past+T]["x"]}_{df_obj.iloc[i+past+T]["y"]}')
                ################## Sample E N D ##################
                sample_list.append(sample)
            df_T = pd.DataFrame(sample_list, columns=df_all.columns)
            df_all = pd.concat([df_all, df_T], ignore_index=True)
    df_all.to_csv(os.path.join(save_dir, 'all_data.csv'), index=False)

def gather_all_data(data_dir:str, past:int, maxT:int, minT:int=1, period:int=1, save_dir:str=None):
    if save_dir is None:
        save_dir = data_dir

    column_name = [f'p{i}' for i in range(0,(past+1))] + ['t', 'id', 'index'] + [f'T{i}' for i in range(minT, maxT+1)]
    df_all = pd.DataFrame(columns=column_name)
    obj_folders = os.listdir(data_dir)
    cnt = 0
    for objf in obj_folders:
        cnt += 1
        print(f'\rProcess {cnt}/{len(obj_folders)}', end='    ')
        df_scene = pd.read_csv(os.path.join(data_dir, objf, 'data.csv'))
        all_obj_id = df_scene['id'].unique()
        for i in range(len(all_obj_id)):
            obj_id = all_obj_id[i]
            df_obj = df_scene[df_scene['id'] == obj_id]
            
            sample_list = []
            for i in range(len(df_obj)-past*period-maxT): # each sample
                if random.randint(1,2) == 2:
                    continue
                sample = []
                ################## Sample START ##################
                for j in range(past+1):
                    obj_past = f'{df_obj.iloc[i+j*period]["x"]}_{df_obj.iloc[i+j*period]["y"]}_{df_obj.iloc[i+j*period]["t"]}'
                    sample.append(obj_past)
                sample.append(df_obj.iloc[i+past]['t'])
                sample.append(df_obj.iloc[i+past+maxT]['id'])
                sample.append(int(df_obj.iloc[i+past+maxT]['index']))
                for T in range(minT, maxT+1):
                    sample.append(f'{df_obj.iloc[i+past+T]["x"]}_{df_obj.iloc[i+past+T]["y"]}')
                ################## Sample E N D ##################
                sample_list.append(sample)
            df_T = pd.DataFrame(sample_list, columns=df_all.columns)
            df_all = pd.concat([x for x in [df_all, df_T] if not x.empty], ignore_index=True)
    df_all.to_csv(os.path.join(save_dir, 'all_data.csv'), index=False)


def save_fig_to_array(fig, save_path):
    import matplotlib
    import pickle
    matplotlib.use('agg')
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    fig_in_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    fig_in_np = fig_in_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    with open(save_path, 'wb') as f:
        pickle.dump(fig_in_np, f)


def save_SID_data(index_list:list, save_path:str, sim_time_per_scene:int):
    # SID - Single-target Interaction Dataset
    def index2map(index):
        # index: [map_idx, path_idx, interact]
        # map_idx=2 - blocked map
        assert (10>=index>=1),("Index must be an integer from 1 to 10.")
        map_dict = {1:[1,1,False], 2:[1,1,True],
                    3:[1,2,False], 4:[1,2,True],
                    5:[1,3,False], 6:[1,3,True],

                    7: [2,1,False], 8: [2,1,True],
                    9: [2,3,False], 10:[2,3,True]
                    }
        return map_dict[index]

    cnt = 0 # NOTE cnt is used as index for SID
    overall_sim_time = sim_time_per_scene * len(index_list)
    for idx in index_list:
        map_idx, path_idx, interact = index2map(idx) # map parameters
        stagger, vmax, target_size, ts = (0.2, 1, 0.5, 0.2) # object parameters

        graph = sid_object.Graph(map_idx)
        path  = graph.get_path(path_idx)
        if interact:
            dyn_obs_path = graph.get_obs_path(ts)
        else:
            dyn_obs_path = [(-1,-1)]

        for _ in range(sim_time_per_scene):
            cnt += 1
            print(f'\rSimulating: {cnt}/{overall_sim_time}', end='   ')

            obj = sid_object.MovingObject(path[0], stagger)
            obj.run(path, ts, vmax, dyn_obs_path=dyn_obs_path)
            t_list   = []
            id_list  = []
            idx_list = []
            x_list   = []
            y_list   = []
            for j, tr in enumerate(obj.traj): # NOTE j is the time step
                obs_idx = min(j, len(dyn_obs_path)-1)
                obs_shape = patches.Circle(dyn_obs_path[obs_idx], radius=target_size/2, fc='k')

                # images containing everything
                fig, ax = plt.subplots()
                graph.plot_map(ax, clean=True) ### NOTE change this if needed
                ax.add_patch(obs_shape)
                ax.set_aspect('equal')
                ax.axis('off')
                
                fig.set_size_inches(4, 4) # XXX depends on your dpi!
                fig.tight_layout(pad=0)
                fig_size = fig.get_size_inches()*fig.dpi # w, h

                boundary = np.array(graph.boundary_coords)
                x_in_px = int(fig_size[0] * tr[0] / (max(boundary[:,0])-min(boundary[:,0])))
                y_in_px = int(fig_size[1] - fig_size[1] * tr[1] / (max(boundary[:,1])-min(boundary[:,1])))

                t_list.append(j)
                id_list.append(cnt)
                idx_list.append(cnt)
                x_list.append(x_in_px)
                y_list.append(y_in_px)

                if save_path is None:
                    plt.show()
                elif (interact or j==0):
                    folder = os.path.join(save_path, f'{cnt}/')
                    Path(folder).mkdir(parents=True, exist_ok=True)
                    plt.savefig(os.path.join(folder,f'{j}.png'))
                plt.close()
            df = pd.DataFrame({'t':t_list, 'id':id_list, 'index':idx_list, 'x':x_list, 'y':y_list}).sort_values(by='t', ignore_index=True)
            df.to_csv(os.path.join(save_path, f'{cnt}/', 'data.csv'), index=False)
  
    print()

def save_SID_data_v2(index_list:list, save_path:str, sim_time_per_track:int=100):
    # SID - Single-target Interaction Dataset v2
    # index_list case index 1~5
    assert (sim_time_per_track>=100), ('Should be no less than 100 trajectories for each.')
    
    stagger, vmax, ts = (0.2, 1, 0.2) # object parameters
    for idx in index_list: # each case is an individual sub-dataset

        graph = sid_object_v2.Graph(idx)

        fig, ax = plt.subplots()
        graph.plot_map(ax, clean=True) ### NOTE change this
        ax.set_aspect('equal')
        ax.axis('off')

        fig.set_size_inches(4, 4) # XXX depends on your dpi!
        fig.tight_layout(pad=0)
        fig_size = fig.get_size_inches()*fig.dpi # w, h
        boundary = np.array(graph.boundary_coords)

        if save_path is None:
            plt.show()
        else:
            folder = os.path.join(save_path, f'{idx}/', f'{idx}/')
            Path(folder).mkdir(parents=True, exist_ok=True)
            plt.savefig(os.path.join(folder,f'{idx}.png'), 
                        bbox_inches='tight', pad_inches=0)
            plt.close()

        t_list   = []
        id_list  = []
        idx_list = []
        x_list   = []
        y_list   = []
        cnt = 0 # NOTE cnt is used as index for SID v2
        t = 0 # accumulated time
        for track_idx in range(3): # left, straight, right
            sim_per_track = [sim_time_per_track*x for x in graph.proportion]
            for _ in range(sim_per_track[track_idx]):
                cnt += 1
                print(f'\rSimulating: {index_list.index(idx)+1}/{len(index_list)}, {cnt}/{sim_time_per_track*3}', end='   ')

                path  = graph.get_random_path(track_idx+1)
                obj = sid_object_v2.MovingObject(path[0], stagger)
                obj.run(path, ts, vmax)
                for tr in obj.traj:
                    x_in_px = int(fig_size[0] * tr[0] / (max(boundary[:,0])-min(boundary[:,0])))
                    y_in_px = int(fig_size[1] - fig_size[1] * tr[1] / (max(boundary[:,1])-min(boundary[:,1])))
                    t_list.append(t)
                    id_list.append(cnt)
                    idx_list.append(idx)
                    x_list.append(x_in_px)
                    y_list.append(y_in_px)
                    t += 1

        df = pd.DataFrame({'t':t_list, 'id':id_list, 'index':idx_list, 'x':x_list, 'y':y_list}).sort_values(by='t', ignore_index=True)
        df.to_csv(os.path.join(save_path, f'{idx}/', f'{idx}/', 'data.csv'), index=False)
  
    print()


def save_MSMD_data(index_list:list, save_path:str, sim_time_per_scene:int):
    # MSMD - Multiple Scene Multimodal Dataset
    cnt = 0 # used as ID here
    overall_sim_time = sim_time_per_scene * len(index_list)
    for idx in index_list:
        boundary_coords, obstacle_list, nchoices = msmd_object.return_Map(index=idx) # map parameters
        ts = 0.2
        
        ### For each index, save one static environment
        graph = msmd_object.Graph(boundary_coords, obstacle_list, inflation=0)
        fig, ax = plt.subplots()
        graph.plot_map(ax, clean=True) ### NOTE change this
        ax.set_aspect('equal')
        ax.axis('off')

        fig.set_size_inches(4, 4) # XXX depends on your dpi!
        fig.tight_layout(pad=0)
        fig_size = fig.get_size_inches()*fig.dpi # w, h
        boundary = np.array(graph.boundary_coords)

        if save_path is None:
            plt.show()
        else:
            folder = os.path.join(save_path, f'{idx}/')
            Path(folder).mkdir(parents=True, exist_ok=True)
            plt.savefig(os.path.join(folder,f'{idx}.png'), 
                        bbox_inches='tight', pad_inches=0)
            plt.close()

        t_list = []   # time or time step
        id_list = []
        idx_list = [] # more information (e.g. scene index)
        x_list = []   # x coordinate
        y_list = []   # y coordinate
        
        t = 0 # accumulated time for each scene (index)
        choice_list = list(range(1,nchoices+1))
        for ch in choice_list:
            for i in range(sim_time_per_scene//nchoices):
                cnt += 1
                print(f'\rSimulating: {cnt}/{overall_sim_time}   ', end='')
                stagger = 0.4   + (random.randint(0, 20)/10-1) * 0.2
                vmax = 1        + (random.randint(0, 20)/10-1) * 0.3
                ref_path = msmd_object.get_ref_path(index=idx, choice=ch, reverse=(i<((sim_time_per_scene//nchoices)//2)))

                obj = msmd_object.MovingObject(ref_path[0], stagger=stagger)
                obj.run(ref_path, ts, vmax)

                ### Generate images
                for tr in obj.traj:
                    x_in_px = int(fig_size[0] * tr[0] / (max(boundary[:,0])-min(boundary[:,0])))
                    y_in_px = int(fig_size[1] - fig_size[1] * tr[1] / (max(boundary[:,1])-min(boundary[:,1])))
                    t_list.append(t)
                    id_list.append(cnt)
                    idx_list.append(idx)
                    x_list.append(x_in_px)
                    y_list.append(y_in_px)
                    t += 1
        df = pd.DataFrame({'t':t_list, 'id':id_list, 'index':idx_list, 'x':x_list, 'y':y_list}).sort_values(by='t', ignore_index=True)
        df.to_csv(os.path.join(save_path, f'{idx}/', 'data.csv'), index=False)
    print()

def save_GCD_data(index_list:None, save_path:str, sim_time_per_scene:int):
    # GCD - General Crossing Dataset

    ts = 0.2 # sampling time
    car_dir = 'nswe'
    car_act = 'lsr'
    human_dir = ['nl','nr','sl','sr','wu','wd','eu','ed']
    human_act = list(range(1,10))
    overall_sim_time = sim_time_per_scene * (len(car_dir)*len(car_act) + len(human_dir)*len(human_act))
    cnt = 0

    for object_type in ['human', 'car']:

        if object_type == 'car':
            this_dir, this_act = car_dir, car_act
        else:
            this_dir, this_act = human_dir, human_act

        graph = gcd_object.Graph(inflate_margin=0)
        fig, ax = plt.subplots()
        graph.plot_map(ax, clean=True)
        ax.set_aspect('equal')
        ax.axis('off')

        fig.set_size_inches(4, 4) # XXX depends on your dpi!
        fig.tight_layout(pad=0)
        fig_size = fig.get_size_inches()*fig.dpi # w, h
        boundary = np.array(graph.boundary_coords) # for converting coordinates to pixel indices

        if save_path is None:
            plt.show()
        else:
            folder = os.path.join(save_path, f'{object_type}/')
            Path(folder).mkdir(parents=True, exist_ok=True)
            plt.savefig(os.path.join(folder,f'{object_type}.png'), 
                        bbox_inches='tight', pad_inches=0)
            plt.close()

        t_list = []   # time or time step
        id_list = []
        idx_list = [] # more information (e.g. scene index)
        x_list = []   # x coordinate
        y_list = []   # y coordinate

        t = 0
        for td in this_dir:
            for ta in this_act:
                for i in range(sim_time_per_scene):
                    cnt += 1
                    print(f'\rSimulating: {cnt}/{overall_sim_time}   ', end='')
                    if object_type == 'human':
                        stagger = 0.3   + (random.randint(0, 20)/10-1) * 0.2
                        vmax = 1        + (random.randint(0, 20)/10-1) * 0.3
                    else:
                        stagger = 0.1
                        vmax = 2
                    ref_path = gcd_object.get_ref_path(object_type, td, ta)

                    obj = gcd_object.MovingObject(ref_path[0], stagger=stagger)
                    obj.run(ref_path, ts, vmax)

                    ### Generate traj
                    for tr in obj.traj:
                        x_in_px = int(fig_size[0] * tr[0] / (max(boundary[:,0])-min(boundary[:,0])))
                        y_in_px = int(fig_size[1] - fig_size[1] * tr[1] / (max(boundary[:,1])-min(boundary[:,1])))
                        t_list.append(t)
                        id_list.append(cnt)
                        idx_list.append(object_type)
                        x_list.append(x_in_px)
                        y_list.append(y_in_px)
                        t += 1
        df = pd.DataFrame({'t':t_list, 'id':id_list, 'index':idx_list, 'x':x_list, 'y':y_list}).sort_values(by='t', ignore_index=True)
        df.to_csv(os.path.join(save_path, f'{object_type}/', 'data.csv'), index=False)
    print()

def save_BSD_data(index_list:list, save_path:str, sim_time_per_scene:int, test=False):
    # BSD - Bookstore Simulation Dataset

    ts = 0.2 # sampling time
    overall_sim_time = sim_time_per_scene * len(index_list)
    cnt = 0
    graph = bookstore_object.Graph(None)
    for start_idx in index_list:
        t = 0
        t_list = []   # time or time step
        id_list = []
        idx_list = [] # more information (e.g. scene index)
        x_list = []   # x coordinate
        y_list = []   # y coordinate
        for _ in range(sim_time_per_scene):
            cnt += 1
            print(f'\rSimulating: {cnt}/{overall_sim_time}   ', end='')
            if not test:
                num_traversed_nodes = random.choice(list(range(5,8)))
            else:
                num_traversed_nodes = 10
            ref_path = graph.get_path(start_node_index=start_idx, num_traversed_nodes=num_traversed_nodes)

            stagger = 8 + random.randint(1,5)
            vmax = 40 + random.randint(1,30)
            obj = bookstore_object.MovingObject(ref_path[0], stagger=stagger)
            obj.run(ref_path, ts, vmax)

            ### Generate traj
            for tr in obj.traj:
                t_list.append(t)
                id_list.append(cnt)
                idx_list.append(start_idx)
                x_list.append(tr[0])
                y_list.append(tr[1])
                t += 1
        Path(os.path.join(save_path, f'{start_idx}')).mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({'t':t_list, 'id':id_list, 'index':idx_list, 'x':x_list, 'y':y_list}).sort_values(by='t', ignore_index=True)
        df.to_csv(os.path.join(save_path, f'{start_idx}', 'data.csv'), index=False)
    print()

def save_WSD_data(index_list:list, save_path:str, sim_time_per_scene:int, test=False):
    # WSD - Warehouse Simulation Dataset
    import pathlib
    root_dir = pathlib.Path(__file__).resolve().parents[2]
    map_path = os.path.join(root_dir, 'src', 'dataset_warehouse_sim/warehouse_sim_original', 'label.png')

    ts = 0.2 # sampling time
    overall_sim_time = sim_time_per_scene * len(index_list)
    cnt = 0
    graph = warehouse_object.Graph(map_path)
    for start_idx in index_list:
        t = 0
        t_list = []   # time or time step
        id_list = []
        idx_list = [] # more information (e.g. scene index)
        x_list = []   # x coordinate
        y_list = []   # y coordinate
        for _ in range(sim_time_per_scene):
            cnt += 1
            print(f'\rSimulating: {cnt}/{overall_sim_time}   ', end='')
            if not test:
                num_traversed_nodes = random.choice(list(range(8,12)))
            else:
                num_traversed_nodes = 15
            ref_path = graph.get_path(start_node_index=start_idx, num_traversed_nodes=num_traversed_nodes)

            stagger = 3 + random.randint(1,5)
            vmax = 10 + random.randint(1,5)
            obj = warehouse_object.MovingObject(ref_path[0], stagger=stagger)
            obj.run(ref_path, ts, vmax)

            ### Generate traj
            for tr in obj.traj:
                t_list.append(t)
                id_list.append(cnt)
                idx_list.append(start_idx)
                x_list.append(tr[0])
                y_list.append(tr[1])
                t += 1
        Path(os.path.join(save_path, f'{start_idx}')).mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({'t':t_list, 'id':id_list, 'index':idx_list, 'x':x_list, 'y':y_list}).sort_values(by='t', ignore_index=True)
        df.to_csv(os.path.join(save_path, f'{start_idx}', 'data.csv'), index=False)

        fig, ax = plt.subplots()
        graph.plot_map(ax)
        ax.set_aspect('equal')
        ax.axis('off')
        fig.set_size_inches(3.3, 2.93) # XXX depends on your dpi!
        fig.tight_layout(pad=0)
        fig.savefig(os.path.join(save_path, f'{start_idx}', 'background.png'))
        plt.close()

    print()

def save_HPD_data(index_list:list, save_path:str, sim_time_per_scene:int, test=False):
    # Hospital Dataset
    import pathlib
    root_dir = pathlib.Path(__file__).resolve().parents[2]
    map_path = os.path.join(root_dir, 'src', 'dataset_sim_hospital/hospital_sim_original', 'label.png')

    ts = 0.2 # sampling time
    overall_sim_time = sim_time_per_scene * len(index_list)
    cnt = 0
    graph = hospital_object.Graph(map_path)
    for start_idx in index_list:
        t = 0
        t_list = []   # time or time step
        id_list = []
        idx_list = [] # more information (e.g. scene index)
        x_list = []   # x coordinate
        y_list = []   # y coordinate
        for _ in range(sim_time_per_scene):
            cnt += 1
            print(f'\rSimulating: {cnt}/{overall_sim_time}   ', end='')
            if not test:
                num_traversed_nodes = random.choice(list(range(5,10)))
            else:
                num_traversed_nodes = 10
            ref_path = graph.get_path(start_node_index=start_idx, num_traversed_nodes=num_traversed_nodes)

            stagger = 1 + random.randint(1, 5)
            vmax = 11 + random.randint(1, 12)
            obj = hospital_object.MovingObject(ref_path[0], stagger=stagger)
            obj.run(ref_path, ts, vmax)

            ### Generate traj
            for tr in obj.traj:
                t_list.append(t)
                id_list.append(cnt)
                idx_list.append(start_idx)
                x_list.append(tr[0])
                y_list.append(tr[1])
                t += 1
        Path(os.path.join(save_path, f'{start_idx}')).mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({'t':t_list, 'id':id_list, 'index':idx_list, 'x':x_list, 'y':y_list}).sort_values(by='t', ignore_index=True)
        df.to_csv(os.path.join(save_path, f'{start_idx}', 'data.csv'), index=False)

        fig, ax = plt.subplots()
        graph.plot_map(ax)
        ax.set_aspect('equal')
        ax.axis('off')
        fig.set_size_inches(4.01, 4.01) # XXX depends on your dpi!
        fig.tight_layout(pad=0)
        fig.savefig(os.path.join(save_path, f'{start_idx}', 'background.png'))
        plt.close()

    print()

def save_ZPD_data(index_list:list, save_path:str, sim_time_per_scene:int, test=False):
    # Hospital Dataset
    import pathlib
    root_dir = pathlib.Path(__file__).resolve().parents[2]
    map_path = os.path.join(root_dir, 'src', 'dataset_sim_zospital/zospital_sim_original', 'label.png')

    ts = 0.2 # sampling time
    overall_sim_time = sim_time_per_scene * len(index_list)
    cnt = 0
    graph = zospital_object.Graph(map_path)
    for start_idx in index_list:
        t = 0
        t_list = []   # time or time step
        id_list = []
        idx_list = [] # more information (e.g. scene index)
        x_list = []   # x coordinate
        y_list = []   # y coordinate
        for _ in range(sim_time_per_scene):
            cnt += 1
            print(f'\rSimulating: {cnt}/{overall_sim_time}   ', end='')
            if not test:
                num_traversed_nodes = random.choice(list(range(5,10)))
            else:
                num_traversed_nodes = 10
            ref_path = graph.get_path(start_node_index=start_idx, num_traversed_nodes=num_traversed_nodes)

            stagger = 1 + random.randint(1, 5)
            vmax = 8 + random.randint(1, 14) # 1m = 20px, reasonable speed is 10~22px/s 
            obj = zospital_object.MovingObject(ref_path[0], stagger=stagger)
            obj.run(ref_path, ts, vmax)

            ### Generate traj
            for tr in obj.traj:
                t_list.append(t)
                id_list.append(cnt)
                idx_list.append(start_idx)
                x_list.append(tr[0])
                y_list.append(tr[1])
                t += 1
        Path(os.path.join(save_path, f'{start_idx}')).mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({'t':t_list, 'id':id_list, 'index':idx_list, 'x':x_list, 'y':y_list}).sort_values(by='t', ignore_index=True)
        df.to_csv(os.path.join(save_path, f'{start_idx}', 'data.csv'), index=False)

        fig, ax = plt.subplots()
        graph.plot_map(ax)
        ax.set_aspect('equal')
        ax.axis('off')
        fig.set_size_inches(3.21, 3.21) # XXX depends on your dpi!
        fig.tight_layout(pad=0)
        fig.savefig(os.path.join(save_path, f'{start_idx}', 'background.png'))
        plt.close()

    print()

def save_HPDBIG_data(index_list:list, save_path:str, sim_time_per_scene:int, test=False):
    # Hospital Dataset
    import pathlib
    root_dir = pathlib.Path(__file__).resolve().parents[2]
    map_path = os.path.join(root_dir, 'src', 'dataset_sim_hospital_big/hospital_big_sim_original', 'label.png')

    ts = 0.2 # sampling time
    overall_sim_time = sim_time_per_scene * len(index_list)
    cnt = 0
    graph = hospital_object.Graph(map_path)
    for start_idx in index_list:
        t = 0
        t_list = []   # time or time step
        id_list = []
        idx_list = [] # more information (e.g. scene index)
        x_list = []   # x coordinate
        y_list = []   # y coordinate
        for _ in range(sim_time_per_scene):
            cnt += 1
            print(f'\rSimulating: {cnt}/{overall_sim_time}   ', end='')
            if not test:
                num_traversed_nodes = random.choice(list(range(5,10)))
            else:
                num_traversed_nodes = 10
            ref_path = graph.get_path(start_node_index=start_idx, num_traversed_nodes=num_traversed_nodes)

            stagger = 1 + random.randint(1, 5)
            vmax = 7 + random.randint(1, 5) # 1m = 10px, reasonable speed is 8~12 px/s 
            obj = hospital_object.MovingObject(ref_path[0], stagger=stagger)
            obj.run(ref_path, ts, vmax)

            ### Generate traj
            for tr in obj.traj:
                t_list.append(t)
                id_list.append(cnt)
                idx_list.append(start_idx)
                x_list.append(tr[0])
                y_list.append(tr[1])
                t += 1
        Path(os.path.join(save_path, f'{start_idx}')).mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({'t':t_list, 'id':id_list, 'index':idx_list, 'x':x_list, 'y':y_list}).sort_values(by='t', ignore_index=True)
        df.to_csv(os.path.join(save_path, f'{start_idx}', 'data.csv'), index=False)

        fig, ax = plt.subplots()
        graph.plot_map(ax)
        ax.set_aspect('equal')
        ax.axis('off')
        fig.set_size_inches(2.81, 5.91) # XXX depends on your dpi!
        fig.tight_layout(pad=0)
        fig.savefig(os.path.join(save_path, f'{start_idx}', 'background.png'))
        plt.close()

    print()

def save_ALD_data(index_list:list, save_path:str, sim_time_per_scene:int, test=False):
    # ALD - Assemble Simulation Dataset
    import pathlib
    root_dir = pathlib.Path(__file__).resolve().parents[2]
    map_path = os.path.join(root_dir, 'src', 'dataset_assemble_sim/', 'label.png')

    ts = 0.2 # sampling time
    overall_sim_time = sim_time_per_scene * len(index_list)
    cnt = 0
    netgraph = assemble_object.return_netgraph_ped(inversed_y=True, y_max=400)
    graph = assemble_object.Graph(netgraph, map_path)
    for start_idx in index_list:
        t = 0
        t_list = []   # time or time step
        id_list = []
        idx_list = [] # more information (e.g. scene index)
        x_list = []   # x coordinate
        y_list = []   # y coordinate
        for _ in range(sim_time_per_scene):
            cnt += 1
            print(f'\rSimulating: {cnt}/{overall_sim_time}   ', end='')
            if not test:
                num_traversed_nodes = random.choice(list(range(8,12)))
            else:
                num_traversed_nodes = 15
            ref_path = graph.get_path(start_node_index=start_idx, num_traversed_nodes=num_traversed_nodes)

            stagger = 3 + random.randint(1,5)
            vmax = 10 + random.randint(1,5)
            obj = assemble_object.MovingObject(ref_path[0], stagger=stagger)
            obj.run(ref_path, ts, vmax)

            ### Generate traj
            for tr in obj.traj:
                t_list.append(t)
                id_list.append(cnt)
                idx_list.append(start_idx)
                x_list.append(tr[0])
                y_list.append(tr[1])
                t += 1
        Path(os.path.join(save_path, f'{start_idx}')).mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({'t':t_list, 'id':id_list, 'index':idx_list, 'x':x_list, 'y':y_list}).sort_values(by='t', ignore_index=True)
        df.to_csv(os.path.join(save_path, f'{start_idx}', 'data.csv'), index=False)

        fig, ax = plt.subplots()
        graph.plot_map(ax)
        ax.set_aspect('equal')
        ax.axis('off')
        fig.set_size_inches(6.4, 4) # XXX depends on your dpi!
        fig.tight_layout(pad=0)
        fig.savefig(os.path.join(save_path, f'{start_idx}', 'label.png'))
        plt.close()

    print()



