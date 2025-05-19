import os, sys
from pathlib import Path

from utils import utils_data

print("Generate synthetic segmentation dataset.")

'''
run "raw_data_proc.py" first
'''

root_dir = Path(__file__).resolve().parents[1]
save_path = os.path.join(root_dir, 'Data/ZPD_1t20_train/') # save in folder

past = 4
minT = 1
maxT = 20
sim_time_per_scene = 30 # times
start_node_list = list(range(1,22))

utils_data.save_ZPD_data(start_node_list, save_path, sim_time_per_scene, test=False)
print('CSV records for each index generated.')

utils_data.gather_all_data(save_path, past, maxT=maxT, minT=minT) # go through all the obj folders and put them together in one CSV
print('Final CSV generated!')

