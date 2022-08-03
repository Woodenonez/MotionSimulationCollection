import os, sys
from pathlib import Path

from util import utils_data

print("Generate synthetic segmentation dataset.")

'''
run "raw_data_proc.py" first
'''

root_dir = Path(__file__).resolve().parents[1]
save_path = os.path.join(root_dir, 'Data/BSD_1t20_train/') # save in folder

past = 4
minT = 1
maxT = 20
sim_time_per_scene = 10 # times
start_node_list = list(range(20))

utils_data.save_BSD_data(start_node_list, save_path, sim_time_per_scene)
print('CSV records for each index generated.')

utils_data.gather_all_data_position(save_path, past, maxT=maxT, minT=minT) # go through all the obj folders and put them together in one CSV
print('Final CSV generated!')

