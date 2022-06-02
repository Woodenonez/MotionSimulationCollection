import os, sys
from pathlib import Path

from util import utils_data

print("Generate synthetic segmentation dataset.")

root_dir = Path(__file__).resolve().parents[1]
save_path = os.path.join(root_dir, 'Data/MSMD_1t10_test2/') # save in folder

past = 4
minT = 1
maxT = 10
sim_time_per_scene = 5 # times
# index_list = [1,2,3,4,5,6,7,8,9] # each index should have at least 30 trajectories
index_list = [10,11,12] # new scene for test

utils_data.save_MSMD_data(index_list, save_path, sim_time_per_scene)
print('CSV records for each index generated.')

utils_data.gather_all_data_trajectory(save_path, past, maxT=maxT, minT=minT) # go through all the obj folders and put them together in one CSV
print('Final CSV generated!')