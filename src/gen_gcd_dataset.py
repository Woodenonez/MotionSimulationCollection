import os, sys
from pathlib import Path

from util import utils_data

print("Generate synthetic segmentation dataset.")

root_dir = Path(__file__).resolve().parents[1]
save_path = os.path.join(root_dir, 'Data/GCD_1t20_train/') # save in folder

past = 4
minT = 1
maxT = 20
sim_time_per_scene = 5 # times

utils_data.save_GCD_data(None, save_path, sim_time_per_scene)
print('CSV records for each index generated.')

### Data structure V1
# utils_data.gather_all_data_trajectory(save_path, past, maxT=maxT, minT=minT) # go through all the obj folders and put them together in one CSV
# print('Final CSV generated!')

### Data structure V2
utils_data.gather_all_data(save_path, past, maxT=maxT, minT=minT) # go through all the obj folders and put them together in one CSV
print('Final CSV generated!')