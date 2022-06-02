import os
from pathlib import Path

from util import utils_data

print("Generate synthetic segmentation dataset.")

root_dir = Path(__file__).resolve().parents[1]
save_path = os.path.join(root_dir, 'Data/SID_20_train/') # save in folder

past = 4
minT = 20
maxT = 20
sim_time_per_scene = 2 # times
index_list = list(range(1,13)) # list(range(1,13)) # 1~12 # [1,3,5] a simple crossing

# utils_data.save_SID_data(index_list, save_path, sim_time_per_scene)
# print('CSV records for each index generated.')

utils_data.gather_all_data_position(save_path, past, maxT=maxT, minT=minT) # go through all the obj folders and put them together in one CSV
print('Final CSV generated!')