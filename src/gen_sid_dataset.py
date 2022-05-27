import os
from pathlib import Path

from util import utils_data

print("Generate synthetic segmentation dataset.")

root_dir = Path(__file__).resolve().parents[1]
save_path = os.path.join(root_dir, 'Data/SID_train_1t10/') # save in folder

past = 4
minT = 1
maxT = 10
sim_time_per_scene = 20 # times
index_list = list(range(1,13)) #[1, 3, 5] #list(range(1,13)) # 1~12 # [1,3,5] a simple crossing

utils_data.save_SID_data(index_list, save_path, sim_time_per_scene)

utils_data.gen_csv_trackers(save_path) # generate CSV tracking files first
print('CSV records for each object generated.')

utils_data.gather_all_data_trajectory(save_path, past, maxT=maxT, minT=minT, dynamic_env=True) # go through all the obj folders and put them together in one CSV
print('Final CSV generated!')