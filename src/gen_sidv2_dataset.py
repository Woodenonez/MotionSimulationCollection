import os
import sys
from pathlib import Path

from utils import utils_data

print("Generate synthetic segmentation dataset.")

root_dir = Path(__file__).resolve().parents[1]
save_path = os.path.join(root_dir, 'Data/SIDv2_1t10_test/') # save in folder

past = 4
minT = 1
maxT = 20
sim_time_per_track = 100 # times
index_list = [2]

utils_data.save_SID_data_v2(index_list, save_path, sim_time_per_track)
print('CSV records for each index generated.')

for case_idx in index_list:
    utils_data.gather_all_data(os.path.join(save_path, str(case_idx)), past, maxT=maxT, minT=minT) # go through all the obj folders and put them together in one CSV
    print('Final CSV generated!')