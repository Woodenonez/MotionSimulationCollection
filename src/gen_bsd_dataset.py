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

utils_data.gather_all_data_trajectory(save_path, past, maxT=maxT, minT=minT) # go through all the obj folders and put them together in one CSV
print('Final CSV generated!')

