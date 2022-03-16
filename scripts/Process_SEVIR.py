""" 
Description:
-----------
This script will help users process the HDF files into tabular data (using parallel processing). 
The quntiles calculated are q = [0,1,10,25,50,75,90,99,100]

Note: Thes data are already made and can be found in the 'datasets' directory. 

how to run:
-----------
python -u Process_SEVIR.py

Author:
----------- 
Randy Chase
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd  
from tqdm import tqdm 
from sevir_methods import *
import multiprocessing as mp

# On some Linux systems setting file locking to false is also necessary:
import os
os.environ["HDF5_USE_FILE_LOCKING"]='FALSE' 

# Enter path to the SEVIR data location
DATA_PATH    = '/Your/Path/To/Sevir/data'
CATALOG_PATH = '/Your/Path/To/Sevir/CATALOG.csv' 

# Read catalog
catalog = pd.read_csv(CATALOG_PATH,parse_dates=['time_utc'],low_memory=False)
# Desired image types
img_types = set(['vis','ir069','ir107','vil'])
# Group by event id, and filter to only events that have all desired img_types
events = catalog.groupby('id').filter(lambda x: img_types.issubset(set(x['img_type']))).groupby('id')
event_ids = list(events.groups.keys())


#init pool, this is a pool of processors to do our processing               
pool = mp.Pool(processes=18)

#if you want to run this from the terminal use the progress bar
d_list = []
for d in tqdm(pool.imap_unordered(retrieve_stats_par,event_ids), total=len(event_ids)):
    d_list.append(d)

#clean up pool and workers 
pool.close()
pool.join()
del pool 

####the parallel function gives us things as single event dataframes. We need to concatentate them 
#initialize them with the first dataframe 
df_ir = d_list[0][0]
df_wv = d_list[0][1]
df_vis = d_list[0][2]
df_vil = d_list[0][3]
df_li = d_list[0][4]

#loop over the rest to concat. 
for d in tqdm(d_list[1:]):
    #if len(d) == 1, that means its nan, so skip. 
    if len(d) == 1:
        continue
    else:
        df_ir = pd.concat([df_ir,d[0]])
        df_wv = pd.concat([df_wv,d[1]])
        df_vis = pd.concat([df_vis,d[2]])
        df_vil = pd.concat([df_vil,d[3]])
        df_li = pd.concat([df_li,d[4]])

# save them out 
df_ir.to_csv('/ourdisk/hpc/ai2es/datasets/dont_archive/sevir/IR_stats_master.csv')
df_wv.to_csv('/ourdisk/hpc/ai2es/datasets/dont_archive/sevir/WV_stats_master.csv')
df_vis.to_csv('/ourdisk/hpc/ai2es/datasets/dont_archive/sevir/VIS_stats_master.csv')
df_vil.to_csv('/ourdisk/hpc/ai2es/datasets/dont_archive/sevir/VIL_stats_master.csv')
df_li.to_csv('/ourdisk/hpc/ai2es/datasets/dont_archive/sevir/LI_stats_master.csv')