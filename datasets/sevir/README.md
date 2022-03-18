# What's in this folder?

This folder houses all the processed SEVIR datasets for the manuscript. More speicfically, it contains (besides this file) the following files:

1. ```CATALOG.csv```
    - This is the metadata file associated with the original dataset. There is a lot of meta-data in here. 
2. ```IR_stats_master.csv```
    - This csv file contains all of the engineered features for the clean infrared channel on GOES. This was produced by me (RJC) from the original images. This is a result of the ```Process_SEVIR.py``` script.
3. ```LI_stats_master.csv```
    - This csv file contains all of the engineered features for the count of lightning flashes observed by GOES. This was produced by me (RJC). This is a result of the ```Process_SEVIR.py``` script.
4. ```onestorm.nc```
    - This netCDF4 file contains 12 timesteps from 1 storm in 06 August 2018. This day had the most numerous lightning flashes in it (>5000 in the first time step). 
5. ```VIL_stats_master.csv```
    - This csv file contains all of the engineered features for the vertically integrated liquid (VIL) from NEXRAD. This was produced by me (RJC). This is a result of the ```Process_SEVIR.py``` script.
6. ```VIS_stats_master.csv```
    - This csv file contains all of the engineered features for the red visibile channel from GOES. This was produced by me (RJC). This is a result of the ```Process_SEVIR.py``` script.
7. ```WV_stats_master.csv```
    - This csv file contains all of the engineered features for the mid-tropospheric water vapor channel from GOES. This was produced by me (RJC). This is a result of the ```Process_SEVIR.py``` script.