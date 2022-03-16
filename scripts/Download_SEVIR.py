""" 
Description:
-----------
This script will help users download THE ENTIRE SEVIR dataset. 
!NOTE NOTE NOTE! 
This data is about 1 TB in size so please do not run this on your machine unless you have that storage space and 
or you can even download that much data on your internet plan (my internet plan has a CAP of 1 TB per month).

how to run:
-----------
python -u Download_SEVIR.py

Author:
----------- 
Randy Chase
"""

#this is the amazon web service package
import s3fs
#this is a progress bar so you can see where you are in downloading
import tqdm 

# Use the anonymous credentials to access public data
fs = s3fs.S3FileSystem(anon=True)

#these are all the variable keys in the dataset
var_list = ['ir069','ir107','lght','processed','vil','vis']

#we need to build the paths to each file. 
paths = []
for vari in var_list:
  paths = paths + fs.ls('s3://sevir/data/'+ vari + '/2018/')+fs.ls('s3://sevir/data/'+ vari + '/2019/')

#add in the CATALOG dataset (which has info on the time and date of each event)
paths = ['sevir/CATALOG.csv'] + paths 

#okay, loop over all paths and download them. 
for i,p in enumerate(tqdm.tqdm(paths)):
    #make savepath
    s = '/Put/Your/Destination/Path/Here' + p[6:]
    fs.get(p,s)