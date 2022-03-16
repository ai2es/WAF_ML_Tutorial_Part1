"""
Description:
-----------
This script contains methods to do the feature engineering. In other words, the functions in here help us
process the > 10,000 storms in an timely manner. 

Author:
----------- 
Randy Chase
"""
import numpy as np
import h5py
import pandas as pd
import warnings
#suppress warnings because i want to see my progress bar on one line 
warnings.filterwarnings('ignore')


# Enter path to the SEVIR data location
DATA_PATH    = '/path/to/sevir/data'
CATALOG_PATH = '/path/to/sevir/CATALOG.csv' 

def read_data(sample_event, img_type, data_path=DATA_PATH,fillna=True):
    """
    This function was written by the creators of the SEVIR data. 

    Reads single SEVIR event for a given image type.
    
    Parameters
    ----------
    sample_event   pd.DataFrame
        SEVIR catalog rows matching a single ID
    img_type   str
        SEVIR image type
    data_path  str
        Location of SEVIR data
    
    Returns
    -------
    np.array
       LxLx49 tensor containing event data
    """
    fn = sample_event[sample_event.img_type==img_type].squeeze().file_name
    fi = sample_event[sample_event.img_type==img_type].squeeze().file_index
    with h5py.File(data_path + '/' + fn,'r') as hf:
        data=hf[img_type][fi] 
    if fillna:
        if (img_type =='vil') or (img_type =='vis'):
            data = np.asarray(data,dtype=np.float32)
            data[data < 0] = np.nan
        else:
            data = np.asarray(data,dtype=np.float32)
            data[data < -30000] = np.nan
    return data

def lght_to_grid(data,fillna=True):
    """
    This function was written by the creators of the SEVIR data. 

    Converts SEVIR lightning data stored in Nx5 matrix to an LxLx49 tensor representing
    flash counts per pixel per frame
    
    Parameters
    ----------
    data  np.array
       SEVIR lightning event (Nx5 matrix)
       
    Returns
    -------
    np.array 
       LxLx49 tensor containing pixel counts
    """
    FRAME_TIMES = np.arange(-120.0,125.0,5) * 60 # in seconds
    out_size = (48,48,len(FRAME_TIMES))
    if data.shape[0]==0:
        return np.zeros(out_size,dtype=np.float32)

    # filter out points outside the grid
    x,y=data[:,3],data[:,4]
    m=np.logical_and.reduce( [x>=0,x<out_size[0],y>=0,y<out_size[1]] )
    data=data[m,:]
    if data.shape[0]==0:
        return np.zeros(out_size,dtype=np.float32)

    # Filter/separate times
    # compute z coodinate based on bin locaiton times
    t=data[:,0]
    z=np.digitize(t,FRAME_TIMES)-1
    z[z==-1]=0 # special case:  frame 0 uses lght from frame 1

    x=data[:,3].astype(np.int64)
    y=data[:,4].astype(np.int64)

    k=np.ravel_multi_index(np.array([y,x,z]),out_size)
    n = np.bincount(k,minlength=np.prod(out_size))
    d = np.reshape(n,out_size).astype(np.float32)
    if fillna:
        data = np.asarray(data,dtype=np.float32)
        d[d<0] = np.nan
    return d

def read_lght_data( sample_event, data_path=DATA_PATH):
    """
    This function was written by the creators of the SEVIR data. 

    Reads lght data from SEVIR and maps flash counts onto a grid  
    
    Parameters
    ----------
    sample_event   pd.DataFrame
        SEVIR catalog rows matching a single ID
    data_path  str
        Location of SEVIR data
    
    Returns
    -------
    np.array 
       LxLx49 tensor containing pixel counts for selected event
    
    """
    fn = sample_event[sample_event.img_type=='lght'].squeeze().file_name
    id = sample_event[sample_event.img_type=='lght'].squeeze().id
    with h5py.File(data_path + '/' + fn,'r') as hf:
        data      = hf[id][:] 
    return lght_to_grid(data)

def retrieve_stats_par(event):
    """
    Takes a single event, opens all data with that event, drops 0s/nans, calculates percentiles,counts lightning. 
    Designed to be modular for parallel use (i.e., multiprocessing) 
    
    Parameters
    ----------
    event   string
        SEVIR catalog group, from events.groups.keys()

    
    Returns
    -------
    list 
       list of dataframes, each entry is for a variable
    
    """

    # Read catalog
    catalog = pd.read_csv(CATALOG_PATH,parse_dates=['time_utc'],low_memory=False)
    # Desired image types
    img_types = set(['vis','ir069','ir107','vil'])
    # Group by event id, and filter to only events that have all desired img_types
    events = catalog.groupby('id').filter(lambda x: img_types.issubset(set(x['img_type']))).groupby('id')

    #this prevents certain files from breaking it
    try:
        #get event details 
        sample_event = events.get_group(event)

        #get meta to pass through 
        meta = grab_meta(sample_event)

        #load data
        ir609 = read_data(sample_event, 'ir069')
        ir107 = read_data(sample_event, 'ir107')
        vil = read_data(sample_event, 'vil')
        vil = get_right_units_vil(vil)
        vis = read_data(sample_event, 'vis')
        lght = read_lght_data(sample_event)

        #get traditional ML params (features; X)
        q = [0,1,10,25,50,75,90,99,100]
        ir = make_stats_df(ir107,q,meta)
        wv = make_stats_df(ir609,q,meta)
        vis = make_stats_df(vis,q,meta)
        vil = make_stats_df(vil,q,meta)
        #get labels (y)
        li = get_counts_df(lght,meta)
        #return list of dataframes to concat 
        return [ir,wv,vis,vil,li]
    except:
        #return nan if broken 
        return [np.nan]

def make_stats_df(data,q,meta):
    """ Abstract percentile function """
    mat = np.nanpercentile(data,q,axis=(0,1)).T
    df = pd.DataFrame(mat)
    df = df.set_index(meta['time'])
    header = np.asarray(q,dtype=np.str)
    header = np.char.rjust(header, 3,'0')
    header = np.char.rjust(header, 4,'q')
    df.columns = header 
    df['event'] = meta['event_t']
    return df 
def get_counts_df(data,meta):
    mat = np.nansum(data,axis=(0,1))[:,np.newaxis]
    df = pd.DataFrame(mat)
    df = df.set_index(meta['time'])
    df.columns = ['c'] 
    df['event'] = meta['event_t']
    return df 

def grab_meta(sample_event):
    event_type = np.tile(np.asarray([sample_event.event_type.iloc[0],]),(49,1))
    time = np.tile(np.asarray(np.datetime64(sample_event.time_utc.iloc[0])),(49,))
    time_off = sample_event.minute_offsets.iloc[0]
    time_off = np.asarray(time_off.split(':'),dtype=np.int64)
    timedelta = pd.Timedelta(minutes=1)
    time = time + timedelta*time_off
    meta = {'event_t':event_type,'time':time}
    return meta 

def get_right_units_vil(vil):
    """they scaled VIL weird, so this unscales it"""
    tmp = np.zeros(vil.shape)
    idx = np.where(vil <=5)
    tmp[idx] = 0
    idx = np.where((vil>5)*(vil <= 18))
    tmp[idx] = (vil[idx] -2)/90.66
    idx = np.where(vil>18)
    tmp[idx] = np.exp((vil[idx] - 83.9)/38.9)
    return tmp