""" 
Description:
-----------
This script hosts many helper functions to make notebooks cleaner. The hope is to not distract users with ugly code. 

Author:
----------- 
Randy Chase
"""

import numpy as np
import pandas as pd
import matplotlib.patheffects as path_effects

#outlines for text 
pe1 = [path_effects.withStroke(linewidth=2,
                             foreground="k")]
pe2 = [path_effects.withStroke(linewidth=2,
                             foreground="w")]

def show_vals(da,ax):
    vals = da.values
    x = np.arange(0,vals.shape[0])
    y = np.arange(0,vals.shape[1])
    X,Y = np.meshgrid(x,y)
    X = np.ravel(X)
    Y = np.ravel(Y)
    V = np.ravel(vals)
    for i in np.arange(0,len(X)):
        fillstr = np.asarray(np.round(V[i],2),dtype=str)
        fillstr = np.char.ljust(fillstr,4,'0')
        if np.round(V[i],2) > 0.5:
            ax.text(X[i]-0.2,Y[i],fillstr,color='k')
        else:
            ax.text(X[i]-0.2,Y[i],fillstr,color='w')
    return

def draw_zoom_window(ax,a,b):
    ax.plot([a,a,a+10,a+10,a],[b,b+10,b+10,b,b],'-k',lw=3)
    ax.plot([a,a,a+10,a+10,a],[b,b+10,b+10,b,b],'-',color='dodgerblue',lw=2)
    return a,b

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

def plot_feature_loc(da,ax,q = [0,1,10,25,50,75,90,99,100]):
    """ This will plot representative pixels matching the quantiles given """
    vals = np.nanpercentile(da,q)
    xs = []
    ys = []
    for v in vals:
        local_idx = np.where(np.round(da.values,1) == np.round(v,1))
        if len(local_idx[0]) > 1:
            ii = np.random.choice(np.arange(0,len(local_idx[0])),size=1)
            xs.append(local_idx[0][ii[0]])
            ys.append(local_idx[1][ii[0]])
        else:
            ii = 0
            xs.append(local_idx[0][ii])
            ys.append(local_idx[1][ii])
            
    markerlist = ['min','$01$','$10$','$25$','$50$','$75$','$90$','$99$','max']
    zlist = list(zip(xs,ys))
    for i,(x,y) in enumerate(zlist):
        ax.text(y,x,markerlist[i],path_effects=pe2)
    return 

def adjust_keys(df,keyadd,dask=False,dropevent=False):
    if dask:
        keys = df.columns
        newkeys = []
        newkeys.append('dtime')
        newkeys = newkeys + list(keys[1:-1]+keyadd)
        newkeys.append(keys[-1])
    else:
        keys = df.keys()
        newkeys = list(keys[:-1]+keyadd)
        newkeys.append(keys[-1])
        
    df.columns = newkeys
    if dropevent:
        df = df.drop(columns='event')
        
    if dask:
        df['dtime'] = df['dtime'].astype(np.datetime64)
    return df 

def clear_nan(X,y):
    tmp = np.hstack([X,y.reshape([y.shape[0],1])])
    df_tmp = pd.DataFrame(tmp)
    df_tmp = df_tmp.dropna(how='any')
    tmp = df_tmp.to_numpy()
    X = tmp[:,:-1]
    y = tmp[:,-1:]
    y = np.asarray(y.squeeze(),dtype=int)
    return X,y

def load_n_combine_df(path_to_data='../datasets/',features_to_keep=np.arange(0,36,1),class_labels=True):
    df_ir = pd.read_csv(path_to_data + 'IR_stats_master.csv',index_col=0,low_memory=False,parse_dates=True)
    df_wv = pd.read_csv(path_to_data + 'WV_stats_master.csv',index_col=0,low_memory=False,parse_dates=True)
    df_vis = pd.read_csv(path_to_data + 'VIS_stats_master.csv',index_col=0,low_memory=False,parse_dates=True)
    df_vil = pd.read_csv(path_to_data + 'VIL_stats_master.csv',index_col=0,low_memory=False,parse_dates=True)
    df_li = pd.read_csv(path_to_data + 'LI_stats_master.csv',index_col=0,low_memory=False,parse_dates=True)

    #get rid of that outlier 
    df_wv = df_wv.where(df_wv.q000 > -10000)
    
    #get rid of NaNs
    idx_keep = np.where(~df_vis.isna().all(axis=1).values)[0]
    df_ir = df_ir.iloc[idx_keep]
    df_wv = df_wv.iloc[idx_keep]
    df_vis = df_vis.iloc[idx_keep]
    df_vil = df_vil.iloc[idx_keep]
    df_li = df_li.iloc[idx_keep]

    #make sure idx are in order 
    df_ir = df_ir.sort_index()
    df_wv = df_wv.sort_index()
    df_vis = df_vis.sort_index()
    df_vil = df_vil.sort_index()
    df_li = df_li.sort_index()

    #adjust keys so merging doesnt make keys confusing
    df_ir = adjust_keys(df_ir,'_ir')
    df_wv = adjust_keys(df_wv,'_wv')
    df_vis = adjust_keys(df_vis,'_vi')
    df_vil = adjust_keys(df_vil,'_vl')
    df_li = adjust_keys(df_li,'_li')

    #drop event column 
    df_ir= df_ir.drop(columns='event')
    df_wv= df_wv.drop(columns='event')
    df_vis= df_vis.drop(columns='event')
    df_vil= df_vil.drop(columns='event')
    df_li = df_li.drop(columns='event')
    
    #slice on time 
    train_slice = slice('2017-01-01','2019-06-01')
    other_slice = slice('2019-06-01','2019-12-31')

    df_ir_tr = df_ir[train_slice]
    df_ir_ot = df_ir[other_slice]
    df_wv_tr = df_wv[train_slice]
    df_wv_ot = df_wv[other_slice]
    df_vis_tr = df_vis[train_slice]
    df_vis_ot = df_vis[other_slice]
    df_vil_tr = df_vil[train_slice]
    df_vil_ot = df_vil[other_slice]
    df_li_tr = df_li[train_slice]
    df_li_ot = df_li[other_slice]


    idx = np.arange(0,df_ir_ot.shape[0])
    #set random seed for reproducability 
    np.random.seed(seed=42)
    idx_v = np.random.choice(idx,size=int(idx.shape[0]/2),replace=False)
    idx_v.sort()
    idx_t = np.setdiff1d(idx,idx_v)
    idx_t.sort()

    df_ir_va = df_ir_ot.iloc[idx_v]
    df_ir_te = df_ir_ot.iloc[idx_t]
    df_wv_va = df_wv_ot.iloc[idx_v]
    df_wv_te = df_wv_ot.iloc[idx_t]
    df_vis_va = df_vis_ot.iloc[idx_v]
    df_vis_te = df_vis_ot.iloc[idx_t]
    df_vil_va = df_vil_ot.iloc[idx_v]
    df_vil_te = df_vil_ot.iloc[idx_t]
    df_li_va = df_li_ot.iloc[idx_v]
    df_li_te = df_li_ot.iloc[idx_t]
    
    X_train = np.hstack([df_ir_tr.to_numpy()*1e-2,df_wv_tr.to_numpy()*1e-2,df_vis_tr.to_numpy()*1e-4,df_vil_tr.to_numpy()])
    X_validate = np.hstack([df_ir_va.to_numpy()*1e-2,df_wv_va.to_numpy()*1e-2,df_vis_va.to_numpy()*1e-4,df_vil_va.to_numpy()])
    X_test= np.hstack([df_ir_te.to_numpy()*1e-2,df_wv_te.to_numpy()*1e-2,df_vis_te.to_numpy()*1e-4,df_vil_te.to_numpy()])

    #filter nans 
    idx_train = np.isnan(X_train)
    #choose 
    X_train = X_train[:,features_to_keep]
    X_validate = X_validate[:,features_to_keep]
    X_test = X_test[:,features_to_keep]

    #make class labels
    if class_labels:
        y_train = np.zeros(X_train.shape[0],dtype=int)
        y_train[np.where(df_li_tr.c_li.values >= 1)] = 1

        y_validate = np.zeros(X_validate.shape[0],dtype=int)
        y_validate[np.where(df_li_va.c_li.values >= 1)] = 1

        y_test = np.zeros(X_test.shape[0],dtype=int)
        y_test[np.where(df_li_te.c_li.values >= 1)] = 1
    else:
        y_train = df_li_tr.c_li.values
        y_validate = df_li_va.c_li.values
        y_test = df_li_te.c_li.values



    #clean out nans 
    X_train,y_train = clear_nan(X_train,y_train)
    X_validate,y_validate = clear_nan(X_validate,y_validate)
    X_test,y_test = clear_nan(X_test,y_test)
    
    return (X_train,y_train),(X_validate,y_validate),(X_test,y_test)