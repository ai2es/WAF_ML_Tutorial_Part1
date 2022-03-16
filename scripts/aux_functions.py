""" 
Description:
-----------
This script hosts many helper functions to make notebooks cleaner. The hope is to not distract users with ugly code. 

Author:
----------- 
Randy Chase
"""

import numpy as np
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