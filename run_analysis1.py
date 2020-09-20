import time
from collections import defaultdict
import os
import math
import time
import pickle
import copy 
import sys

import pandas as pd 
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.mixture import GaussianMixture as GMM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.linear_model import LinearRegression
from scipy.stats import multivariate_normal, norm
import scipy.spatial as spatial
import tensorflow as tf
# sudo apt-get install python3-tk # needs this
import matplotlib
matplotlib.use('Agg') # so a display is not needed (e.g., if running Ubuntu with WSL 2 or in Docker container)
from matplotlib import pyplot as plt, cm as cm, mlab as mlab
from matplotlib.patches import Patch
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
#import seaborn as sns; sns.set()
from IPython.core.debugger import Pdb

from seshat import *
    
# Call generate_core_data (see seshat module defined in seshat.py) to load data for subsequent analyses
print('Calling generate_core_data')
worldRegions,NGAs,PC_matrix,CC_df, CC_times, CC_scaled, PC1_df,regionDict,t_min,t_max,pc1_min,pc1_max,flowInfo, movArrayOut,velArrayOut,movArrayIn,velArrayIn,flowInfoInterp,movArrayOutInterp,r0,minPoints,dGrid,u0Vect,v0Vect,velScaling = generate_core_data()


###############################################################################
# Figure 2
# A histogram of PC1 values (projections onto PC1 of pooled imputations)
###############################################################################
num_bins = 50
#n, bins, patches = plt.hist(PC_matrix[:,0], num_bins, density=0, facecolor='blue', alpha=0.5)
#n, bins, patches = plt.hist(PC_matrix[:,0], num_bins, density=0, facecolor='grey')
#n, bins, patches = plt.hist(PC_matrix[:,0], num_bins, facecolor='grey')
n, bins, patches = plt.hist(PC_matrix[:,0], np.arange(-6.4,4.2,.2), facecolor='blue',alpha=0.75)
#ax = plt.axes()
#ax.set_facecolor('white')
plt.xlabel("PC1 Value",size=15)
plt.ylabel("Counts",size=15)
plt.savefig("pc1_histogram.pdf")
plt.close()

########################################################################
# Figure 3
# Markov transition matrix heatmap
########################################################################

d_x_in = velArrayIn[:,:2,0] #The ending point of "IN" vector in the first 2 PC space
d_y_in = velArrayIn[:,2:,0] #The ending point of "OUT" vector in the other 7 PC space
d_v_in = velArrayIn[:,:2,1] #The "IN" velocity in the first 2 PC space
d_w_in = velArrayIn[:,2:,1] #The "IN" velocity in the other 7 PC space
d_xy_in = velArrayIn[:,:,0] #The ending point of "OUT" vector in 9 PC space

pos_v_not_nan_in = np.where(~np.isnan(d_v_in))[0][::2].astype(np.int32) #Position of non-NaN points due to starting point
pos_v_nan_in = np.where(np.isnan(d_v_in))[0][::2].astype(np.int32) #Position of NaN points due to starting point

n_obs_in = len(pos_v_not_nan_in)

d_xy_tf_in = tf.constant(d_xy_in[pos_v_not_nan_in,:],dtype=tf.float32)
d_v_tf_in = tf.constant(d_v_in[pos_v_not_nan_in,:],dtype=tf.float32) #Removed NaN already

d_x_notnan_in = d_x_in[pos_v_not_nan_in,:]
d_xy_notnan_in = d_xy_in[pos_v_not_nan_in,:]
d_v_notnan_in = d_v_in[pos_v_not_nan_in,:]

d_x_out = velArrayOut[:,:2,0] #The starting point of "OUT" vector in the first 2 PC space
d_y_out = velArrayOut[:,2:,0] #The starting point of "OUT" vector in the other 7 PC space
d_v_out = velArrayOut[:,:2,1] #The "OUT" velocity in the first 2 PC space
d_w_out = velArrayOut[:,2:,1] #The "OUT" velocity in the other 7 PC space
d_xy_out = velArrayOut[:,:,0] #The starting point of "OUT" vector in 9 PC space

pos_v_not_nan_out = np.where(~np.isnan(d_v_out))[0][::2].astype(np.int32) #Position of non-NaN points due to end point
pos_v_nan_out = np.where(np.isnan(d_v_out))[0][::2].astype(np.int32) #Position of NaN points due to end point

n_obs_out = len(pos_v_not_nan_out)

#Removing NaN
d_x_notnan_out = d_x_out[pos_v_not_nan_out,:]
d_xy_notnan_out = d_xy_out[pos_v_not_nan_out,:]
d_v_notnan_out = d_v_out[pos_v_not_nan_out,:]

init_PC1 = d_x_out[pos_v_nan_in,0]
final_PC1 = d_x_out[pos_v_nan_out,0]
PC1 = d_x_notnan_out[:,0]
vel_PC1 = d_v_notnan_out[:,0]

def cum_transition(PC1,vel_PC1_annual,init_PC1, years_move=100,n_bin=6,n_iter=10 ,flag_barrier=False, graph=True,
					graph_name='' ,flag_rm_jump=False, transition_prob_input=None,ratio_init_input=None):
    vel_PC1 = vel_PC1_annual*years_move
    left_end = np.min(PC1)#np.min(PC1+vel_PC1)
    right_end = np.max(PC1)# np.max(PC1+vel_PC1)
    width = (right_end-left_end)/n_bin
    center_list =( np.linspace(left_end,right_end-width,n_bin) + np.linspace(left_end+width,right_end,n_bin) )/2

    if ratio_init_input is None:
        count_init_PC1 = np.zeros(n_bin)
        for i in range(n_bin):
            loc = (left_end+i*width <=init_PC1)*(init_PC1<left_end+(i+1)*width)
            count_init_PC1[i] = np.sum(loc)
        ratio_init = count_init_PC1/np.sum(count_init_PC1)
        
    else:
        ratio_init = ratio_init_input
        
    if transition_prob_input is None:
        transition_count_matrix = np.zeros([n_bin,n_bin])
        transition_prob_matrix = np.zeros([n_bin,n_bin])
        for i in range(n_bin):
            loc_origin = (left_end+i*width <=PC1)*(PC1<left_end+(i+1)*width)
            num_origin_i = np.sum(loc_origin)
            if num_origin_i == 0:
                print('No observation starting in bin %i'%i)
            center = left_end+(i+.5)*width
            dest = center+vel_PC1[loc_origin]
            if flag_barrier:
                loc_dest_right = (dest>right_end)
                transition_count_matrix[i,-1]=transition_count_matrix[i,-1]+np.sum(loc_dest_right)
                loc_dest_left = (dest<left_end)
                transition_count_matrix[i,0]=transition_count_matrix[i,0]+np.sum(loc_dest_left)
            for j in range(n_bin):
                loc_dest_j = (left_end+j*width <=dest)*(dest<left_end+(j+1)*width)
                transition_count_matrix[i,j] = transition_count_matrix[i,j] + np.sum(loc_dest_j)
                if flag_rm_jump:
                    transition_count_matrix[:5,7:] = 0
                
                transition_prob_matrix[i,j] = transition_count_matrix[i,j]/num_origin_i
    else:
        transition_count_matrix = None
        transition_prob_matrix = transition_prob_input
            
    #transition_count_matrix,transition_prob_matrix = calc_transition_matrix(PC1,vel_PC1,n_bin)
    #---------
    
    dist_transition = []  
    dist_cum_transition = []      

    dist_i = ratio_init.reshape([-1,1])
    dist_cum_i = ratio_init.reshape([-1,1])
        
    for i in range(n_iter):
        dist_transition.append(dist_i)
        dist_cum_transition.append(dist_cum_i)
        dist_i = np.matmul(transition_prob_matrix.T,dist_i)
        dist_cum_i = dist_cum_i + dist_i
            
    return transition_count_matrix, transition_prob_matrix,ratio_init,dist_transition,dist_cum_transition
        
n_bin=10
years_move=500
left_end = np.min(PC1)
right_end = np.max(PC1)
width = (right_end-left_end)/n_bin
center_list =( np.linspace(left_end,right_end-width,n_bin) + np.linspace(left_end+width,right_end,n_bin) )/2

end_year_NGA = np.array(flowInfo.loc[pos_v_nan_out]['Time']).astype(float)
start_year_NGA =np.array(flowInfo.loc[pos_v_nan_in]['Time']).astype(float) 
duration_NGA = end_year_NGA-start_year_NGA
n_iter_list =   np.floor(duration_NGA/years_move).astype(int)

flag_barrier=1
n_iter=10
for i in range(30):
    init = init_PC1[i] #doesn't matter
    transition_count_matrix, transition_prob_matrix,ratio_init,dist_transition,dist_cum_transition = cum_transition(PC1,vel_PC1,init, years_move=years_move,n_bin=n_bin,n_iter=n_iter,flag_barrier=flag_barrier,graph=False ,flag_rm_jump=False,ratio_init_input=None )    

plt.imshow(transition_prob_matrix, cmap='hot')
plt.xticks(np.arange(n_bin),np.around(center_list,decimals=1) )
plt.yticks(np.arange(n_bin),np.around(center_list,decimals=1) )
plt.colorbar()
plt.xlabel("PC1 Value",size=15)
plt.ylabel("PC1 Value",size=15)
plt.savefig(os.path.join("heatmap-of-transition-matrix.pdf"))

print("Done with run_analysis1.py")
