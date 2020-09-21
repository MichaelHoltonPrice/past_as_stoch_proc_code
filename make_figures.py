import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg') # so a display is not needed (e.g., if running Ubuntu with WSL 2 or in Docker container)
from matplotlib import pyplot as plt
from seshat import *
    
# Call generate_core_data (see seshat module defined in seshat.py) to load data for subsequent analyses
print('Calling generate_core_data')
worldRegions,NGAs,PC_matrix,CC_df, CC_times, CC_scaled, PC1_df,regionDict,t_min,t_max,pc1_min,pc1_max,flowInfo, movArrayOut,velArrayOut,movArrayIn,velArrayIn,flowInfoInterp,movArrayOutInterp,r0,minPoints,dGrid,u0Vect,v0Vect,velScaling = generate_core_data()

###############################################################################
# Figure 1
# Average score of observations on PC2 in a sliding window along PC1
###############################################################################
print('Making Figure 1')
PC1=velArrayOut[:,0,0]
PC2=velArrayOut[:,1,0]
PC1_vel = velArrayOut[:,0,1]*100
PC2_vel = velArrayOut[:,1,1]*100

window_width =1.0
overlap = .5

score_list = []
vel_list = []
score_std_list = []
vel_std_list = []

score_error_list = []
vel_error_list = []

center_list = []

PC1_min = np.min(PC1)
PC1_max = np.max(PC1)

n_window = np.ceil( (PC1_max - PC1_min - window_width)/(window_width-overlap) ).astype(int)

for i in range(n_window):
  window = np.array([PC1_min+i*(window_width-overlap), PC1_min+i*(window_width-overlap)+window_width])
  center = np.mean(window)
  loc = (window[0]<=PC1) * (PC1<window[1])
  
  PC2_in_window = PC2[loc]
  PC2_vel_in_window = PC2_vel[loc]
  PC2_vel_in_window = PC2_vel_in_window[~np.isnan(PC2_vel_in_window)]
    
  score = np.mean(PC2_in_window)
  vel = np.mean(PC2_vel_in_window)
  score_std = np.std(PC2_in_window)
  vel_std = np.std(PC2_vel_in_window)

  score_error = score_std/np.sqrt(len(PC2_in_window) )
  vel_error = vel_std/np.sqrt( len(PC2_vel_in_window) )
  
  center_list.append(center)
  score_list.append(score)
  vel_list.append(vel)
  score_std_list.append(score_std)
  vel_std_list.append(vel_std)
  
  score_error_list.append(score_error)
  vel_error_list.append(vel_error)
   
plt.axis()
plt.xlim(-6,5)
#plt.ylim(-3,3)
plt.plot(center_list, score_list, 'b-o')
plt.errorbar(center_list, score_list, yerr=score_error_list, capthick=2, capsize=3)
plt.xlabel("PC1",size=15)
plt.ylabel("PC2",size=15)
plt.savefig("PC2_vs_PC1.pdf")
plt.close()


###############################################################################
# Figure 2
# A histogram of PC1 values (projections onto PC1 of pooled imputations)
###############################################################################
print('Making Figure 2')
n, bins, patches = plt.hist(PC_matrix[:,0], np.arange(-6.4,4.2,.2), facecolor='blue',alpha=0.75)
plt.xlabel("PC1 Value",size=15)
plt.ylabel("Counts",size=15)
plt.savefig("PC1_histogram.pdf")
plt.close()

########################################################################
# Figure 3
# Markov transition matrix heatmap
########################################################################
print('Making Figure 3')
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


###############################################################################
# Figure S1
# Movement plot for Moralizing Gods
###############################################################################
print('Making Figure S1')

lineWidth = .01
headWidth = .15
headLength = .1

mgMissCol = 'grey'
mgAbsnCol = 'blue'
mgPresCol = 'green'
mgScatCol = 'red'

plt.figure(figsize=(17,8.85))
plt.axis('scaled')
plt.xlim(-6,5)
plt.ylim(-3,3)
plt.xticks(size=25)
plt.yticks(size=25)

# Add vertical bars for the information thresholds at -2.5 and -.5
plt.plot([-2.5,-2.5],[-3,+3],color='black')
plt.plot([-0.5,-0.5],[-3,+3],color='black')

# Load, respectively, the data generated by the R code in ./mhg_code and the
# data provided by Whitehouse et al. (2019) in their Nature paper. The data are
# identical for the 12 NGAs in the Nature data, but the data frame mg_df
# contains data for additional NGAs. nat_df is only used to determine which
# NGAs were used in the Nature publication.
mg_df = pd.read_csv(os.path.join('mhg_code','data_used_for_nature_analysis.csv'))
nat_df = pd.read_csv(os.path.join('mhg_code','other_csv','41586_2019_1043_MOESM6_ESM_sheet1.csv'))
NGAs_nat = np.unique(nat_df['NGA'].values)
mg_tab = pd.DataFrame(columns=['NGA','Start','Stop','Value','Nature'])

mg_one_list=[]
for i in range(0,movArrayOut.shape[0]):
  if not np.isnan(movArrayOut[i,0,1]):
    nga = flowInfo['NGA'][i]
    time = flowInfo['Time'][i]
    if mg_df.loc[ (mg_df['NGA']==nga) & (mg_df['Time']==time)]['MoralisingGods'].shape[0]>0:
      if mg_df.loc[ (mg_df['NGA']==nga) & (mg_df['Time']==time)]['MoralisingGods'].values[0]==0:
        rgb = mgAbsnCol
      elif mg_df.loc[ (mg_df['NGA']==nga) & (mg_df['Time']==time)]['MoralisingGods'].values[0]==1:
        rgb = mgPresCol
        if not nga in mg_one_list:
          rgb0 = 'orange'
          if nga in NGAs_nat:
            rgb0 = mgScatCol
          else:
            rgb0 = 'orange'
          plt.scatter(velArrayOut[i,0,0],velArrayOut[i,1,0], color=rgb0,zorder=2)
          mg_one_list.append( nga )
      else:
        rgb = mgMissCol

    plt.arrow(movArrayOut[i,0,0],movArrayOut[i,1,0],movArrayOut[i,0,1],movArrayOut[i,1,1],width=lineWidth,head_width=headWidth,head_length=headLength,color=rgb,alpha=.5,zorder=1)

    # Next, plot interpolated points (if necessary)
    # Doing this all very explicitly to make the code clearer
    dt = velArrayOut[i,0,2]
    if dt > 100:
      for n in range(0,int(dt / 100) - 1):
        pc1 = movArrayOut[i,0,0] + velArrayOut[i,0,1]*(float(n+1))*100.
        pc2 = movArrayOut[i,1,0] + velArrayOut[i,1,1]*(float(n+1))*100.
        plt.scatter(pc1,pc2,s=10,color=rgb,alpha=.5,zorder=1)

plt.xlabel("PC1", size=25)
plt.ylabel("PC2", size=25)
plt.savefig("pc12_movement_plot_colored_by_MoralisingGods.pdf")
plt.close()

print("Done with make_figures.py")
