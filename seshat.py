import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

def tailored_svd(data):
    # perform singular value decomposition on the given data matrix
    #center the data
    mean = np.mean(data, axis=0)
    data -= mean
    P, D, Q = np.linalg.svd(data, full_matrices=False)
    return P, D, Q

# Create the grid for PC1 and PC2 in a standalone file to avoid replicating code
def createGridForPC12(dGrid,flowArray):
    # Remove endpoints
    ind = [True if not np.isnan(flowArray[i,0,1]) else False for i in range(flowArray.shape[0])]
    fa = flowArray[ind,:,:]
    points2D = fa[:,range(0,2),0]

    u0Min = np.floor(np.min(points2D[:,0] - dGrid) / dGrid) * dGrid # PC1 min
    u0Max = np.ceil(np.max(points2D[:,0] + dGrid) / dGrid) * dGrid # PC1 max
    v0Min = np.floor(np.min(points2D[:,1] - dGrid) / dGrid) * dGrid # PC1 min
    v0Max = np.ceil(np.max(points2D[:,1] + dGrid) / dGrid) * dGrid # PC1 max
    u0Vect = np.arange(u0Min,u0Max,dGrid)
    v0Vect = np.arange(v0Min,v0Max,dGrid)
    return u0Vect,v0Vect

def generate_core_data():
    # Read csv data files
    CC_file = "data1.csv" #20 imputed sets
    PC1_file = "data2.csv" #Turchin's PC1s
    #polity_file = os.path.abspath(os.path.join("./..","data","scraped_seshat.csv")) #Info on polities spans and gaps
    CC_df = pd.read_csv(CC_file) # A pandas dataframe
    PC1_df = pd.read_csv(PC1_file) # A pandas dataframe
#polity_df = pd.read_csv(polity_file) # A pandas dataframe

    # Create a dictionary that maps from World Region to Late, Intermediate, and Early NGAs
    regionDict = {"Africa":["Ghanaian Coast","Niger Inland Delta","Upper Egypt"]}
    regionDict["Europe"] = ["Iceland","Paris Basin","Latium"]
    regionDict["Central Eurasia"] = ["Lena River Valley","Orkhon Valley","Sogdiana"]
    regionDict["Southwest Asia"] = ["Yemeni Coastal Plain","Konya Plain","Susiana"]
    regionDict["South Asia"] = ["Garo Hills","Deccan","Kachi Plain"]
    regionDict["Southeast Asia"] = ["Kapuasi Basin","Central Java","Cambodian Basin"]
    regionDict["East Asia"] = ["Southern China Hills","Kansai","Middle Yellow River Valley"]
    regionDict["North America"] = ["Finger Lakes","Cahokia","Valley of Oaxaca"]
    regionDict["South America"] = ["Lowland Andes","North Colombia","Cuzco"]
    regionDict["Oceania-Australia"] = ["Oro PNG","Chuuk Islands","Big Island Hawaii"]

    worldRegions = list(regionDict.keys()) # List of world regions

    # Define some plotting parameters
    t_min = -10000
    t_max = 2000
    pc1_min = -7
    pc1_max = 7
    pc2_min = -7
    pc2_max = 7

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    # Do the singular value decomposition
    # Subset only the 9 CCs and convert to a numpy array 
    CC_names = ['PolPop', 'PolTerr', 'CapPop', 'levels', 'government','infrastr', 'writing', 'texts', 'money']
    CC_array = CC_df.loc[:, CC_names].values

    # Normalize the data (across 20 imputations, not within each imputation)
    CC_scaled = StandardScaler().fit_transform(CC_array)
    CC_times = CC_df.loc[:, ['Time']].values

    # Do a singular value decomposition
    P, D, Q = tailored_svd(CC_scaled)

    # For each polity, project onto the principle components
    # PC_matrix is 8280 x 9 = (414*20) x 9
    PC_matrix = np.matmul(CC_scaled, Q.T)

    NGAs = CC_df.NGA.unique().tolist() # list of unique NGAs from the dataset

    # Create the data for the flow analysis. The inputs for this data creation are
    # the complexity characteristic dataframe, CC_df [8280 x 13], and the matrix of
    # principal component projections, PC_matrix [8280 x 9]. Each row is an imputed
    # observation for 8280 / 20 = 414 unique polity configurations. CC_df provides
    # key information for each observation, such as NGA and Time.
    #
    # Four arrays are created: movArrayOut, velArrayIn, movArrayIn, and velArrayIn.
    # All four arrays have the dimensions 414 x 9 x 2. mov stands for movements and
    # vel for velocity. 414 is the numbers of observations, 9 is the number of PCs,
    # and the final axis has two elements: (a) the PC value and (b) the change in
    # the PC value going to the next point in the NGA's time sequence (or, for vel,
    # the change divided by the time difference). The "Out" arrays give the
    # movement (or velocity) away from a point and the "In" arrays give the
    # movement (or velocity) towards a point. The difference is set to NA for the
    # last point in each "Out" sequence and the first point in each "In" sequence.
    # In addition, NGA name and time are stored in the dataframe flowInfo (the needed
    # "supporting" info for each  observation).

    # Generate the "Out" datasets
    movArrayOut = np.empty(shape=(0,9,2)) # Initialize the movement array "Out" 
    velArrayOut = np.empty(shape=(0,9,3)) # Initialize the velocity array "Out" [location, movement / duration, duration]
    flowInfo = pd.DataFrame(columns=['NGA','Time']) # Initialize the info dataframe

    # Iterate over NGAs to populate movArrayOut, velArrayOut, and flowInfo
    for nga in NGAs:
        indNga = CC_df["NGA"] == nga # boolean vector for slicing by NGA
        times = sorted(np.unique(CC_df.loc[indNga,'Time'])) # Vector of unique times
        for i_t,t in enumerate(times):
            ind = indNga & (CC_df['Time']==t) # boolean vector for slicing also by time
            newInfoRow = pd.DataFrame(data={'NGA': [nga], 'Time': [t]})
            flowInfo = flowInfo.append(newInfoRow,ignore_index=True)
            newArrayEntryMov = np.empty(shape=(1,9,2))
            newArrayEntryVel = np.empty(shape=(1,9,3))
            for p in range(movArrayOut.shape[1]):
                newArrayEntryMov[0,p,0] = np.mean(PC_matrix[ind,p]) # Average across imputations
                newArrayEntryVel[0,p,0] = np.mean(PC_matrix[ind,p]) # Average across imputations
                if i_t < len(times) - 1:
                    nextTime = times[i_t + 1]
                    nextInd = indNga & (CC_df['Time']==nextTime) # boolean vector for slicing also by time
                    nextVal = np.mean(PC_matrix[nextInd,p])
                    newArrayEntryMov[0,p,1] = nextVal - newArrayEntryMov[0,p,0]
                    newArrayEntryVel[0,p,1] = newArrayEntryMov[0,p,1]/(nextTime-t)
                    newArrayEntryVel[0,p,2] = (nextTime-t)
                else:
                    newArrayEntryMov[0,p,1] = np.nan
                    newArrayEntryVel[0,p,1] = np.nan
                    newArrayEntryVel[0,p,2] = np.nan
            movArrayOut = np.append(movArrayOut,newArrayEntryMov,axis=0)
            velArrayOut = np.append(velArrayOut,newArrayEntryVel,axis=0)

    # Modify movement and velocity arrays to be for movements in rather than movements out
    movArrayIn = np.copy(movArrayOut)
    velArrayIn = np.copy(velArrayOut)
    movArrayIn[:,:,1] = np.nan
    velArrayIn[:,:,1] = np.nan

    ind = np.where([True if np.isnan(movArrayOut[i,0,1]) else False for i in range(movArrayOut.shape[0])])[0]
    loVect = np.insert(ind[0:(-1)],0,0)
    hiVect = ind - 1
    for lo,hi in zip(loVect,hiVect):
        for k in range(lo,hi+1):
            movArrayIn[k+1,:,1] = movArrayOut[k,:,1]
            velArrayIn[k+1,:,1] = velArrayOut[k,:,1]
            velArrayIn[k+1,:,2] = velArrayOut[k,:,2]



    # Next, create interpolated arrays by iterating over NGAs
    movArrayOutInterp = np.empty(shape=(0,9,2)) # Initialize the flow array 
    flowInfoInterp = pd.DataFrame(columns=['NGA','Time']) # Initialize the info dataframe
    interpTimes = np.arange(-9600,1901,100)
    for nga in NGAs:
        indNga = CC_df["NGA"] == nga # boolean vector for slicing by NGA
        times = sorted(np.unique(CC_df.loc[indNga,'Time'])) # Vector of unique times
        for i_t,t in enumerate(interpTimes):
            if t >= min(times) and t <= max(times): # Is the time in the NGAs range?
                newInfoRow = pd.DataFrame(data={'NGA': [nga], 'Time': [t]})
                flowInfoInterp = flowInfoInterp.append(newInfoRow,ignore_index=True)
                newArrayEntry = np.empty(shape=(1,9,2))
                for p in range(movArrayOutInterp.shape[1]):
                    # Interpolate using flowArray
                    indFlow = flowInfo['NGA'] == nga
                    tForInterp = np.array(flowInfo['Time'][indFlow],dtype='float64')
                    pcForInterp = movArrayOut[indFlow,p,0]
                    currVal = np.interp(t,tForInterp,pcForInterp)
                    newArrayEntry[0,p,0] = currVal
                    if i_t < len(interpTimes) - 1:
                        nextTime = interpTimes[i_t + 1]
                        nextVal = np.interp(nextTime,tForInterp,pcForInterp)
                        newArrayEntry[0,p,1] = nextVal - currVal
                    else:
                        newArrayEntry[0,p,1] = np.nan
                movArrayOutInterp = np.append(movArrayOutInterp,newArrayEntry,axis=0)

    r0 = 1.5
    minPoints = 20
    dGrid = .2
    u0Vect,v0Vect = createGridForPC12(dGrid,velArrayOut)
    velScaling = 100
    return worldRegions,NGAs,PC_matrix,CC_df,CC_times, CC_scaled, PC1_df,regionDict,t_min,t_max,pc1_min,pc1_max,flowInfo,movArrayOut,velArrayOut,movArrayIn,velArrayIn,flowInfoInterp,movArrayOutInterp,r0,minPoints,dGrid,u0Vect,v0Vect,velScaling
