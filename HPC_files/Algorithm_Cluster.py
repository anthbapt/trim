import os
import scipy as sp
import numpy as np
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})
from sklearn.feature_selection import mutual_info_regression
from computation import create_node_edge_incidence_matrix
import networkx as nx


#set up loop for cluster
__TASK_ID__ = os.environ.get('SGE_TASK_ID')

if __name__ == '__main__':
    if __TASK_ID__:
        i = int(__TASK_ID__)

#import network and incidence matrix
G=nx.read_gml("network_100_i",destringizer=int)
N=G.number_of_nodes()
E=G.number_of_edges()
Elist=nx.edges(G)
Elist=np.array(Elist)
Elist = Elist + 1
Elist = Elist.tolist()
K=np.load("Kmat_100_i.npy")


# Node
n_nodes = N
# Edge list
edge_list = Elist
n_edges = len(edge_list)
# Incidence matrix for the structural network
B = create_node_edge_incidence_matrix(edge_list)
# Incidence matrix for the triadic interactions

# tlen length of the times series to be analysed (multiple of num) should be smaller or equal
# than the length of the timeseries from data
# function that calculates MI Sigma MIz and MIC (conditional MI) 
# from  mutual information calculated for continuous variables
def mutual_information_analysis_continuous(timeseries,I,num,tlen):
    timeseries = timeseries[:,-tlen:]
    dtlen=int(np.floor(tlen/num))
    X = np.asarray(timeseries[I[0],:])
    Y = np.asarray(timeseries[I[1],:])
    Z = np.asarray(timeseries[I[2],:])
    Xa = np.zeros((tlen,2))
    Xa[:,0] = X
    MI = mutual_info_regression(Xa, Y, discrete_features=False)
    MI = MI[0]
    idx = Z.argsort();
    X_sort = X[idx];
    Y_sort = Y[idx];
    Xn = np.zeros((dtlen,2))
    MIz = np.zeros((num))
    for i in range(num):
        Xn[:,0] = X_sort[i*dtlen:(i+1)*dtlen]; 
        y = Y_sort[i*dtlen:(i+1)*dtlen]; 
        mi = mutual_info_regression(Xn, y, discrete_features=False)
        MIz[i] = mi[0]
    MIC = np.mean(MIz)
    Sigma = np.std(MIz)
    T=np.max(MIz)-np.min(MIz)
    Tn=np.max(abs(MIz[0:num-1])-MIz[1:num])

    return MI, MIz,MIC,Sigma, T,Tn


# tlen length of the times series to be analysed (multiple of num) should be smaller or equal
# than the length of the timeseries from data
# num number of bins
# function that calculates MI Sigma MIz and MIC (conditional MI) 
# from  mutual information calculated for continuous variables
def correlation_analysis_continuous(timeseries,I,num,tlen):
    timeseries = timeseries[:,-tlen:]
    dtlen = int(np.floor(tlen/num))
    X = np.asarray(timeseries[I[0],:])
    Y = np.asarray(timeseries[I[1],:])
    Z = np.asarray(timeseries[I[2],:])
    idx = Z.argsort();
    X_sort = X[idx];
    Y_sort = Y[idx];
    Cz = np.zeros((num))
    Xaus=np.zeros((dtlen,2))
    for i in range(num):
        Xaus[:,0] = X_sort[i*dtlen:(i+1)*dtlen]; 
        Xaus[:,1] = Y_sort[i*dtlen:(i+1)*dtlen]; 
        C=np.cov(Xaus)
        Cz[i]=C[0,1]
    C = np.mean(Cz)
    Sigma = np.var(Cz)
    T=np.max(Cz)-np.min(Cz)
    Tn=np.max(abs(Cz[0:num-1])-Cz[1:num])
    return C, Cz,Sigma, T, Tn

#function that calculates the  Theta variable (normalized z-score)
def null_model_results(M, Cov, timeseries,I, num, Sigma, T, Tn, nrunmax, Gaussian_version, Mutual_version):
    I2 = I
    if (Gaussian_version==True):
        MT = M[I]
        CovT = Cov[I]
        CovT = CovT[:,I]
        mult_dist = sp.stats.multivariate_normal(mean = MT, cov = CovT)
        I2 = [0,1,2]
      
    Sigma_null_list = []
    T_null_list = []
    Tn_null_list = []
    
    for n in range(nrunmax):
        if(Gaussian_version==False): 
            null_timeseries = np.array(timeseries).copy()
            np.random.shuffle(null_timeseries[I2[2], :])
        elif(Gaussian_version==True):    
            null_timeseries = mult_dist.rvs(tlen)
            null_timeseries = np.transpose(null_timeseries)
            
        if(Mutual_version==False):
            X_null, Xz_null, Sigma_null, T_null, Tn_null = correlation_analysis_continuous(null_timeseries,I2,num,tlen)
        elif(Mutual_version==True):
            X_null, Xz_null, MIC_null, Sigma_null, T_null, Tn_null = mutual_information_analysis_continuous(null_timeseries,I2,num,tlen)
        Sigma_null_list.append(Sigma_null)
        T_null_list.append(T_null)
        Tn_null_list.append(Tn_null)
    Sigma_null_list = np.array(Sigma_null_list)
    T_null_list=np.array(T_null_list)
    Tn_null_list=np.array(Tn_null_list)
    Sigma_mean_null = np.mean(Sigma_null_list)
    T_mean_null=np.mean(T_null_list)
    Tn_mean_null=np.mean(Tn_null_list)
    std_null = np.std(Sigma_null_list)
    T_std_null=np.std(T_null_list)
    Tn_std_null=np.std(Tn_null_list)
    Theta = abs(Sigma-Sigma_mean_null)/(std_null)
    Theta_T=abs(T-T_mean_null)/T_std_null
    Theta_Tn=abs(Tn-Tn_mean_null)/Tn_std_null
    P = np.count_nonzero(Sigma_null_list[Sigma_null_list>Sigma])/nrunmax
    P = P if P>0 else 1/nrunmax
    P_T = np.count_nonzero(T_null_list[T_null_list>T])/nrunmax
    P_T = P_T if P_T>0 else 1/nrunmax
    P_Tn = np.count_nonzero(Tn_null_list[Tn_null_list>Tn])/nrunmax
    P_Tn = P_Tn if P_Tn>0 else 1/nrunmax
    return  X_null, Xz_null, Theta, Theta_T, Theta_Tn, Sigma,Sigma_null_list, P,P_T,P_Tn


# main function that given a timeseries between N variables, selects the three time series in the triple of nodes
# I (in the original labelling of the edge list) and calculates the MI, the
# the conditional mutual information MIC, Sigma and Theta (with a null model taking nrunmax iterations)
# tlen length of the times series to be analysed (multiple of num) should be smaller or equal
# than the length of the timeseries from data
# num number of bins
# version 1: gaussian model from covariance and mean of the three timeseries
# version 2: reshuffling of the Z timeseries
def Theta_score_null_model(timeseries, I, num, tlen, nrunmax, Gaussian_version=True, Mutual_version=True):
    I = np.array(I)-1
    timeseries = timeseries[:, -tlen:]
    Cov = np.cov(timeseries)
    M = np.mean(timeseries, axis=1)
    MIC = -1

    if (Mutual_version == True):
        X, Xz, MIC, Sigma, T, Tn = mutual_information_analysis_continuous(timeseries, I, num, tlen)
    elif (Mutual_version == False):
        X, Xz, Sigma, T, Tn = correlation_analysis_continuous(timeseries, I, num, tlen)
    X_null, Xz_null, Theta, Theta_T, Theta_Tn, Sigma, Sigma_null_list, P,P_T,P_Tn = null_model_results(M, Cov, \
                                        timeseries, I, num, Sigma, T, Tn, nrunmax, Gaussian_version, Mutual_version)

    return X, Xz, Xz_null, MIC, Theta, Theta_T, Theta_Tn, Sigma, Sigma_null_list, P, P_T, P_Tn


ts_list = []
P_list1 = []
num_list = []

timeseries = np.load("timeseries_100_i.npy")
timeseries = timeseries[:,::3]
timeseries = timeseries[:,-400000:]

#get list of edges for analysis: random edges without triadic interaction and edges with positive and negative triadic interactions

x1 = [Elist[j] for j in range(len(Elist))]
tlen = 11000

#do the analysis for all triples from the choice of edges and all other nodes

j = 0

for l in x1:
    j=j+1
    if 0 ==0:    
        I = [l[0],l[1],i] # triple without triadic interaction,but with a link between X and Y, 10 bins
        if I[0]!=I[2] and I[1]!=I[2]:
            tlen, num, nrunmax = 7000, round(tlen/100), 10
            #tlen used to be 7000
        
            X, Xz2, Xz_null2, MIC, Theta2, Theta2_T, Theta2_Tn, Sigma, Sigma_null_list, P, P_T, P_Tn = Theta_score_null_model(timeseries, I, num, tlen, nrunmax, False, True)
            if P == 0.1:
                nrunmax=100
                X, Xz2, Xz_null2, MIC, Theta2, Theta2_T, Theta2_Tn, Sigma, Sigma_null_list, P, P_T, P_Tn = Theta_score_null_model(timeseries, I, num, tlen, nrunmax, False, True)
                if P == 0.01:
                    nrunmax=1000
                    X, Xz2, Xz_null2, MIC, Theta2, Theta2_T, Theta2_Tn, Sigma, Sigma_null_list, P, P_T, P_Tn = Theta_score_null_model(timeseries, I, num, tlen, nrunmax, False, True)
            str2='output_ran_'+str(l[0])+'-'+str(l[1])+'_'+str(i)+'.txt'
            with open(str2, "a") as f:
                print(l,i,Theta2, Theta2_T, Theta2_Tn, P, P_T, P_Tn,X,MIC,Sigma,end='\n', file=f)