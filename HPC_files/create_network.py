import numpy as np
import random
import seaborn as sns
from model import NDwTIs
from computation import create_node_edge_incidence_matrix
import networkx as nx

sns.set_style("whitegrid", {'axes.grid' : False})

# randomly generate the network, here BA graph
#G = nx.barabasi_albert_graph(100,1.1)
#G = nx.gnp_random_graph(100,0.022)
G = nx.gnm_random_graph(100,300)
#G = nx.read_gml("network_100_f",destringizer=int)
N=G.number_of_nodes()
E=G.number_of_edges()
Elist=nx.edges(G)
Elist=np.array(Elist)
Elist = Elist + 1

# generate the nodes and edges involved in the triadic interaction

# save the network
nx.write_gml(G,"network_100_i")
# create the incidence matrix K for the regulatory graph

n_nodes = N
edge_list = Elist
n_edges = len(edge_list)
# Incidence matrix for the structural network
B = create_node_edge_incidence_matrix(edge_list)

E=G.number_of_edges()
N=G.number_of_nodes()
#c1=round(N/(2*8))
#c2=round(N/(2*8))
#u = np.random.uniform(0, 1, (E,N))
#K = np.zeros((E,N))
#for i in range(E):
#    for j in range(N):
#        K[i,j] = 1*np.heaviside(c1/(N*E)-u[i,j],1)+(-1)*np.heaviside((c1+c2)/(N*E)-u[i,j],1)*np.heaviside(u[i,j]-c2/(N*E),1)
Elist=nx.edges(G)
Elist=np.array(Elist)
Elist = Elist + 1
T= round(N/4) #number of triadic nodes
K = np.zeros((E,N))
R1=np.zeros(T,int)
R2=np.zeros(T,int)
D1=[]
# generate the nodes and edges involved in the triadic interaction
for i in range(T):
    W1 = random.randrange(E)
    while W1 in D1:
        W1 = random.randrange(E)
    D1.append(W1)
    R2[i]= random.randrange(N)
    while R2[i]+1==Elist[D1[i]][0] or R2[i]+1==Elist[D1[i]][1]:
        R2[i]= random.randrange(N)
    print(Elist[D1[i]],R2[i])
Elist = Elist.tolist()
# save the network
nx.write_gml(G,"network_100_i")
# create the incidence matrix K for the regulatory graph
K = np.zeros((E,N))
for i in range(T):
    R3=random.randrange(2)
    R3=2*R3-1
    K[D1[i],R2[i]]=R3
np.save("Kmat_100_i.npy",K)
# calculate timeseries, split the modelling into 3 parts to reduce computational time
    

model = NDwTIs(
    B=B, K=K, w_pos=8, w_neg=0.5,
    threshold=1e-3, alpha=0.001, noise_std=1e-2,
    x_init=np.zeros(n_nodes), dt=1e-2, t_max=300.)

timeseries1= model.run()

timeseries1 = np.array(timeseries1)
#timeseries1 = timeseries1[:,-200000::5]

np.save("timeseries_100_i-1",timeseries1)

model = NDwTIs(
    B=B, K=K, w_pos=8, w_neg=0.5,
    threshold=1e-3, alpha=0.001, noise_std=1e-2,
    x_init=timeseries1[:,-1], dt=1e-2, t_max=300.)

print(timeseries1[:,-1])
  
timeseries2 = model.run()

timeseries2 = np.array(timeseries2)
#timeseries2 = timeseries2[:,-200000::5]

np.save("timeseries_100_i-2",timeseries2)

model = NDwTIs(
    B=B, K=K, w_pos=8, w_neg=0.5,
    threshold=1e-3, alpha=0.001, noise_std=1e-2,
    x_init=timeseries2[:,-1], dt=1e-2, t_max=300.)
print(timeseries2[:,-1])
timeseries3 = model.run()

timeseries3 = np.array(timeseries3)
#timeseries3 = timeseries3[:,-200000::5]

np.save("timeseries_100_i-3",timeseries3)

model = NDwTIs(
    B=B, K=K, w_pos=8, w_neg=0.5,
    threshold=1e-3, alpha=0.001, noise_std=1e-2,
    x_init=timeseries3[:,-1], dt=1e-2, t_max=300.)
print(timeseries2[:,-1])
timeseries4 = model.run()

timeseries4 = np.array(timeseries4)
#timeseries4 = timeseries3[:,-200000::5]

np.save("timeseries_100_i-4",timeseries4)

model = NDwTIs(
    B=B, K=K, w_pos=8, w_neg=0.5,
    threshold=1e-3, alpha=0.001, noise_std=1e-2,
    x_init=timeseries4[:,-1], dt=1e-2, t_max=300.)
print(timeseries2[:,-1])
timeseries5 = model.run()

timeseries5 = np.array(timeseries5)
#timeseries4 = timeseries3[:,-200000::5]

np.save("timeseries_100_i-5",timeseries5)

model = NDwTIs(
    B=B, K=K, w_pos=8, w_neg=0.5,
    threshold=1e-3, alpha=0.001, noise_std=1e-2,
    x_init=timeseries5[:,-1], dt=1e-2, t_max=300.)
print(timeseries2[:,-1])
timeseries6 = model.run()

timeseries6 = np.array(timeseries6)
#timeseries4 = timeseries3[:,-200000::5]

np.save("timeseries_100_i-6",timeseries6)


# concatenate timeseries and then save it
timeseries = np.concatenate((timeseries1, timeseries2,timeseries3,timeseries4,timeseries5,timeseries6),axis=1)
timeseries = timeseries[:,::5]
np.save("timeseries_100_i",timeseries)