import re
import pandas
import matplotlib.pyplot as plt
import re
import pandas as pd
import numpy as np

def edge(s):
    """
    Turn txt output file into list

    Args:
        s (txt file): output file

    Returns:
        list: Output values in list format
    """
    f = open(s,'r')
    e = f.read()
    for i in range(1,10):
        e = e.replace('\n',' ')
    e1 = re.split(r' ', e)
    e1.pop()
    return e1


def MI_MIC_merge(data1, data2, data3 = None, key = 'Sigma', display = None, label = None):
    
    
    data = pd.concat([data1, data2, data3]).reset_index(drop=True)
    plt.figure(figsize=(9,7))
    if key in {'P', 'P_T', 'P_Tn'}:
        if display == 3:
            x1 = data1['CMI'].astype(float)
            x2 = data2['CMI'].astype(float)
            y1 = data1['MI'].astype(float)
            y2 = data2['MI'].astype(float)
            z1 = np.log(1/data1[key].astype(float))
            z2 = np.log(1/data2[key].astype(float))
            zs = np.concatenate([z1, z2], axis=0)
            min_, max_ = zs.min(), zs.max()
            plt.scatter(x1, y1, c=z1, marker='o', s = 75, alpha=0.8, cmap = 'gist_rainbow')
            plt.clim(min_, max_)
            plt.scatter(x2, y2, c=z3, marker='*', s= 50, edgecolor='black', alpha = 0.8, cmap = 'gist_rainbow')            
            plt.clim(min_, max_)
            plt.xlim(xmin=0)
            plt.ylim(ymin=0)
            plt.xlabel('CMI', fontsize=10, color='black')
            plt.ylabel('MI', fontsize=10,color='black')
            if label != None:
                plt.title(label[0])
            cbar = plt.colorbar()
            cbar.ax.set_xlabel(r'$\mathregular{\ln{(1/i)}}$'.replace('i', key), rotation=0, fontsize=10)
            plt.show()
        else:
            x = data['CMI'].astype(float)
            y = data['MI'].astype(float)
            z = np.log(1/data[key].astype(float))
            plt.scatter(x, y, c=z, s = 5, alpha=0.8, cmap = 'gist_rainbow')
            plt.xlim(xmin=0)
            plt.ylim(ymin=0)
            if display == 1:
                plt.xlim(0, max(data1['CMI'].astype(float)) + 0.1*max(data1['CMI'].astype(float)))
                plt.ylim(0, max(data1['MI'].astype(float)) + 0.1*max(data1['MI'].astype(float)))
            if display == 2:
                plt.xlim(0, max(data2['CMI'].astype(float))+ 0.1*max(data2['CMI'].astype(float)))
                plt.ylim(0, max(data2['MI'].astype(float))+ 0.1*max(data2['MI'].astype(float)))
            
            plt.xlabel('CMI', fontsize=10, color='black')
            plt.ylabel('MI', fontsize=10,color='black')
            if label != None:
                plt.title(label[0])
            cbar = plt.colorbar()
            cbar.ax.set_xlabel(r'$\mathregular{\ln{(1/i)}}$'.replace('i', key), rotation=0, fontsize=10)
            plt.show()
    
    else:
        if display == 3:
            x1 = data1['CMI'].astype(float)
            x2 = data2['CMI'].astype(float)
            x3 = data3['CMI'].astype(float)
            y1 = data1['Theta_Sigma'].astype(float)
            y2 = data2['Theta_Sigma'].astype(float)
            y3 = data3['Theta_Sigma'].astype(float)
            z1 = data1[key].astype(float)  
            z2 = data2[key].astype(float)
            z3 = data3[key].astype(float)
            zs = np.concatenate([z1, z2, z3], axis=0)
            min_, max_ = zs.min(), zs.max()
            plt.scatter(x1, y1, c=z1, marker='s', s = 10, alpha=0.8, cmap = 'copper')
            plt.clim(min_, max_)
            plt.scatter(x2, y2, c=z2, marker='.', s = 75, alpha=0.8, cmap = 'copper')
            plt.clim(min_, max_)
            plt.scatter(x3, y3, c=z3, marker='X', s= 50, edgecolor='black', alpha = 0.8, cmap = 'copper')
            plt.xlim(xmin=0)
            plt.ylim(ymin=1)
            plt.yscale('log')
            plt.gca().axes.set_yticks([1,5,10,15], [1,5,10,15])
            plt.tick_params(axis='both', which='major', labelsize=15)
            plt.xlabel('CMI', fontsize=20, color='black')
            plt.ylabel(r'$\mathregular{\Theta_\Sigma}$', fontsize=20,color='black')
            if label != None:
                plt.title(label[0])
            cbar = plt.colorbar()
            if key == 'Sigma':
                cbar.ax.set_xlabel(r'$\mathregular{\i}$'.replace('i', key), rotation=0, fontsize=20)
            else:
                cbar.ax.set_xlabel(r'$\mathregular{i}$'.replace('i', key), rotation=0, fontsize=20)
            cbar.ax.tick_params(labelsize=15) 
            plt.show()
        if display == 2.5:
            x1 = data1['CMI'].astype(float)
            x2 = data2['CMI'].astype(float)
            x3 = data3['CMI'].astype(float)
            y1 = data1['Theta_Sigma'].astype(float)
            y2 = data2['Theta_Sigma'].astype(float)
            y3 = data3['Theta_Sigma'].astype(float)
            z1 = data1[key].astype(float)  
            z2 = data2[key].astype(float)
            z3 = data3[key].astype(float)
            zs = np.concatenate([z1, z2, z3], axis=0)
            min_, max_ = zs.min(), zs.max()
            plt.scatter(x1, y1, c=z1, marker='.', s = 120, alpha=0.8, cmap = 'copper')
            plt.scatter(x2, y2, c=z2, marker='*', s= 75, edgecolor='black', alpha = 0.8, cmap = 'copper')
            plt.scatter(x3, y3, c=z3, marker='X', s= 75, edgecolor='black', alpha = 0.8, cmap = 'copper')
            plt.clim(min_, max_)
            plt.xlim(xmin=0)
            plt.ylim(ymin=1)
            plt.yscale('log')
            plt.tick_params(axis='both', which='major', labelsize=15)
            plt.gca().axes.set_yticks([1,5,10,15], [1,5,10,15])
            plt.xlabel('CMI', fontsize=20, color='black')
            plt.ylabel(r'$\mathregular{\Theta_\Sigma}$', fontsize=20,color='black')
            if label != None:
                plt.title(label[0])
            cbar = plt.colorbar()
            if key == 'Sigma':
                cbar.ax.set_xlabel(r'$\mathregular{\i}$'.replace('i', key), rotation=0, fontsize=20)
            else:
                cbar.ax.set_xlabel(r'$\mathregular{i}$'.replace('i', key), rotation=0, fontsize=20)
            cbar.ax.tick_params(labelsize=15) 
            plt.show()
            
def edge_processing(names:list):
    edges = []
    for i in names:
        edges.append(edge(i))
    for i in range(len(edges)):
        for j in range(len(edges[i])):
            edges[i][j]=float(edges[i][j])
    edges=np.array(edges)
    edges=edges.reshape(12,8,7)
    return edges


def entropy(timeseries, I):
    """
    calculates entropy for triple

    Args:
        timeseries: timeseries data of network
        I: chosen triple

    Returns:
        S: Entropy
    """
    timeseries[I[0]-1,:] = X
    timeseries[I[1]-1,:] = Y
    timeseries[I[2]-1,:] = Z
    num = 5
    tlen = len(timeseries[0])
    nrunmax = 100
    MI, MIz, MIz_null, MIC, Theta_S, Theta2_T, Theta2_Tn, Sigma, Sigma_null_list, P, P_T, P_Tn = ifc.Theta_score_null_model(timeseries, I, num, tlen, nrunmax, True, True)
    x = range(1, num+1)
    th1, th2, c = decision_tree(x, MIz, disp_fig=False, disp_txt_rep=False,
              disp_tree=False)
    I = np.array(I)-1
    X, Y, Z = ifc.timeseries_quantile(timeseries, I, num, tlen)
    X_a = X[Z < th1]
    Y_a = Y[Z < th1]
    P_sep = 10
    n = np.zeros((10,10))
    X_min, Y_min, X_max, Y_max = np.min(X_a), np.min(Y_a), np.max(X_a),  np.max(Y_a)
    ival_X = (X_max-X_min)/P_sep
    ival_Y = (Y_max-Y_min)/P_sep
    for i in range(len(X_a)):
        x = np.floor((X_a[i]-X_min)/ival_X)
        if x == 10.0: x = 9.0
        y = np.floor((Y_a[i]-Y_min)/ival_Y)
        if y == 10.0: y = 9.0
        n[int(x),int(y)] += 1
    Y_1 = 0
    for i in range(10):
        for j in range(10):
            Y_1 += (n[i,j]/len(X_a))**2        
    X_b = X[np.logical_and(Z <= th2, Z >= th1)]
    Y_b = Y[np.logical_and(Z <= th2, Z >= th1)]
    n = np.zeros((10,10))
    X_min, Y_min, X_max, Y_max = np.min(X_b), np.min(Y_b), np.max(X_b),  np.max(Y_b)
    ival_X = (X_max-X_min)/P_sep
    ival_Y = (Y_max-Y_min)/P_sep
    for i in range(len(X_b)):
        x = np.floor((X_b[i]-X_min)/ival_X)
        if x == 10.0: x = 9.0
        y = np.floor((Y_b[i]-Y_min)/ival_Y)
        if y == 10.0: y = 9.0
        n[int(x),int(y)] += 1
    Y_2 = 0
    for i in range(10):
        for j in range(10):
            Y_2 += (n[i,j]/len(X_b))**2
    X_c = X[Z > th2]
    Y_c = Y[Z > th2]
    n = np.zeros((10,10))
    X_min, Y_min, X_max, Y_max = np.min(X_c), np.min(Y_c), np.max(X_c),  np.max(Y_c)
    ival_X = (X_max-X_min)/P_sep
    ival_Y = (Y_max-Y_min)/P_sep
    for i in range(len(X_c)):
        x = np.floor((X_c[i]-X_min)/ival_X)
        if x == 10.0: x = 9.0
        y = np.floor((Y_c[i]-Y_min)/ival_Y)
        if y == 10.0: y = 9.0
        n[int(x),int(y)] += 1
    Y_3 = 0
    for i in range(10):
        for j in range(10):
            Y_3 += (n[i,j]/len(X_b))**2
    S = - (np.log(Y_1)+np.log(Y_2)+np.log(Y_3))
    return S

def get_distances(csv_results,G):
    values = pd.read_csv(csv_results)
    length = []
    for i in range(len(values['node1'])):
        try:
            p1 = nx.shortest_path_length(G,int(values['node1'][i]-1),int(values['reg'][i]-1))
        except:
            p1 = 20
        try:
            p2 = nx.shortest_path_length(G,int(values['node2'][i]-1),int(values['reg'][i]-1))
        except:
            p2 = 20
        length.append(np.min([p1,p2]))
    return length
    