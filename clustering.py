import matplotlib
matplotlib.use('Agg')
import h5py
import matplotlib.pyplot as plt
import scipy.io
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
import scipy.io as sio
import methods
from pytictoc import TicToc
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
import seaborn as sns
import cv2
import matplotlib.patches as patches
import random
data=np.loadtxt('spontaneous_activity_subject8.txt')

print(data.shape)

cluster_list=np.loadtxt('cluster_list_k50.txt')

neuron_list=np.zeros((len(cluster_list),20))
k=0
for i in cluster_list:
    i=np.int(i)
    file_data=np.loadtxt('./data/clusters/k50/cluster'+str(i)+'.txt')
    values=random.sample(file_data,20)
    neuron_list[k,:]=values
    k=k+1


T=data.shape[0]
tau=1
E=3
L=range(10,100,10)
emsemble=50


def example(X,Y,T,tau,E,L,emsemble):
    rhox=np.zeros((len(L),emsemble))
    rhoy=np.zeros((len(L),emsemble))
    for i in range(len(L)):
        for j in range (emsemble):

            y_pred, y_target, x_pred, x_target = methods.compute_xmap(X,Y,T,E,tau,L[i])
            rhox[i,j]=methods.compute_corr(y_pred,y_target)
            rhoy[i,j]=methods.compute_corr(x_pred,x_target)
    rhox=np.mean(rhox,1)
    rhoy=np.mean(rhoy,1)
    return rhox,rhoy,L

t=TicToc()
t.tic()
for i in range(len(neuron_list)):
    for j in range(i,len(neuron_list)):
        Rx=np.zeros(len(L))
        Ry=np.zeros(len(L))
        for k in range(len(neuron_list[0])):
            x=np.transpose(data[:,np.int(neuron_list[i,k])])
            y=np.transpose(data[:,np.int(neuron_list[j,k])])
            rhox, rhoy, L=example(x,y,T,tau,E,L,emsemble)
            Rx=Rx+rhox
            Ry=Ry+rhoy
        Rx=Rx/np.float(len(neuron_list[0]))
        Ry=Ry/np.float(len(neuron_list[0]))
        fig2=plt.figure()
        ax2=fig2.add_subplot(111)
        ax2.plot(L,Rx)
        ax2.plot(L,Ry)
        plt.ylim(-.5, 1)
        plt.title('Between cluster'+str(cluster_list[i])+'and cluster'+str(cluster_list[j]))
        plt.legend(('Y|Mx','X|My'))
        plt.savefig('/home/catia/Desktop/ccm-clustering/results/ccm_neurons/cluster'+str(np.int(cluster_list[i]))+'cluster'+str(np.int(cluster_list[j]))+'.png')

t.toc()
