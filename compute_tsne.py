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

cluster_list=np.loadtxt('cluster_list_k50.txt')

n_points_10perc=0
for i in cluster_list:
    #print(i)
    i=np.int(i)
    file_data=np.loadtxt('./data/clusters/k50_10perc/cluster'+str(i)+'.txt')
    n_points_10perc+=len(file_data)

data_10perc=np.zeros((data.shape[0],n_points_10perc))
k=0
for i in cluster_list:
    i=np.int(i)
    file_data=np.loadtxt('./data/clusters/k50_10perc/cluster'+str(i)+'.txt')
    for j in file_data:
        data_10perc[:,k]=data[:,np.int(j)]
        k=k+1
print(data_10perc.shape)

n_points_289each=0
for i in cluster_list:
    i=np.int(i)
    file_data=np.loadtxt('./data/clusters/k50_289each/cluster'+str(i)+'.txt')
    n_points_289each+=len(file_data)

data_289each=np.zeros((data.shape[0],n_points_289each))
k=0
for i in cluster_list:
    i=np.int(i)
    file_data=np.loadtxt('./data/clusters/k50_289each/cluster'+str(i)+'.txt')
    for j in file_data:
        data_289each[:,k]=data[:,np.int(j)]
        k=k+1
print(data_289each.shape)

X_embbeded_10perc=TSNE(n_components=2).fit_transform(np.transpose(data_10perc))
mat = np.matrix(X_embbeded_10perc)
with open('x_embbeded_10perc.txt','wb') as f:
    for line in mat:
        np.savetxt(f, line, fmt='%.2f')

X_embbeded_289each=TSNE(n_components=2).fit_transform(np.transpose(data_289each))
mat = np.matrix(X_embbeded_289each)
with open('x_embbeded_289each.txt','wb') as f:
    for line in mat:
        np.savetxt(f, line, fmt='%.2f')
