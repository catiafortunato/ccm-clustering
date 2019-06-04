import h5py
import numpy as np
import scipy.io as sio

def read_reference_file(str_file):
    mat_contents=sio.loadmat(str_file)
    anat_stack=mat_contents['anat_stack_norm']
    anat_yx=mat_contents['anat_yx_norm']
    anat_yz=mat_contents['anat_yz_norm']
    anat_zx=mat_contents['anat_zx_norm']

    return anat_stack, anat_yx, anat_yz, anat_zx

def rand_color(num_clusters):
    colorlist=np.zeros((num_clusters,3))
    for i in range(num_clusters):
        color=list(np.random.choice(range(1),size=3))
        while color in colorlist:
            color= list(np.random.choice(range(1),size=3))
        colorlist[i,:]=color
    return colorlist

def plot_cluster_in_anat(anat_yx, anat_yz, anat_zx, position, neurons, color):
    for i in neurons:
        [x,y,z]=position[i]
        anat_yx[y-2,x-2,:]=color
        anat_yz[y-2,z-2,:]=color
        anat_zx[138-z,x-2,:]=color
    return anat_yx, anat_yz, anat_zx