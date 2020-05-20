import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imshow
from PIL import Image
from sklearn.cluster import KMeans


def kmeans_fit(X,n_clusters):
    '''
    Fits a KMeans model with the provided data and number of clusters.
    input:
    X: data to be clustered
    n_clusters: number of clusters to use
    output:
    array of cluster labels
    inertia of the clusters
    '''
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)
    return kmeans.labels_,kmeans.inertia_

def add_labels_to_df(labels,filepath):
    '''
    Adds the cluster labels to a dataframe consisting of other metadata
    about the items.
    imput:
    labels: kmeans cluster labels provided by the kmeans_fit function
    filepath: filepath to the dataframe of metadata where the kmeans 
    label will be added
    output:
    None (resaves the dataframe with the kmeans label)
    '''
    labels = pd.DataFrame(labels)
    df = pd.read_csv(filepath)
    labels.columns = ['kmeans_label']
    merged = pd.concat([df,labels],axis=1,join_axes=[df.index])
    merged.to_csv(filepath,index=False)

def show_cluster(n_clusters,filepath):
    '''
    Saves a plot per cluster with example images from the cluster.
    input:
    n_clusters: number of clusters to plot
    filepath: filbath to dataframe with image clusters
    output: None (saves plot images)
    '''
    for cluster in range(n_clusters):
        fig,ax = plt.subplots(4,4,figsize=(10,10))
        fig.suptitle('Cluster {}'.format(cluster+1),fontsize=28)
        df = pd.read_csv(filepath)
        wines = df['name'].loc[df['kmeans_label']==cluster]
        wines = wines.tolist()[::-4]
        for i,ax in enumerate(ax.flatten()):
            image = Image.open('../images/{}.jpg'.format(wines[i]))
            ax.set_xticks([]) 
            ax.set_yticks([]) 
            ax.grid()
            ax.imshow(image)
        plt.savefig('../figures/cnnv2_cluster{}.jpg'.format(cluster+1))

def elbow_plot(X,num_k):
    '''
    Plots the error for a range of cluster numbers Each cluster number 
    calls the kmeans_fit method to calculate the error depicted 
    in the elbow plot.
    input:
    X: data to be clustered
    num_k: max number of clusters, 1-num_k will be utlized in the elbo plot
    
    output:
    elbow plot
    '''
    fig, ax = plt.subplots(figsize=(8,8))

    rss_arr = []
    for i in range(1,num_k):
        _,rss = kmeans_fit(X,i)
        rss_arr.append(rss)
    ax.plot(range(1,num_k),rss_arr)
    ax.set_xlabel('k Number of Clusters')
    ax.set_ylabel('Error')
    plt.show()



if __name__ == '__main__':
    X = np.load('../data/64x64/image_array_cnn.npy')
    # fname = np.load('../data/50x50/file_array_2d.npy')
    # X = X[:1000,:]
    # fname = fname[:1000]
    n_clusters = 10
    labels, rss = kmeans_fit(X,n_clusters)
    
    filepath = '../data/64x64/sorted_df.csv'
    add_labels_to_df(labels,filepath)
    # show_cluster(fname,labels,n_clusters)
    # elbow_plot(X,25)