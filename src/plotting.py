import numpy as np
import pandas as pd
import glob
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams.update({'font.size':20})

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# from data_pipelines import ImagePipeline

def metadata_heatmap():
    pass

def sat_histogram():
    '''
    Make 
    '''
    pass

def tt_holdout_error_curves(tb_csv_dir):
    '''
    Test, train and holdout error curves.
    
    '''
    test = pd.read_csv('{}/test_friday_am.csv'.format(tb_csv_dir))
    val = pd.read_csv('{}/validation_friday_am.csv'.format(tb_csv_dir))

    x_vals = test['Test']
    
    fig, axes = plt.subplots(1,2,figsize=(24,10))
    axes[0].plot(x_vals,test['avg'], label='Train')
    axes[0].plot(x_vals,val['avg'], label='Test')
    axes[0].axvline(4,color='k', label='Chosen Epoch')
    axes[0].set_title('RGB CNN Error vs Epoch')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('RMSE')
    axes[0].legend()
    axes[0].set_ylim([0.03,0.08])

    axes[1].plot(x_vals,test['gavg'], label='Train')
    axes[1].plot(x_vals,val['gavg'], label='Test')
    axes[1].axvline(2, color='k', label='Chosen Epoch')
    axes[1].set_title('Gray CNN Error vs Epoch')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('RMSE')
    axes[1].legend()
    axes[1].set_ylim([0.03,0.08])

    plt.savefig('../images/tt_error.png',dpi=200)
    plt.close('all')
    return val

def pca_var(exp_variance_ratio):
    x_vals = range(len(exp_variance_ratio))
    y_vals = np.cumsum(exp_variance_ratio)

    fig, ax = plt.subplots(1, figsize=(10,10))
    ax.plot(x_vals, y_vals, label='PCA')
    ax.axhline(0.90, color='k', label='0.90 Goal')
    ax.set_ylabel('Cumulative Variance Ratio')
    ax.set_xlabel('Number of Components')
    ax.set_title('PCA Explained Variance vs Number of Components')
    ax.legend()

    plt.savefig('../images/pca_var.png', dpi=200)
    plt.close('all')

def pca_var_all(rgb_ratio_list, gray_ratio_list):

    x_vals = range(len(rgb_ratio_list))

    rgb_vals = np.cumsum(rgb_ratio_list)
    gray_vals = np.cumsum(gray_ratio_list)

    fig, ax = plt.subplots(1, figsize=(10,10))
    ax.plot(x_vals, rgb_vals, label='Color Image')
    ax.plot(x_vals, gray_vals, label='Gray Image')
    ax.axhline(0.90, color='k', label='0.90 Goal')
    ax.set_ylim([0,1])
    ax.set_ylabel('Cumulative Variance Ratio')
    ax.set_xlabel('Number of Components')
    ax.set_title('PCA Explained Variance\nRGB and Gray')
    ax.legend()

    plt.savefig('../images/pca_var_all.png', dpi=200)
    plt.close('all')

def elbow_plot(latents, max_k,title):
    '''
    Plot elbow plot
    '''
    fig, ax = plt.subplots(1, figsize=(10,10))

    rss_lst = []
    for k in range(1, max_k):
        kmeans = KMeans(k, init=latents[:k,:],random_state=33)
        kmeans.fit(latents)
        rss_lst.append(kmeans.inertia_)
    
    ax.plot(range(1, max_k), rss_lst)
    ax.set_ylabel('RSS')
    ax.set_xlabel('Number of Clusters')
    ax.set_title('Elbow Plot for PCA')

    plt.savefig('../images/pca_kmeans_elbow_{}.png'.format(title), dpi=200)
    plt.close('all')

def silhouette(latents, num_clusters):
    pass

def six_hist(sil_values, labels, color):
    fig, axes = plt.subplots(3,2, figsize=(10,10))
    cluster_vals = np.unique(labels)

    for ax, cluster in zip(axes.flatten(), cluster_vals):
        ax.hist(sil_values[np.where(labels == cluster)], bins=50)
        # ax.set_title('Cluster {}'.format(cluster))
        ax.set_xlabel('Silhouette Score')
        ax.set_ylabel('Count')
    
    plt.savefig('../images/six_sil_pca_{}.png'.format(color), dpi=200)
    plt.close('all')

def plot_before_after(test,test_decoded,n=10):
    '''
    Plots the image and reconstructed image.
    Input:
    test: test dataset of image arrays
    test_decoded: reconstructed test dataset image arrays (predict results)
    Output: None (saves figure to a file)
    '''
    plt.figure(figsize=(n*2, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(test[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(test_decoded[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.savefig('../images/before_after.png',dpi=100)
    plt.close('all')

if __name__ == "__main__":
    path = '/home/conslag/Documents/galvanize/capstones/homes-like-this/data/models'
    df = tt_holdout_error_curves(path)
    