import numpy as np
import pandas as pd
import glob

from os import listdir
import os
from os.path import isfile, join

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

from bokeh.io import show, output_file, export_png
from bokeh.plotting import figure, output_file, gmap
from bokeh.models import ColumnDataSource, GMapOptions

import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.style.use('ggplot')
plt.rcParams.update({'font.size':20})




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

def silhouette(X, n_clusters, color):
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    clusterer.fit(X)
    cluster_labels = clusterer.labels_
    
    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')
    plt.savefig('../images/sil_{}_{}clusters.png'.format(color, n_clusters))

def cluster_plot_7x3(images, labels, num_clusters, title):
    fig, axes = plt.subplots(3,num_clusters,figsize=(24,12))
    for cluster in range(num_clusters):
        subset_idx = np.where(labels == cluster)
        random_idx = np.random.choice(subset_idx[0],3,replace=False)
        axes[0,cluster].set_title('Cluster {}'.format(cluster+1))
        for idx, rand_idx in enumerate(random_idx):
            axes[idx,cluster].imshow(images[rand_idx])
            axes[idx,cluster].set_axis_off()
    plt.savefig('../images/7x3_cluster_{}.png'.format(title), dpi=200)

def similar_img_plot(images, n, title):

    fig, axes = plt.subplots(n,n,figsize=(12,12))
    for ax, img in zip(axes.flatten(), images):
        ax.imshow(img)
        ax.set_axis_off()

    plt.savefig('../images/{}x{}_cluster_{}.png'.format(n,n,title), dpi=200)


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

def plot_before_after(test,test_decoded,fname,n=10):
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
        plt.imshow(test[2*i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(test_decoded[2*i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.savefig('../images/before_after_{}.png'.format(fname),dpi=100)
    plt.close('all')


def listing_map(lat_lst, long_lst, export_as_png=False):

    output_file("../images/gmap.html")
    map_style = '''[
    {
        "featureType": "road",
        "elementType": "labels",
        "stylers": [
            {
                "visibility": "off"
            }
        ]
    },
    {
        "featureType": "poi",
        "elementType": "labels",
        "stylers": [
            {
                "visibility": "off"
            }
        ]
    },
    {
        "featureType": "transit",
        "elementType": "labels.text",
        "stylers": [
            {
                "visibility": "off"
            }
        ]
    }
    ]
    '''
    map_options = GMapOptions(lat=39.755123, lng=-104.980635, map_type="roadmap", zoom=10, styles=map_style)

    p = gmap(os.environ['GCP_API_KEY'], map_options, title="Denver")

    source = ColumnDataSource(
        data=dict(lat=lat_lst,
                lon=long_lst)
    )

    p.circle(x="lon", y="lat", size=10, fill_color="red", fill_alpha=0.8, source=source)

    if export_as_png:
        export_png(p, filename="../images/listing_map.png")
    else:
        show(p)

def bar_chart_desc(tick_labels, values, xlabel, ylabel, title):
    desc_idx = np.argsort(np.array(values))
    desc_y = values[desc_idx][::-1]
    desc_x = tick_labels[desc_idx][::-1]


    fig, ax = plt.subplots(1,figsize=(10,8))
    ax.bar(desc_x, desc_y)
    ax.set_xticklabels(desc_x, rotation=25, ha='right')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

def plot_1x4_resize_img(sq_imgs, size=(20,5)):
    '''
    Plots 1x4 of varying resizing
    '''
    shape_lst = ['32 x 32','64 x 64','128 x 128','256 x 256']

    fig, axs = plt.subplots(1,4,figsize=size)
    for idx, (ax, img) in enumerate(zip(axs, sq_imgs)):
        ax.imshow(img, cmap='gray')
        ax.set_xlabel(shape_lst[idx])
        ax.set_xticks([]); ax.set_yticks([])


def plot_1x2_img(imgs, labels, size=(20,8)):
    '''
    Plots 1x2 of varying resizing
    '''

    fig, axs = plt.subplots(1,2,figsize=size)
    for idx, (ax, img) in enumerate(zip(axs, imgs)):
        ax.imshow(img)
        ax.set_xlabel(labels[idx])
        ax.set_xticks([]); ax.set_yticks([])


def cm_plot(cm,title):
    '''
    Plots heatmap of Confusion matrix for NB classifier
    '''
    cats = ['Denver','Arvada','Aurora','Lakewood','Centennial','Westminster','Thornton']
    y_range = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]

    cm_round = np.round(cm,decimals=2)
    heat_map = sb.heatmap(cm_round, annot=True)
    plt.xticks(range(len(cats)), labels = cats, rotation=25)
    plt.title(title)
    plt.yticks(y_range, labels = cats, rotation=0, ha='right')

if __name__ == "__main__":
    lats = [39.588828, 39.745790, 39.809788, 39.730797, 39.646233, 39.870414, 39.873274, 39.692519, 39.843736, 39.853361]
    longs = [-104.931643, -105.125263, -105.106349, -105.087823, -104.780792, -105.215168, -105.093423, -105.142520,-105.023142, -105.085782]

    listing_map(lats,longs)

    # df = tt_holdout_error_curves(path)
    