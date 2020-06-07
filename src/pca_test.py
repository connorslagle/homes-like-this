# general imports
import numpy as np
import pdb

# sklearn imports
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

# other python files
from data_pipelines import ImagePipeline
from cnn_new import Autoencoder
from plotting import pca_var_all, elbow_plot, six_hist



if __name__ == "__main__":
    # import data
    rgb = Autoencoder(gray_imgs=False)
    gray = Autoencoder()

    rgb.load_Xy('2020-06-04')
    gray.load_Xy('2020-06-04')

    X_rgb = rgb.X_rgb
    X_gray = gray.X_gray

    # center data
    # scaler = StandardScaler()
    # X_rbg_ss = scaler.fit_transform(X_rgb['train'])
    # X_gray_ss = scaler.fit_transform(X_gray['train'])


    # init pca
    pca_rbg = PCA(n_components=128, random_state=33)
    pca_rbg.fit(X_rgb['train'])

    pca_gray = PCA(n_components=128, random_state=33)
    pca_gray.fit(X_gray['train'])

    pca_var_all(pca_rbg.explained_variance_ratio_, pca_gray.explained_variance_ratio_)

    # cluster with kmeans, look at top 9 ims
    elbow_plot(pca_rbg.components_, 50, 'color')
    elbow_plot(pca_gray.components_, 50, 'gray')

    kmeans_rbg = KMeans(n_clusters=20, init=pca_rbg.components_[:,:20], random_state=33)
    kmeans_rbg.fit(pca_rbg.components_[:,:20])
    rbg_labels = kmeans_rbg.labels_
    # pdb.set_trace()
    # rbg_sil = silhouette_samples(X_rbg_ss, rbg_labels)
    avg_sil_score = silhouette_score(pca_rbg.components_,rbg_labels,metric='cosine')
    print(avg_sil_score)

    
    # kmeans_gray = KMeans(7, random_state=33, algorithm='full')
    # kmeans_gray.fit(pca_gray.components_)
    # gray_labels = kmeans_gray.labels_
    # gray_sil = silhouette_samples(pca_gray.components_, gray_labels)

    # six_hist(rbg_sil, rbg_labels, 'color')
    # six_hist(gray_sil, gray_labels, 'gray')


