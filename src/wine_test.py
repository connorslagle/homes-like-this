import pandas as pd
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras import backend as K
from theano import function
import matplotlib.pyplot as plt
import kmeans_test 
import pickle

import tensorflow as tf
from cnn import load_and_featurize_data, ImagePipeline, load_data, fname_to_city



class Autoencoder():

    def __init__(self,model=None):
        '''
        If a model is provided, it initializes the model for later use.
        input:
        model (optional): trained cnn model
        output: None
        '''
        if model != None:
            self.model = model

    def build_autoencoder_model(self):
        '''
        If a model was not provided when instantiating the class, this method
        builds the autoencoder model.
        input: None
        output: None
        '''
            
        autoencoder = Sequential()
        
        # encoder layers
        autoencoder.add(Conv2D(128,(3,3), activation='relu', padding='same',input_shape=(64,64,3)))
        autoencoder.add(MaxPooling2D((2,2), padding = 'same'))
        autoencoder.add(Conv2D(64,(3,3), activation='relu', padding='same'))
        autoencoder.add(MaxPooling2D((2,2), padding = 'same'))
        autoencoder.add(Conv2D(32,(3,3), activation='relu', padding='same'))
        autoencoder.add(MaxPooling2D((2,2), padding = 'same'))
        autoencoder.add(Conv2D(8,(3,3), activation='relu', padding='same'))
        autoencoder.add(MaxPooling2D((2,2), padding = 'same'))

        autoencoder.add(Flatten())
        autoencoder.add(Reshape((4,4,8)))

        # decoder layers
        autoencoder.add(Conv2D(8,(3,3), activation='relu', padding='same'))
        autoencoder.add(UpSampling2D((2,2)))
        autoencoder.add(Conv2D(32,(3,3), activation='relu', padding='same'))
        autoencoder.add(UpSampling2D((2,2)))
        autoencoder.add(Conv2D(64,(3,3),activation='relu', padding='same'))
        autoencoder.add(UpSampling2D((2,2)))
        autoencoder.add(Conv2D(128,(3,3), activation='relu', padding='same'))
        autoencoder.add(UpSampling2D((2,2)))
        autoencoder.add(Conv2D(3,(3,3), activation='sigmoid', padding='same'))

        # autoencoder.summary()

        autoencoder.compile(optimizer='adam', loss='mean_squared_error')

        self.model = autoencoder


    def fit(self,train,test,batch_size,epochs):
        '''
        After the autoencoder model has been built, this method fits the
        model using a train and test dataset. It also stores the model
        loss and val_loss history in a history attribute.
        input:
        train: training dataset of image arrays (64x64x3)
        test: test dataset of image arrays (64x64x3)
        batch_size: batch size for model fit
        epochs: number of epochs to execute
        output: None                
        '''
        self.model.fit(train, train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(test,test))
        self.history = self.model.history.history

    def get_rmse(self,test):
        '''
        Calculates the RMSE of the model on the test dataset after the model
        is trained. 
        
        input:
        test: test dataset of image arrays (64x64x3)
        output:
        RMSE of test dataset
        '''
        return self.model.evaluate(test,test)

    def predict(self,X):
        '''
        Using the trained autoencoder, predicts on the provided image 
        array and returns the reconstructed images.
        input:
        X: dataset of image arrays (64x64x3)
    
        output:
        reconstructed images
        '''
        return self.model.predict(X)

    def plot_before_after(self,test,test_decoded,n=10):
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
            plt.imshow(X[i])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(test_decoded[i])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        plt.savefig('before_after.png')

    def get_layers(self,X,layer_index):
        '''
        Extracts a layer of the autoencoder model. Used to get the flattened
        encoded image for clustering.
        input:
        X: dataset of image arrays (64x64x3)
        layer: index of the desired layer
        output:
        extracted layer array
        '''
        batches = np.split(X,5)
        for i,batch in enumerate(batches):
            get_layer_output = K.function([self.model.layers[0].input],
                                        [self.model.layers[layer_index].output])
            layer_output = get_layer_output([batch])[0]

            if i == 0:
                final_layers = layer_output
            else:
                final_layers = np.vstack((final_layers,layer_output))
        self.encoding = final_layers
        return self.encoding

    def plot_loss(self):
        '''
        Plots the loss and val_loss of the trained model.
        input: None
        Output: None - saves the plot to a file
        '''
        fig,ax = plt.subplots(figsize=(8,6))
        ax.plot(cnn.model.history.history['loss'])
        ax.plot(cnn.model.history.history['val_loss'])
        ax.set_title('CNN Autoencoder Model Loss')
        ax.set_ylabel('Loss')
        ax.set_xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper right')
        plt.savefig('cnn_loss.png')

    def execute_kmeans(self,n_clusters,df_filepath):
        '''
        Executes the kmeans_fit method in the kmeans.py file and
        also adds the kmeans_label assigment to the provided image
        dataframe path.
        inputs:
        n_clusters: number of clusters in the kmeans model
        df_filepath: filepath to add the cluster assignments
        output: None
        '''
        labels,inertia = kmeans.kmeans_fit(self.encoding,n_clusters=n_clusters)
        kmeans.add_labels_to_df(labels,df_filepath)

    def show_cluster(self,n_clusters,df_filepath):
        '''
        Executes the show_cluster method in the kmeans.py file. Saves a plots per
        cluser with example images from the cluster.
        inputs:
        n_clusters: number of clusters in the kmeans model
        df_filepath: filepath to dataframe with image clusters 
        output: None
        '''
        kmeans.show_cluster(n_clusters,df_filepath)

if __name__=='__main__':
    # X = np.load('../data/64x64/image_array_cnn.npy')
    # # X = X[:500]
    # train, test = train_test_split(X, test_size=0.2)

    # my params -------------
    # needed for tf gpu
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.compat.v1.Session(config=config)

    img_size = 64
    img_dim = 3
    df = pd.read_csv('../data/metadata/2020-05-14_pg1_3_all.csv')

    X_train, X_test, Y_train, Y_test, X_holdout, Y_holdout = load_and_featurize_data('../data/proc_images/color/{}/'.format(img_size))

    #building model
    

    #if building the model from scratch
    # cnn = Autoencoder()
    # cnn.build_autoencoder_model()
    # batch_size = 100
    # epochs = 2
    # cnn.fit(train,test,batch_size,epochs)
    # scores = cnn.get_rmse(test)
    # print(scores)
    # cnn.plot_loss()

    #if passing in a model
    filename = 'model200'
    model = pickle.load(open(filename, 'rb'))
    cnn = Autoencoder(model)

    #plot before and after images
    # X_decoded = cnn.predict(X)
    # cnn.plot_before_after(X,X_decoded,10)

    layers = cnn.get_layers(X,8)

    df_filepath = '../data/64x64/sorted_df.csv'
    n_clusters = 7
    cnn.execute_kmeans(n_clusters,df_filepath)
    cnn.show_cluster(n_clusters,df_filepath)