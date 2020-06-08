import numpy as np
import pandas as pd
# from bson.objectid import ObjectId
from datetime import date
import os
import pickle
from skimage import io
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.filters import sobel
np.random.seed(1337)  # for reproducibility

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Reshape
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# for test autoencoder
import kmeans_test
import matplotlib.pyplot as plt



class ImagePipeline():
    '''
    Class for importing, processing and featurizing images.
    '''

    def __init__(self, image_dir, gray_imgs=True):
        self.image_dir = image_dir
        self.save_dir = '../data/proc_images/'

        # Empty lists to fill with img names/arrays
        self.img_lst2 = []
        self.img_names2 = []

        # Featurization outputs
        self.features = None
        self.labels = None
        self.gray_images = gray_imgs


    def _empty_variables(self):
        """
        Reset all the image related instance variables
        """
        self.img_lst2 = []
        self.img_names2 = []
        self.features = None
        self.labels = None

    def read(self, batch_mode=False, batch_size=1000,batch_resize_size=(32,32)):
        '''
        Reads image/image names to self variables. Has batch importer modes, to save computer memory.

        Batch import mode PROCESSES images - needed to reset class lists.

        Review before processing.
        '''

        self._empty_variables()

        img_names = os.listdir(self.image_dir)
        
        
        if batch_mode:
            num_batches = (len(img_names) // batch_size) + 1
            for batch in range(num_batches):
                self.img_lst2 = []
                self.img_names2 = []

                remaining = len(img_names) - batch*batch_size

                if remaining >= batch_size:
                    names = img_names[batch*batch_size:(batch+1)*batch_size]
                    self.img_names2.append(names)
                    img_lst = [io.imread(os.path.join(self.image_dir, fname)) for fname in names]
                    self.img_lst2.append(img_lst)
                else:
                    names = img_names[batch*batch_size:]
                    self.img_names2.append(names)
                    img_lst = [io.imread(os.path.join(self.image_dir, fname)) for fname in names]
                    self.img_lst2.append(img_lst)


                self._square_image()
                if self.gray_images:
                    self._gray_image()

                self._resize(batch_resize_size)
                
                self.save()
        
        else:
            self.img_names2.append(img_names)
            img_lst = [io.imread(os.path.join(self.image_dir, fname)) for fname in img_names]
            self.img_lst2.append(img_lst)

        self.img_lst2 = self.img_lst2[0]

        

    def _square_image(self):
        '''
        Squares image based on largest side length.
        '''
        cropped_lst = []
        for img in self.img_lst2[0]:
            # breakpoint()
            y_len, x_len, _ = img.shape

            crop_len = min([x_len,y_len])
            x_crop = [int((x_len/2) - (crop_len/2)), int((x_len/2) + (crop_len/2))]
            y_crop = [int((y_len/2) - (crop_len/2)), int((y_len/2) + (crop_len/2))]
            if y_len >= crop_len:
                cropped_lst.append(img[y_crop[0]:y_crop[1], x_crop[0]:x_crop[1]])
            else:
                cropped_lst.append(img[x_crop[0]:x_crop[1], y_crop[0]:y_crop[1]])
        self.img_lst2 = cropped_lst

    def _gray_image(self):
        '''
        Grayscales img
        '''
        gray_imgs = [rgb2gray(elem) for elem in self.img_lst2]
        self.img_lst2 = gray_imgs


    def _filter_image(self, filter='sobel'):
        '''
        Filters grey img
        '''
        filter_imgs = [sobel(elem) for elem in self.img_lst2]
        self.img_lst2 = filter_imgs

    def _resize(self, shape):
        """
        Resize all images in self.img_lst2 to specified size (prefer base 2 numbers (32,64,128))
        """

        new_img_lst2 = []
        for image in self.img_lst2:
            new_img_lst2.append(resize(image, shape))

        self.img_lst2 = new_img_lst2
        self.shape = shape[0]


    def save(self):
        '''
        Saves images to save_dir. Subdir is img side length.
        '''
        if self.gray_images:
            gray_tag = 'gray'
        else:
            gray_tag = 'color'
        for fname, img in zip(self.img_names2[0], self.img_lst2):

            io.imsave(os.path.join('{}{}/{}/'.format(self.save_dir,gray_tag,self.shape), fname), img)

    def _vectorize_features(self):
        """
        Take a list of images and vectorize all the images. Returns a feature matrix where each
        row represents an image
        """
        imgs = [np.ravel(img) for img in self.img_lst2]
        
        self.features = np.r_['0', imgs]


    def _vectorize_labels(self):
        """
        Convert file names to a list of y labels (in the example it would be either cat or dog, 1 or 0)
        """
        # Get the labels with the dimensions of the number of image files
        self.labels = self.img_names2[0]

    def vectorize(self):
        """
        Return (feature matrix, the response) if output is True, otherwise set as instance variable.
        Run at the end of all transformations
        """
        self._vectorize_features()
        self._vectorize_labels()

def load_data(file_dir, use_filter=False):
    '''
    Load images from specified directory.

    Outputs featurized (raveled) images for NB Classification model.
    '''

    img_pipe = ImagePipeline(file_dir)
    img_pipe.read()
    if use_filter:
        img_pipe._filter_image()
    img_pipe.vectorize()
    # breakpoint()
    X_from_pipe = img_pipe.features
    y_from_pipe = img_pipe.labels
    return X_from_pipe, y_from_pipe

def fname_to_city(df, X_in, y_in, cities_dict):
    '''
    Searches dataframe for filenames in y -> creates target with city as
    categories.
    
    Returns: city_target and matching X
    '''
    city = []
    idx = []
    for elem in y_in: 
        if elem in df.image_file.values: 
            city.append(df.city[df.image_file == elem].values[0])
            idx.append(y_in.index(elem))

    X_match = X_in[idx,:]

    numerical_target = [cities_dict[key] for key in city]

    return X_match, numerical_target

def load_and_featurize_data(from_file, img_size, image_dim = 3):
    target_ = {'Denver': 0, 'Arvada': 1, 'Aurora': 2, 'Lakewood':3,
                 'Centennial': 4,'Westminster':5, 'Thornton':6}
    
    # image size
    img_rows, img_cols = img_size, img_size

    # the data, shuffled and split between train and test sets
    X_feat, y_target = load_data(from_file)

    X_new, target = fname_to_city(df, X_feat, y_target, target_)

    X_tt, X_holdout, y_tt, y_holdout = train_test_split(X_new, np.array(target), stratify=np.array(target))
    X_train, X_test, y_train, y_test = train_test_split(X_tt, y_tt,stratify=y_tt)

    # reshape input into format Conv2D layer likes
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, image_dim)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, image_dim)
    X_holdout = X_holdout.reshape(X_holdout.shape[0], img_rows, img_cols, image_dim)

    # don't change conversion or normalization
    X_train = X_train.astype('float32') # data was uint8 [0-255]
    X_test = X_test.astype('float32')  # data was uint8 [0-255]
    X_holdout = X_holdout.astype('float32')
    X_train /= 255 # normalizing (scaling from 0 to 1)
    X_test /= 255  # normalizing (scaling from 0 to 1)
    X_holdout /= 255

    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    print(X_holdout.shape[0], 'holdout samples')

    # convert class vectors to binary class matrices (don't change)
    Y_train = to_categorical(y_train, nb_classes) # cool
    Y_test = to_categorical(y_test, nb_classes)   
    Y_holdout = to_categorical(y_holdout, nb_classes)
    # in Ipython you should compare Y_test to y_test
    return X_train, X_test, Y_train, Y_test, X_holdout, Y_holdout

def define_model(nb_filters, kernel_size, input_shape, pool_size):
    model = Sequential() # model is a linear stack of layers (don't change)

    # note: the convolutional layers and dense layers require an activation function
    # see https://keras.io/activations/
    # and https://en.wikipedia.org/wiki/Activation_function
    # options: 'linear', 'sigmoid', 'tanh', 'relu', 'softplus', 'softsign'

    # first Conv/pool
    model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]),
                        padding='same', 
                        input_shape=input_shape)) #first conv. layer  KEEP
    model.add(Activation('relu')) # Activation specification necessary for Conv2D and Dense layers

    model.add(Conv2D(2*nb_filters, (kernel_size[0], kernel_size[1]), padding='same')) #2nd conv. layer KEEP
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=pool_size)) # decreases size, helps prevent overfitting
    model.add(Dropout(0.5)) # zeros out some fraction of inputs, helps prevent overfitting

    # Second Conv/pool
    # model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]),
    #                     padding='same')) #first conv. layer  KEEP
    # model.add(Activation('relu')) # Activation specification necessary for Conv2D and Dense layers

    # model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]), padding='same')) #2nd conv. layer KEEP
    # model.add(Activation('relu'))

    # model.add(MaxPooling2D(pool_size=pool_size)) # decreases size, helps prevent overfitting
    # model.add(Dropout(0.5)) # zeros out some fraction of inputs, helps prevent overfitting

    # Third Conv/pool
    # model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]),
    #                     padding='same')) #first conv. layer  KEEP
    # model.add(Activation('relu')) # Activation specification necessary for Conv2D and Dense layers

    # model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]), padding='same')) #2nd conv. layer KEEP
    # model.add(Activation('relu'))

    # model.add(MaxPooling2D(pool_size=pool_size)) # decreases size, helps prevent overfitting
    # model.add(Dropout(0.5)) # zeros out some fraction of inputs, helps prevent overfitting

    model.add(Flatten()) # necessary to flatten before going into conventional dense layer  KEEP
    print('Model flattened out to ', model.output_shape)

    # now start a typical neural network
    model.add(Dense(32)) # (only) 32 neurons in this layer, really?   KEEP
    model.add(Activation('relu'))

    model.add(Dropout(0.5)) # zeros out some fraction of inputs, helps prevent overfitting

    model.add(Dense(nb_classes)) # 10 final nodes (one for each class)  KEEP
    model.add(Activation('softmax')) # softmax at end to pick between classes 0-9 KEEP
    
    # many optimizers available, see https://keras.io/optimizers/#usage-of-optimizers
    # suggest you KEEP loss at 'categorical_crossentropy' for this multiclass problem,
    # and KEEP metrics at 'accuracy'
    # suggest limiting optimizers to one of these: 'adam', 'adadelta', 'sgd'
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    return model


def build_autoencoder_model(img_size):
        '''
        If a model was not provided when instantiating the class, this method
        builds the autoencoder model.
        input: None
        output: None
        '''
        img_size = int(img_size)
            
        autoencoder = Sequential()
        
        # encoder layers
        autoencoder.add(Conv2D(128,(3,3), activation='relu', padding='same',input_shape=(img_size,img_size,3)))
        autoencoder.add(MaxPooling2D((2,2), padding = 'same'))
        autoencoder.add(Conv2D(64,(3,3), activation='relu', padding='same'))
        autoencoder.add(MaxPooling2D((2,2), padding = 'same'))
        autoencoder.add(Conv2D(32,(3,3), activation='relu', padding='same'))
        autoencoder.add(MaxPooling2D((2,2), padding = 'same'))
        autoencoder.add(Conv2D(16,(3,3), activation='relu', padding='same'))    # added
        autoencoder.add(MaxPooling2D((2,2), padding = 'same'))  # added
        autoencoder.add(Conv2D(8,(3,3), activation='relu', padding='same'))
        autoencoder.add(MaxPooling2D((2,2), padding = 'same'))

        autoencoder.add(Flatten())
        autoencoder.add(Reshape((4,4,8)))

        # decoder layers
        autoencoder.add(Conv2D(8,(3,3), activation='relu', padding='same'))
        autoencoder.add(UpSampling2D((2,2)))
        autoencoder.add(Conv2D(16,(3,3), activation='relu', padding='same'))    #added
        autoencoder.add(UpSampling2D((2,2)))    # added
        autoencoder.add(Conv2D(32,(3,3), activation='relu', padding='same'))
        autoencoder.add(UpSampling2D((2,2)))
        autoencoder.add(Conv2D(64,(3,3),activation='relu', padding='same'))
        autoencoder.add(UpSampling2D((2,2)))
        autoencoder.add(Conv2D(128,(3,3), activation='relu', padding='same'))
        autoencoder.add(UpSampling2D((2,2)))
        autoencoder.add(Conv2D(3,(3,3), activation='sigmoid', padding='same'))

        # autoencoder.summary()

        autoencoder.compile(optimizer='adam', loss='mean_squared_error')

        return autoencoder



if __name__ == '__main__':
    # needed for tf gpu
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.compat.v1.Session(config=config)

    img_size = 128
    '''
    If first time:
    '''
    # img_dim = 3
    # df = pd.read_csv('../data/metadata/2020-05-14_pg1_3_all.csv')


    # # # important inputs to the model: don't changes the ones marked KEEP 
    # # batch_size = 10  # number of training samples used at a time to update the weights
    # nb_classes = 7   # number of output possibilites: [0 - 9] KEEP
    # # nb_epoch = 10       # number of passes through the entire train dataset before weights "final"
    # # img_rows, img_cols = img_size, img_size  # the size of the MNIST images KEEP
    # # input_shape = (img_rows, img_cols, img_dim)  # 1 channel image input (grayscale) KEEP
    # # nb_filters = 10  # number of convolutional filters to use
    # # pool_size = (2, 2) # pooling decreases image size, reduces computation, adds translational invariance
    # # kernel_size = (3, 3) # convolutional kernel size, slides over image to learn features

    # X_train, X_test, Y_train, Y_test, X_holdout, Y_holdout = load_and_featurize_data('../data/proc_images/color/{}/'.format(img_size), img_size)

    # # pickle datasets
    # X_test_filename, X_train_filename = '2020-05-14_color_{}_Xtest.pkl'.format(img_size), '2020-05-14_color_{}_Xtrain.pkl'.format(img_size)

    # with open('../data/pkl/{}'.format(X_test_filename), 'wb') as f:
    #     pickle.dump(X_test,f)
    
    # with open('../data/pkl/{}'.format(X_train_filename), 'wb') as f:
    #     pickle.dump(X_train, f)

    '''
    runing with pkl'd X mats
    '''
    X_test_filename, X_train_filename = '2020-05-14_color_{}_Xtest.pkl'.format(img_size), '2020-05-14_color_{}_Xtrain.pkl'.format(img_size)

    # unpickle
    with open('../data/pkl/{}'.format(X_test_filename), 'rb') as f:
        X_test = pickle.load(f)
    
    with open('../data/pkl/{}'.format(X_train_filename), 'rb') as f:
        X_train = pickle.load(f)
    

    model = build_autoencoder_model(img_size)
    print(model.summary())

    # fitting
    model.fit(X_train, X_train,
                epochs=2,
                batch_size=10,
                validation_data=(X_test,X_test))

    # model.evaluate(X_test, X_test)

    # X_decoded = model.predict(X_train)


    
    # # during fit process watch train and test error simultaneously
    # model.fit(X_train, X_train, batch_size=batch_size, epochs=nb_epoch,
    #         verbose=1, validation_data=(X_test, Y_test))

    # score = model.evaluate(X_test, Y_test, verbose=0)
    # print('Test score:', score[0])
    # print('Test accuracy:', score[1]) # this is the one we care about