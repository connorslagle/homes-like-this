import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# to make functions work in tensorflow container -have to bring in
from datetime import date
import os
from skimage import io
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.filters import sobel

from tensorflow import keras
from tensorflow.keras.layers import Activation, Dense, Input
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


from cnn import load_and_featurize_data


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



if __name__ == "__main__":
    np.random.seed(1) # get consistent results from a stochastic training process
