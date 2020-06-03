import pymongo
import numpy as np
import pandas as pd
from datetime import date
import os
from skimage import io
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.filters import sobel



# class DataHandler():

#     def __init__(self):


class MongoImporter():
    '''
    Class to import/format metadata from mongodb to pandas df. Has option to save to csv for easier processing.
    '''

    def __init__(self):
        self.conn = pymongo.MongoClient('localhost', 27017)
        self.db = self.conn['listings']

    def _from_collection(self):
        '''
        Pulls metadata from specified collection, returns raw documents
        '''

        if self.from_search_coll:
            coll = self.db['search_metadata']
        else:
            coll = self.db['listing_metadata']

        cur = coll.find({})
        raw_docs = [doc for doc in cur]
        cur.close()
        return raw_docs

    def _square_docs(self):
        '''
        Pops unwanted columns, truncates search_dict by 'listing_href' entries.

        Returns list of square dictionaries (len(values) is the same for all keys)
            allows for pd.DataFrame.from_dict
        
        '''

        raw_docs = self._from_collection()

        # if data from search_metadata collection
        if self.from_search_coll:
            fields_to_pop = ['_id', 'search_url', 'lotsqft']
            
            new_list = []
            for doc in raw_docs:
                if 'image_id' not in doc.keys():
                    new_list.append(doc)
            
            raw_docs = new_list

            [[doc.pop(field) for field in fields_to_pop] for doc in raw_docs]
            max_length = [min([len(v) for v in doc.values()]) for doc in raw_docs]

            sq_docs = []
            for idx, doc in enumerate(raw_docs):
                sq_docs.append({k: v[:max_length[idx]] for k,v in doc.items()})
        else:
            new_list = []
            for doc in raw_docs:
                if len(doc.keys()) == 5:
                    new_list.append(doc)

            raw_docs=new_list

            fields_to_pop = ['_id', 'aux_metadata', 'prop_desc']

            # complicated list comps - consider revising
            [[doc.pop(field) for field in fields_to_pop] for doc in raw_docs]
            [[doc['image_id'].append(doc['image_id'][0]) for _ in range(len(doc['image_urls'])-1)] for doc in raw_docs]
            sq_docs = raw_docs

        return sq_docs

    def _concat_docs(self):
        '''
        Concatenates list of sq. dicts to single pandas df. 
        '''

        self.from_search_coll = True

        search_docs = self._square_docs()
        search_df = pd.DataFrame()

        # append search_metadata to single df
        for doc in search_docs:
            temp = pd.DataFrame.from_dict(doc)
            search_df = search_df.append(temp, ignore_index=True)
        self.search_df = search_df

        self.from_search_coll = False
        listing_docs = self._square_docs()
        listing_df = pd.DataFrame()

        # append listing_metadata to single df
        for doc in listing_docs:
            temp = pd.DataFrame.from_dict(doc)
            listing_df = listing_df.append(temp, ignore_index=True)
        self.listing_df = listing_df

    def _join_dfs(self):
        '''
        Join dfs on image file name.

        returns joined df
        '''
        full_df = self.listing_df.join(self.search_df.set_index('listing_id'), on='image_id', how='right')
        full_df.reset_index(inplace=True)
        return full_df
        
    def load_docs(self, from_search_coll=True):
        self.from_search_coll = from_search_coll
        self._concat_docs()
        df = self._join_dfs()
        self.df = self._format_df(df)

        return self.df
    
    def _format_df(self, df):
        '''
        Pipeline for df data cleaning. Returns formatted df.
        '''

        # reindex df, drop old index
        df.drop('index', axis=1, inplace=True)

        #drop nans
        df.dropna(inplace=True)

        # image url to filename
        df.image_urls = df.image_urls.str.split('/')
        df.image_urls = [elem[-1] for elem in df.image_urls]

        # listing href to address/city/zip
        df.listing_href = df.listing_href.str.split('/')
        df.listing_href = [elem[-1] for elem in df.listing_href]
        
        state = 'CO'

        df.listing_href = df.listing_href.str.split(state)
        df['temp2'] = [elem[-1] for elem in df.listing_href]
        df['temp1'] = [elem[-2] for elem in df.listing_href]

        df.temp1 = df.temp1.str.split('_')
        df.temp2 = df.temp2.str.split('_')

        drop_lst = []
        for idx, elem in enumerate(df.temp1):
            if len(elem) < 3:
                drop_lst.append(idx)

        df.drop(drop_lst, inplace=True)

        df['address'] = [elem[-3] for elem in df.temp1]
        df['city'] = [elem[-2] for elem in df.temp1]
        df['state'] = [state for elem in df.listing_href]
        df['zipcode'] = [elem[1] for elem in df.temp2]
        df.drop('listing_href', axis=1, inplace=True)
        df.drop('temp1', axis=1, inplace=True)
        df.drop('temp2', axis=1, inplace=True)

        df.columns = ['listing_id', 'image_file','prop_type', 'listing_price',
                        'beds','baths', 'sqft', 'address','city','state','zipcode']

        # drop duplicate images
        df.drop_duplicates('image_file',inplace=True)

        # if not in big 7 cities - drop, some misclassified on Realtor.com
        cities = ['Denver','Aurora','Arvada','Thornton','Lakewood','Centennial','Westminster']

        mask = [(elem not in cities) for elem in df.city]

        df.drop(df.city[mask].index, inplace=True)
        return df

    def to_csv(self, file_name):
        '''
        Save to csv file for easier data processing down the line.
        '''
        today_date = str(date.today())
        file_path = '../data/metadata/{}_{}'.format(today_date,file_name)

        self.df.to_csv(file_path)

class ImagePipeline(MongoImporter):
    '''
    Class for importing, processing and featurizing images.
    '''

    def __init__(self, image_dir, gray_imgs=True):
        self.image_dir = image_dir
        self.save_dir = '../data/proc_images/'
        self.city_dict = {'Denver': 0, 'Arvada': 1, 'Aurora': 2, 'Lakewood':3,
                        'Centennial': 4,'Westminster':5, 'Thornton':6}

        # Empty lists to fill with img names/arrays
        self.img_lst2 = []
        self.img_names2 = []

        # Featurization outputs
        self.features = None
        self.labels = None
        self.gray_images = gray_imgs


    def _empty_variables(self):
        '''
        Reset all the image related instance variables
        '''
        self.img_lst2 = []
        self.img_names2 = []
        self.features = None
        self.labels = None

    def read(self, batch_mode=False, batch_size=1000,batch_resize_size=(128,128)):
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
        '''
        Resize all images in self.img_lst2 to specified size (prefer base 2 numbers (32,64,128))
        '''

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
        '''
        Take a list of images and vectorize all the images. Returns a feature matrix where each
        row represents an image
        '''
        imgs = [np.ravel(img) for img in self.img_lst2]
        
        self.features = np.r_['0', imgs]


    def _vectorize_labels(self):
        '''
        Convert file names to a list of y labels (in the example it would be either cat or dog, 1 or 0)
        '''
        # Get the labels with the dimensions of the number of image files
        self.labels = self.img_names2[0]

    def vectorize(self):
        '''
        Return (feature matrix, the response) if output is True, otherwise set as instance variable.
        Run at the end of all transformations
        '''
        self._vectorize_features()
        self._vectorize_labels()

    def build_Xy(self):
        '''
        Returns X,y mats for cnn
        '''
        self.read()
        self.vectorize()
        self.df = self.load_docs()

        city = []
        idx = []
        for elem in self.labels:
            if elem in self.df.image_file.values:
                city.append(self.df.city[self.df.image_file == elem].values[0])
                idx.append(self.labels.index(elem))

        self.X = self.features[idx,:]
        self.y = [self.city_dict[key] for key in city]

        X_tt, self.X_holdout, y_tt, self.y_holdout = train_test_split(self.X, np.array(self.y), stratify=np.array(self.y))
        X_train, X_test, y_train, y_test = train_test_split(X_tt, y_tt,stratify=y_tt)





def load_and_featurize_data(from_file, img_size, image_dim = 3):

    

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


if __name__ == "__main__":
    # importer = MongoImporter()
    # df = importer.load_docs()
    # importer.to_csv('pg1_3_all.csv')

    img_pipe = ImagePipeline('../data/listing_images/full/',gray_imgs=False)
    img_pipe.read(batch_mode=True, batch_size=500,batch_resize_size=(256,256))
    # img_pipe.resize((64,64))
    # img_pipe.save()