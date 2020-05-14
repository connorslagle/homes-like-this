import pymongo
import numpy as np
import pandas as pd
from bson.objectid import ObjectId
from datetime import date



class MongoImporter():
    '''
    Class to import/format metadata from mongodb.
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
        Pops unwanted columns, truncates search_dict by 'listing_href' entries,
        
        '''
        raw_docs = self._from_collection()

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

            [[doc.pop(field) for field in fields_to_pop] for doc in raw_docs]
            [[doc['image_id'].append(doc['image_id'][0]) for _ in range(len(doc['image_urls'])-1)] for doc in raw_docs]
            sq_docs = raw_docs

        return sq_docs

    def _concat_docs(self):
        '''
        Concatenates documents to single pandas df
        '''
        self.from_search_coll = True
        # breakpoint()
        search_docs = self._square_docs()
        search_df = pd.DataFrame()
        # breakpoint()
        for doc in search_docs:
            temp = pd.DataFrame.from_dict(doc)
            search_df = search_df.append(temp, ignore_index=True)
        self.search_df = search_df

        self.from_search_coll = False
        listing_docs = self._square_docs()
        listing_df = pd.DataFrame()

        for doc in listing_docs:
            temp = pd.DataFrame.from_dict(doc)
            listing_df = listing_df.append(temp, ignore_index=True)
        self.listing_df = listing_df

    def _join_dfs(self):
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
        df_copy = df.copy()

        
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
        # breakpoint()
        df.drop(drop_lst, inplace=True)
        # breakpoint()
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

        # if not in big 7 cities - drop
        cities = ['Denver','Aurora','Arvada','Thornton','Lakewood','Centennial','Westminster']

        mask = [(elem not in cities) for elem in df.city]

        df.drop(df.city[mask].index, inplace=True)

        return df

    def to_csv(self, file_name):
        '''
        output csv with todays date
        '''
        today_date = str(date.today())
        file_path = f'../data/metadata/{today_date}_{file_name}'

        self.df.to_csv(file_path)

    
# class ImagePipeline():


if __name__ == "__main__":
    importer = MongoImporter()
    df = importer.load_docs()
    importer.to_csv('pg1_3_all.csv')
    
    '''
    # df = pd.DataFrame.from_dict(listing_docs[0])
    # df2 = pd.DataFrame.from_dict(search_docs[0])
    # df3 = df.join(df2.set_index('listing_id'), on='image_id', how='right') 
    '''