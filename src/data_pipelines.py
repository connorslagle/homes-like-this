import pymongo
import numpy as np
import pandas as pd
from bson.objectid import ObjectId


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
            sq_docs = [{k: v[:len(doc['listing_href'])] for k,v in doc.items()} for doc in raw_docs]
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

    def load_docs(self, from_search_coll=True):
        self.from_search_coll = from_search_coll
        return self._square_docs()

    
# class ImagePipeline():


if __name__ == "__main__":
    importer = MongoImporter()
    search_docs = importer.load_docs()
    listing_docs = importer.load_docs(False)

    # df = pd.DataFrame.from_dict(listing_docs[0])
    # df2 = pd.DataFrame.from_dict(search_docs[0])
    # df3 = df.join(df2.set_index('listing_id'), on='image_id', how='right') 