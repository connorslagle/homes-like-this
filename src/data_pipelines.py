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
        self.db = conn['listings']

    def _from_search_page(self):
        '''
        Pulls search_page metadata, returns raw documents
        '''
        coll = self.db['search_metadata']
        return coll.find({})

    def _from_listing_page(self):
        '''
        Pulls listing_page metadata, returns raw documents
        '''
        coll = self.db['listing_metadata']
        return coll.find({})

    def _square_docs(self, from_search_coll=True):
        '''
        Pops unwanted columns, truncates dict by 'listing_href' entries
        '''
        if from_search_coll:
            raw_docs = self._from_search_page()

            fields_to_pop = ['_id', 'search_url', 'lotsqft']

            if 'image_id' not in raw_docs.keys():
                [raw_docs.pop(field) for field in fields_to_pop]
                docs = {k: v[:len(raw_docs['listing_href'])] for k,v in raw_docs.items()}

        else:
            raw_docs = self._from_listing_page()
        

    
class ImagePipeline():


if __name__ == "__main__":
    