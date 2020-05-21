# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html

import os
from urllib.parse import urlparse

import scrapy
from scrapy.pipelines.images import ImagesPipeline
from scrapy.exceptions import DropItem
from scrapy_selenium import SeleniumRequest
from PIL import Image
import pymongo



class MetadataPipeline():
    '''
    Metadata retrieved from 'SearchPageItem' (container for scraped data
    from search page).
    
    Metadata will be stored in mongodb, in db 'listings'
    
    db will have collections 'search_metadata', and 'listing_metadata',
    each containing info scraped from search, listing pgs, respectfully.
    '''

    def open_spider(self, spider):
        self.conn = pymongo.MongoClient('localhost', 27017)
        self.db = self.conn['listings']

    def close_spider(self, spider):
        self.conn.close()

    def process_item(self, item, spider):
        if 'image_urls' not in item.keys():
            collection = self.db['search_metadata']
            collection.insert(dict(item))
        else:
            collection = self.db['listing_metadata']
            collection.insert(dict(item))

        return item


class MyImagesPipeline(ImagesPipeline): 
    '''
    Images from 'ListingItem' (container for imgs from listing page)
    '''

    def file_path(self, request, response=None, info=None):
        return 'full/' + os.path.basename(urlparse(request.url).path)

    def get_media_requests(self, item, info):
        if 'image_urls' in item.keys():
            for image_url in item['image_urls']:
                yield scrapy.http.Request(image_url)

    def item_completed(self, results, item, info):
        image_paths = [x['path'] for ok, x in results if ok]
        if not image_paths:
            raise DropItem("Item contains no images")
        item['image_paths'] = image_paths
        return item