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
import pymongo


class MetadataPipeline():
    '''
    Metadata will be stored in mongodb, in db 'listing_metadata'
    db will have collections 'by_search_page', and 'by_listing'
    '''
    def __init__(self):
        self.conn = pymongo.MongoClient('localhost', 27017)
        db = self.conn['listings']
        self.collection = db['metadata']

    def process_item(self, item, spider):
        self.collection.insert(dict(item))
        return item


class MyImagesPipeline(ImagesPipeline):

    def get_media_requests(self, item, info):
        for image_url in item['image_urls']:
            yield SeleniumRequest(url=image_url)

    def item_completed(self, results, item, info):
        image_paths = [x['path'] for ok, x in results if ok]
        if not image_paths:
            raise DropItem("Item contains no images")
        item['image_paths'] = image_paths
        return item