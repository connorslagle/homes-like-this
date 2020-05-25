# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html

import os
from urllib.parse import urlparse
import json

import scrapy
from scrapy.exporters import JsonItemExporter
from scrapy.pipelines.images import ImagesPipeline
from scrapy.exceptions import DropItem
from scrapy_selenium import SeleniumRequest
from PIL import Image
import pymongo
import boto3
import botocore
from datetime import datetime



class MetadataPipeline():
    '''
    Metadata retrieved from 'SearchPageItem' (container for scraped data
    from search page).
    
    Metadata will be stored in mongodb, in db 'listings'
    
    db will have collections 'search_metadata', and 'listing_metadata',
    each containing info scraped from search, listing pgs, respectfully.
    '''
    '''
    mongodb options
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
    
    '''
    S3 options
    '''

    # def open_spider(self, spider):
    #     self.boto3_connection = boto3.resource('s3')
    #     self.client = boto3.client('s3')
    #     self.bucket_name = 'homes-like-this'

    # def close_spider(self, spider):
    #     pass

    # def process_item(self, item, spider):
    #     time_now = datetime.now()
    #     time_str = '{}_{}_{}_{}'.format(str(time_now.date()), time_now.hour, time_now.minute, time_now.second)

    #     if 'image_urls' not in item.keys():
    #         search_f = open('../../../data/jsondump/search_{}.json'.format(time_str),'wb')
    #         search_exp = JsonItemExporter(search_f)
    #         search_exp.start_exporting()
    #         search_exp.export_item(item)
    #         search_exp.finish_exporting()

    #         search_file_path = search_f.name
    #         search_file_name = os.path.basename(search_f.name)

    #         search_f.close()
    #         self.client.upload_file(search_file_path, self.bucket_name, 'search_metadata/{}'.format(search_file_name))

    #         # with open(file_name, 'rb') as f:
    #         #     self.client.upload_file()
    #         #     self.s3object = self.boto3_connection.Object(self.bucket_name, 'search_metadata/{}'.format(file_name))
    #         #     self.s3object.put(
    #         #         Body=(f.readlines())
    #         #     )
    #     else:
    #         f = open('../../../data/jsondump/listing_{}.json'.format(time_str),'wb')
    #         item_exp = JsonItemExporter(f)
    #         item_exp.start_exporting()
    #         item_exp.export_item(item)
    #         item_exp.finish_exporting()

    #         file_path = f.name
    #         file_name = os.path.basename(f.name)

    #         f.close()
    #         self.client.upload_file(file_path, self.bucket_name, 'listing_metadata/{}'.format(file_name))
    #         # with open(file_name, 'rb') as f:
    #         #     self.s3object = self.boto3_connection.Object(self.bucket_name, 'listing_metadata/{}'.format(file_name))
    #         #     self.s3object.put(
    #         #         Body=(f.readlines())
    #         #     )

    #     return item


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