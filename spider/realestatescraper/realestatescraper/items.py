# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class ListingItem(scrapy.Item):
    '''
    Container for data scraped from listing webpage.
    '''

    # define the fields for your item here:

    # images
    images = scrapy.Field()
    image_paths = scrapy.Field()
    image_urls = scrapy.Field()
    image_id = scrapy.Field()

    # aux metadata
    aux_metadata = scrapy.Field()
    from_url = scrapy.Field()

    # property description
    prop_desc = scrapy.Field()

class SearchPageItem(scrapy.Item):
    '''
    Container for data scraped from search webpage.
    '''

    # define the fields for your item here:
    
    # meta data from search page
    search_url = scrapy.Field()
    listing_id = scrapy.Field()
    listing_href = scrapy.Field()
    prop_type = scrapy.Field()
    price = scrapy.Field()
    beds = scrapy.Field()
    baths = scrapy.Field()
    sqft = scrapy.Field()
    lotsqft = scrapy.Field()