# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class ListingItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()

    # images
    images = scrapy.Field()
    image_paths = scrapy.Field()
    image_urls = scrapy.Field()
    image_id = scrapy.Field()

    # aux metadata
    prop_status = scrapy.Field()
    price_sqft = scrapy.Field()
    time_on_web = scrapy.Field()
    prop_type = scrapy.Field()
    year_built = scrapy.Field()
    prop_style = scrapy.Field()
    prop_desc = scrapy.Field()

class SearchPageItem(scrapy.Item):
    # define the fields for your item here like:
    
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

