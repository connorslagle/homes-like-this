import scrapy
from scrapy.loader import ItemLoader
from scrapy.loader.processors import Compose
from scrapy_selenium import SeleniumRequest
from ..items import ListingItem


class ListingSpider(scrapy.Spider):
    name = "temp"

    def start_requests(self):
        allowed_domains = ['realtor.com']
        start_urls = ['https://www.realtor.com/realestateandhomes-detail/6810-Deatonhill-Dr-Apt-114_Austin_TX_78745_M80627-96866']
        for url in start_urls:
            yield SeleniumRequest(url=url, callback=self.parse_result)

    def parse_result(self, response):
        
        l = ItemLoader(item=ListingItem(), response=response)

        imgs_xpath = "//section[@id='ldp-hero-container']/div/div/div[1]/div[1]/div/img/@data-src"

        single_elem = Compose(lambda x: x[0])
        l.add_xpath('image_urls', imgs_xpath, single_elem)
        yield l.load_item()
        # new imgs xpath //section[@id='ldp-hero-container']/div/div/div[1]/div[1]/div/img/@data-src