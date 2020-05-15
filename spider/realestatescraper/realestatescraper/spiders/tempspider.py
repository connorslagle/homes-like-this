import scrapy
from scrapy.loader import ItemLoader
from scrapy.loader.processors import Compose
from scrapy_selenium import SeleniumRequest
from ..items import ListingItem


class ListingSpider(scrapy.Spider):
    '''
    Temporary spider for testing functionality/xpaths
    '''
    name = "temp"

    def start_requests(self):
        allowed_domains = ['realtor.com']
        start_urls = ['https://www.realtor.com/realestateandhomes-detail/6810-Deatonhill-Dr-Apt-114_Austin_TX_78745_M80627-96866']
        for url in start_urls:
            test_dict = {'test':{'inner':[2]}}
            yield SeleniumRequest(url=url, callback=self.parse_result, cb_kwargs=test_dict)

    def parse_result(self, response, test):
        value=test['inner'][0]
        print(f'\n{value}\n')
