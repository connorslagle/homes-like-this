import scrapy
from scrapy.loader import ItemLoader
from scrapy.loader.processors import Compose
from scrapy_selenium import SeleniumRequest
from ..items import SearchPageItem


class ListingSpider(scrapy.Spider):
    '''
    Another temporary Spider to test functionality.
    '''
    
    name = "test"

    def start_requests(self):
        allowed_domains = ['realtor.com']
        start_urls = ['https://www.realtor.com/realestateandhomes-search/Denver_CO/pg-1']
        for url in start_urls:
            yield SeleniumRequest(url=url, callback=self.parse_result)

    def parse_result(self, response):
        
        '''
        List of Listings pages: extract metadata and load to SearchPageItem container
        '''
        self.search_page_url = response.url
        url_page = response.url.split('-')[-1]
        url_city = response.url.split('/')[-2]

        base_xpath= "//ul[@data-testid='property-list-container']/li/div/div[2]/div[3]"

        self.href = response.xpath(f'{base_xpath}/a/@href').extract()
        
        for idx, listing in enumerate(self.href):
            l = ItemLoader(item=SearchPageItem(), response=response)
            l.add_value('listing_id',f'{url_city}_{url_page}_{idx}')
            l.add_value('search_url', self.search_page_url)
            l.add_value('search_city', url_city)
            l.add_value('search_page', url_page)

            idx_elem = Compose(lambda x: x[idx])

            l.add_value('listing_href', listing)
            l.add_xpath('prop_type', f'{base_xpath}/a/div/div[1]/div/span/text()', idx_elem)
            l.add_xpath('price', f'{base_xpath}/a/div/div[2]/span/text()', idx_elem)
            l.add_xpath('beds', f'{base_xpath}/div[1]/div/a/div[1]/div/ul/li[1]/span[1]/text()', idx_elem)
            l.add_xpath('baths', f'{base_xpath}/div[1]/div/a/div[1]/div/ul/li[2]/span[1]/text()', idx_elem)
            l.add_xpath('sqft', f'{base_xpath}/div[1]/div/a/div[1]/div/ul/li[3]/span[1]/text()', idx_elem)
            l.add_xpath('lotsqft', f'{base_xpath}/div[1]/div/a/div[1]/div/ul/li[4]/span[1]/text()', idx_elem)
            yield l.load_item()