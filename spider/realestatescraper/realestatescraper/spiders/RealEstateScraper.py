import scrapy
from scrapy_selenium import SeleniumRequest

class ListingSpider(scrapy.Spider):
    name = 'listings'

    def start_requests(self):
        possible_urls = ['https://www.realtor.com/realestateandhomes-search/Aurora_CO/pg-',
                            'https://www.redfin.com/city/30839/CO/Aurora/page-1',
                            'https://www.homes.com/aurora-co/homes-for-sale/p1',
                            'https://www.zillow.com/aurora-co/1_p']


        pages = range(2,43)
        urls = ['https://www.realtor.com/realestateandhomes-search/Aurora_CO/pg-1']

        for url in urls:
            yield SeleniumRequest(url=url, callback=self.parse_result)
    
    def parse_result(self, response):
        page = response.url.split('/')[-2]
        filename = f'realtor_test_{page}.html'
        with open(filename, 'wb') as f:
            f.write(response.body)
        self.log(f'Saved file {filename}')
