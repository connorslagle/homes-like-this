import scrapy

class ListingSpider(scrapy.Spider):
    name = 'listings'

    def start_requests(self):
        possible_urls = ['https://www.realtor.com/realestateandhomes-search/Aurora_CO/pg-',
                            'https://www.redfin.com/city/30839/CO/Aurora/page-1',
                            'https://www.homes.com/aurora-co/homes-for-sale/p1']


        pages = range(2,43)
        urls = ['https://www.zillow.com/aurora-co/1_p']

        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)
    
    def parse(self, response):
        page = response.url.split('/')[-1]
        filename = f'zillow_test_{page}.html'
        with open(filename, 'wb') as f:
            f.write(response.body)
        self.log(f'Saved file {filename}')
