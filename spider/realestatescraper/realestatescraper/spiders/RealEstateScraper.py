import scrapy

class ListingSpider(scrapy.Spider):
    name = 'listings'

    def start_requests(self):
        base_ = 'https://www.realtor.com/realestateandhomes-search/Aurora_CO/pg-'
        pages = range(2,43)
        urls = ['https://www.homes.com/aurora-co/homes-for-sale/p1']

        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)
    
    def parse(self, response):
        page = response.url.split('/')[-1]
        filename = f'aurora_{page}.html'
        with open(filename, 'wb') as f:
            f.write(response.body)
        self.log(f'Saved file {filename}')
