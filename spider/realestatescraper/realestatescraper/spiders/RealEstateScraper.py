import scrapy

class ListingSpider(scrapy.Spider):
    name = 'listings'

    def start_requests(self):
        urls = [
            'https://www.realtor.com/realestateandhomes-search/Aurora_CO/pg-1'
            # 'https://www.realtor.com/realestateandhomes-search/Denvver_CO',
            # 'https://www.realtor.com/realestateandhomes-search/Littleton_CO',
            # 'https://www.realtor.com/realestateandhomes-search/Thornton_CO',
            # 'https://www.realtor.com/realestateandhomes-search/Greenwood-Village_CO',
            # 'https://www.realtor.com/realestateandhomes-search/Boulder_CO',
            # 'https://www.realtor.com/realestateandhomes-search/Westminster_CO'
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)
    
    def parse(self, response):
        page = response.url.split('-')[-1]
        filename = f'aurora_{page}.html'
        with open(filename, 'wb') as f:
            f.write(response.body)
        self.log(f'Saved file {filename}')
