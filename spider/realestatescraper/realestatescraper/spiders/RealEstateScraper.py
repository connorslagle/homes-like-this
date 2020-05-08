import scrapy
from scrapy_selenium import SeleniumRequest

class ListingSpider(scrapy.Spider):
    name = 'large_metro'
    
    def start_requests(self):
        allowed_domains = ['realtor.com']
        '''
        Start URLs for the 7 largest cities in Denver Metro (> 100,000 inhabitants)
        Aurora, Thornton, Arvada, Centennial, Denver, Lakewood, Westminster

        Medium (10,000 - 100,000 inhabitants): 
        Berkley, Brighton, Broomfield, Castle Rock, Columbine (CDP), Commerce City, 
        Englewood, Federal Heights, Golden, Greenwood Village, Highlands Ranch (CDP), 
        Ken Caryl (CDP), Littleton, Northglenn, Parker, Sherrelwood (CDP)
        '''
        
        start_urls = ['https://www.realtor.com/realestateandhomes-search/Aurora_CO/pg-1',
                      'https://www.realtor.com/realestateandhomes-search/Arvada_CO/pg-1',
                      'https://www.realtor.com/realestateandhomes-search/Centennial_CO/pg-1',
                      'https://www.realtor.com/realestateandhomes-search/Denver_CO/pg-1',
                      'https://www.realtor.com/realestateandhomes-search/Lakewood_CO/pg-1',
                      'https://www.realtor.com/realestateandhomes-search/Thornton_CO/pg-1',
                      'https://www.realtor.com/realestateandhomes-search/Westminster_CO/pg-1']

        for url in start_urls:
            yield SeleniumRequest(url=url, callback=self.parse_result)
    
    def parse_result(self, response):
        page = response.url.split('/')[-2]
        filename = f'realtor_test_{page}.html'
        with open(filename, 'wb') as f:
            f.write(response.body)
        self.log(f'Saved file {filename}')


if __name__ == "__main__":
    possible_urls = ['https://www.realtor.com/realestateandhomes-search/Aurora_CO/pg-',
                    'https://www.redfin.com/city/30839/CO/Aurora/page-1',
                    'https://www.homes.com/aurora-co/homes-for-sale/p1',
                    'https://www.zillow.com/aurora-co/1_p']

    # CSS selector for next page, Realtor.com (a.pagination-direction)