import scrapy
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from scrapy_selenium import SeleniumRequest

class ListingCrawler(CrawlSpider):
    name = 'crawl-listings'
    

    def start_requests(self):
        allowed_domains = ['realtor.com']
        start_urls = ['https://www.realtor.com/realestateandhomes-search/Aurora_CO/pg-1']

        for url in start_urls:
            yield SeleniumRequest(url=url, callback=self.parse_result)
    
    def parse_result(self, response):
        '''
        Collect raw HTML of search page
        '''
        page = response.url.split('-')[-1]
        filename = f'realtor_aurora_pg{page}.html'
        with open(filename, 'wb') as f:
            f.write(response.body)
        self.log(f'Saved file {filename}')

        selenium_driver = response.request.meta['driver']


        '''
        Navigate to 
        '''
        


if __name__ == "__main__":
    possible_urls = ['https://www.realtor.com/realestateandhomes-search/Aurora_CO/pg-',
                    'https://www.redfin.com/city/30839/CO/Aurora/page-1',
                    'https://www.homes.com/aurora-co/homes-for-sale/p1',
                    'https://www.zillow.com/aurora-co/1_p']

    # CSS selector for next page, Realtor.com (a.pagination-direction)
    '''
    Property Card Information:
    listing xpaths:
        href:           //ul[@data-testid='property-list-container']/li/div/div[2]/div[3]/a/@href
        type:           //ul[@data-testid='property-list-container']/li/div/div[2]/div[3]/a/div/div[1]/div/span/text()
        price:          //ul[@data-testid='property-list-container']/li/div/div[2]/div[3]/a/div/div[2]/span/text()
        beds:           //ul[@data-testid='property-list-container']/li/div/div[2]/div[3]/div[1]/div/a/div[1]/div/ul/li[1]/span[1]/text()
        baths:          //ul[@data-testid='property-list-container']/li/div/div[2]/div[3]/div[1]/div/a/div[1]/div/ul/li[2]/span[1]/text()
        sq.ft:          //ul[@data-testid='property-list-container']/li/div/div[2]/div[3]/div[1]/div/a/div[1]/div/ul/li[3]/span[1]/text()
        lot:            //ul[@data-testid='property-list-container']/li/div/div[2]/div[3]/div[1]/div/a/div[1]/div/ul/li[4]/span[1]/text()
        address:        //ul[@data-testid='property-list-container']/li/div/div[2]/div[3]/div[1]/div/a/div[2]/text()
        city_state_zip: //ul[@data-testid='property-list-container']/li/div/div[2]/div[3]/div[1]/div/a/div[2]/div/text()


    location: //ul[@data-testid='property-list-container']/li/div/div[2]/div[1]
    scrapy command: response.xpath("//ul[@data-testid='property-list-container']/li/div/div[2]/div[1]").extract()
    html path for property: //ul[@data-testid='property-list-container']/li/div/div[2]/div[1]/a/@href
    scrapy cmd to follow href response.urljoin(href string)
    '''


    # //a[@rel="noopener"]/picture
    # when in listing
    # 1. click //div[@class="slick-list"]
    # 2. driver.execute_script("window.scrollTo({left:0, top:document.body.scrollHeight, behavior:'smooth'});")
    # 3. driver.execute_script("window.scrollTo({left:0, top:-(document.body.scrollHeight), behavior:'smooth'});")