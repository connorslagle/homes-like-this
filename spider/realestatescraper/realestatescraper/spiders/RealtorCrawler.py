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
    # //a[@rel="noopener"]/picture
    # when in listing
    # 1. click //div[@class="slick-list"]
    # 2. driver.execute_script("window.scrollTo({left:0, top:document.body.scrollHeight, behavior:'smooth'});")
    # 3. driver.execute_script("window.scrollTo({left:0, top:-(document.body.scrollHeight), behavior:'smooth'});")