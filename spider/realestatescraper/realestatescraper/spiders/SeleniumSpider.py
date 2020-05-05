import scrapy
from selenium import webdriver


class ProductSpider(scrapy.Spider):
    name = "listing_spider"
    allowed_domains = ['realtor.com']
    start_urls = ['https://www.realtor.com/realestateandhomes-search/Aurora_CO/pg-1']

    def __init__(self):
        self.driver = webdriver.Firefox()

    def parse(self, response):
        self.driver.get(response.url)



        while True:

            page = response.url.split('-')[-1]
            filename = f'realtor_test_selenium_{page}.html'
            with open(filename, 'wb') as f:
                f.write(response.body)
            self.log(f'Saved file {filename}')

            next_pg = self.driver.find_element_by_css_selector('a.pagination-direction')

            try:
                next_pg.click()
                page = response.url.split('-')[-1]
                filename = f'realtor_test_selenium_{page}.html'
                with open(filename, 'wb') as f:
                    f.write(response.body)
                self.log(f'Saved file {filename}')
                # get the data and write it to scrapy items
            except:
                break

        self.driver.close()