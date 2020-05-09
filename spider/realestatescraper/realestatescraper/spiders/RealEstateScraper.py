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
        
        # start_urls = ['https://www.realtor.com/realestateandhomes-search/Aurora_CO/pg-1']

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
        '''
        List of Listings pages: extract metadata and send to file (json)
        '''
        url_page = response.url.split('-')[-1]
        url_city = response.url.split('/')[-2]

        base_xpath= "//ul[@data-testid='property-list-container']/li/div/div[2]/div[3]"

        href = response.xpath(f'{base_xpath}/a/@href').extract()
        prop_type = response.xpath(f'{base_xpath}/a/div/div[1]/div/span/text()').extract()
        price = response.xpath(f'{base_xpath}/a/div/div[2]/span/text()').extract()
        beds = response.xpath(f'{base_xpath}/div[1]/div/a/div[1]/div/ul/li[1]/span[1]/text()').extract()
        baths = response.xpath(f'{base_xpath}/div[1]/div/a/div[1]/div/ul/li[2]/span[1]/text()').extract()
        sqft = response.xpath(f'{base_xpath}/div[1]/div/a/div[1]/div/ul/li[3]/span[1]/text()').extract()
        lotsqft = response.xpath(f'{base_xpath}/div[1]/div/a/div[1]/div/ul/li[4]/span[1]/text()').extract()
        address = response.xpath(f'{base_xpath}/div[1]/div/a/div[2]/text()').extract()
        city = response.xpath(f'{base_xpath}/div[1]/div/a/div[2]/div/text()').extract()

        
        '''
        getting images from listings
        '''

        # for idx, listing in enumerate(href):
        #     # go back a page response.meta['driver'].execute_script("window.history.go(-1)")

        listing_zip = zip(href, prop_type, price, beds, baths, sqft, lotsqft, address, city)

        scraped_info = {'fields':['href', 'prop_type', 'price', 'beds', 'baths', 'sqft', 'lotsqft', 'address', 'city']}
        for listing, data in enumerate(listing_zip):
            key = f'{url_city}_pg{url_page}_listing{listing}'
            scraped_info[key] = data
        yield scraped_info

        '''
        Nav to next listing page to scrape more
        '''
        li_num = 9
        next_page_xpath=f"//div[@data-testid='srp-body']/section[1]/div[1]/ul/li[{li_num}]/a/@href"
        next_page = response.xpath(next_page_xpath).extract()
        while not bool(next_page):
            if li_num == 0:
                break
            li_num -= 1
            next_page_xpath=f"//div[@data-testid='srp-body']/section[1]/div[1]/ul/li[{li_num}]/a/@href"
            next_page = response.xpath(next_page_xpath).extract()

        if next_page:
            yield SeleniumRequest(url=response.urljoin(next_page[0]), callback=self.parse_result)

        # page = response.url.split('/')[-2]
        # filename = f'realtor_test_{page}.html'
        # with open(filename, 'wb') as f:
        #     f.write(response.body)
        # self.log(f'Saved file {filename}')


if __name__ == "__main__":
    '''
    Property Card Information:
    extract cmds with: response.xpath(<xpath>).extract()
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

    individual listing page xpaths:
        photos (60x60):         //div[@class='ldp-hero-carousel-wrap']/div[1]/div/div/img/@src
        description:            //div[@id='ldp-detail-overview']/div[2]/p/text()

    '''

