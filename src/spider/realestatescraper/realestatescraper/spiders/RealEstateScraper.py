import scrapy
from scrapy.loader import ItemLoader
from scrapy.loader.processors import Compose
from scrapy_selenium import SeleniumRequest
from ..items import ListingItem, SearchPageItem

class ListingSpider(scrapy.Spider):
    '''
    Main spider - used to collect imgs and metadata from Realtor.com

    To use: 
        - Make sure MongoDB Docker container running - metadata stored there
        - Navigate to /homes-like-this/spider/realestatescraper dir in bash
        - type: scrapy crawl large_metro

    Will take a while, with delays can crawl @ 10 imgs/min

    Note:
        - When stopping spider, in bash ctrl-C once and LET SPIDER CLOSE, not fast,
            but otherwise Selenium reqs still in memory, build up overtime
    '''

    name = 'large_metro'
    
    def start_requests(self):
        allowed_domains = ['realtor.com']
        '''
        Start URLs for the 7 largest cities in Denver Metro (> 100,000 inhabitants)
        Aurora, Thornton, Arvada, Centennial, Denver, Lakewood, Westminster
        '''
        
        # start_urls = ['https://www.realtor.com/realestateandhomes-search/Aurora_CO/pg-1']
        start_urls = [
                'https://www.realtor.com/realestateandhomes-search/Aurora_CO/pg-3',
                'https://www.realtor.com/realestateandhomes-search/Arvada_CO/pg-3',
                'https://www.realtor.com/realestateandhomes-search/Centennial_CO/pg-3',
                'https://www.realtor.com/realestateandhomes-search/Denver_CO/pg-3',
                'https://www.realtor.com/realestateandhomes-search/Lakewood_CO/pg-3',
                'https://www.realtor.com/realestateandhomes-search/Thornton_CO/pg-3',
                'https://www.realtor.com/realestateandhomes-search/Westminster_CO/pg-3']

        for url in start_urls:
            yield SeleniumRequest(url=url, callback=self.parse_result)
    
    def parse_result(self, response, metadata_item=None, listing_counter=0):
        '''
        This method is called after a request from Slenium, returns the 'response' as well as
        other kw_args passed from previous loop

        input: response from SeleniumRequest, metadata_item and listing_counter (from cb_kwargs)
                for entering 'listing' if statement
        
        output: scraped data (via yield) and next SeleniumRequest
        '''

        if metadata_item == None:
            '''
            List of Listings pages: extract metadata and load to SearchPageItem container
            '''

            url_page = response.url.split('-')[-1]
            url_city = response.url.split('/')[-2]

            l = ItemLoader(item=SearchPageItem(), response=response)

            base_xpath= "//ul[@data-testid='property-list-container']/li/div/div[2]/div[3]"
            href = response.xpath(f'{base_xpath}/a/@href').extract()
            
            # load data to 'item' - container used to process data in pipeline
            for idx, listing in enumerate(href):
                l.add_value('listing_id',f'{url_city}_{url_page}_{idx}')
                l.add_value('listing_href', listing)

            l.add_value('search_url', response.url)
            l.add_xpath('prop_type', f'{base_xpath}/a/div/div[1]/div/span/text()')
            l.add_xpath('price', f'{base_xpath}/a/div/div[2]/span/text()')
            l.add_xpath('beds', f'{base_xpath}/div[1]/div/a/div[1]/div/ul/li[1]/span[1]/text()')
            l.add_xpath('baths', f'{base_xpath}/div[1]/div/a/div[1]/div/ul/li[2]/span[1]/text()')
            l.add_xpath('sqft', f'{base_xpath}/div[1]/div/a/div[1]/div/ul/li[3]/span[1]/text()')
            l.add_xpath('lotsqft', f'{base_xpath}/div[1]/div/a/div[1]/div/ul/li[4]/span[1]/text()')
            metadata_item = l.load_item()
            yield metadata_item
            
            # increase counter to signal entering next if statement on callback
            listing_counter += 1 
            yield SeleniumRequest(url=response.urljoin(metadata_item['listing_href'][0]), callback=self.parse_result, 
                    cb_kwargs={'metadata_item': metadata_item, 'listing_counter': listing_counter})
        
        if listing_counter <= len(metadata_item['listing_href'])-1:
            '''
            Scrapes images and auxilliary metatdata from listing webpage. Repeats until all listings on
            search page scraped.
            '''

            img_l = ItemLoader(item=ListingItem(), response=response)

            listing_id = metadata_item['listing_id'][listing_counter-1]
            img_id = f'{listing_id}'

            img_l.add_xpath('aux_metadata',"//div[@id='ldp-detail-overview']/div[1]/div/ul/li/div[@class]/text()")
            img_l.add_xpath('prop_desc', "//div[@id='ldp-detail-overview']/div[2]/p/text()")
            
            img_l.add_value('image_id', img_id)
            img_l.add_xpath( 'image_urls',
                "//section[@id='ldp-hero-container']/div/div/div[1]/div[1]/div/img/@data-src")
            yield img_l.load_item()

            listing_counter += 1
            if listing_counter < len(metadata_item['listing_href']):
                yield SeleniumRequest(url=response.urljoin(metadata_item['listing_href'][listing_counter]), callback=self.parse_result, 
                    cb_kwargs={'metadata_item': metadata_item, 'listing_counter': listing_counter})
            else:
                yield SeleniumRequest(url=metadata_item['search_url'][0], callback=self.parse_result, 
                    cb_kwargs={'metadata_item': metadata_item, 'listing_counter': listing_counter})
                    
        elif listing_counter > len(metadata_item['listing_href']):
            '''
            Moves to next search page. Restarts the scrape by not passing 'listing_counter'.
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

            if bool(next_page):
                yield SeleniumRequest(url=response.urljoin(next_page[0]), callback=self.parse_result)



if __name__ == "__main__":
    '''
    Usefull html xpaths for extracted data:

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
        prop details:           //div[@id='ldp-detail-overview']/div[1]/div/ul/li/div[@class]/text()
            ['status', 'price_sqft', 'time_on_web', 'type', 'built', 'style']

    Shell cmds:
        from scrapy_selenium import SeleniumRequest
        fetch(SeleniumRequest(url='https://www.realtor.com/realestateandhomes-search/Aurora_CO/pg-1'))

    '''

