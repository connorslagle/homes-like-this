import scrapy
from scrapy.loader import ItemLoader
from scrapy.loader.processors import Compose
from scrapy_selenium import SeleniumRequest
from ..items import ListingItem, SearchPageItem

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
        
        start_urls = ['https://www.realtor.com/realestateandhomes-search/Arvada_CO/pg-1']

        # start_urls = ['https://www.realtor.com/realestateandhomes-search/Aurora_CO/pg-1',
        #         'https://www.realtor.com/realestateandhomes-search/Arvada_CO/pg-1',
        #         'https://www.realtor.com/realestateandhomes-search/Centennial_CO/pg-1',
        #         'https://www.realtor.com/realestateandhomes-search/Denver_CO/pg-1',
        #         'https://www.realtor.com/realestateandhomes-search/Lakewood_CO/pg-1',
        #         'https://www.realtor.com/realestateandhomes-search/Thornton_CO/pg-1',
        #         'https://www.realtor.com/realestateandhomes-search/Westminster_CO/pg-1']

        for url in start_urls:
            yield SeleniumRequest(url=url, callback=self.parse_result)
    
    def parse_result(self, response, metadata_item=None):
        if metadata_item == None:
            '''
            List of Listings pages: extract metadata and load to SearchPageItem container
            '''
            self.search_page_url = response.url
            url_page = response.url.split('-')[-1]
            url_city = response.url.split('/')[-2]

            base_xpath= "//ul[@data-testid='property-list-container']/li/div/div[2]/div[3]"

            self.href = response.xpath(f'{base_xpath}/a/@href').extract()
            l = ItemLoader(item=SearchPageItem(), response=response)

            for idx, listing in enumerate(self.href):
                l.add_value('listing_id',f'{url_city}_{url_page}_{idx}')
                l.add_value('search_url', self.search_page_url)
                l.add_value('search_city', url_city)
                l.add_value('search_page', url_page)

                idx_elem = Compose(lambda x: x[idx])

                l.add_value('listing_href', listing)
                l.add_xpath('prop_type', f'{base_xpath}/a/div/div[1]/div/span/text()', idx_elem)
                l.add_xpath('price', f'{base_xpath}/a/div/div[2]/span/text()', idx_elem)
                l.add_xpath('beds', f'{base_xpath}/div[1]/div/a/div[1]/div/ul/li[1]/span[1]/text()', idx_elem)
                l.add_xpath('baths', f'{base_xpath}/div[1]/div/a/div[1]/div/ul/li[2]/span[1]/text()', idx_elem)
                l.add_xpath('sqft', f'{base_xpath}/div[1]/div/a/div[1]/div/ul/li[3]/span[1]/text()', idx_elem)
                l.add_xpath('lotsqft', f'{base_xpath}/div[1]/div/a/div[1]/div/ul/li[4]/span[1]/text()', idx_elem)
            self.metadata_item = l.load_item()
            yield self.metadata_item
            
            self.listing_counter = 0 
            yield SeleniumRequest(url=response.urljoin(self.href[0]), callback=self.parse_result, cb_kwargs={'metadata_item': 1})
        
        if self.listing_counter <= len(self.href):
            '''
            getting images from listings
            '''
            img_l = ItemLoader(item=ListingItem(), response=response)
            listing_id = self.metadata_item.get('listing_id')[self.listing_counter]
            img_id = f'{listing_id}_{self.listing_counter}'

            img_l.add_value('image_id', img_id)
            img_l.add_xpath( 'image_urls',
                "//section[@id='ldp-hero-container']/div/div/div[1]/div[1]/div/img/@data-src")
            yield img_l.load_item()

            self.listing_counter += 1
            yield SeleniumRequest(url=response.urljoin(self.href[self.listing_counter]), callback=self.parse_result, cb_kwargs={'metadata_item': 1})
        
        else:
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
        prop details:           //div[@id='ldp-detail-overview']/div[1]/div/ul/div[1]/div/div/li/div[2]/text()
            ['status', 'price_sqft', 'time_on_web', 'type', 'built', 'style']


    '''

