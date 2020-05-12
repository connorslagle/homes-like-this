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
        
        # start_urls = ['https://www.realtor.com/realestateandhomes-search/Aurora_CO/pg-1']

        start_urls = ['https://www.realtor.com/realestateandhomes-search/Aurora_CO/pg-1',
                # 'https://www.realtor.com/realestateandhomes-search/Arvada_CO/pg-1',
                'https://www.realtor.com/realestateandhomes-search/Centennial_CO/pg-1',
                'https://www.realtor.com/realestateandhomes-search/Denver_CO/pg-1',
                'https://www.realtor.com/realestateandhomes-search/Lakewood_CO/pg-1',
                'https://www.realtor.com/realestateandhomes-search/Thornton_CO/pg-1',
                'https://www.realtor.com/realestateandhomes-search/Westminster_CO/pg-1']

        for url in start_urls:
            yield SeleniumRequest(url=url, callback=self.parse_result)
    
    def parse_result(self, response, metadata_item=None):
        if metadata_item == None:
            '''
            List of Listings pages: extract metadata and load to SearchPageItem container
            '''
            self.search_page_url = response.url
            self.url_page = response.url.split('-')[-1]
            self.url_city = response.url.split('/')[-2]

            l = ItemLoader(item=SearchPageItem(), response=response)

            base_xpath= "//ul[@data-testid='property-list-container']/li/div/div[2]/div[3]"
            self.href = response.xpath(f'{base_xpath}/a/@href').extract()
            

            for idx, listing in enumerate(self.href):
                l.add_value('listing_id',f'{self.url_city}_{self.url_page}_{idx}')
                l.add_value('listing_href', listing)

            l.add_value('search_url', self.search_page_url)
            l.add_xpath('prop_type', f'{base_xpath}/a/div/div[1]/div/span/text()')
            l.add_xpath('price', f'{base_xpath}/a/div/div[2]/span/text()')
            l.add_xpath('beds', f'{base_xpath}/div[1]/div/a/div[1]/div/ul/li[1]/span[1]/text()')
            l.add_xpath('baths', f'{base_xpath}/div[1]/div/a/div[1]/div/ul/li[2]/span[1]/text()')
            l.add_xpath('sqft', f'{base_xpath}/div[1]/div/a/div[1]/div/ul/li[3]/span[1]/text()')
            l.add_xpath('lotsqft', f'{base_xpath}/div[1]/div/a/div[1]/div/ul/li[4]/span[1]/text()')
            metadata_item = l.load_item()
            print(f'\nSending metadata from search page: City:{self.url_city}\t Page:{self.url_page}')
            print(f'href is {len(self.href)} long\n')
            yield metadata_item
            
            self.listing_counter = 0 
            yield SeleniumRequest(url=response.urljoin(self.href[0]), callback=self.parse_result, cb_kwargs={'metadata_item': metadata_item})
        
        if self.listing_counter <= len(self.href)-1:
            print(f'\nCollecting Listing Data: City:{self.url_city}\tPage:{self.url_page}\tListing#:{self.listing_counter}\n')
            '''
            getting images from listings
            '''
            img_l = ItemLoader(item=ListingItem(), response=response)

            listing_id = response.cb_kwargs.get('metadata_item').get('listing_id')[self.listing_counter]
            img_id = f'{listing_id}_{self.listing_counter}'

            aux_metadata = response.xpath("//div[@id='ldp-detail-overview']/div[1]/div/ul/li/div[@class]/text()").extract()
            if len(aux_metadata) != 0:
                img_l.add_value('prop_status', aux_metadata[0])
                img_l.add_value('price_sqft', aux_metadata[1])
                img_l.add_value('time_on_web', aux_metadata[2])
                img_l.add_value('prop_type', aux_metadata[3])
                img_l.add_value('year_built', aux_metadata[4])
                img_l.add_value('prop_style', aux_metadata[5])
            img_l.add_xpath('prop_desc', "//div[@id='ldp-detail-overview']/div[2]/p/text()")
            
            img_l.add_value('image_id', img_id)
            img_l.add_xpath( 'image_urls',
                "//section[@id='ldp-hero-container']/div/div/div[1]/div[1]/div/img/@data-src")
            yield img_l.load_item()

            self.listing_counter += 1

            if self.listing_counter < len(self.href):
                print(f'\nGoing to next Listing: City:{self.url_city}\tPage:{self.url_page}\tListing#:{self.listing_counter}\n')
                yield SeleniumRequest(url=response.urljoin(self.href[self.listing_counter]), callback=self.parse_result, cb_kwargs=response.cb_kwargs)
            else:
                yield SeleniumRequest(url=self.search_page_url, callback=self.parse_result, cb_kwargs={'metadata_item': response.cb_kwargs})
        elif self.listing_counter > len(self.href):
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
                print(f'\nGoing to next Page: City:{self.url_city}\tPage:{self.url_page}\n')
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
        prop details:           //div[@id='ldp-detail-overview']/div[1]/div/ul/li/div[@class]/text()
            ['status', 'price_sqft', 'time_on_web', 'type', 'built', 'style']


    '''

