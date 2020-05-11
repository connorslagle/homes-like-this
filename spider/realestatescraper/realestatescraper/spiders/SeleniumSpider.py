import scrapy
from scrapy_selenium import SeleniumRequest
from ..items import RealestatescraperItem

class ProductSpider(scrapy.Spider):
    name = "test_spider"
    
    
    def start_request(self):
        allowed_domains = ['realtor.com']
        start_urls = ['https://www.realtor.com/realestateandhomes-search/Boston_MA/pg-1']
        for url in start_urls:
            yield SeleniumRequest(url=url, callback=self.parse_result)

    def parse_result(self, response):
        item = RealestatescraperItem()
        item['image_urls'] = ['https://ap.rdcpix.com/f2d3d8da16bfc6272857bb8432fd0821l-m4195757090xd-r3_w60_h60_q80.jpg',
                'https://ap.rdcpix.com/f2d3d8da16bfc6272857bb8432fd0821l-m1553618064xd-r3_w60_h60_q80.jpg',
                'https://ap.rdcpix.com/f2d3d8da16bfc6272857bb8432fd0821l-m3390361382xd-r3_w60_h60_q80.jpg',
                'https://ap.rdcpix.com/f2d3d8da16bfc6272857bb8432fd0821l-m694287565xd-r3_w60_h60_q80.jpg',
                'https://ap.rdcpix.com/f2d3d8da16bfc6272857bb8432fd0821l-m314216804xd-r3_w60_h60_q80.jpg',
                'https://ap.rdcpix.com/f2d3d8da16bfc6272857bb8432fd0821l-m4168513700xd-r3_w60_h60_q80.jpg',
                'https://ap.rdcpix.com/f2d3d8da16bfc6272857bb8432fd0821l-m254909824xd-r3_w60_h60_q80.jpg',
                'https://ap.rdcpix.com/f2d3d8da16bfc6272857bb8432fd0821l-m2850122435xd-r3_w60_h60_q80.jpg',
                'https://ap.rdcpix.com/f2d3d8da16bfc6272857bb8432fd0821l-m4293566320xd-r3_w60_h60_q80.jpg',
                'https://ap.rdcpix.com/f2d3d8da16bfc6272857bb8432fd0821l-m3828111939xd-r3_w60_h60_q80.jpg',
                'https://ap.rdcpix.com/f2d3d8da16bfc6272857bb8432fd0821l-m87540920xd-r3_w60_h60_q80.jpg',
                'https://ap.rdcpix.com/f2d3d8da16bfc6272857bb8432fd0821l-m2740238186xd-r3_w60_h60_q80.jpg',
                'https://ap.rdcpix.com/f2d3d8da16bfc6272857bb8432fd0821l-m3177530126xd-r3_w60_h60_q80.jpg',
                'https://ap.rdcpix.com/f2d3d8da16bfc6272857bb8432fd0821l-m3699680085xd-r3_w60_h60_q80.jpg',
                'https://ap.rdcpix.com/f2d3d8da16bfc6272857bb8432fd0821l-m2105159420xd-r3_w60_h60_q80.jpg',
                'https://ap.rdcpix.com/f2d3d8da16bfc6272857bb8432fd0821l-m2574086931xd-r3_w60_h60_q80.jpg',
                'https://ap.rdcpix.com/f2d3d8da16bfc6272857bb8432fd0821l-m94802870xd-r3_w60_h60_q80.jpg',
                'https://ap.rdcpix.com/f2d3d8da16bfc6272857bb8432fd0821l-m4021695627xd-r3_w60_h60_q80.jpg',
                'https://ap.rdcpix.com/f2d3d8da16bfc6272857bb8432fd0821l-m80416574xd-r3_w60_h60_q80.jpg',
                'https://ap.rdcpix.com/f2d3d8da16bfc6272857bb8432fd0821l-m1113940022xd-r3_w60_h60_q80.jpg',
                'https://ap.rdcpix.com/f2d3d8da16bfc6272857bb8432fd0821l-m1842568115xd-r3_w60_h60_q80.jpg',
                'https://ap.rdcpix.com/f2d3d8da16bfc6272857bb8432fd0821l-m1254826498xd-r3_w60_h60_q80.jpg',
                'https://ap.rdcpix.com/f2d3d8da16bfc6272857bb8432fd0821l-m3178044250xd-r3_w60_h60_q80.jpg',
                'https://ap.rdcpix.com/f2d3d8da16bfc6272857bb8432fd0821l-m782603140xd-r3_w60_h60_q80.jpg',
                'https://ap.rdcpix.com/f2d3d8da16bfc6272857bb8432fd0821l-m2994640551xd-r3_w60_h60_q80.jpg',
                'https://ap.rdcpix.com/f2d3d8da16bfc6272857bb8432fd0821l-m2433103184xd-r3_w60_h60_q80.jpg',
                'https://ap.rdcpix.com/f2d3d8da16bfc6272857bb8432fd0821l-m4286653922xd-r3_w60_h60_q80.jpg',
                'https://ap.rdcpix.com/f2d3d8da16bfc6272857bb8432fd0821l-m2332058624xd-r3_w60_h60_q80.jpg',
                'https://ap.rdcpix.com/f2d3d8da16bfc6272857bb8432fd0821l-m1478366693xd-r3_w60_h60_q80.jpg',
                'https://ap.rdcpix.com/f2d3d8da16bfc6272857bb8432fd0821l-m2992851727xd-r3_w60_h60_q80.jpg',
                'https://ap.rdcpix.com/f2d3d8da16bfc6272857bb8432fd0821l-m863422880xd-r3_w60_h60_q80.jpg',
                'https://ap.rdcpix.com/f2d3d8da16bfc6272857bb8432fd0821l-m319271888xd-r3_w60_h60_q80.jpg']
        yield item