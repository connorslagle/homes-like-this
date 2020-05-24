# -*- coding: utf-8 -*-
# Scrapy settings for realestatescraper project
#
# For simplicity, this file contains only settings considered important or
# commonly used. You can find more settings consulting the documentation:
#
#     https://docs.scrapy.org/en/latest/topics/settings.html
#     https://docs.scrapy.org/en/latest/topics/downloader-middleware.html
#     https://docs.scrapy.org/en/latest/topics/spider-middleware.html

from shutil import which

# Log identifier
BOT_NAME = 'realestatescraper'
SPIDER_MODULES = ['realestatescraper.spiders']
NEWSPIDER_MODULE = 'realestatescraper.spiders'

# Configure a delay for requests for the same website (default: 0)
# See https://docs.scrapy.org/en/latest/topics/settings.html#download-delay
# See also autothrottle settings and docs
DOWNLOAD_DELAY = 20
# The download delay setting will honor only one of:
CONCURRENT_REQUESTS_PER_DOMAIN = 1
CONCURRENT_REQUESTS_PER_IP = 1

# Enable and configure the AutoThrottle extension (disabled by default)
# See https://docs.scrapy.org/en/latest/topics/autothrottle.html
AUTOTHROTTLE_ENABLED = True
# # The initial download delay
AUTOTHROTTLE_START_DELAY = 30
# # The maximum download delay to be set in case of high latencies
AUTOTHROTTLE_MAX_DELAY = 60
# # The average number of requests Scrapy should be sending in parallel to
# # each remote server
AUTOTHROTTLE_TARGET_CONCURRENCY = 1
# # Enable showing throttling stats for every response received:
AUTOTHROTTLE_DEBUG = True

# # Enable or disable downloader middlewares
# # See https://docs.scrapy.org/en/latest/topics/downloader-middleware.html
DOWNLOADER_MIDDLEWARES = {
    'realestatescraper.middlewares.RealestatescraperDownloaderMiddleware': 543,
    'scrapy_selenium.SeleniumMiddleware': 600,
    'scrapy_proxy_pool.middlewares.ProxyPoolMiddleware': None,
    'scrapy_proxy_pool.middlewares.BanDetectionMiddleware': None
}

# Configure item pipelines
# See https://docs.scrapy.org/en/latest/topics/item-pipeline.html
ITEM_PIPELINES = {
   'realestatescraper.pipelines.MyImagesPipeline': 400,
   'realestatescraper.pipelines.MetadataPipeline': 300,
}

# MEDIA_ALLOW_REDIRECTS = True

# PROXY_POOL_ENABLED = True

# # image pipeline settings
# IMAGES_STORE = '/home/conslag/Documents/galvanize/capstones/homes-like-this/data/listing_images'

# make thumbnail images
IMAGES_THUMBS = {
    'small': (64, 64),
}

# Scrapy-selenium
SELENIUM_DRIVER_NAME = 'firefox'
SELENIUM_DRIVER_EXECUTABLE_PATH = which('geckodriver')
SELENIUM_DRIVER_ARGUMENTS=['-headless']  # '--headless' if using chrome instead of firefox

# S3 storage alternative
IMAGES_STORE = 's3://homes-like-this/listing_images'


'''
Unused Options
'''
ROBOTSTXT_OBEY = True



# data storage
# FEEDS = {
#     '/home/conslag/Documents/galvanize/capstones/homes-like-this/data/%(name)s/%(time)s.jsonl' : {
#         'format': 'jsonlines',
#         'encoding': 'utf8',
#         'store_empty': False,
#     }
# }

# Crawl responsibly by identifying yourself (and your website) on the user-agent
USER_AGENT = 'BingPreview'

# Configure maximum concurrent requests performed by Scrapy (default: 16)
#CONCURRENT_REQUESTS = 32

# Disable cookies (enabled by default)
# COOKIES_ENABLED = False

# Disable Telnet Console (enabled by default)
# TELNETCONSOLE_ENABLED = False

# Override the default request headers:
# DEFAULT_REQUEST_HEADERS = {
#   'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
#   'Accept-Language': 'en',
#}

# Enable or disable spider middlewares
# See https://docs.scrapy.org/en/latest/topics/spider-middleware.html
# SPIDER_MIDDLEWARES = {
#    'realestatescraper.middlewares.RealestatescraperSpiderMiddleware': 543,
#}

# Enable or disable extensions
# See https://docs.scrapy.org/en/latest/topics/extensions.html
# EXTENSIONS = {
#    'scrapy.extensions.telnet.TelnetConsole': None,
#}

# Enable and configure HTTP caching (disabled by default)
# See https://docs.scrapy.org/en/latest/topics/downloader-middleware.html#httpcache-middleware-settings
#HTTPCACHE_ENABLED = True
#HTTPCACHE_EXPIRATION_SECS = 0
#HTTPCACHE_DIR = 'httpcache'
#HTTPCACHE_IGNORE_HTTP_CODES = []
#HTTPCACHE_STORAGE = 'scrapy.extensions.httpcache.FilesystemCacheStorage'
