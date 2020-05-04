from selenium import webdriver
from wayback import WaybackClient, WaybackSession
from bs4 import BeautifulSoup
from pymongo import MongoClient



if __name__ == "__main__":
    live_url = 'https://www.realtor.com/realestateandhomes-search/Aurora_CO'

    '''
    Wayback Machine API - used to get old versions of webpage for scraping
    '''
    
    # client = WaybackClient()
    # records = client.search(live_url)

    '''
    Using Selenium to automate navigation on Realtor.com
    '''

    web_driver = webdriver.Firefox()

    
    '''
    Parse HTML with beautifulSoup
    '''
    # with open('../spider/realestatescraper/aurora_.html', 'r') as f:
    #     soup = BeautifulSoup(f, "html.parser")
