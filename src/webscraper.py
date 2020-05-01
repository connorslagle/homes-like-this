from selenium import webdriver
from wayback import WaybackClient, WaybackSession
from bs4 import BeautifulSoup
from pymongo import MongoClient



if __name__ == "__main__":
    live_url = 'https://www.realtor.com/realestateandhomes-search/Aurora_CO'

    client = WaybackClient()
    records = client.search(live_url)

