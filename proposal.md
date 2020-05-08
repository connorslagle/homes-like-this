# Capstone II Proposal
## Homes Like This - Property Search Optimizer
Finding your dream home is hard. Even after deciding on the number of bedrooms, number of bathrooms, type of cooling system, etc., we have to sift through countless online photos to select homes we 'could see ourselves in'. Well, what if you could automate the second part? What if you could filter homes based on pictures of rooms you can already 'see yourself in'?

I think this would be pretty useful. To acomplish this, I plan to use the following data:

- Webscraped property listing images and meta data (location, listing price, # beds, # baths, etc.) 
    - Initially for the largest 7 cities around the Denver Metro
- LSUN Database images of house rooms

With this data, I plan to build a recommender that recommends where to look for homes based on current listings and a user input photo of a room they like. I plan to do this with the following steps:

- Featurize property listing photos (NB Classifier and/or CNN Autoencoder + K-means Clustering)
- Combine photo data with listing meta data
- Recommend location with highest concentration (listings/zip code) of homes with greatest similarity to innput photo
- Deploy in web application

## Preliminary EDA
Decided on Realtor.com as webpage to scrape - this website updates frequently with MLS and shows ~ 40 listings per page (reducing req. # of website requests). The main page gives metadata for all listings - each listing page gives ~15-20 photos (size 350x350). Scraper (using Scrapy) has sucessfully scraped main page - updating to collect images.

## Minimum Viable Product

**MVP:** Featurize images, combine photo data with listing metadata, train/test recommender for 7 major cities around Denver, CO. Deploy in Web Application (providing top X zipcodes to search)

**MVP+:** Plot listings on geomap

**MVP++:** Apply to other metros.