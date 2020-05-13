# homes-like-this
Predict where to search for homes based on input pictures.

# Project Outline

1. Motivating question

Can one optimize a home search by filtering listings based on images of rooms they already like?

2. Brief thesis on method to tackle question

In this project, I propose a machine learning algorithm... need to complete after body paragraphs

3. Description of Data

Data was collected from [Realtor.com](https://www.realtor.com/) via webscraping (Scrapy/Selenium); collecting listing images and metadata then storing the metadata in a NoSQL (MongoDB) database.

For a proof of concept, listings were scraped from municipalities around the Denver Metro Area with a [population > 100,000](https://en.wikipedia.org/wiki/Denver_metropolitan_area#Places_with_over_100,000_inhabitants). 


Municipality | Pop. (2018, est.) | Listings Avail. (5/13/2020) | Listings Scraped | Images Scraped
|---|---:|---:|---:|---:|
Denver | 727,000 | 3,700 | 30 | 700
Aurora | 374,000 | 1,900 | 30 | 700
Lakewood | 156,000 | 510 | 30 | 700
Thornton | 139,000 | 700 | 30 | 700
Arvada | 120,000 | 500 | 30 | 700
Westminster | 113,000 | 320 | 30 | 700
Centennial | 110,000 | 340 | 30 | 700
**Total** | **1,740,000** | **8,000** | **210** | **4,900**



## Capstone 2

Goals for Capstone 2: 
- Scalable webscraper (done)
- Data Cleaning Pipelines
- Featurize images (NB, or CNN) and hard cluster
- Label clusters, test with pictures of my house

## Capstone 3

Goals for Capstone 3: 
- Scrape more data (more cities/listings -> run on AWS)
- Combine image features with metadata -> predict where to look by 
- 

# Part 1:
