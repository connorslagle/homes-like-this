#!/bin/bash

# script to install all reqs for scraping in AWS

# Ubuntu
cd ~
# anaconda

# python stuff
pip install pymongo && pip install pandas && pip install numpy && pip install PIL && install boto3 && install botocore
# scrapy
pip install Scrapy && pip install scrapy-selenium
# docker stuff
docker run --name mongoserver -p 27017:27017 -v "$PWD":/home/data -d mongo

# resource .bashrc
source ~/.bashrc


