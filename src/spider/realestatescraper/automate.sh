#!/bin/bash

# script for controlling scrape on AWS:
# Start up protocol:
# 	- change img dir in spider settings
# 	- run install script from src/aws_setup/
# 	- copy AWS creds to .bashrc
# 	- source ~/.bashrc
# 	- start mongo DB
# 	- run full scrape, all cities (this script)
# 	- screen out of terminal 
# 	- 'screen -S my-session' to start, ctrl-a,d to disconnect
# 	- 'screen -R my-session' to re-attach

docker start mongoserver
scrapy crawl large_metro
docker exec mongoserver sh -c 'exec mongodump --db listings --gzip --archive' > dump_`date "+%Y-%m-%d"`.gz


