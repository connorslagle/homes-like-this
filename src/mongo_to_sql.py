


# db imports
import pymongo

import psycopg2 as pg2

class mongoToSql():
    """
    Class for xfering mongo db listing data to SQL
    """

    def __init__(self):
        self.mongo_conn = pymongo.MongoClient('localhost', 27017)
        self.psql_conn = pg2.connect(dbname='db_name', user='postgres', host='localhost', port='5432')