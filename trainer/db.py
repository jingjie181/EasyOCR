import configparser
import logging
import numpy as np
import pandas as pd
import psycopg2
from psycopg2 import pool

config = configparser.ConfigParser()
config.read("config.ini", encoding="utf-8")


class PoolAgent:
    def __init__(self, settings):
        self.settings = settings
        self.postgreSQL_pool = psycopg2.pool.SimpleConnectionPool(
            minconn=1, maxconn=2, **self.settings
        )

    def query(self, text):
        res = None
        while res is None:
            try:
                if self.postgreSQL_pool:
                    logging.info("Connection pool created successfully")

                # Use getconn() to Get Connection from connection pool
                ps_connection = self.postgreSQL_pool.getconn()

                if ps_connection:
                    logging.info("successfully recived connection from connection pool ")
                    res = pd.read_sql_query(text, ps_connection)
                    # Use this method to release the connection object and send back to connection pool
                    self.postgreSQL_pool.putconn(ps_connection)
                    logging.info("Put away a PostgreSQL connection")
            except (Exception, psycopg2.DatabaseError) as error:
                logging.error(f"Error while connecting to PostgreSQL. Retrying.")
                pass
        return res



aimbox = PoolAgent(settings=config['AIMBOX'])
aimbox_rds = PoolAgent(settings=config["AIMBOX-RDS"])
