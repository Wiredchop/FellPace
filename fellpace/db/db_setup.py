"""This module loads the database file into a sqlite3 connection object and adds useful functions to the connection object."""
import math
import sqlite3
from pathlib import Path


class XPercentile:
    percentile = 0.2
    def __init__(self):
        self.values = []

    def step(self, value):
        self.values.append(value)

    def finalize(self):
        self.values.sort()
        upper_quartile_index = round(self.percentile * (len(self.values)))-1
        return self.values[upper_quartile_index]

class std_dev:
    def __init__(self):
        self.n = 0
        self.sum = 0
        self.sq = 0

    def step(self, value):
        self.n += 1
        self.sum += value
        self.sq += value * value

    def finalize(self):
        return math.sqrt(self.sq/self.n - (self.sum/self.n * self.sum/self.n))

def setup_db(path_to_db_file: Path) -> sqlite3.Connection:
    """return a connection to the fellpace DB"""
    
    con = sqlite3.connect(path_to_db_file)
    con.create_aggregate("XPercentile", 1, XPercentile)
    con.create_aggregate('stddev', 1, std_dev)
    return con