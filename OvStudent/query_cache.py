# query_cache.py

from collections import OrderedDict
from ttl_cache import TTLCache
import os

class QueryCache(TTLCache):
    def __init__(self, maxsize=1000, ttl=3600):
        super().__init__(maxsize=maxsize, ttl=ttl)
