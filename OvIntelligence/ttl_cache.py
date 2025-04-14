# ttl_cache.py

import time
from collections import OrderedDict
import os

class TTLCache(OrderedDict):
    def __init__(self, maxsize=1000, ttl=3600):
        super().__init__()
        self.maxsize = maxsize
        self.ttl = ttl

    def __getitem__(self, key):
        value, timestamp = super().__getitem__(key)
        if time.time() - timestamp > self.ttl:
            del self[key]
            raise KeyError(key)
        return value

    def __setitem__(self, key, value):
        super().__setitem__(key, (value, time.time()))
        if len(self) > self.maxsize:
            self.popitem(last=False)