# rate_limiter.py

import time
import os

class RateLimiter:
    def __init__(self, limit_seconds):
        self.limit_seconds = limit_seconds
        self.last_request_time = None

    def can_make_request(self):
        if not self.last_request_time:
            return True
        time_since_last = time.time() - self.last_request_time
        return time_since_last >= self.limit_seconds

    def time_until_next_request(self):
        if not self.last_request_time:
            return 0
        time_since_last = time.time() - self.last_request_time
        return max(0, self.limit_seconds - time_since_last)

    def record_request(self):
        self.last_request_time = time.time()
