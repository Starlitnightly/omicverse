# system_monitor.py

import psutil
import time
from datetime import timedelta
import os

class SystemMonitor:
    @staticmethod
    def get_system_stats():
        process = psutil.Process(os.getpid())
        memory = psutil.virtual_memory()
        return {
            'memory_usage': process.memory_info().rss / 1024 / 1024,  # MB
            'cpu_percent': psutil.cpu_percent(interval=1),
            'uptime': time.time() - psutil.boot_time(),
            'system_memory': {
                'total': memory.total / (1024 ** 3),        # GB
                'available': memory.available / (1024 ** 3),  # GB
                'percent': memory.percent
            }
        }

    @staticmethod
    def format_uptime(seconds):
        return str(timedelta(seconds=int(seconds)))
