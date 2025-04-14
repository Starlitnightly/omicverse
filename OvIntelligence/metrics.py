# metrics.py

from prometheus_client import Counter, Histogram, Gauge
from typing import Dict
from dataclasses import dataclass, field
import os
import psutil
import os


@dataclass
class PerformanceMetrics:
    _instance: 'PerformanceMetrics' = field(default=None, init=False, repr=False)

    # Prometheus Metrics
    query_counter: Counter = field(init=False)
    query_latency: Histogram = field(init=False)
    cache_hits: Counter = field(init=False)
    model_calls: Dict[str, Counter] = field(default_factory=dict, init=False)
    memory_usage: Gauge = field(init=False)
    request_duration: Histogram = field(init=False)

    def __post_init__(self):
        if PerformanceMetrics._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            # Initialize Prometheus Metrics
            self.query_counter = Counter('rag_queries_total', 'Total number of queries processed')
            self.query_latency = Histogram('rag_query_duration_seconds', 'Query processing duration')
            self.cache_hits = Counter('rag_cache_hits_total', 'Number of cache hits')
            self.memory_usage = Gauge('rag_memory_usage_bytes', 'Memory usage in bytes')
            self.request_duration = Histogram(
                'rag_request_duration_seconds',
                'Request duration in seconds',
                buckets=(0.1, 0.5, 1.0, 2.0, 5.0)
            )
            PerformanceMetrics._instance = self

    @staticmethod
    def get_instance():
        if PerformanceMetrics._instance is None:
            PerformanceMetrics()
        return PerformanceMetrics._instance

    # Methods to record metrics
    def record_query(self, duration: float):
        self.query_counter.inc()
        self.query_latency.observe(duration)

    def record_cache_hit(self):
        self.cache_hits.inc()

    def record_model_call(self, model_name: str):
        sanitized_name = model_name.replace('.', '_').replace(':', '_').replace('-', '_')
        metric_name = f'rag_model_calls_{sanitized_name}'

        if model_name not in self.model_calls:
            self.model_calls[model_name] = Counter(
                metric_name,
                f'Number of calls to model {model_name}'
            )
        self.model_calls[model_name].inc()

    def record_memory_usage(self):
        process = psutil.Process(os.getpid())
        self.memory_usage.set(process.memory_info().rss)

    def record_request_time(self, duration: float):
        self.request_duration.observe(duration)
