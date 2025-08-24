"""Base API client with common functionality for all bioinformatics databases."""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
from urllib.parse import urljoin

import httpx
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..config import settings


logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple rate limiter implementation."""
    
    def __init__(self, calls_per_second: float):
        self.calls_per_second = calls_per_second
        self.min_interval = 1.0 / calls_per_second
        self.last_call = 0.0
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire permission to make a request."""
        async with self._lock:
            current_time = time.time()
            time_since_last_call = current_time - self.last_call
            
            if time_since_last_call < self.min_interval:
                sleep_time = self.min_interval - time_since_last_call
                await asyncio.sleep(sleep_time)
            
            self.last_call = time.time()
    
    def acquire_sync(self):
        """Synchronous version of acquire."""
        current_time = time.time()
        time_since_last_call = current_time - self.last_call
        
        if time_since_last_call < self.min_interval:
            sleep_time = self.min_interval - time_since_last_call
            time.sleep(sleep_time)
        
        self.last_call = time.time()


class BaseAPIClient(ABC):
    """Base class for all API clients."""
    
    def __init__(
        self,
        base_url: str,
        rate_limit: Optional[float] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.rate_limit = rate_limit or settings.api.default_rate_limit
        self.timeout = timeout or settings.api.default_timeout
        self.max_retries = max_retries or settings.api.max_retries
        
        # Rate limiter
        self.rate_limiter = RateLimiter(self.rate_limit)
        
        # Session for synchronous requests
        self.session = self._create_session()
        
        # Client for async requests
        self._async_client: Optional[httpx.AsyncClient] = None
    
    def _create_session(self) -> requests.Session:
        """Create a requests session with retry logic."""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST", "PUT", "DELETE"],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set default headers
        session.headers.update(self.get_default_headers())
        
        return session
    
    @property
    async def async_client(self) -> httpx.AsyncClient:
        """Get or create async client."""
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(
                timeout=self.timeout,
                headers=self.get_default_headers(),
            )
        return self._async_client
    
    @abstractmethod
    def get_default_headers(self) -> Dict[str, str]:
        """Get default headers for requests."""
        return {
            "User-Agent": "BioinformaticsDataCollector/0.1.0",
            "Accept": "application/json",
        }
    
    def build_url(self, endpoint: str) -> str:
        """Build full URL from endpoint."""
        if endpoint.startswith("http"):
            return endpoint
        return urljoin(self.base_url + "/", endpoint.lstrip("/"))
    
    def request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> requests.Response:
        """Make a synchronous HTTP request."""
        self.rate_limiter.acquire_sync()
        
        url = self.build_url(endpoint)
        
        # Merge headers
        request_headers = self.get_default_headers()
        if headers:
            request_headers.update(headers)
        
        logger.debug(f"{method} {url}")
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                json=json,
                headers=request_headers,
                timeout=self.timeout,
                **kwargs,
            )
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise
    
    async def request_async(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> httpx.Response:
        """Make an asynchronous HTTP request."""
        await self.rate_limiter.acquire()
        
        url = self.build_url(endpoint)
        
        # Merge headers
        request_headers = self.get_default_headers()
        if headers:
            request_headers.update(headers)
        
        logger.debug(f"{method} {url}")
        
        client = await self.async_client
        
        try:
            response = await client.request(
                method=method,
                url=url,
                params=params,
                json=json,
                headers=request_headers,
                **kwargs,
            )
            response.raise_for_status()
            return response
        except httpx.RequestException as e:
            logger.error(f"Async request failed: {e}")
            raise
    
    def get(self, endpoint: str, **kwargs) -> requests.Response:
        """Make a GET request."""
        return self.request("GET", endpoint, **kwargs)
    
    async def get_async(self, endpoint: str, **kwargs) -> httpx.Response:
        """Make an async GET request."""
        return await self.request_async("GET", endpoint, **kwargs)
    
    def post(self, endpoint: str, **kwargs) -> requests.Response:
        """Make a POST request."""
        return self.request("POST", endpoint, **kwargs)
    
    async def post_async(self, endpoint: str, **kwargs) -> httpx.Response:
        """Make an async POST request."""
        return await self.request_async("POST", endpoint, **kwargs)
    
    def close(self):
        """Close the session."""
        self.session.close()
        if self._async_client:
            asyncio.create_task(self._async_client.aclose())
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._async_client:
            await self._async_client.aclose()
