"""GEO (Gene Expression Omnibus) API client."""

import logging
from typing import Any, Dict, List, Optional, Union
import xml.etree.ElementTree as ET

from ..base import BaseAPIClient
from ...config import settings


logger = logging.getLogger(__name__)


class GEOClient(BaseAPIClient):
    """Client for NCBI GEO API.
    
    GEO is a public repository that archives and freely distributes
    high-throughput gene expression and other functional genomics data.
    
    API Documentation: https://www.ncbi.nlm.nih.gov/geo/info/geo_paccess.html
    """
    
    def __init__(self, **kwargs):
        base_url = kwargs.pop("base_url", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils")
        self.api_key = settings.api.ncbi_api_key
        self.database = "gds"  # GEO DataSets database
        self.geo_base_url = "https://www.ncbi.nlm.nih.gov/geo/query"
        super().__init__(
            base_url=base_url,
            rate_limit=kwargs.get("rate_limit", 3),  # NCBI limit: 3 req/sec
            **kwargs
        )
    
    def get_default_headers(self) -> Dict[str, str]:
        """Get GEO-specific headers."""
        return {
            "User-Agent": "BioinformaticsCollector/1.0",
            "Accept": "application/json,application/xml,text/plain",
        }
    
    def _get_base_params(self) -> Dict[str, str]:
        """Get base parameters for all requests."""
        params = {
            "db": self.database,
            "retmode": "json",
        }
        if self.api_key:
            params["api_key"] = self.api_key
        return params
    
    def search(self, query: str, max_results: int = 20) -> Dict[str, Any]:
        """Search GEO for datasets.
        
        Args:
            query: Search query (e.g., "breast cancer[title]", "GSE1234[ACCN]")
            max_results: Maximum number of results
        
        Returns:
            Search results with GEO IDs
        """
        endpoint = "/esearch.fcgi"
        params = self._get_base_params()
        params.update({
            "term": query,
            "retmax": max_results,
            "usehistory": "y"
        })
        
        response = self.get(endpoint, params=params)
        return response.json()
    
    def get_dataset_summary(self, geo_id: Union[str, int]) -> Dict[str, Any]:
        """Get dataset summary by GEO ID.
        
        Args:
            geo_id: GEO dataset ID (e.g., "200000001" or "GDS1234")
        
        Returns:
            Dataset summary
        """
        # Convert GDS accession to ID if needed
        if isinstance(geo_id, str) and geo_id.startswith("GDS"):
            search_result = self.search(f"{geo_id}[ACCN]", max_results=1)
            if search_result.get("esearchresult", {}).get("idlist"):
                geo_id = search_result["esearchresult"]["idlist"][0]
            else:
                raise ValueError(f"GEO dataset {geo_id} not found")
        
        endpoint = "/esummary.fcgi"
        params = self._get_base_params()
        params["id"] = str(geo_id)
        
        response = self.get(endpoint, params=params)
        data = response.json()
        
        # Extract dataset from result
        if "result" in data and str(geo_id) in data["result"]:
            return data["result"][str(geo_id)]
        return {}
    
    def get_series_matrix(self, gse_accession: str) -> Dict[str, Any]:
        """Get series matrix file data.
        
        Args:
            gse_accession: GEO Series accession (e.g., "GSE1234")
        
        Returns:
            Parsed matrix data
        """
        # Use SOFT format for series data
        url = f"{self.geo_base_url}/acc.cgi"
        params = {
            "acc": gse_accession,
            "targ": "self",
            "form": "text",
            "view": "brief"
        }
        
        response = self.session.get(url, params=params)
        response.raise_for_status()
        
        parsed = self._parse_soft_format(response.text)
        
        # Ensure series section exists
        if 'series' not in parsed and parsed:
            # If there's no explicit series section, wrap everything in it
            parsed = {'series': parsed}
        
        return parsed
    
    def get_sample_data(self, gsm_accession: str) -> Dict[str, Any]:
        """Get sample data.
        
        Args:
            gsm_accession: GEO Sample accession (e.g., "GSM1234")
        
        Returns:
            Sample metadata and data
        """
        url = f"{self.geo_base_url}/acc.cgi"
        params = {
            "acc": gsm_accession,
            "targ": "self",
            "form": "text",
            "view": "full"
        }
        
        response = self.session.get(url, params=params)
        response.raise_for_status()
        
        parsed = self._parse_soft_format(response.text)
        
        # Ensure sample section exists
        if 'sample' not in parsed and parsed:
            # If there's no explicit sample section, wrap everything in it
            parsed = {'sample': parsed}
        
        return parsed
    
    def get_platform_data(self, gpl_accession: str) -> Dict[str, Any]:
        """Get platform annotation data.
        
        Args:
            gpl_accession: GEO Platform accession (e.g., "GPL570")
        
        Returns:
            Platform information
        """
        url = f"{self.geo_base_url}/acc.cgi"
        params = {
            "acc": gpl_accession,
            "targ": "self",
            "form": "text",
            "view": "brief"
        }
        
        response = self.session.get(url, params=params)
        response.raise_for_status()
        
        parsed = self._parse_soft_format(response.text)
        
        # Ensure platform section exists
        if 'platform' not in parsed and parsed:
            # If there's no explicit platform section, wrap everything in it
            parsed = {'platform': parsed}
        
        return parsed
    
    def search_datasets_by_gene(self, gene_symbol: str, organism: str = "Homo sapiens",
                               max_results: int = 50) -> List[Dict[str, Any]]:
        """Search for datasets containing a specific gene.
        
        Args:
            gene_symbol: Gene symbol (e.g., "TP53")
            organism: Organism name
            max_results: Maximum results
        
        Returns:
            List of relevant datasets
        """
        query = f'"{gene_symbol}"[Gene Symbol] AND "{organism}"[Organism]'
        search_results = self.search(query, max_results)
        
        datasets = []
        if "esearchresult" in search_results:
            dataset_ids = search_results["esearchresult"].get("idlist", [])
            
            for dataset_id in dataset_ids[:max_results]:
                try:
                    summary = self.get_dataset_summary(dataset_id)
                    if summary:
                        datasets.append(summary)
                except Exception as e:
                    logger.error(f"Failed to get dataset {dataset_id}: {e}")
        
        return datasets
    
    def get_expression_data(self, gse_accession: str, sample_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get expression data for a GEO series.
        
        Args:
            gse_accession: GEO Series accession
            sample_ids: Optional list of sample IDs to filter
        
        Returns:
            Expression data matrix
        """
        # Get series data
        matrix_data = self.get_series_matrix(gse_accession)
        series_data = matrix_data.get('series', matrix_data)
        
        # Extract expression values
        expression_data = {
            "series_id": gse_accession,
            "title": series_data.get("series_title", ""),
            "platform": series_data.get("series_platform_id", ""),
            "samples": {},
            "features": []
        }
        
        # Parse sample data if available - check for different key formats
        sample_list = None
        sample_titles = None
        
        if "sample_geo_accession" in series_data:
            sample_list = series_data["sample_geo_accession"]
            if isinstance(sample_list, str):
                sample_list = [sample_list]
        elif "series_sample_id" in series_data:
            sample_list = series_data["series_sample_id"]
            if isinstance(sample_list, str):
                sample_list = sample_list.split(",")
                sample_list = [s.strip() for s in sample_list]
        
        # Get sample titles if available
        if "sample_title" in series_data:
            sample_titles = series_data["sample_title"]
            if isinstance(sample_titles, str):
                sample_titles = [sample_titles]
        
        if sample_list:
            if sample_ids:
                # Filter to requested samples
                indices = [i for i, s in enumerate(sample_list) if s in sample_ids]
                sample_list = [sample_list[i] for i in indices]
                if sample_titles:
                    sample_titles = [sample_titles[i] for i in indices if i < len(sample_titles)]
            
            for i, sample_id in enumerate(sample_list):
                sample_data = {}
                if sample_titles and i < len(sample_titles):
                    sample_data["sample_title"] = sample_titles[i]
                expression_data["samples"][sample_id] = sample_data
        
        return expression_data
    
    def _parse_soft_format(self, text: str) -> Dict[str, Any]:
        """Parse SOFT format text into structured data.
        
        Args:
            text: SOFT format text
        
        Returns:
            Parsed data dictionary
        """
        data = {}
        current_section = None
        current_subsection = {}
        data_table = {}
        in_data_table = False
        
        for line in text.split('\n'):
            line = line.strip()
            
            if not line or line.startswith('#'):
                continue
            
            if line.startswith('^'):
                # New section
                if current_section and current_subsection:
                    data[current_section] = current_subsection
                current_section = line[1:].lower()
                current_subsection = {}
                in_data_table = False
            
            elif line.startswith('!'):
                # Metadata line - handle both tab and = separators
                in_data_table = False  # Not in data table anymore
                if '\t' in line:
                    parts = line[1:].split('\t')
                    key = parts[0].lower().replace(' ', '_')
                    values = [v.strip('"') for v in parts[1:] if v]
                    
                    # Handle duplicate keys - append to list
                    if key in current_subsection:
                        # Convert to list if not already
                        if not isinstance(current_subsection[key], list):
                            current_subsection[key] = [current_subsection[key]]
                        # Append new value(s)
                        if len(values) == 1:
                            current_subsection[key].append(values[0])
                        else:
                            current_subsection[key].extend(values)
                    else:
                        # First occurrence
                        if len(values) == 1:
                            current_subsection[key] = values[0]
                        elif len(values) > 1:
                            current_subsection[key] = values
                elif ' = ' in line:
                    key, value = line[1:].split(' = ', 1)
                    key = key.lower().replace(' ', '_')
                    
                    # Handle multi-value fields
                    if key in current_subsection:
                        if not isinstance(current_subsection[key], list):
                            current_subsection[key] = [current_subsection[key]]
                        current_subsection[key].append(value)
                    else:
                        current_subsection[key] = value
            
            elif '\t' in line and not line.startswith('^') and not line.startswith('!'):
                # Data table row (only if not a metadata line)
                parts = line.split('\t')
                if parts[0]:
                    # This looks like data table content
                    in_data_table = True
                    row_id = parts[0]
                    if len(parts) > 1:
                        data_table[row_id] = parts[1:]
        
        # Save last section
        if current_section and current_subsection:
            data[current_section] = current_subsection
        elif current_subsection and not current_section:
            # If we have data but no section was defined, return flat structure
            data = current_subsection
        
        # Add data table if we found one
        if data_table:
            data["data_table"] = data_table
        
        return data
    
    def download_supplementary_files(self, gse_accession: str) -> List[str]:
        """Get URLs for supplementary files.
        
        Args:
            gse_accession: GEO Series accession
        
        Returns:
            List of supplementary file URLs
        """
        # Try to get actual file links from the HTML page
        url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gse_accession}"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            
            # Simple regex to find FTP links
            import re
            ftp_pattern = r'(ftp://[^\s<>"]+(?:\.tar|\.txt\.gz|\.gz|\.zip))'
            urls = re.findall(ftp_pattern, response.text)
            
            if urls:
                # Also add the direct download link
                urls.append(f"https://www.ncbi.nlm.nih.gov/geo/download/?acc={gse_accession}&format=file")
                return list(set(urls))  # Remove duplicates
        except:
            pass
        
        # Fallback to standard FTP patterns
        gse_prefix = gse_accession[:6] + "nnn"
        return [
            f"ftp://ftp.ncbi.nlm.nih.gov/geo/series/{gse_prefix}/{gse_accession}/suppl/",
            f"https://www.ncbi.nlm.nih.gov/geo/download/?acc={gse_accession}&format=file"
        ]