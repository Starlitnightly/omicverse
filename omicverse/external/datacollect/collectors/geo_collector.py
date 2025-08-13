"""GEO data collector."""

import logging
import json
from typing import Any, Dict, List, Optional, Union

from omicverse.external.datacollect.api.geo import GEOClient
from omicverse.external.datacollect.models.genomic import Gene, Expression
from .base import BaseCollector
from omicverse.external.datacollect.config.config import settings


logger = logging.getLogger(__name__)


class GEOCollector(BaseCollector):
    """Collector for GEO gene expression data."""
    
    def __init__(self, db_session=None):
        api_client = GEOClient()
        super().__init__(api_client, db_session)
    
    def collect_single(self, identifier: str, **kwargs) -> Dict[str, Any]:
        """Collect data for a single GEO identifier.
        
        Args:
            identifier: GEO accession (GSE, GSM, GPL, or GDS)
            **kwargs: Additional parameters
        
        Returns:
            Collected data
        """
        # Determine type of accession
        if identifier.startswith("GSE"):
            return self.collect_series(identifier)
        elif identifier.startswith("GSM"):
            return self.collect_sample(identifier)
        elif identifier.startswith("GPL"):
            return self.collect_platform(identifier)
        elif identifier.startswith("GDS"):
            return self.collect_dataset(identifier)
        else:
            raise ValueError(f"Unknown GEO accession type: {identifier}")
    
    def collect_series(self, gse_accession: str) -> Dict[str, Any]:
        """Collect GEO series data.
        
        Args:
            gse_accession: GEO Series accession (e.g., "GSE1234")
        
        Returns:
            Series data including samples and expression values
        """
        logger.info(f"Collecting GEO series {gse_accession}")
        
        # Get series metadata
        series_data = self.api_client.get_series_matrix(gse_accession)
        
        # Get expression data
        expression_data = self.api_client.get_expression_data(gse_accession)
        
        # Get supplementary file URLs
        supp_files = self.api_client.download_supplementary_files(gse_accession)
        
        data = {
            "accession": gse_accession,
            "type": "series",
            "title": expression_data.get("title", ""),
            "platform": expression_data.get("platform", ""),
            "summary": series_data.get("series", {}).get("series_summary", ""),
            "overall_design": series_data.get("series", {}).get("series_overall_design", ""),
            "sample_count": len(expression_data.get("samples", {})),
            "samples": list(expression_data.get("samples", {}).keys()),
            "supplementary_files": supp_files,
            "raw_data": series_data
        }
        
        return data
    
    def collect_sample(self, gsm_accession: str) -> Dict[str, Any]:
        """Collect GEO sample data.
        
        Args:
            gsm_accession: GEO Sample accession (e.g., "GSM1234")
        
        Returns:
            Sample metadata and expression data
        """
        logger.info(f"Collecting GEO sample {gsm_accession}")
        
        sample_data = self.api_client.get_sample_data(gsm_accession)
        
        # Extract sample information
        sample_info = sample_data.get("sample", {})
        
        data = {
            "accession": gsm_accession,
            "type": "sample",
            "title": sample_info.get("sample_title", ""),
            "source": sample_info.get("sample_source_name_ch1", ""),
            "organism": sample_info.get("sample_organism_ch1", ""),
            "platform": sample_info.get("sample_platform_id", ""),
            "series": sample_info.get("sample_series_id", ""),
            "characteristics": self._extract_characteristics(sample_info),
            "raw_data": sample_data
        }
        
        return data
    
    def collect_platform(self, gpl_accession: str) -> Dict[str, Any]:
        """Collect GEO platform data.
        
        Args:
            gpl_accession: GEO Platform accession (e.g., "GPL570")
        
        Returns:
            Platform annotation data
        """
        logger.info(f"Collecting GEO platform {gpl_accession}")
        
        platform_data = self.api_client.get_platform_data(gpl_accession)
        
        platform_info = platform_data.get("platform", {})
        
        data = {
            "accession": gpl_accession,
            "type": "platform",
            "title": platform_info.get("platform_title", ""),
            "technology": platform_info.get("platform_technology", ""),
            "organism": platform_info.get("platform_organism", ""),
            "manufacturer": platform_info.get("platform_manufacturer", ""),
            "description": platform_info.get("platform_description", ""),
            "raw_data": platform_data
        }
        
        return data
    
    def collect_dataset(self, gds_id: str) -> Dict[str, Any]:
        """Collect GEO dataset.
        
        Args:
            gds_id: GEO DataSet ID or accession
        
        Returns:
            Dataset information
        """
        logger.info(f"Collecting GEO dataset {gds_id}")
        
        dataset = self.api_client.get_dataset_summary(gds_id)
        
        data = {
            "accession": dataset.get("accession", gds_id),
            "type": "dataset",
            "title": dataset.get("title", ""),
            "summary": dataset.get("summary", ""),
            "gpl": dataset.get("gpl", ""),
            "gse": dataset.get("gse", []),
            "sample_count": dataset.get("n_samples", 0),
            "pubmed_id": dataset.get("pubmed_id", []),
            "raw_data": dataset
        }
        
        return data
    
    def save_to_database(self, data: Dict[str, Any]) -> Any:
        """Save GEO data to database.
        
        Args:
            data: Collected GEO data
        
        Returns:
            Saved database object(s)
        """
        data_type = data.get("type")
        
        if data_type == "series":
            return self.save_series_to_database(data)
        elif data_type == "sample":
            return self.save_sample_to_database(data)
        else:
            logger.warning(f"Database saving not implemented for {data_type}")
            return None
    
    def save_series_to_database(self, data: Dict[str, Any]) -> List[Expression]:
        """Save series data to database as expression records."""
        saved_expressions = []
        
        accession = data["accession"]
        platform = data.get("platform", "")
        
        # Create expression records for each sample
        for sample_id in data.get("samples", []):
            # Check if expression record exists
            existing = self.db_session.query(Expression).filter_by(
                dataset_id=accession,
                sample_id=sample_id
            ).first()
            
            if not existing:
                expression = Expression(
                    id=self.generate_id("geo_expr", accession, sample_id),
                    source="GEO",
                    dataset_id=accession,
                    sample_id=sample_id,
                    platform=platform,
                    gene_id=None,  # Will be populated when expression values are added
                    expression_value=0.0  # Placeholder
                )
                self.db_session.add(expression)
                saved_expressions.append(expression)
        
        if saved_expressions:
            self.db_session.commit()
            logger.info(f"Saved {len(saved_expressions)} expression records for {accession}")
        
        return saved_expressions
    
    def save_sample_to_database(self, data: Dict[str, Any]) -> Expression:
        """Save sample data to database."""
        sample_id = data["accession"]
        series_id = data.get("series", "")
        
        # Check if expression record exists
        existing = self.db_session.query(Expression).filter_by(
            sample_id=sample_id
        ).first()
        
        if existing:
            logger.info(f"Updating existing sample {sample_id}")
            expression = existing
        else:
            expression = Expression(
                id=self.generate_id("geo_sample", sample_id),
                source="GEO",
                dataset_id=series_id,
                sample_id=sample_id,
                platform=data.get("platform", ""),
                gene_id=None,
                expression_value=0.0
            )
            self.db_session.add(expression)
        
        # Update metadata
        expression.tissue = data.get("source", "")
        expression.condition = data.get("title", "")
        
        # Extract cell type from characteristics if available
        characteristics = data.get("characteristics", {})
        if "cell_type" in characteristics:
            expression.cell_type = characteristics["cell_type"]
        
        self.db_session.commit()
        logger.info(f"Saved sample {sample_id}")
        
        return expression
    
    def collect_batch(self, identifiers: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Collect data for multiple identifiers."""
        results = []
        for identifier in identifiers:
            try:
                data = self.collect_single(identifier, **kwargs)
                results.append(data)
            except Exception as e:
                logger.error(f"Failed to collect GEO data for {identifier}: {e}")
        return results
    
    def search_by_gene(self, gene_symbol: str, organism: str = "Homo sapiens",
                      max_results: int = 20) -> List[Dict[str, Any]]:
        """Search for datasets containing a gene.
        
        Args:
            gene_symbol: Gene symbol
            organism: Organism name
            max_results: Maximum results
        
        Returns:
            List of relevant datasets
        """
        logger.info(f"Searching GEO for gene {gene_symbol} in {organism}")
        
        datasets = self.api_client.search_datasets_by_gene(
            gene_symbol, organism, max_results
        )
        
        # Enrich with additional data
        enriched_datasets = []
        for dataset in datasets:
            try:
                # Get more details
                gds_id = dataset.get("uid")
                if gds_id:
                    detailed = self.collect_dataset(gds_id)
                    enriched_datasets.append(detailed)
            except Exception as e:
                logger.error(f"Failed to enrich dataset: {e}")
                enriched_datasets.append(dataset)
        
        return enriched_datasets
    
    def collect_expression_values(self, gse_accession: str, gene_symbols: List[str]) -> Dict[str, Dict[str, float]]:
        """Collect expression values for specific genes.
        
        Args:
            gse_accession: GEO Series accession
            gene_symbols: List of gene symbols
        
        Returns:
            Dictionary mapping samples to gene expression values
        """
        logger.info(f"Collecting expression values for {len(gene_symbols)} genes from {gse_accession}")
        
        # This would require downloading and parsing the full expression matrix
        # For now, return a placeholder structure
        expression_data = self.api_client.get_expression_data(gse_accession)
        
        result = {}
        for sample_id in expression_data.get("samples", {}):
            result[sample_id] = {gene: 0.0 for gene in gene_symbols}
        
        return result
    
    def _extract_characteristics(self, sample_info: Dict[str, Any]) -> Dict[str, str]:
        """Extract sample characteristics."""
        characteristics = {}
        
        # Look for characteristic fields
        for key, value in sample_info.items():
            if key.startswith("sample_characteristics_ch"):
                if isinstance(value, list):
                    for item in value:
                        if ": " in item:
                            char_key, char_value = item.split(": ", 1)
                            characteristics[char_key.lower().replace(" ", "_")] = char_value
                elif ": " in str(value):
                    char_key, char_value = str(value).split(": ", 1)
                    characteristics[char_key.lower().replace(" ", "_")] = char_value
        
        return characteristics