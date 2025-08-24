"""UCSC Genome Browser data collector."""

import logging
from typing import Any, Dict, List, Optional

from src.api.ucsc import UCSCClient
from src.models.genomic import Gene, Variant
from .base import BaseCollector
from ..config.config import settings


logger = logging.getLogger(__name__)


class UCSCCollector(BaseCollector):
    """Collector for UCSC Genome Browser data."""
    
    def __init__(self, db_session=None):
        api_client = UCSCClient()
        super().__init__(api_client, db_session)
        self.default_genome = "hg38"  # Human genome GRCh38
    
    def collect_single(self, identifier: str, **kwargs) -> Dict[str, Any]:
        """Collect data for a single identifier.
        
        Args:
            identifier: Gene name or genomic position (chr:start-end)
            **kwargs: Additional parameters (genome, tracks)
        
        Returns:
            Collected data
        """
        genome = kwargs.get('genome', self.default_genome)
        
        # Check if identifier is a position or gene name
        if ':' in identifier and '-' in identifier:
            return self.collect_region(identifier, genome, **kwargs)
        else:
            return self.collect_gene(identifier, genome, **kwargs)
    
    def collect_gene(self, gene_name: str, genome: str = None, **kwargs) -> Dict[str, Any]:
        """Collect UCSC data for a gene.
        
        Args:
            gene_name: Gene symbol
            genome: Genome assembly
            **kwargs: Additional parameters
        
        Returns:
            Gene data from UCSC
        """
        if genome is None:
            genome = self.default_genome
        
        logger.info(f"Collecting UCSC data for gene {gene_name} in {genome}")
        
        # Search for gene
        search_result = self.api_client.search(genome, gene_name, type="gene")
        
        if not search_result.get("matches"):
            raise ValueError(f"Gene {gene_name} not found in {genome}")
        
        # Get the first match
        match = search_result["matches"][0]
        
        data = {
            "gene": gene_name,
            "genome": genome,
            "search_result": match
        }
        
        # Parse position if available
        if "position" in match:
            chrom, positions = match["position"].split(":")
            start, end = positions.split("-")
            start, end = int(start), int(end)
            
            data["chromosome"] = chrom
            data["start"] = start
            data["end"] = end
            
            # Get gene predictions
            gene_data = self.api_client.get_data(
                genome, "ncbiRefSeqCurated", chrom, start, end
            )
            data["gene_predictions"] = gene_data
            
            # Get sequence
            sequence = self.api_client.get_sequence(genome, chrom, start, end)
            data["sequence"] = sequence
            
            # Get conservation scores
            conservation = self.api_client.get_conservation_scores(
                genome, chrom, start, end
            )
            data["conservation"] = conservation
            
            # Get regulatory elements
            regulatory = self.api_client.get_regulatory_elements(
                genome, chrom, start, end
            )
            data["regulatory_elements"] = regulatory
            
            # Get SNPs in gene region
            snps = self.api_client.get_snp_data(genome, chrom, start, window=(end-start)//2)
            data["snps"] = snps
        
        return data
    
    def collect_region(self, position: str, genome: str = None, **kwargs) -> Dict[str, Any]:
        """Collect UCSC data for a genomic region.
        
        Args:
            position: Genomic position (chr:start-end)
            genome: Genome assembly
            **kwargs: Additional parameters
        
        Returns:
            Region data
        """
        if genome is None:
            genome = self.default_genome
        
        logger.info(f"Collecting UCSC data for region {position} in {genome}")
        
        # Parse position
        chrom, positions = position.split(":")
        start, end = positions.split("-")
        start, end = int(start), int(end)
        
        data = {
            "position": position,
            "genome": genome,
            "chromosome": chrom,
            "start": start,
            "end": end
        }
        
        # Get sequence
        sequence = self.api_client.get_sequence(genome, chrom, start, end)
        data["sequence"] = sequence
        
        # Get all genes in region
        gene_data = self.api_client.get_data(
            genome, "ncbiRefSeqCurated", chrom, start, end
        )
        data["genes"] = gene_data
        
        # Get SNPs
        snps = self.api_client.get_snp_data(genome, chrom, (start+end)//2, window=(end-start)//2)
        data["snps"] = snps
        
        # Get conservation
        conservation = self.api_client.get_conservation_scores(genome, chrom, start, end)
        data["conservation"] = conservation
        
        # Get regulatory elements
        regulatory = self.api_client.get_regulatory_elements(genome, chrom, start, end)
        data["regulatory_elements"] = regulatory
        
        # Get additional tracks if specified
        if "tracks" in kwargs:
            tracks_data = {}
            for track in kwargs["tracks"]:
                try:
                    track_data = self.api_client.get_data(genome, track, chrom, start, end)
                    tracks_data[track] = track_data
                except Exception as e:
                    logger.warning(f"Could not get data for track {track}: {e}")
            data["custom_tracks"] = tracks_data
        
        return data
    
    def collect_batch(self, identifiers: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Collect data for multiple identifiers."""
        results = []
        for identifier in identifiers:
            try:
                data = self.collect_single(identifier, **kwargs)
                results.append(data)
            except Exception as e:
                logger.error(f"Failed to collect UCSC data for {identifier}: {e}")
        return results
    
    def save_to_database(self, data: Dict[str, Any]) -> Optional[Gene]:
        """Save UCSC gene data to database.
        
        Args:
            data: Collected UCSC data
        
        Returns:
            Saved Gene instance or None
        """
        if "gene" not in data:
            logger.warning("No gene information to save")
            return None
        
        gene_name = data["gene"]
        
        # Check if gene exists
        existing = self.db_session.query(Gene).filter_by(
            symbol=gene_name
        ).first()
        
        if existing:
            logger.info(f"Updating existing gene {gene_name}")
            gene = existing
        else:
            gene = Gene(
                id=self.generate_id("ucsc_gene", gene_name),
                gene_id=f"ucsc:{gene_name}",
                source="UCSC"
            )
        
        # Update fields
        gene.symbol = gene_name
        gene.chromosome = data.get("chromosome")
        gene.start_position = data.get("start")
        gene.end_position = data.get("end")
        
        # Add description from search result
        if "search_result" in data:
            gene.description = data["search_result"].get("description", "")
        
        # Store additional data as JSON
        if "gene_predictions" in data:
            gene.ucsc_data = {
                "genome": data.get("genome"),
                "predictions": data["gene_predictions"],
                "conservation": data.get("conservation"),
                "regulatory_elements": data.get("regulatory_elements")
            }
        
        if not existing:
            self.db_session.add(gene)
        
        self.db_session.commit()
        logger.info(f"Saved gene {gene_name} to database")
        
        return gene
    
    def search_genes(self, query: str, genome: str = None) -> List[Dict[str, Any]]:
        """Search for genes in UCSC.
        
        Args:
            query: Search query
            genome: Genome assembly
        
        Returns:
            Search results
        """
        if genome is None:
            genome = self.default_genome
        
        logger.info(f"Searching UCSC for '{query}' in {genome}")
        
        results = self.api_client.search(genome, query, type="gene")
        return results.get("matches", [])
    
    def get_genome_info(self, genome: str = None) -> Dict[str, Any]:
        """Get information about a genome assembly.
        
        Args:
            genome: Genome assembly
        
        Returns:
            Genome information
        """
        if genome is None:
            genome = self.default_genome
        
        # Get chromosome info
        chromosomes = self.api_client.get_chromosomes(genome)
        
        # Get available tracks
        tracks = self.api_client.get_tracks(genome)
        
        return {
            "genome": genome,
            "chromosomes": chromosomes,
            "tracks": tracks
        }
