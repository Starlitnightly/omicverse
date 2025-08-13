"""BLAST data collector."""

import logging
from typing import Any, Dict, List, Optional

from omicverse.external.datacollect.api.blast import BLASTClient
from omicverse.external.datacollect.models.genomic import Gene
from omicverse.external.datacollect.models.protein import Protein
from .base import BaseCollector
from ..config import settings


logger = logging.getLogger(__name__)


class BLASTCollector(BaseCollector):
    """Collector for BLAST sequence similarity searches."""
    
    def __init__(self, db_session=None):
        api_client = BLASTClient()
        super().__init__(api_client, db_session)
        self.default_program = "blastn"
        self.default_database = "nt"
    
    def collect_single(self, identifier: str, **kwargs) -> Dict[str, Any]:
        """Collect BLAST results for a sequence.
        
        Args:
            identifier: Sequence string or accession number
            **kwargs: Additional parameters (program, database, is_accession)
        
        Returns:
            BLAST search results
        """
        is_accession = kwargs.get('is_accession', False)
        
        if is_accession:
            # Fetch sequence by accession
            seq_data = self.api_client.search_by_accession(identifier)
            sequence = self._extract_sequence_from_fasta(seq_data['sequence'])
            accession = identifier
        else:
            sequence = identifier
            accession = kwargs.get('accession', 'unknown')
        
        program = kwargs.get('program', self.default_program)
        database = kwargs.get('database', self.default_database)
        
        logger.info(f"Running BLAST search with program {program} against {database}")
        
        # Run BLAST search
        blast_results = self.api_client.blast_and_wait(
            sequence=sequence,
            program=program,
            database=database,
            **kwargs
        )
        
        # Process and enrich results
        data = {
            "query_accession": accession,
            "query_sequence": sequence[:100] + "..." if len(sequence) > 100 else sequence,
            "query_length": blast_results.get("query_len", len(sequence)),
            "program": blast_results.get("program"),
            "database": blast_results.get("database"),
            "version": blast_results.get("version"),
            "statistics": blast_results.get("statistics", {}),
            "hit_count": len(blast_results.get("hits", []))
        }
        
        # Process top hits
        hits = blast_results.get("hits", [])
        top_hits = []
        
        for hit in hits[:10]:  # Process top 10 hits
            # Extract best HSP
            best_hsp = None
            if hit.get("hsps"):
                best_hsp = max(hit["hsps"], key=lambda x: x.get("bit_score", 0))
            
            hit_info = {
                "accession": hit.get("accession"),
                "definition": hit.get("def"),
                "length": hit.get("length"),
                "best_evalue": best_hsp.get("evalue") if best_hsp else None,
                "best_bit_score": best_hsp.get("bit_score") if best_hsp else None,
                "best_identity": best_hsp.get("percent_identity") if best_hsp else None,
                "best_coverage": self._calculate_coverage(
                    best_hsp.get("query_from", 0),
                    best_hsp.get("query_to", 0),
                    data["query_length"]
                ) if best_hsp else 0,
                "hsp_count": len(hit.get("hsps", []))
            }
            
            # Add alignment details for top hit
            if len(top_hits) == 0 and best_hsp:
                hit_info["alignment"] = {
                    "query_from": best_hsp.get("query_from"),
                    "query_to": best_hsp.get("query_to"),
                    "hit_from": best_hsp.get("hit_from"),
                    "hit_to": best_hsp.get("hit_to"),
                    "identity": best_hsp.get("identity"),
                    "positive": best_hsp.get("positive"),
                    "gaps": best_hsp.get("gaps"),
                    "align_len": best_hsp.get("align_len")
                }
            
            # Parse organism from definition if available
            hit_info["organism"] = self._extract_organism(hit.get("def", ""))
            
            top_hits.append(hit_info)
        
        data["top_hits"] = top_hits
        
        # Categorize hits by identity threshold
        identity_categories = {
            "high_identity": [],  # > 95%
            "medium_identity": [],  # 80-95%
            "low_identity": []  # < 80%
        }
        
        for hit in top_hits:
            identity = hit.get("best_identity", 0)
            if identity > 95:
                identity_categories["high_identity"].append(hit["accession"])
            elif identity > 80:
                identity_categories["medium_identity"].append(hit["accession"])
            else:
                identity_categories["low_identity"].append(hit["accession"])
        
        data["identity_categories"] = identity_categories
        
        # Store full results if requested
        if kwargs.get('include_full_results'):
            data["full_results"] = blast_results
        
        return data
    
    def _extract_sequence_from_fasta(self, fasta: str) -> str:
        """Extract sequence from FASTA format.
        
        Args:
            fasta: FASTA formatted text
        
        Returns:
            Sequence string
        """
        lines = fasta.strip().split('\n')
        sequence_lines = [line for line in lines if not line.startswith('>')]
        return ''.join(sequence_lines)
    
    def _calculate_coverage(self, query_from: int, query_to: int, query_len: int) -> float:
        """Calculate query coverage percentage.
        
        Args:
            query_from: Start position in query
            query_to: End position in query
            query_len: Total query length
        
        Returns:
            Coverage percentage
        """
        if query_len == 0:
            return 0
        coverage_len = abs(query_to - query_from) + 1
        return (coverage_len / query_len) * 100
    
    def _extract_organism(self, definition: str) -> str:
        """Extract organism name from sequence definition.
        
        Args:
            definition: Sequence definition line
        
        Returns:
            Organism name or empty string
        """
        # Common patterns in NCBI definitions
        if '[' in definition and ']' in definition:
            start = definition.rfind('[')
            end = definition.rfind(']')
            if start < end:
                return definition[start+1:end]
        return ""
    
    def collect_batch(self, identifiers: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Run BLAST searches for multiple sequences.
        
        Args:
            identifiers: List of sequences or accession numbers
            **kwargs: Additional parameters
        
        Returns:
            List of BLAST results
        """
        results = []
        for identifier in identifiers:
            try:
                data = self.collect_single(identifier, **kwargs)
                results.append(data)
            except Exception as e:
                logger.error(f"Failed to run BLAST for {identifier}: {e}")
                results.append({
                    "query": identifier,
                    "error": str(e)
                })
        return results
    
    def save_to_database(self, data: Dict[str, Any]) -> Any:
        """Save BLAST results to database.
        
        Args:
            data: BLAST results
        
        Returns:
            Saved entity (Gene or Protein)
        """
        query_accession = data.get("query_accession", "unknown")
        program = data.get("program", "")
        
        # Determine if this is nucleotide or protein data
        is_protein = program in ["blastp", "blastx"]
        
        if is_protein:
            return self._save_protein_blast(data)
        else:
            return self._save_gene_blast(data)
    
    def _save_gene_blast(self, data: Dict[str, Any]) -> Gene:
        """Save nucleotide BLAST results.
        
        Args:
            data: BLAST results
        
        Returns:
            Saved Gene instance
        """
        query_accession = data.get("query_accession", "unknown")
        
        # Check if gene exists
        existing = self.db_session.query(Gene).filter_by(
            gene_id=f"blast:{query_accession}"
        ).first()
        
        if existing:
            logger.info(f"Updating existing gene for accession {query_accession}")
            gene = existing
        else:
            gene = Gene(
                id=self.generate_id("blast_gene", query_accession),
                gene_id=f"blast:{query_accession}",
                source="BLAST"
            )
        
        # Update gene information from top hit if available
        if data.get("top_hits"):
            top_hit = data["top_hits"][0]
            gene.description = top_hit.get("definition", "")
            gene.organism = top_hit.get("organism", "")
        
        # Store BLAST results in JSON field
        gene.annotations = {
            "blast_results": {
                "program": data.get("program"),
                "database": data.get("database"),
                "query_length": data.get("query_length"),
                "hit_count": data.get("hit_count"),
                "top_hits": data.get("top_hits", [])[:5],  # Store top 5 hits
                "identity_categories": data.get("identity_categories", {}),
                "statistics": data.get("statistics", {})
            }
        }
        
        if not existing:
            self.db_session.add(gene)
        
        self.db_session.commit()
        logger.info(f"Saved BLAST results for gene {query_accession}")
        
        return gene
    
    def _save_protein_blast(self, data: Dict[str, Any]) -> Protein:
        """Save protein BLAST results.
        
        Args:
            data: BLAST results
        
        Returns:
            Saved Protein instance
        """
        query_accession = data.get("query_accession", "unknown")
        
        # Check if protein exists
        existing = self.db_session.query(Protein).filter_by(
            uniprot_accession=query_accession
        ).first()
        
        if existing:
            logger.info(f"Updating existing protein for accession {query_accession}")
            protein = existing
        else:
            protein = Protein(
                id=self.generate_id("blast_protein", query_accession),
                uniprot_accession=query_accession,
                sequence="",  # Required field
                sequence_length=data.get("query_length", 0),
                source="BLAST"
            )
        
        # Update protein information from top hit if available
        if data.get("top_hits"):
            top_hit = data["top_hits"][0]
            protein.description = top_hit.get("definition", "")
            protein.organism = top_hit.get("organism", "")
        
        # Store BLAST results in annotations
        if not protein.annotations:
            protein.annotations = {}
        
        protein.annotations["blast_results"] = {
            "program": data.get("program"),
            "database": data.get("database"),
            "query_length": data.get("query_length"),
            "hit_count": data.get("hit_count"),
            "top_hits": data.get("top_hits", [])[:5],
            "identity_categories": data.get("identity_categories", {}),
            "statistics": data.get("statistics", {})
        }
        
        if not existing:
            self.db_session.add(protein)
        
        self.db_session.commit()
        logger.info(f"Saved BLAST results for protein {query_accession}")
        
        return protein
    
    def find_homologs(self, sequence: str, identity_threshold: float = 30.0,
                     coverage_threshold: float = 50.0, **kwargs) -> List[Dict[str, Any]]:
        """Find homologous sequences.
        
        Args:
            sequence: Query sequence
            identity_threshold: Minimum identity percentage
            coverage_threshold: Minimum coverage percentage
            **kwargs: Additional BLAST parameters
        
        Returns:
            List of homologous sequences
        """
        # Run BLAST search
        blast_data = self.collect_single(sequence, **kwargs)
        
        homologs = []
        for hit in blast_data.get("top_hits", []):
            if (hit.get("best_identity", 0) >= identity_threshold and
                hit.get("best_coverage", 0) >= coverage_threshold):
                homologs.append({
                    "accession": hit["accession"],
                    "definition": hit["definition"],
                    "organism": hit["organism"],
                    "identity": hit["best_identity"],
                    "coverage": hit["best_coverage"],
                    "evalue": hit["best_evalue"],
                    "bit_score": hit["best_bit_score"]
                })
        
        logger.info(f"Found {len(homologs)} homologs above thresholds")
        return homologs