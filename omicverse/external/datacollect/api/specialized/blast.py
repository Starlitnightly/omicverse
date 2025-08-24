"""NCBI BLAST API client."""

import logging
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

from ..base import BaseAPIClient
from ...config.config import settings


logger = logging.getLogger(__name__)


class BLASTClient(BaseAPIClient):
    """Client for NCBI BLAST API.
    
    BLAST (Basic Local Alignment Search Tool) finds regions of similarity
    between biological sequences. The program compares nucleotide or protein
    sequences to sequence databases and calculates statistical significance.
    
    API Documentation: https://blast.ncbi.nlm.nih.gov/Blast.cgi?CMD=Web&PAGE_TYPE=BlastDocs&DOC_TYPE=DeveloperInfo
    """
    
    def __init__(self, **kwargs):
        base_url = kwargs.pop("base_url", "https://blast.ncbi.nlm.nih.gov/Blast.cgi")
        super().__init__(
            base_url=base_url,
            rate_limit=kwargs.get("rate_limit", 3),  # NCBI recommends max 3 requests per second
            **kwargs
        )
        self.max_retries = kwargs.get("max_retries", 60)  # Max retries for checking results
        self.retry_delay = kwargs.get("retry_delay", 3)  # Seconds between status checks
    
    def get_default_headers(self) -> Dict[str, str]:
        """Get BLAST-specific headers."""
        return {
            "Accept": "application/xml",
            "User-Agent": "BioinformaticsDataCollector/1.0"
        }
    
    def blast_sequence(self, sequence: str, program: str = "blastn",
                      database: str = "nt", **kwargs) -> str:
        """Submit a BLAST search.
        
        Args:
            sequence: Query sequence (nucleotide or protein)
            program: BLAST program (blastn, blastp, blastx, tblastn, tblastx)
            database: Target database (nt, nr, refseq_rna, refseq_protein, swissprot, etc.)
            **kwargs: Additional BLAST parameters
        
        Returns:
            Request ID (RID) for retrieving results
        """
        params = {
            "CMD": "Put",
            "PROGRAM": program,
            "DATABASE": database,
            "QUERY": sequence,
            "FORMAT_TYPE": "XML",
            "EXPECT": kwargs.get("expect", 10),
            "HITLIST_SIZE": kwargs.get("hitlist_size", 50),
            "DESCRIPTIONS": kwargs.get("descriptions", 100),
            "ALIGNMENTS": kwargs.get("alignments", 100)
        }
        
        # Add optional parameters
        if kwargs.get("filter"):
            params["FILTER"] = kwargs["filter"]
        if kwargs.get("word_size"):
            params["WORD_SIZE"] = kwargs["word_size"]
        if kwargs.get("matrix"):
            params["MATRIX"] = kwargs["matrix"]
        if kwargs.get("gap_open"):
            params["GAPCOSTS"] = f"{kwargs['gap_open']} {kwargs.get('gap_extend', 1)}"
        if kwargs.get("composition_based_statistics"):
            params["COMPOSITION_BASED_STATISTICS"] = kwargs["composition_based_statistics"]
        
        response = self.session.post(self.base_url, data=params)
        response.raise_for_status()
        
        # Parse response to get RID
        content = response.text
        rid = None
        rtoe = None
        
        for line in content.split('\n'):
            if line.strip().startswith('RID ='):
                rid = line.split('=')[1].strip()
            elif line.strip().startswith('RTOE ='):
                rtoe = int(line.split('=')[1].strip())
        
        if not rid:
            raise ValueError("Failed to get Request ID from BLAST submission")
        
        logger.info(f"BLAST search submitted with RID: {rid}, estimated time: {rtoe} seconds")
        return rid
    
    def check_status(self, rid: str) -> str:
        """Check the status of a BLAST search.
        
        Args:
            rid: Request ID from blast_sequence
        
        Returns:
            Status (WAITING, READY, FAILED, UNKNOWN)
        """
        params = {
            "CMD": "Get",
            "FORMAT_OBJECT": "SearchInfo",
            "RID": rid
        }
        
        response = self.session.get(self.base_url, params=params)
        response.raise_for_status()
        
        content = response.text
        
        if "Status=WAITING" in content:
            return "WAITING"
        elif "Status=READY" in content:
            return "READY"
        elif "Status=FAILED" in content:
            return "FAILED"
        else:
            return "UNKNOWN"
    
    def get_results(self, rid: str, format_type: str = "XML") -> Any:
        """Get BLAST search results.
        
        Args:
            rid: Request ID from blast_sequence
            format_type: Output format (XML, Text, HTML, etc.)
        
        Returns:
            Search results in specified format
        """
        params = {
            "CMD": "Get",
            "RID": rid,
            "FORMAT_TYPE": format_type
        }
        
        response = self.session.get(self.base_url, params=params)
        response.raise_for_status()
        
        if format_type == "XML":
            return self._parse_xml_results(response.text)
        else:
            return response.text
    
    def _parse_xml_results(self, xml_content: str) -> Dict[str, Any]:
        """Parse BLAST XML results.
        
        Args:
            xml_content: XML response from BLAST
        
        Returns:
            Parsed results dictionary
        """
        try:
            root = ET.fromstring(xml_content)
            
            results = {
                "program": root.findtext(".//BlastOutput_program", ""),
                "version": root.findtext(".//BlastOutput_version", ""),
                "database": root.findtext(".//BlastOutput_db", ""),
                "query_id": root.findtext(".//BlastOutput_query-ID", ""),
                "query_def": root.findtext(".//BlastOutput_query-def", ""),
                "query_len": int(root.findtext(".//BlastOutput_query-len", "0")),
                "hits": []
            }
            
            # Parse hits
            iterations = root.find(".//BlastOutput_iterations")
            if iterations:
                iteration = iterations.find("Iteration")
                if iteration:
                    hits = iteration.find("Iteration_hits")
                    if hits:
                        for hit in hits.findall("Hit"):
                            hit_data = {
                                "num": int(hit.findtext("Hit_num", "0")),
                                "id": hit.findtext("Hit_id", ""),
                                "def": hit.findtext("Hit_def", ""),
                                "accession": hit.findtext("Hit_accession", ""),
                                "length": int(hit.findtext("Hit_len", "0")),
                                "hsps": []
                            }
                            
                            # Parse HSPs (High-scoring Segment Pairs)
                            hsps = hit.find("Hit_hsps")
                            if hsps:
                                for hsp in hsps.findall("Hsp"):
                                    hsp_data = {
                                        "num": int(hsp.findtext("Hsp_num", "0")),
                                        "bit_score": float(hsp.findtext("Hsp_bit-score", "0")),
                                        "score": int(hsp.findtext("Hsp_score", "0")),
                                        "evalue": float(hsp.findtext("Hsp_evalue", "0")),
                                        "query_from": int(hsp.findtext("Hsp_query-from", "0")),
                                        "query_to": int(hsp.findtext("Hsp_query-to", "0")),
                                        "hit_from": int(hsp.findtext("Hsp_hit-from", "0")),
                                        "hit_to": int(hsp.findtext("Hsp_hit-to", "0")),
                                        "identity": int(hsp.findtext("Hsp_identity", "0")),
                                        "positive": int(hsp.findtext("Hsp_positive", "0")),
                                        "gaps": int(hsp.findtext("Hsp_gaps", "0")),
                                        "align_len": int(hsp.findtext("Hsp_align-len", "0")),
                                        "qseq": hsp.findtext("Hsp_qseq", ""),
                                        "hseq": hsp.findtext("Hsp_hseq", ""),
                                        "midline": hsp.findtext("Hsp_midline", "")
                                    }
                                    
                                    # Calculate percent identity
                                    if hsp_data["align_len"] > 0:
                                        hsp_data["percent_identity"] = (
                                            hsp_data["identity"] / hsp_data["align_len"] * 100
                                        )
                                    else:
                                        hsp_data["percent_identity"] = 0
                                    
                                    hit_data["hsps"].append(hsp_data)
                            
                            results["hits"].append(hit_data)
            
            # Add statistics
            stats = root.find(".//Statistics")
            if stats:
                results["statistics"] = {
                    "db_num": int(stats.findtext("Statistics_db-num", "0")),
                    "db_len": int(stats.findtext("Statistics_db-len", "0")),
                    "hsp_len": int(stats.findtext("Statistics_hsp-len", "0")),
                    "eff_space": float(stats.findtext("Statistics_eff-space", "0")),
                    "kappa": float(stats.findtext("Statistics_kappa", "0")),
                    "lambda": float(stats.findtext("Statistics_lambda", "0")),
                    "entropy": float(stats.findtext("Statistics_entropy", "0"))
                }
            
            return results
            
        except ET.ParseError as e:
            logger.error(f"Failed to parse BLAST XML: {e}")
            return {"error": str(e), "raw": xml_content}
    
    def blast_and_wait(self, sequence: str, program: str = "blastn",
                      database: str = "nt", **kwargs) -> Dict[str, Any]:
        """Submit BLAST search and wait for results.
        
        Args:
            sequence: Query sequence
            program: BLAST program
            database: Target database
            **kwargs: Additional BLAST parameters
        
        Returns:
            BLAST results
        """
        # Submit BLAST search
        rid = self.blast_sequence(sequence, program, database, **kwargs)
        
        # Wait for results
        retries = 0
        while retries < self.max_retries:
            time.sleep(self.retry_delay)
            status = self.check_status(rid)
            
            if status == "READY":
                return self.get_results(rid)
            elif status == "FAILED":
                raise ValueError(f"BLAST search failed for RID: {rid}")
            elif status == "UNKNOWN":
                logger.warning(f"Unknown status for RID: {rid}")
            
            retries += 1
            if retries % 10 == 0:
                logger.info(f"Still waiting for BLAST results (RID: {rid})...")
        
        raise TimeoutError(f"BLAST search timed out after {self.max_retries * self.retry_delay} seconds")
    
    def search_by_accession(self, accession: str) -> Dict[str, Any]:
        """Search for sequence by accession number.
        
        Args:
            accession: NCBI accession number
        
        Returns:
            Sequence information
        """
        # Use NCBI E-utilities to fetch sequence
        entrez_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {
            "db": "nucleotide",
            "id": accession,
            "rettype": "fasta",
            "retmode": "text"
        }
        
        response = self.session.get(entrez_url, params=params)
        response.raise_for_status()
        
        return {
            "accession": accession,
            "sequence": response.text
        }
    
    def get_database_info(self, database: str = "nt") -> Dict[str, Any]:
        """Get information about a BLAST database.
        
        Args:
            database: Database name
        
        Returns:
            Database information
        """
        params = {
            "CMD": "DisplayDatabases"
        }
        
        response = self.session.get(self.base_url, params=params)
        response.raise_for_status()
        
        # Parse HTML response to extract database info
        # This is a simplified version - actual parsing would be more complex
        return {
            "database": database,
            "available": database in response.text,
            "description": f"NCBI {database} database"
        }
