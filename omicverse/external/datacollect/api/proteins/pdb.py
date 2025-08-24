"""PDB API client."""

import logging
from typing import Any, Dict, List, Optional

from ..base import BaseAPIClient
from ...config import settings


logger = logging.getLogger(__name__)


class PDBClient(BaseAPIClient):
    """Client for RCSB PDB REST API."""
    
    def __init__(self, **kwargs):
        rate_limit = kwargs.pop("rate_limit", 10)
        super().__init__(
            base_url=settings.api.pdb_base_url,
            rate_limit=rate_limit,
            **kwargs
        )
        self.search_url = settings.api.pdb_search_url
    
    def get_default_headers(self) -> Dict[str, str]:
        """Get PDB-specific headers."""
        headers = super().get_default_headers()
        headers.update({
            "Accept": "application/json",
        })
        return headers
    
    def get_entry(self, pdb_id: str) -> Dict[str, Any]:
        """Get PDB entry data.
        
        Args:
            pdb_id: 4-character PDB ID
        
        Returns:
            PDB entry data
        """
        endpoint = f"/rest/v1/core/entry/{pdb_id.upper()}"
        response = self.get(endpoint)
        return response.json()
    
    def get_structure(self, pdb_id: str, format: str = "pdb") -> str:
        """Download structure file.
        
        Args:
            pdb_id: 4-character PDB ID
            format: File format (pdb, cif, mmtf)
        
        Returns:
            Structure file content
        """
        endpoint = f"/download/{pdb_id.upper()}.{format}"
        response = self.get(endpoint)
        return response.text
    
    def search(
        self,
        query: Dict[str, Any],
        return_type: str = "entry",
        rows: int = 25,
        start: int = 0,
    ) -> Dict[str, Any]:
        """Search PDB using structured query.
        
        Args:
            query: Search query in PDB search API format
            return_type: Type of results to return
            rows: Number of results
            start: Starting offset
        
        Returns:
            Search results
        """
        search_request = {
            "query": query,
            "return_type": return_type,
            "request_options": {
                "pager": {
                    "start": start,
                    "rows": rows,
                }
            }
        }
        
        response = self.search_client.post(
            "/graphql",
            json=search_request,
        )
        return response.json()
    
    def text_search(self, query: str, **kwargs) -> Dict[str, Any]:
        """Perform text search.
        
        Args:
            query: Text search query
            **kwargs: Additional search parameters
        
        Returns:
            Search results
        """
        search_query = {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "value": query,
            }
        }
        return self.search(search_query, **kwargs)
    
    def get_ligands(self, pdb_id: str) -> List[Dict[str, Any]]:
        """Get ligands for a PDB entry.
        
        Args:
            pdb_id: 4-character PDB ID
        
        Returns:
            List of ligands
        """
        endpoint = f"/graphql"
        query = {
            "query": f"""
            {{
              nonpolymer_entities(entry_id: "{pdb_id.upper()}") {{
                rcsb_id
                rcsb_nonpolymer_entity {{
                  comp_id
                  name
                  formula
                  formula_weight
                }}
              }}
            }}
            """
        }
        response = self.post(endpoint, json=query)
        data = response.json()
        if "data" in data and "nonpolymer_entities" in data["data"] and data["data"]["nonpolymer_entities"]:
            return data["data"]["nonpolymer_entities"]
        return []
    
    def get_assemblies(self, pdb_id: str) -> List[Dict[str, Any]]:
        """Get biological assemblies.
        
        Args:
            pdb_id: 4-character PDB ID
        
        Returns:
            List of assemblies
        """
        endpoint = f"/rest/v1/core/assembly/{pdb_id.upper()}"
        response = self.get(endpoint)
        return response.json()
    
    def get_polymer_entities(self, pdb_id: str) -> List[Dict[str, Any]]:
        """Get polymer entities (chains).
        
        Args:
            pdb_id: 4-character PDB ID
        
        Returns:
            List of polymer entities
        """
        endpoint = f"/graphql"
        query = {
            "query": f"""
            {{
              polymer_entities(entry_id: "{pdb_id.upper()}") {{
                rcsb_id
                entity_poly {{
                  pdbx_seq_one_letter_code
                  entity_id
                }}
                rcsb_polymer_entity_instance_container_identifiers {{
                  auth_asym_ids
                }}
                rcsb_polymer_entity_container_identifiers {{
                  uniprot_ids
                }}
                rcsb_entity_source_organism {{
                  ncbi_scientific_name
                }}
              }}
            }}
            """
        }
        response = self.post(endpoint, json=query)
        data = response.json()
        if "data" in data and "polymer_entities" in data["data"] and data["data"]["polymer_entities"]:
            return data["data"]["polymer_entities"]
        return []
    
    def sequence_search(
        self,
        sequence: str,
        sequence_type: str = "protein",
        e_value: float = 0.1,
    ) -> Dict[str, Any]:
        """Search by sequence similarity.
        
        Args:
            sequence: Query sequence
            sequence_type: Type of sequence (protein/dna/rna)
            e_value: E-value cutoff
        
        Returns:
            Sequence search results
        """
        search_query = {
            "type": "terminal",
            "service": "sequence",
            "parameters": {
                "value": sequence,
                "sequence_type": sequence_type,
                "evalue_cutoff": e_value,
            }
        }
        return self.search(search_query)
