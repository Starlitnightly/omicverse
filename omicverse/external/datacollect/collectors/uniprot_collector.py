"""UniProt data collector."""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from omicverse.external.datacollect.api.uniprot import UniProtClient
from omicverse.external.datacollect.models.protein import Protein, ProteinFeature, GOTerm
from .base import BaseCollector


logger = logging.getLogger(__name__)


class UniProtCollector(BaseCollector):
    """Collector for UniProt data."""
    
    def __init__(self, db_session=None):
        api_client = UniProtClient()
        super().__init__(api_client, db_session)
    
    def collect_single(self, accession: str, include_features: bool = True) -> Dict[str, Any]:
        """Collect data for a single UniProt entry.
        
        Args:
            accession: UniProt accession
            include_features: Whether to include protein features
        
        Returns:
            Collected data dictionary
        """
        logger.info(f"Collecting UniProt data for {accession}")
        
        # Get main entry data
        entry_data = self.api_client.get_entry(accession)
        
        # Extract relevant fields
        primary_accession = entry_data.get("primaryAccession", accession)
        
        data = {
            "accession": primary_accession,
            "entry_name": entry_data.get("uniProtkbId"),
            "protein_name": self._extract_protein_name(entry_data),
            "gene_name": self._extract_gene_name(entry_data),
            "organism": entry_data.get("organism", {}).get("scientificName"),
            "organism_id": entry_data.get("organism", {}).get("taxonId"),
            "sequence": entry_data.get("sequence", {}).get("value", ""),
            "sequence_length": entry_data.get("sequence", {}).get("length", 0),
            "molecular_weight": entry_data.get("sequence", {}).get("molWeight", 0),
            "function_description": self._extract_function(entry_data),
            "go_terms": self._extract_go_terms(entry_data),
            "pdb_ids": self._extract_pdb_ids(entry_data),
        }
        
        # Add features if requested
        if include_features:
            data["features"] = entry_data.get("features", [])
        
        return data
    
    def collect_batch(self, accessions: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Collect data for multiple UniProt entries.
        
        Args:
            accessions: List of UniProt accessions
            **kwargs: Additional parameters
        
        Returns:
            List of collected data dictionaries
        """
        logger.info(f"Collecting batch data for {len(accessions)} entries")
        
        results = []
        for accession in accessions:
            try:
                data = self.collect_single(accession, **kwargs)
                results.append(data)
            except Exception as e:
                logger.error(f"Failed to collect {accession}: {e}")
                continue
        
        return results
    
    def save_to_database(self, data: Dict[str, Any]) -> Protein:
        """Save UniProt data to database.
        
        Args:
            data: Collected UniProt data
        
        Returns:
            Saved Protein model instance
        """
        # Check if protein already exists
        existing = self.db_session.query(Protein).filter_by(
            accession=data["accession"]
        ).first()
        
        if existing:
            logger.info(f"Updating existing protein {data['accession']}")
            protein = existing
        else:
            protein = Protein(
                id=self.generate_id("uniprot", data["accession"]),
                source="UniProt",
            )
        
        # Update fields
        for field in ["accession", "entry_name", "protein_name", "gene_name", 
                     "organism", "organism_id", "sequence", "sequence_length",
                     "molecular_weight", "function_description"]:
            if field in data:
                setattr(protein, field, data[field])
        
        # Handle PDB IDs
        if data.get("pdb_ids"):
            protein.pdb_ids = ",".join(data["pdb_ids"])
            protein.has_3d_structure = "Y"
        
        # Handle GO terms
        if data.get("go_terms"):
            for go_data in data["go_terms"]:
                go_term = self._get_or_create_go_term(go_data)
                if go_term not in protein.go_terms:
                    protein.go_terms.append(go_term)
        
        # Handle features
        if data.get("features"):
            # Remove existing features
            protein.features.clear()
            
            # Add new features
            for i, feature_data in enumerate(data["features"]):
                feature = ProteinFeature(
                    id=self.generate_id("feature", protein.id, feature_data.get("type"), 
                                      feature_data.get("location", {}).get("start", {}).get("value"),
                                      feature_data.get("location", {}).get("end", {}).get("value"),
                                      feature_data.get("description", ""), i),
                    source="UniProt",
                    feature_type=feature_data.get("type", ""),
                    start_position=feature_data.get("location", {}).get("start", {}).get("value"),
                    end_position=feature_data.get("location", {}).get("end", {}).get("value"),
                    description=feature_data.get("description", ""),
                )
                protein.features.append(feature)
        
        # Save to database
        if not existing:
            self.db_session.add(protein)
        
        self.db_session.commit()
        logger.info(f"Saved protein {protein.accession} to database")
        
        return protein
    
    def _extract_protein_name(self, entry_data: Dict) -> str:
        """Extract protein name from entry data."""
        protein_desc = entry_data.get("proteinDescription", {})
        recommended_name = protein_desc.get("recommendedName", {})
        full_name = recommended_name.get("fullName", {})
        return full_name.get("value", "")
    
    def _extract_gene_name(self, entry_data: Dict) -> Optional[str]:
        """Extract primary gene name."""
        genes = entry_data.get("genes", [])
        if genes:
            primary_gene = genes[0].get("geneName", {})
            return primary_gene.get("value")
        return None
    
    def _extract_function(self, entry_data: Dict) -> Optional[str]:
        """Extract function description."""
        comments = entry_data.get("comments", [])
        for comment in comments:
            if comment.get("commentType") == "FUNCTION":
                texts = comment.get("texts", [])
                if texts:
                    return texts[0].get("value")
        return None
    
    def _extract_go_terms(self, entry_data: Dict) -> List[Dict]:
        """Extract GO terms."""
        go_terms = []
        xrefs = entry_data.get("uniProtKBCrossReferences", [])
        
        for xref in xrefs:
            if xref.get("database") == "GO":
                go_terms.append({
                    "id": xref.get("id"),
                    "properties": xref.get("properties", {})
                })
        
        return go_terms
    
    def _extract_pdb_ids(self, entry_data: Dict) -> List[str]:
        """Extract PDB IDs."""
        pdb_ids = []
        xrefs = entry_data.get("uniProtKBCrossReferences", [])
        
        for xref in xrefs:
            if xref.get("database") == "PDB":
                pdb_ids.append(xref.get("id"))
        
        return pdb_ids
    
    def _get_or_create_go_term(self, go_data: Dict) -> GOTerm:
        """Get or create GO term."""
        go_id = go_data["id"]
        
        go_term = self.db_session.query(GOTerm).filter_by(go_id=go_id).first()
        
        if not go_term:
            # Extract properties from list format
            properties = go_data.get("properties", [])
            props_dict = {}
            if isinstance(properties, list):
                for prop in properties:
                    if isinstance(prop, dict) and "key" in prop and "value" in prop:
                        props_dict[prop["key"]] = prop["value"]
            else:
                props_dict = properties
            
            # Extract GO term name and namespace
            go_term_value = props_dict.get("GoTerm", "")
            # Parse namespace from term (e.g., "C:centrosome" -> namespace="C", name="centrosome")
            if ":" in go_term_value:
                namespace_abbr, name = go_term_value.split(":", 1)
                namespace_map = {
                    "C": "cellular_component",
                    "F": "molecular_function", 
                    "P": "biological_process"
                }
                namespace = namespace_map.get(namespace_abbr, "")
            else:
                name = go_term_value
                namespace = ""
            
            go_term = GOTerm(
                id=self.generate_id("go", go_id),
                source="UniProt",
                go_id=go_id,
                name=name,
                namespace=namespace,
            )
            self.db_session.add(go_term)
        
        return go_term
    
    def search_and_collect(
        self,
        query: str,
        max_results: int = 100,
        fields: Optional[List[str]] = None,
        **kwargs
    ) -> List[Protein]:
        """Search UniProt and collect results.
        
        Args:
            query: UniProt search query
            max_results: Maximum number of results
            fields: Fields to retrieve
            **kwargs: Additional parameters
        
        Returns:
            List of saved Protein instances
        """
        logger.info(f"Searching UniProt with query: {query}")
        
        # Default fields if not specified
        if not fields:
            fields = ["accession", "id", "gene_names", "organism_name", "length"]
        
        # Search
        search_results = self.api_client.search(
            query=query,
            fields=fields,
            size=min(max_results, 500)  # UniProt max is 500 per request
        )
        
        # Collect data for each result
        proteins = []
        for result in search_results.get("results", []):
            accession = result.get("primaryAccession")
            if accession:
                try:
                    protein = self.process_and_save(accession, **kwargs)
                    if protein:
                        proteins.append(protein)
                except Exception as e:
                    logger.error(f"Failed to process {accession}: {e}")
                    continue
        
        logger.info(f"Collected and saved {len(proteins)} proteins")
        return proteins