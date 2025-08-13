"""InterPro data collector."""

import logging
import json
from typing import Any, Dict, List, Optional

from omicverse.external.datacollect.api.interpro import InterProClient
from omicverse.external.datacollect.models.interpro import Domain, DomainLocation
from omicverse.external.datacollect.models.protein import Protein
from .base import BaseCollector
from omicverse.external.datacollect.config.config import settings


logger = logging.getLogger(__name__)


class InterProCollector(BaseCollector):
    """Collector for InterPro domain and family data."""
    
    def __init__(self, db_session=None):
        api_client = InterProClient()
        super().__init__(api_client, db_session)
    
    def collect_single(self, uniprot_accession: str) -> Dict[str, Any]:
        """Collect InterPro annotations for a protein.
        
        Args:
            uniprot_accession: UniProt accession
        
        Returns:
            Collected InterPro data
        """
        logger.info(f"Collecting InterPro data for {uniprot_accession}")
        
        # Get all InterPro entries for the protein
        entries = self.api_client.get_protein_entries(uniprot_accession)
        
        # Get domain predictions
        domains = self.api_client.get_domains_for_protein(uniprot_accession)
        
        # Organize data
        data = {
            "uniprot_accession": uniprot_accession,
            "entries": entries,
            "domains": domains,
            "total_entries": len(entries),
            "domain_count": len([d for d in domains if d["type"] == "domain"]),
            "family_count": len([d for d in domains if d["type"] == "family"]),
        }
        
        return data
    
    def save_to_database(self, data: Dict[str, Any]) -> List[Domain]:
        """Save InterPro data to database.
        
        Args:
            data: Collected InterPro data
        
        Returns:
            List of saved Domain instances
        """
        # Get the protein
        protein = self.db_session.query(Protein).filter_by(
            accession=data["uniprot_accession"]
        ).first()
        
        if not protein:
            logger.warning(f"Protein {data['uniprot_accession']} not found in database")
            return []
        
        saved_domains = []
        
        # Process each domain/family
        for domain_data in data["domains"]:
            # Get or create domain
            domain = self._get_or_create_domain(domain_data)
            saved_domains.append(domain)
            
            # Add protein-domain association if not exists
            if hasattr(protein, 'domains') and domain not in protein.domains:
                protein.domains.append(domain)
            
            # Add domain locations
            for location in domain_data.get("locations", []):
                self._add_domain_location(protein, domain, location)
        
        self.db_session.commit()
        logger.info(f"Saved {len(saved_domains)} domains for {data['uniprot_accession']}")
        
        return saved_domains
    
    def _get_or_create_domain(self, domain_data: Dict) -> Domain:
        """Get or create a domain entry."""
        interpro_id = domain_data["interpro_id"]
        
        domain = self.db_session.query(Domain).filter_by(
            interpro_id=interpro_id
        ).first()
        
        if not domain:
            # Get full entry details
            try:
                entry_details = self.api_client.get_entry(interpro_id)
                description = entry_details.get("description", "")
                member_dbs = json.dumps(entry_details.get("member_databases", {}))
            except:
                description = ""
                member_dbs = "{}"
            
            domain = Domain(
                id=self.generate_id("interpro", interpro_id),
                source="InterPro",
                interpro_id=interpro_id,
                name=domain_data["name"],
                type=domain_data["type"],
                description=description,
                member_databases=member_dbs,
            )
            self.db_session.add(domain)
        
        return domain
    
    def _add_domain_location(self, protein: Protein, domain: Domain, 
                           location_data: Dict) -> None:
        """Add domain location on protein."""
        # InterPro provides fragment locations
        for fragment in location_data.get("fragments", []):
            start = fragment.get("start")
            end = fragment.get("end")
            
            if start and end:
                # Check if location already exists
                existing = self.db_session.query(DomainLocation).filter_by(
                    protein_id=protein.id,
                    domain_id=domain.id,
                    start_position=start,
                    end_position=end
                ).first()
                
                if not existing:
                    location = DomainLocation(
                        id=self.generate_id("domain_loc", protein.id, domain.id, start, end),
                        source="InterPro",
                        protein_id=protein.id,
                        domain_id=domain.id,
                        start_position=start,
                        end_position=end,
                        score=location_data.get("score")
                    )
                    self.db_session.add(location)
    
    def collect_batch(self, identifiers: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Collect data for multiple identifiers."""
        results = []
        for identifier in identifiers:
            try:
                data = self.collect_single(identifier, **kwargs)
                results.append(data)
            except Exception as e:
                logger.error(f"Failed to collect InterPro data for {identifier}: {e}")
        return results
    
    def collect_for_proteins(self, proteins: List[Protein]) -> Dict[str, List[Domain]]:
        """Collect InterPro data for multiple proteins.
        
        Args:
            proteins: List of Protein instances
        
        Returns:
            Dictionary mapping accessions to domains
        """
        results = {}
        
        for protein in proteins:
            try:
                data = self.collect_single(protein.accession)
                domains = self.save_to_database(data)
                results[protein.accession] = domains
            except Exception as e:
                logger.error(f"Failed to collect InterPro data for {protein.accession}: {e}")
                results[protein.accession] = []
        
        return results
    
    def search_by_domain(self, interpro_id: str, taxonomy_id: Optional[int] = None,
                        max_results: int = 100) -> List[Protein]:
        """Find proteins containing a specific InterPro domain.
        
        Args:
            interpro_id: InterPro entry ID
            taxonomy_id: Optional taxonomy filter
            max_results: Maximum results to return
        
        Returns:
            List of proteins with this domain
        """
        logger.info(f"Searching for proteins with domain {interpro_id}")
        
        # Get proteins from InterPro
        result = self.api_client.get_entry_proteins(interpro_id, taxonomy_id)
        
        proteins = []
        count = 0
        
        for protein_data in result.get("results", []):
            if count >= max_results:
                break
                
            accession = protein_data.get("metadata", {}).get("accession")
            if accession:
                # Check if protein exists in our database
                protein = self.db_session.query(Protein).filter_by(
                    accession=accession
                ).first()
                
                if protein:
                    proteins.append(protein)
                    count += 1
        
        logger.info(f"Found {len(proteins)} proteins with domain {interpro_id}")
        return proteins