"""STRING data collector."""

import logging
import json
from typing import Any, Dict, List, Optional, Tuple

from omicverse.external.datacollect.api.string import STRINGClient
from omicverse.external.datacollect.models.protein import Protein
from omicverse.external.datacollect.models.interaction import ProteinInteraction
from .base import BaseCollector
from ..config import settings


logger = logging.getLogger(__name__)


class STRINGCollector(BaseCollector):
    """Collector for STRING protein-protein interaction data."""
    
    def __init__(self, db_session=None):
        api_client = STRINGClient()
        super().__init__(api_client, db_session)
        self.default_species = 9606  # Human by default
        self.default_score_threshold = 400  # Medium confidence
    
    def collect_single(self, identifier: str, **kwargs) -> Dict[str, Any]:
        """Collect data for a single identifier.
        
        Args:
            identifier: Protein identifier
            **kwargs: Additional parameters (species, collect_partners, partner_limit)
        
        Returns:
            Collected data
        """
        species = kwargs.get('species', None)
        collect_partners = kwargs.get('collect_partners', True)
        partner_limit = kwargs.get('partner_limit', 20)
        
        return self.collect_protein(identifier, species, collect_partners, partner_limit)
    
    def collect_protein(self, identifier: str, species: int = None,
                      collect_partners: bool = True, partner_limit: int = 20) -> Dict[str, Any]:
        """Collect STRING data for a single protein.
        
        Args:
            identifier: Protein identifier (UniProt, Ensembl, or gene name)
            species: NCBI taxonomy ID (default: human)
            collect_partners: Whether to collect interaction partners
            partner_limit: Maximum number of partners to collect
        
        Returns:
            Collected interaction data
        """
        if species is None:
            species = self.default_species
            
        logger.info(f"Collecting STRING data for {identifier} (species: {species})")
        
        # Resolve identifier to STRING ID
        string_mappings = self.api_client.get_string_ids([identifier], species)
        
        if not string_mappings:
            raise ValueError(f"Could not map {identifier} to STRING ID")
        
        string_id = string_mappings[0]["stringId"]
        preferred_name = string_mappings[0].get("preferredName", identifier)
        
        data = {
            "identifier": identifier,
            "string_id": string_id,
            "preferred_name": preferred_name,
            "species": species,
            "interactions": [],
            "enrichment": None,
            "homologs": []
        }
        
        if collect_partners:
            # Get interaction partners
            partners = self.api_client.get_interaction_partners(
                [string_id], species, self.default_score_threshold, partner_limit
            )
            
            # Get detailed network
            if partners:
                partner_ids = [p["stringId_B"] for p in partners]
                network = self.api_client.get_network(
                    [string_id] + partner_ids[:10],  # Limit network size
                    species, self.default_score_threshold
                )
                data["network"] = network
            
            data["interactions"] = partners
            data["interaction_count"] = len(partners)
            
            # Get interaction actions (detailed interaction types)
            actions = self.api_client.get_actions(
                [string_id], species, self.default_score_threshold, 50
            )
            data["actions"] = actions
        
        # Get PPI enrichment
        try:
            enrichment = self.api_client.get_ppi_enrichment(
                [string_id], species, self.default_score_threshold
            )
            data["enrichment"] = enrichment
        except Exception as e:
            logger.warning(f"Could not get PPI enrichment: {e}")
        
        return data
    
    def collect_batch(self, identifiers: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Collect data for multiple identifiers."""
        results = []
        for identifier in identifiers:
            try:
                data = self.collect_single(identifier, **kwargs)
                results.append(data)
            except Exception as e:
                logger.error(f"Failed to collect STRING data for {identifier}: {e}")
        return results
    
    def collect_network(self, identifiers: List[str], species: int = None,
                       add_nodes: int = 0) -> Dict[str, Any]:
        """Collect interaction network for multiple proteins.
        
        Args:
            identifiers: List of protein identifiers
            species: NCBI taxonomy ID
            add_nodes: Number of additional nodes to add to network
        
        Returns:
            Network data
        """
        if species is None:
            species = self.default_species
            
        logger.info(f"Collecting STRING network for {len(identifiers)} proteins")
        
        # Map to STRING IDs
        string_mappings = self.api_client.get_string_ids(identifiers, species)
        string_ids = [m["stringId"] for m in string_mappings]
        
        # Get network
        network = self.api_client.get_network(
            string_ids, species, self.default_score_threshold, 
            "functional", add_nodes
        )
        
        # Get enrichment analysis
        enrichment = self.api_client.get_enrichment(string_ids, species)
        
        # Get PPI enrichment
        ppi_enrichment = self.api_client.get_ppi_enrichment(
            string_ids, species, self.default_score_threshold
        )
        
        data = {
            "identifiers": identifiers,
            "string_ids": string_ids,
            "species": species,
            "network": network,
            "enrichment": enrichment,
            "ppi_enrichment": ppi_enrichment,
            "node_count": len(set([e["stringId_A"] for e in network] + 
                                 [e["stringId_B"] for e in network])),
            "edge_count": len(network)
        }
        
        return data
    
    def save_to_database(self, data: Dict[str, Any]) -> List[ProteinInteraction]:
        """Save STRING interaction data to database.
        
        Args:
            data: Collected STRING data
        
        Returns:
            List of saved ProteinInteraction instances
        """
        saved_interactions = []
        
        # Find the main protein
        main_protein = self._get_protein_by_identifier(
            data["identifier"], data["species"]
        )
        
        if not main_protein:
            logger.warning(f"Protein {data['identifier']} not found in database")
            return []
        
        # Update STRING ID if not set
        if not main_protein.string_id:
            main_protein.string_id = data["string_id"]
            self.db_session.commit()
        
        # Process interactions
        for interaction_data in data.get("interactions", []):
            # Get or create partner protein
            partner_string_id = interaction_data["stringId_B"]
            partner_name = interaction_data.get("preferredName_B", partner_string_id)
            
            partner = self._get_or_create_protein_by_string_id(
                partner_string_id, partner_name, data["species"]
            )
            
            # Create interaction
            interaction = self._create_or_update_interaction(
                main_protein, partner, interaction_data
            )
            
            saved_interactions.append(interaction)
        
        # Process detailed actions if available
        for action_data in data.get("actions", []):
            # Find the interaction
            stringA = action_data["stringId_A"]
            stringB = action_data["stringId_B"]
            
            # Update interaction with action details
            interaction = self.db_session.query(ProteinInteraction).filter(
                ((ProteinInteraction.protein1_id == main_protein.id) & 
                 (ProteinInteraction.protein2.has(string_id=stringB))) |
                ((ProteinInteraction.protein2_id == main_protein.id) & 
                 (ProteinInteraction.protein1.has(string_id=stringB)))
            ).first()
            
            if interaction:
                # Add action type to interaction types
                action_type = action_data.get("mode")
                if action_type and interaction.interaction_type:
                    types = set(interaction.interaction_type.split(";"))
                    types.add(action_type)
                    interaction.interaction_type = ";".join(sorted(types))
        
        self.db_session.commit()
        logger.info(f"Saved {len(saved_interactions)} interactions for {data['identifier']}")
        
        return saved_interactions
    
    def _get_protein_by_identifier(self, identifier: str, species: int) -> Optional[Protein]:
        """Get protein by identifier."""
        # Try UniProt accession first
        protein = self.db_session.query(Protein).filter_by(
            accession=identifier
        ).first()
        
        if not protein:
            # Try gene name
            protein = self.db_session.query(Protein).filter_by(
                gene_name=identifier
            ).first()
        
        return protein
    
    def _get_or_create_protein_by_string_id(self, string_id: str, 
                                          preferred_name: str,
                                          species: int) -> Protein:
        """Get or create protein by STRING ID."""
        # Check if protein exists with this STRING ID
        protein = self.db_session.query(Protein).filter_by(
            string_id=string_id
        ).first()
        
        if not protein:
            # Create minimal protein entry
            protein = Protein(
                id=self.generate_id("string_protein", string_id),
                source="STRING",
                accession=string_id,  # Use STRING ID as accession
                protein_name=preferred_name,
                gene_name=preferred_name,
                organism_id=species,
                string_id=string_id,
                sequence="",  # Set empty sequence as it's required
                sequence_length=0
            )
            self.db_session.add(protein)
            self.db_session.flush()
        
        return protein
    
    def _create_or_update_interaction(self, protein1: Protein, protein2: Protein,
                                    interaction_data: Dict) -> ProteinInteraction:
        """Create or update protein interaction."""
        # Check if interaction already exists
        existing = self.db_session.query(ProteinInteraction).filter(
            ((ProteinInteraction.protein1_id == protein1.id) & 
             (ProteinInteraction.protein2_id == protein2.id)) |
            ((ProteinInteraction.protein1_id == protein2.id) & 
             (ProteinInteraction.protein2_id == protein1.id))
        ).first()
        
        if existing:
            # Update scores
            existing.confidence_score = max(
                existing.confidence_score or 0,
                interaction_data.get("score", 0)
            )
            interaction = existing
        else:
            # Create new interaction
            interaction = ProteinInteraction(
                id=self.generate_id("interaction", protein1.id, protein2.id),
                source="STRING",
                protein1_id=protein1.id,
                protein2_id=protein2.id,
                interaction_type="physical",  # Default, will be updated with actions
                confidence_score=interaction_data.get("score", 0),
                evidence=json.dumps({
                    "combined_score": interaction_data.get("score", 0),
                    "neighborhood": interaction_data.get("nscore", 0),
                    "fusion": interaction_data.get("fscore", 0),
                    "cooccurrence": interaction_data.get("pscore", 0),
                    "coexpression": interaction_data.get("ascore", 0),
                    "experimental": interaction_data.get("escore", 0),
                    "database": interaction_data.get("dscore", 0),
                    "textmining": interaction_data.get("tscore", 0)
                })
            )
            self.db_session.add(interaction)
        
        return interaction
    
    def collect_homologs(self, identifier: str, source_species: int = None) -> Dict[str, Any]:
        """Collect homologous proteins in other species.
        
        Args:
            identifier: Protein identifier
            source_species: Source species taxonomy ID
        
        Returns:
            Homolog data
        """
        if source_species is None:
            source_species = self.default_species
            
        logger.info(f"Collecting homologs for {identifier}")
        
        # Map to STRING ID
        string_mappings = self.api_client.get_string_ids([identifier], source_species)
        if not string_mappings:
            raise ValueError(f"Could not map {identifier} to STRING ID")
        
        string_id = string_mappings[0]["stringId"]
        
        # Get homologs
        homologs = self.api_client.get_homology([string_id], source_species)
        
        data = {
            "identifier": identifier,
            "string_id": string_id,
            "source_species": source_species,
            "homologs": homologs,
            "species_count": len(set(h.get("ncbiTaxonId") for h in homologs))
        }
        
        return data
    
    def search_by_name(self, query: str, species: int = None,
                      limit: int = 10) -> List[Dict[str, Any]]:
        """Search for proteins by name.
        
        Args:
            query: Search query
            species: NCBI taxonomy ID
            limit: Maximum results
        
        Returns:
            List of matching proteins
        """
        if species is None:
            species = self.default_species
            
        logger.info(f"Searching STRING for '{query}' in species {species}")
        
        # Resolve query
        results = self.api_client.resolve([query], species)
        
        return results[:limit]