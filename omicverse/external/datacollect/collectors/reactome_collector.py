"""Reactome pathway data collector."""

import logging
from typing import Any, Dict, List, Optional

from omicverse.external.datacollect.api.reactome import ReactomeClient
from omicverse.external.datacollect.models.pathway import Pathway
from omicverse.external.datacollect.models.genomic import Gene
from omicverse.external.datacollect.models.protein import Protein
from .base import BaseCollector
from omicverse.external.datacollect.config.config import settings


logger = logging.getLogger(__name__)


class ReactomeCollector(BaseCollector):
    """Collector for Reactome pathway data."""
    
    def __init__(self, db_session=None):
        api_client = ReactomeClient()
        super().__init__(api_client, db_session)
        self.default_species = "Homo sapiens"
    
    def collect_single(self, identifier: str, **kwargs) -> Dict[str, Any]:
        """Collect Reactome data for a single identifier.
        
        Args:
            identifier: Gene name, protein ID, or pathway ID
            **kwargs: Additional parameters (type, species)
        
        Returns:
            Collected Reactome data
        """
        id_type = kwargs.get('type', 'gene')
        species = kwargs.get('species', self.default_species)
        
        if id_type == 'pathway':
            return self.collect_pathway_data(identifier, **kwargs)
        elif id_type == 'gene':
            return self.collect_gene_pathways(identifier, species, **kwargs)
        elif id_type == 'protein':
            return self.collect_protein_interactions(identifier, **kwargs)
        else:
            raise ValueError(f"Unknown identifier type: {id_type}")
    
    def collect_pathway_data(self, pathway_id: str, **kwargs) -> Dict[str, Any]:
        """Collect detailed pathway information.
        
        Args:
            pathway_id: Reactome pathway ID
            **kwargs: Additional parameters
        
        Returns:
            Pathway data
        """
        logger.info(f"Collecting Reactome pathway data for {pathway_id}")
        
        # Get pathway details
        pathway_details = self.api_client.get_pathway_details(pathway_id)
        
        data = {
            "pathway_id": pathway_details.get("stId"),
            "name": pathway_details.get("displayName"),
            "species": pathway_details.get("species", [{}])[0].get("displayName") if pathway_details.get("species") else None,
            "database_name": pathway_details.get("databaseName"),
            "schema_class": pathway_details.get("schemaClass"),
            "is_in_disease": pathway_details.get("isInDisease", False),
            "is_in_inferred": pathway_details.get("isInferred", False),
            "has_diagram": pathway_details.get("hasDiagram", False)
        }
        
        # Get summation if available
        if pathway_details.get("summation"):
            data["description"] = " ".join([s.get("text", "") for s in pathway_details["summation"]])
        
        # Get literature references
        if pathway_details.get("literatureReference"):
            data["references"] = [
                {
                    "pubmed_id": ref.get("pubMedIdentifier"),
                    "title": ref.get("title"),
                    "journal": ref.get("journal"),
                    "year": ref.get("year")
                }
                for ref in pathway_details["literatureReference"][:5]  # Top 5 references
            ]
        
        # Get pathway hierarchy
        try:
            ancestors = self.api_client.get_entity_ancestors(pathway_id)
            data["hierarchy"] = [
                {
                    "id": ancestor.get("stId"),
                    "name": ancestor.get("displayName")
                }
                for ancestor in ancestors
            ]
        except Exception as e:
            logger.warning(f"Could not get pathway hierarchy: {e}")
            data["hierarchy"] = []
        
        # Get participants if requested
        if kwargs.get('include_participants'):
            try:
                participants = self.api_client.get_pathway_participants(pathway_id)
                data["participants"] = self._process_participants(participants)
            except Exception as e:
                logger.warning(f"Could not get pathway participants: {e}")
                data["participants"] = []
        
        # Get events if requested
        if kwargs.get('include_events'):
            try:
                events = self.api_client.get_pathway_events(pathway_id)
                data["events"] = [
                    {
                        "id": event.get("stId"),
                        "name": event.get("displayName"),
                        "type": event.get("schemaClass")
                    }
                    for event in events[:20]  # Top 20 events
                ]
                data["event_count"] = len(events)
            except Exception as e:
                logger.warning(f"Could not get pathway events: {e}")
                data["events"] = []
        
        # Get orthology information if requested
        if kwargs.get('include_orthology'):
            target_species = kwargs.get('target_species', 'Mus musculus')
            try:
                orthologs = self.api_client.get_orthology(pathway_id, target_species)
                data["orthologs"] = [
                    {
                        "id": orth.get("stId"),
                        "name": orth.get("displayName"),
                        "species": target_species
                    }
                    for orth in orthologs
                ]
            except Exception as e:
                logger.warning(f"Could not get orthology data: {e}")
                data["orthologs"] = []
        
        return data
    
    def collect_gene_pathways(self, gene_name: str, species: str = None, **kwargs) -> Dict[str, Any]:
        """Collect pathways containing a gene.
        
        Args:
            gene_name: Gene name or identifier
            species: Species name
            **kwargs: Additional parameters
        
        Returns:
            Gene pathway data
        """
        if species is None:
            species = self.default_species
        
        logger.info(f"Collecting Reactome pathways for gene {gene_name} in {species}")
        
        # Get pathways for gene
        pathways = self.api_client.get_pathways_by_gene(gene_name, species)
        
        data = {
            "gene": gene_name,
            "species": species,
            "pathway_count": len(pathways),
            "pathways": []
        }
        
        # Process pathways
        pathway_categories = {
            "signal_transduction": [],
            "metabolism": [],
            "gene_expression": [],
            "cell_cycle": [],
            "immune_system": [],
            "disease": [],
            "other": []
        }
        
        for pathway in pathways[:50]:  # Process top 50 pathways
            pathway_info = {
                "id": pathway.get("stId"),
                "name": pathway.get("displayName"),
                "species": pathway.get("species", [{}])[0].get("displayName") if pathway.get("species") else species,
                "has_diagram": pathway.get("hasDiagram", False)
            }
            
            # Categorize pathway
            name_lower = pathway_info["name"].lower()
            if "signal" in name_lower or "transduction" in name_lower:
                pathway_categories["signal_transduction"].append(pathway_info)
            elif "metabol" in name_lower:
                pathway_categories["metabolism"].append(pathway_info)
            elif "gene expression" in name_lower or "transcription" in name_lower:
                pathway_categories["gene_expression"].append(pathway_info)
            elif "cell cycle" in name_lower or "mitosis" in name_lower:
                pathway_categories["cell_cycle"].append(pathway_info)
            elif "immune" in name_lower:
                pathway_categories["immune_system"].append(pathway_info)
            elif "disease" in name_lower or "cancer" in name_lower:
                pathway_categories["disease"].append(pathway_info)
            else:
                pathway_categories["other"].append(pathway_info)
            
            data["pathways"].append(pathway_info)
        
        data["pathway_categories"] = {
            cat: {"count": len(pathways), "pathways": pathways[:5]}  # Top 5 per category
            for cat, pathways in pathway_categories.items()
        }
        
        # Run enrichment analysis if multiple genes provided
        if kwargs.get('gene_list'):
            gene_list = kwargs['gene_list']
            try:
                enrichment = self.api_client.analyze_expression_data(gene_list, species)
                data["enrichment_analysis"] = {
                    "token": enrichment.get("summary", {}).get("token"),
                    "pathways_found": enrichment.get("pathwaysFound"),
                    "identifiers_not_found": enrichment.get("identifiersNotFound", [])
                }
            except Exception as e:
                logger.warning(f"Could not perform enrichment analysis: {e}")
        
        return data
    
    def collect_protein_interactions(self, protein_accession: str, **kwargs) -> Dict[str, Any]:
        """Collect protein-protein interactions.
        
        Args:
            protein_accession: Protein accession (UniProt ID)
            **kwargs: Additional parameters
        
        Returns:
            Protein interaction data
        """
        logger.info(f"Collecting Reactome interactions for protein {protein_accession}")
        
        # Get interactors
        interactors = self.api_client.get_interactors(protein_accession)
        
        data = {
            "protein": protein_accession,
            "interactor_count": len(interactors),
            "interactors": []
        }
        
        # Process interactors
        for interactor in interactors:
            interactor_info = {
                "accession": interactor.get("acc"),
                "gene_name": interactor.get("alias"),
                "score": interactor.get("score"),
                "evidence_count": len(interactor.get("evidences", [])),
                "interaction_type": interactor.get("interactionType")
            }
            
            # Add evidence details
            if interactor.get("evidences"):
                interactor_info["evidences"] = [
                    {
                        "pubmed_id": ev.get("pubmedId"),
                        "interaction_type": ev.get("interactionType")
                    }
                    for ev in interactor["evidences"][:3]  # Top 3 evidences
                ]
            
            data["interactors"].append(interactor_info)
        
        # Get pathways for this protein
        try:
            pathways = self.api_client.get_pathways_by_gene(protein_accession)
            data["pathway_count"] = len(pathways)
            data["top_pathways"] = [
                {
                    "id": p.get("stId"),
                    "name": p.get("displayName")
                }
                for p in pathways[:10]
            ]
        except Exception as e:
            logger.warning(f"Could not get pathways for protein: {e}")
            data["pathway_count"] = 0
            data["top_pathways"] = []
        
        return data
    
    def _process_participants(self, participants: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process pathway participants.
        
        Args:
            participants: Raw participants data
        
        Returns:
            Processed participants list
        """
        processed = []
        
        for participant in participants[:50]:  # Process top 50 participants
            p_info = {
                "id": participant.get("stId"),
                "name": participant.get("displayName"),
                "type": participant.get("schemaClass")
            }
            
            # Add gene names if available
            if participant.get("geneNames"):
                p_info["gene_names"] = participant["geneNames"]
            
            # Add identifier if available
            if participant.get("identifier"):
                p_info["identifier"] = participant["identifier"]
            
            processed.append(p_info)
        
        return processed
    
    def collect_batch(self, identifiers: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Collect data for multiple identifiers."""
        results = []
        for identifier in identifiers:
            try:
                data = self.collect_single(identifier, **kwargs)
                results.append(data)
            except Exception as e:
                logger.error(f"Failed to collect Reactome data for {identifier}: {e}")
                results.append({
                    "identifier": identifier,
                    "error": str(e)
                })
        return results
    
    def save_to_database(self, data: Dict[str, Any]) -> Any:
        """Save Reactome data to database.
        
        Args:
            data: Reactome data
        
        Returns:
            Saved entity (Pathway, Gene, or Protein)
        """
        if "pathway_id" in data:
            return self._save_pathway_data(data)
        elif "gene" in data:
            return self._save_gene_data(data)
        elif "protein" in data:
            return self._save_protein_data(data)
        else:
            logger.warning("Cannot determine data type to save")
            return None
    
    def _save_pathway_data(self, data: Dict[str, Any]) -> Pathway:
        """Save pathway data to database."""
        pathway_id = data["pathway_id"]
        
        # Check if pathway exists
        existing = self.db_session.query(Pathway).filter_by(
            pathway_id=pathway_id
        ).first()
        
        if existing:
            logger.info(f"Updating existing pathway {pathway_id}")
            pathway = existing
        else:
            pathway = Pathway(
                id=self.generate_id("reactome_pathway", pathway_id),
                pathway_id=pathway_id,
                source="Reactome"
            )
        
        # Update fields
        pathway.name = data.get("name")
        pathway.description = data.get("description")
        pathway.organism = data.get("species")
        
        # Store Reactome-specific data
        if not pathway.metadata:
            pathway.metadata = {}
        
        pathway.metadata["reactome"] = {
            "database_name": data.get("database_name"),
            "schema_class": data.get("schema_class"),
            "is_in_disease": data.get("is_in_disease"),
            "has_diagram": data.get("has_diagram"),
            "hierarchy": data.get("hierarchy", []),
            "references": data.get("references", []),
            "event_count": data.get("event_count", 0)
        }
        
        if not existing:
            self.db_session.add(pathway)
        
        self.db_session.commit()
        logger.info(f"Saved Reactome pathway {pathway_id}")
        
        return pathway
    
    def _save_gene_data(self, data: Dict[str, Any]) -> Gene:
        """Save gene pathway data to database."""
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
                id=self.generate_id("reactome_gene", gene_name),
                gene_id=f"reactome:{gene_name}",
                symbol=gene_name,
                source="Reactome"
            )
        
        gene.species = data.get("species")
        
        # Store Reactome pathway data
        if not gene.annotations:
            gene.annotations = {}
        
        gene.annotations["reactome_pathways"] = {
            "pathway_count": data.get("pathway_count", 0),
            "pathways": data.get("pathways", [])[:20],  # Store top 20 pathways
            "pathway_categories": data.get("pathway_categories", {}),
            "enrichment_analysis": data.get("enrichment_analysis", {})
        }
        
        if not existing:
            self.db_session.add(gene)
        
        self.db_session.commit()
        logger.info(f"Saved Reactome data for gene {gene_name}")
        
        return gene
    
    def _save_protein_data(self, data: Dict[str, Any]) -> Protein:
        """Save protein interaction data to database."""
        protein_acc = data["protein"]
        
        # Check if protein exists
        existing = self.db_session.query(Protein).filter_by(
            uniprot_accession=protein_acc
        ).first()
        
        if existing:
            logger.info(f"Updating existing protein {protein_acc}")
            protein = existing
        else:
            protein = Protein(
                id=self.generate_id("reactome_protein", protein_acc),
                uniprot_accession=protein_acc,
                sequence="",  # Required field
                sequence_length=0,
                source="Reactome"
            )
        
        # Store Reactome interaction data
        if not protein.annotations:
            protein.annotations = {}
        
        protein.annotations["reactome_interactions"] = {
            "interactor_count": data.get("interactor_count", 0),
            "interactors": data.get("interactors", [])[:20],  # Store top 20 interactors
            "pathway_count": data.get("pathway_count", 0),
            "top_pathways": data.get("top_pathways", [])
        }
        
        if not existing:
            self.db_session.add(protein)
        
        self.db_session.commit()
        logger.info(f"Saved Reactome data for protein {protein_acc}")
        
        return protein