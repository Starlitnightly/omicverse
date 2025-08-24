"""KEGG data collector."""

import logging
from typing import Any, Dict, List, Optional

from src.api.kegg import KEGGClient
from src.models.pathway import Pathway
from src.models.genomic import Gene
from src.models.protein import Protein
from .base import BaseCollector
from ..config.config import settings


logger = logging.getLogger(__name__)


class KEGGCollector(BaseCollector):
    """Collector for KEGG pathway and gene data."""
    
    def __init__(self, db_session=None):
        api_client = KEGGClient()
        super().__init__(api_client, db_session)
    
    def collect_single(self, pathway_id: str) -> Dict[str, Any]:
        """Collect data for a single KEGG pathway.
        
        Args:
            pathway_id: KEGG pathway ID (e.g., "hsa04110")
        
        Returns:
            Collected pathway data
        """
        logger.info(f"Collecting KEGG pathway {pathway_id}")
        
        # Get pathway information
        pathway_data = self.api_client.get_pathway(pathway_id)
        
        # Get genes in pathway
        genes = self.api_client.get_genes_by_pathway(pathway_id)
        
        # Get linked pathways
        linked_pathways = self.api_client.link("pathway", "pathway", pathway_id)
        
        data = {
            "pathway_id": pathway_id,
            "name": pathway_data.get("NAME", ""),
            "description": pathway_data.get("DESCRIPTION", ""),
            "organism": pathway_data.get("ORGANISM", ""),
            "category": pathway_data.get("CLASS", ""),
            "genes": genes,
            "gene_count": len(genes),
            "linked_pathways": linked_pathways,
            "raw_data": pathway_data
        }
        
        return data
    
    def collect_gene(self, gene_id: str) -> Dict[str, Any]:
        """Collect data for a KEGG gene.
        
        Args:
            gene_id: KEGG gene ID (e.g., "hsa:7157")
        
        Returns:
            Gene data
        """
        logger.info(f"Collecting KEGG gene {gene_id}")
        
        # Get gene information
        gene_data = self.api_client.get_gene(gene_id)
        
        # Get pathways containing this gene
        pathways = self.api_client.get_pathways_by_gene(gene_id)
        
        # Try to get UniProt mapping
        uniprot_mapping = self.api_client.conv("uniprot", "genes", [gene_id])
        
        data = {
            "kegg_id": gene_id,
            "name": gene_data.get("NAME", ""),
            "definition": gene_data.get("DEFINITION", ""),
            "organism": gene_data.get("ORGANISM", ""),
            "pathways": pathways,
            "pathway_count": len(pathways),
            "uniprot_ids": list(uniprot_mapping.values()),
            "raw_data": gene_data
        }
        
        return data
    
    def save_to_database(self, data: Dict[str, Any]) -> Pathway:
        """Save pathway data to database."""
        # Check if pathway already exists
        existing = self.db_session.query(Pathway).filter_by(
            pathway_id=data["pathway_id"]
        ).first()
        
        if existing:
            logger.info(f"Updating existing pathway {data['pathway_id']}")
            pathway = existing
        else:
            pathway = Pathway(
                id=self.generate_id("kegg_pathway", data["pathway_id"]),
                source="KEGG",
            )
        
        # Update fields
        pathway.pathway_id = data["pathway_id"]
        pathway.name = data["name"]
        pathway.description = data["description"]
        pathway.organism = data["organism"]
        pathway.category = data["category"]
        pathway.database = "KEGG"
        
        # Add genes if they exist in our database
        for gene_link in data["genes"]:
            kegg_gene_id = gene_link["target"]
            
            # Try to find gene in our database
            gene = self.db_session.query(Gene).filter_by(
                kegg_id=kegg_gene_id
            ).first()
            
            if gene and hasattr(pathway, 'genes') and gene not in pathway.genes:
                pathway.genes.append(gene)
        
        # Save to database
        if not existing:
            self.db_session.add(pathway)
        
        self.db_session.commit()
        logger.info(f"Saved pathway {pathway.pathway_id} to database")
        
        return pathway
    
    def save_gene_to_database(self, data: Dict[str, Any]) -> Gene:
        """Save KEGG gene data to database."""
        # Extract organism and gene symbol
        kegg_id = data["kegg_id"]
        organism_code, gene_num = kegg_id.split(":", 1)
        
        # Check if gene already exists
        existing = self.db_session.query(Gene).filter_by(
            kegg_id=kegg_id
        ).first()
        
        if existing:
            logger.info(f"Updating existing gene {kegg_id}")
            gene = existing
        else:
            gene = Gene(
                id=self.generate_id("kegg_gene", kegg_id),
                gene_id=kegg_id,  # Set the required gene_id field
                source="KEGG",
            )
        
        # Update fields
        gene.kegg_id = kegg_id
        gene.symbol = data["name"].split(",")[0] if data["name"] else gene_num
        gene.description = data["definition"]
        gene.organism = data["organism"]
        
        # Add UniProt associations if available
        for uniprot_id in data.get("uniprot_ids", []):
            # Clean UniProt ID (remove 'up:' prefix)
            if uniprot_id.startswith("up:"):
                uniprot_id = uniprot_id[3:]
                
            # Find protein in database
            protein = self.db_session.query(Protein).filter_by(
                accession=uniprot_id
            ).first()
            
            if protein:
                # Update protein's gene information if not set
                if not protein.gene_name:
                    protein.gene_name = gene.symbol
        
        # Save to database
        if not existing:
            self.db_session.add(gene)
        
        self.db_session.commit()
        logger.info(f"Saved gene {gene.kegg_id} to database")
        
        return gene
    
    def collect_batch(self, identifiers: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Collect data for multiple identifiers."""
        results = []
        for identifier in identifiers:
            try:
                data = self.collect_single(identifier, **kwargs)
                results.append(data)
            except Exception as e:
                logger.error(f"Failed to collect KEGG data for {identifier}: {e}")
        return results
    
    def search_pathways(self, query: str, organism: str = "hsa") -> List[Pathway]:
        """Search for pathways by keyword.
        
        Args:
            query: Search query
            organism: Organism code (default: hsa for human)
        
        Returns:
            List of matching pathways
        """
        logger.info(f"Searching KEGG pathways for '{query}' in {organism}")
        
        # Search pathways
        results = self.api_client.find(f"{organism}", query)
        
        pathways = []
        for result in results:
            pathway_id = result["entry_id"]
            if pathway_id.startswith(organism):
                try:
                    data = self.collect_single(pathway_id)
                    pathway = self.save_to_database(data)
                    pathways.append(pathway)
                except Exception as e:
                    logger.error(f"Failed to process pathway {pathway_id}: {e}")
        
        return pathways
    
    def collect_pathways_for_protein(self, protein: Protein) -> List[Pathway]:
        """Collect KEGG pathways for a protein.
        
        Args:
            protein: Protein instance
        
        Returns:
            List of pathways
        """
        pathways = []
        
        # Convert UniProt to KEGG gene ID
        kegg_mapping = self.api_client.conv("hsa", "uniprot", [f"up:{protein.accession}"])
        
        for uniprot_id, kegg_id in kegg_mapping.items():
            # Get pathways for this gene
            pathway_links = self.api_client.get_pathways_by_gene(kegg_id)
            
            for link in pathway_links:
                pathway_id = link["target"]
                try:
                    data = self.collect_single(pathway_id)
                    pathway = self.save_to_database(data)
                    
                    # Add protein-pathway association
                    if hasattr(pathway, 'proteins') and protein not in pathway.proteins:
                        pathway.proteins.append(protein)
                        self.db_session.commit()
                    
                    pathways.append(pathway)
                except Exception as e:
                    logger.error(f"Failed to process pathway {pathway_id}: {e}")
        
        return pathways
