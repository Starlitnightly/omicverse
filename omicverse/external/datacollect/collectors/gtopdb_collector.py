"""Guide to PHARMACOLOGY data collector."""

import logging
from typing import Any, Dict, List, Optional

from omicverse.external.datacollect.api.gtopdb import GtoPdbClient
from omicverse.external.datacollect.models.genomic import Gene
from omicverse.external.datacollect.models.protein import Protein
from omicverse.external.datacollect.models.disease import Disease
from .base import BaseCollector
from omicverse.external.datacollect.config.config import settings


logger = logging.getLogger(__name__)


class GtoPdbCollector(BaseCollector):
    """Collector for Guide to PHARMACOLOGY data."""
    
    def __init__(self, db_session=None):
        api_client = GtoPdbClient()
        super().__init__(api_client, db_session)
    
    def collect_single(self, identifier: str, **kwargs) -> Dict[str, Any]:
        """Collect GtoPdb data for a single identifier.
        
        Args:
            identifier: Gene symbol, target ID, ligand ID, or disease name
            **kwargs: Additional parameters (type)
        
        Returns:
            Collected GtoPdb data
        """
        id_type = kwargs.get('type', 'gene')
        
        if id_type == 'gene':
            return self.collect_gene_targets(identifier, **kwargs)
        elif id_type == 'target':
            return self.collect_target_data(int(identifier), **kwargs)
        elif id_type == 'ligand':
            return self.collect_ligand_data(int(identifier), **kwargs)
        elif id_type == 'disease':
            return self.collect_disease_data(identifier, **kwargs)
        else:
            raise ValueError(f"Unknown identifier type: {id_type}")
    
    def collect_gene_targets(self, gene_symbol: str, **kwargs) -> Dict[str, Any]:
        """Collect drug target data for a gene.
        
        Args:
            gene_symbol: Gene symbol
            **kwargs: Additional parameters
        
        Returns:
            Gene target data
        """
        logger.info(f"Collecting GtoPdb targets for gene {gene_symbol}")
        
        # Get targets for gene
        targets = self.api_client.get_target_by_gene(gene_symbol)
        
        data = {
            "gene_symbol": gene_symbol,
            "target_count": len(targets),
            "targets": [],
            "target_families": {},
            "target_types": {},
            "associated_ligands": [],
            "approved_drugs": []
        }
        
        # Process targets
        for target in targets:
            target_info = {
                "id": target.get("targetId"),
                "name": target.get("name"),
                "abbreviation": target.get("abbreviation"),
                "type": target.get("type"),
                "family": target.get("familyName"),
                "family_id": target.get("familyId"),
                "species": target.get("species"),
                "subunit_composition": target.get("subunitComposition"),
                "functional_characteristics": target.get("functionalCharacteristics")
            }
            
            data["targets"].append(target_info)
            
            # Count by family
            family = target_info["family"]
            if family:
                if family not in data["target_families"]:
                    data["target_families"][family] = 0
                data["target_families"][family] += 1
            
            # Count by type
            target_type = target_info["type"]
            if target_type:
                if target_type not in data["target_types"]:
                    data["target_types"][target_type] = 0
                data["target_types"][target_type] += 1
            
            # Get interactions for target if requested
            if kwargs.get('include_interactions') and target_info["id"]:
                try:
                    interactions = self.api_client.get_interactions(target_id=target_info["id"])
                    for interaction in interactions[:20]:  # Top 20 interactions
                        ligand_info = {
                            "ligand_id": interaction.get("ligandId"),
                            "ligand_name": interaction.get("ligandName"),
                            "type": interaction.get("type"),
                            "action": interaction.get("action"),
                            "affinity_type": interaction.get("affinityType"),
                            "affinity_value": interaction.get("affinityValue"),
                            "affinity_units": interaction.get("affinityUnits"),
                            "is_approved_drug": interaction.get("approved", False)
                        }
                        
                        data["associated_ligands"].append(ligand_info)
                        
                        if ligand_info["is_approved_drug"]:
                            data["approved_drugs"].append({
                                "id": ligand_info["ligand_id"],
                                "name": ligand_info["ligand_name"]
                            })
                except Exception as e:
                    logger.warning(f"Could not get interactions for target {target_info['id']}: {e}")
        
        return data
    
    def collect_target_data(self, target_id: int, **kwargs) -> Dict[str, Any]:
        """Collect detailed target data.
        
        Args:
            target_id: GtoPdb target ID
            **kwargs: Additional parameters
        
        Returns:
            Target data
        """
        logger.info(f"Collecting GtoPdb data for target {target_id}")
        
        # Get target details
        target = self.api_client.get_target(target_id)
        
        data = {
            "target_id": target.get("targetId"),
            "name": target.get("name"),
            "abbreviation": target.get("abbreviation"),
            "systematic_name": target.get("systematicName"),
            "type": target.get("type"),
            "family": target.get("familyName"),
            "family_id": target.get("familyId"),
            "species": target.get("species"),
            "gene_symbol": target.get("geneSymbol"),
            "gene_id": target.get("geneId"),
            "uniprot_id": target.get("uniprotId"),
            "ensembl_id": target.get("ensemblId"),
            "subunits": target.get("subunits", []),
            "tissue_distribution": target.get("tissueDistribution"),
            "functional_characteristics": target.get("functionalCharacteristics"),
            "physiological_function": target.get("physiologicalFunction"),
            "pathophysiology": target.get("pathophysiology"),
            "clinical_significance": target.get("clinicalSignificance")
        }
        
        # Get interactions if requested
        if kwargs.get('include_interactions'):
            try:
                interactions = self.api_client.get_interactions(target_id=target_id)
                data["interactions"] = self._process_interactions(interactions)
                data["interaction_count"] = len(interactions)
            except Exception as e:
                logger.warning(f"Could not get interactions: {e}")
                data["interactions"] = []
                data["interaction_count"] = 0
        
        # Get associated diseases if available
        if target.get("diseaseAssociations"):
            data["disease_associations"] = target["diseaseAssociations"]
        
        # Get references
        if target.get("references"):
            data["references"] = [
                {
                    "pubmed_id": ref.get("pubmedId"),
                    "title": ref.get("title"),
                    "year": ref.get("year")
                }
                for ref in target["references"][:10]  # Top 10 references
            ]
        
        return data
    
    def collect_ligand_data(self, ligand_id: int, **kwargs) -> Dict[str, Any]:
        """Collect detailed ligand data.
        
        Args:
            ligand_id: GtoPdb ligand ID
            **kwargs: Additional parameters
        
        Returns:
            Ligand data
        """
        logger.info(f"Collecting GtoPdb data for ligand {ligand_id}")
        
        # Get ligand details
        ligand = self.api_client.get_ligand(ligand_id)
        
        data = {
            "ligand_id": ligand.get("ligandId"),
            "name": ligand.get("name"),
            "abbreviation": ligand.get("abbreviation"),
            "inn": ligand.get("inn"),  # International Nonproprietary Name
            "type": ligand.get("type"),
            "species": ligand.get("species"),
            "is_approved": ligand.get("approved", False),
            "is_withdrawn": ligand.get("withdrawn", False),
            "is_labeled": ligand.get("labeled", False),
            "radioactivity": ligand.get("radioactivity"),
            "molecular_formula": ligand.get("molecularFormula"),
            "molecular_weight": ligand.get("molecularWeight"),
            "smiles": ligand.get("smiles"),
            "inchi": ligand.get("inchi"),
            "inchi_key": ligand.get("inchiKey"),
            "pubchem_cid": ligand.get("pubchemCid"),
            "drugbank_id": ligand.get("drugbankId"),
            "chembl_id": ligand.get("chemblId"),
            "bioactivity_comments": ligand.get("bioactivityComments"),
            "clinical_use_comments": ligand.get("clinicalUseComments")
        }
        
        # Get interactions if requested
        if kwargs.get('include_interactions'):
            try:
                interactions = self.api_client.get_interactions(ligand_id=ligand_id)
                data["interactions"] = self._process_interactions(interactions)
                data["interaction_count"] = len(interactions)
                
                # Extract target information
                data["targets"] = []
                for interaction in interactions:
                    target_info = {
                        "id": interaction.get("targetId"),
                        "name": interaction.get("targetName"),
                        "type": interaction.get("targetType"),
                        "species": interaction.get("targetSpecies")
                    }
                    if target_info not in data["targets"]:
                        data["targets"].append(target_info)
            except Exception as e:
                logger.warning(f"Could not get interactions: {e}")
                data["interactions"] = []
                data["targets"] = []
        
        # Get synonyms
        if ligand.get("synonyms"):
            data["synonyms"] = ligand["synonyms"]
        
        # Get database links
        if ligand.get("databaseLinks"):
            data["database_links"] = ligand["databaseLinks"]
        
        return data
    
    def collect_disease_data(self, disease_name: str, **kwargs) -> Dict[str, Any]:
        """Collect disease-related pharmacology data.
        
        Args:
            disease_name: Disease name or ID
            **kwargs: Additional parameters
        
        Returns:
            Disease pharmacology data
        """
        logger.info(f"Collecting GtoPdb data for disease {disease_name}")
        
        # Search for disease
        diseases = self.api_client.get_diseases()
        
        matching_disease = None
        for disease in diseases:
            if disease_name.lower() in disease.get("name", "").lower():
                matching_disease = disease
                break
        
        if not matching_disease:
            # Try to parse as ID
            try:
                disease_id = int(disease_name)
                matching_disease = self.api_client.get_disease(disease_id)
            except (ValueError, Exception):
                raise ValueError(f"Disease {disease_name} not found")
        
        data = {
            "disease_id": matching_disease.get("diseaseId"),
            "name": matching_disease.get("name"),
            "description": matching_disease.get("description"),
            "mesh_id": matching_disease.get("meshId"),
            "efo_id": matching_disease.get("efoId"),
            "orphanet_id": matching_disease.get("orphanetId"),
            "targets": [],
            "drugs": []
        }
        
        # Get disease targets
        if matching_disease.get("diseaseId"):
            try:
                targets = self.api_client.get_disease_targets(matching_disease["diseaseId"])
                for target in targets:
                    target_info = {
                        "id": target.get("targetId"),
                        "name": target.get("name"),
                        "type": target.get("type"),
                        "family": target.get("familyName"),
                        "association_type": target.get("associationType"),
                        "association_description": target.get("associationDescription")
                    }
                    data["targets"].append(target_info)
                
                data["target_count"] = len(targets)
            except Exception as e:
                logger.warning(f"Could not get disease targets: {e}")
                data["target_count"] = 0
        
        # Get approved drugs for disease targets
        for target in data["targets"]:
            if target["id"]:
                try:
                    interactions = self.api_client.get_interactions(target_id=target["id"])
                    for interaction in interactions:
                        if interaction.get("approved"):
                            drug_info = {
                                "id": interaction.get("ligandId"),
                                "name": interaction.get("ligandName"),
                                "target": target["name"],
                                "action": interaction.get("action")
                            }
                            if drug_info not in data["drugs"]:
                                data["drugs"].append(drug_info)
                except Exception as e:
                    logger.warning(f"Could not get drugs for target {target['id']}: {e}")
        
        data["drug_count"] = len(data["drugs"])
        
        return data
    
    def _process_interactions(self, interactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process interaction data."""
        processed = []
        
        for interaction in interactions[:50]:  # Process top 50 interactions
            int_info = {
                "ligand_id": interaction.get("ligandId"),
                "ligand_name": interaction.get("ligandName"),
                "target_id": interaction.get("targetId"),
                "target_name": interaction.get("targetName"),
                "type": interaction.get("type"),
                "action": interaction.get("action"),
                "action_comment": interaction.get("actionComment"),
                "selectivity": interaction.get("selectivity"),
                "endogenous": interaction.get("endogenous", False),
                "primary_target": interaction.get("primaryTarget", False),
                "concentration_range": interaction.get("concentrationRange"),
                "affinity_type": interaction.get("affinityType"),
                "affinity_value": interaction.get("affinityValue"),
                "affinity_units": interaction.get("affinityUnits"),
                "assay_description": interaction.get("assayDescription"),
                "references": interaction.get("references", [])[:3]  # Top 3 references
            }
            processed.append(int_info)
        
        return processed
    
    def collect_batch(self, identifiers: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Collect data for multiple identifiers."""
        results = []
        for identifier in identifiers:
            try:
                data = self.collect_single(identifier, **kwargs)
                results.append(data)
            except Exception as e:
                logger.error(f"Failed to collect GtoPdb data for {identifier}: {e}")
                results.append({
                    "identifier": identifier,
                    "error": str(e)
                })
        return results
    
    def save_to_database(self, data: Dict[str, Any]) -> Any:
        """Save GtoPdb data to database.
        
        Args:
            data: GtoPdb data
        
        Returns:
            Saved entity (Gene, Protein, or Disease)
        """
        if "gene_symbol" in data:
            return self._save_gene_data(data)
        elif "target_id" in data and data.get("gene_symbol"):
            return self._save_target_data(data)
        elif "disease_id" in data:
            return self._save_disease_data(data)
        else:
            logger.warning("Cannot determine data type to save")
            return None
    
    def _save_gene_data(self, data: Dict[str, Any]) -> Gene:
        """Save gene target data."""
        gene_symbol = data["gene_symbol"]
        
        # Check if gene exists
        existing = self.db_session.query(Gene).filter_by(
            symbol=gene_symbol
        ).first()
        
        if existing:
            logger.info(f"Updating existing gene {gene_symbol}")
            gene = existing
        else:
            gene = Gene(
                id=self.generate_id("gtopdb_gene", gene_symbol),
                gene_id=f"gtopdb:{gene_symbol}",
                symbol=gene_symbol,
                source="GtoPdb"
            )
        
        # Store GtoPdb-specific data
        if not gene.annotations:
            gene.annotations = {}
        
        gene.annotations["gtopdb_targets"] = {
            "target_count": data.get("target_count", 0),
            "targets": data.get("targets", [])[:10],  # Store top 10 targets
            "target_families": data.get("target_families", {}),
            "target_types": data.get("target_types", {}),
            "approved_drugs": data.get("approved_drugs", [])[:10]  # Store top 10 drugs
        }
        
        if not existing:
            self.db_session.add(gene)
        
        self.db_session.commit()
        logger.info(f"Saved GtoPdb data for gene {gene_symbol}")
        
        return gene
    
    def _save_target_data(self, data: Dict[str, Any]) -> Protein:
        """Save target data as protein."""
        uniprot_id = data.get("uniprot_id", f"gtopdb_target_{data['target_id']}")
        
        # Check if protein exists
        existing = self.db_session.query(Protein).filter_by(
            uniprot_accession=uniprot_id
        ).first()
        
        if existing:
            logger.info(f"Updating existing protein {uniprot_id}")
            protein = existing
        else:
            protein = Protein(
                id=self.generate_id("gtopdb_protein", uniprot_id),
                uniprot_accession=uniprot_id,
                sequence="",  # Required field
                sequence_length=0,
                source="GtoPdb"
            )
        
        protein.name = data.get("name")
        protein.gene_name = data.get("gene_symbol")
        protein.organism = data.get("species")
        
        # Store GtoPdb target data
        if not protein.annotations:
            protein.annotations = {}
        
        protein.annotations["gtopdb_target"] = {
            "target_id": data.get("target_id"),
            "abbreviation": data.get("abbreviation"),
            "type": data.get("type"),
            "family": data.get("family"),
            "tissue_distribution": data.get("tissue_distribution"),
            "functional_characteristics": data.get("functional_characteristics"),
            "clinical_significance": data.get("clinical_significance"),
            "interaction_count": data.get("interaction_count", 0),
            "interactions": data.get("interactions", [])[:20],  # Store top 20 interactions
            "disease_associations": data.get("disease_associations", [])
        }
        
        if not existing:
            self.db_session.add(protein)
        
        self.db_session.commit()
        logger.info(f"Saved GtoPdb target data for protein {uniprot_id}")
        
        return protein
    
    def _save_disease_data(self, data: Dict[str, Any]) -> Disease:
        """Save disease pharmacology data."""
        disease_name = data["name"]
        disease_id = f"gtopdb:{data['disease_id']}"
        
        # Check if disease exists
        existing = self.db_session.query(Disease).filter_by(
            disease_id=disease_id
        ).first()
        
        if existing:
            logger.info(f"Updating existing disease {disease_name}")
            disease = existing
        else:
            disease = Disease(
                id=self.generate_id("gtopdb_disease", str(data["disease_id"])),
                disease_id=disease_id,
                name=disease_name,
                source="GtoPdb"
            )
        
        disease.description = data.get("description")
        disease.mesh_id = data.get("mesh_id")
        disease.efo_id = data.get("efo_id")
        disease.orphanet_id = data.get("orphanet_id")
        
        # Store GtoPdb pharmacology data
        if not disease.therapeutic_areas:
            disease.therapeutic_areas = []
        
        disease.therapeutic_areas.append("pharmacology")
        
        # Store drug and target data
        if not disease.annotations:
            disease.annotations = {}
        
        disease.annotations = {
            "gtopdb_pharmacology": {
                "target_count": data.get("target_count", 0),
                "targets": data.get("targets", [])[:20],  # Store top 20 targets
                "drug_count": data.get("drug_count", 0),
                "drugs": data.get("drugs", [])[:20]  # Store top 20 drugs
            }
        }
        
        if not existing:
            self.db_session.add(disease)
        
        self.db_session.commit()
        logger.info(f"Saved GtoPdb data for disease {disease_name}")
        
        return disease