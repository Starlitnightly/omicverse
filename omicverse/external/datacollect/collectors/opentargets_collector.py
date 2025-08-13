"""OpenTargets Platform data collector."""

import logging
from typing import Any, Dict, List, Optional
import json

from src.api.opentargets import OpenTargetsClient
from src.models.genomic import Gene
from src.models.disease import Disease
from .base import BaseCollector
from config.config import settings


logger = logging.getLogger(__name__)


class OpenTargetsCollector(BaseCollector):
    """Collector for OpenTargets Platform target-disease association data."""
    
    def __init__(self, db_session=None):
        api_client = OpenTargetsClient()
        super().__init__(api_client, db_session)
    
    def collect_single(self, identifier: str, **kwargs) -> Dict[str, Any]:
        """Collect data for a single identifier.
        
        Args:
            identifier: Target (Ensembl ID), disease (EFO ID), or drug (ChEMBL ID)
            **kwargs: Additional parameters (type)
        
        Returns:
            Collected data
        """
        id_type = kwargs.get('type', 'target')
        
        if id_type == 'target':
            return self.collect_target_data(identifier, **kwargs)
        elif id_type == 'disease':
            return self.collect_disease_data(identifier, **kwargs)
        elif id_type == 'drug':
            return self.collect_drug_data(identifier, **kwargs)
        else:
            raise ValueError(f"Unknown identifier type: {id_type}")
    
    def collect_target_data(self, target_id: str, **kwargs) -> Dict[str, Any]:
        """Collect comprehensive target data.
        
        Args:
            target_id: Ensembl gene ID
            **kwargs: Additional parameters
        
        Returns:
            Target data
        """
        logger.info(f"Collecting OpenTargets Platform data for target {target_id}")
        
        # Get target info
        target_info = self.api_client.get_target(target_id)
        
        if not target_info:
            raise ValueError(f"Target {target_id} not found")
        
        data = {
            "target_id": target_id,
            "target_info": target_info,
            "symbol": target_info.get("approvedSymbol"),
            "name": target_info.get("approvedName")
        }
        
        # Get disease associations
        associations = self.api_client.get_associations(
            target_id, size=kwargs.get('association_limit', 100)
        )
        
        if associations and "associatedDiseases" in associations:
            assoc_data = associations["associatedDiseases"]
            if isinstance(assoc_data, dict) and "rows" in assoc_data:
                data["disease_associations"] = assoc_data["rows"]
                data["association_count"] = assoc_data.get("count", len(assoc_data["rows"]))
            else:
                data["disease_associations"] = []
                data["association_count"] = 0
        
        # Get known drugs
        known_drugs = self.api_client.get_known_drugs_for_target(target_id)
        data["known_drugs"] = known_drugs
        data["drug_count"] = known_drugs.get("count", 0) if known_drugs else 0
        
        # Get similar targets if requested
        if kwargs.get('include_similar'):
            similar = self.api_client.get_similar_targets(
                target_id, 
                threshold=kwargs.get('similarity_threshold', 0.5),
                size=kwargs.get('similar_limit', 20)
            )
            data["similar_targets"] = similar
        
        # Get tractability and safety info
        if target_info.get("tractability"):
            data["tractability"] = target_info["tractability"]
        if target_info.get("safety"):
            data["safety"] = target_info["safety"]
        
        return data
    
    def collect_disease_data(self, disease_id: str, **kwargs) -> Dict[str, Any]:
        """Collect disease data.
        
        Args:
            disease_id: Disease ID (EFO)
            **kwargs: Additional parameters
        
        Returns:
            Disease data
        """
        logger.info(f"Collecting OpenTargets Platform data for disease {disease_id}")
        
        # Get disease info
        disease_info = self.api_client.get_disease(disease_id)
        
        if not disease_info:
            raise ValueError(f"Disease {disease_id} not found")
        
        data = {
            "disease_id": disease_id,
            "disease_info": disease_info,
            "name": disease_info.get("name"),
            "description": disease_info.get("description")
        }
        
        # Store therapeutic areas
        if disease_info.get("therapeuticAreas"):
            data["therapeutic_areas"] = disease_info["therapeuticAreas"]
        
        # Store disease hierarchy
        if disease_info.get("parents"):
            data["parent_diseases"] = disease_info["parents"]
        if disease_info.get("children"):
            data["child_diseases"] = disease_info["children"]
        
        return data
    
    def collect_drug_data(self, drug_id: str, **kwargs) -> Dict[str, Any]:
        """Collect drug data.
        
        Args:
            drug_id: ChEMBL drug ID
            **kwargs: Additional parameters
        
        Returns:
            Drug data
        """
        logger.info(f"Collecting OpenTargets Platform data for drug {drug_id}")
        
        # Get drug info
        drug_info = self.api_client.get_drug_info(drug_id)
        
        if not drug_info:
            raise ValueError(f"Drug {drug_id} not found")
        
        data = {
            "drug_id": drug_id,
            "drug_info": drug_info,
            "name": drug_info.get("name"),
            "drug_type": drug_info.get("drugType"),
            "max_phase": drug_info.get("maximumClinicalTrialPhase")
        }
        
        # Get mechanisms of action
        if drug_info.get("mechanismsOfAction"):
            moa = drug_info["mechanismsOfAction"]
            data["mechanisms"] = moa.get("rows", []) if isinstance(moa, dict) else []
            data["target_count"] = len(set(
                target["id"] 
                for mech in data["mechanisms"] 
                for target in mech.get("targets", [])
            ))
        
        # Get indications
        if drug_info.get("indications"):
            ind = drug_info["indications"]
            data["indications"] = ind.get("rows", []) if isinstance(ind, dict) else []
            data["indication_count"] = len(data["indications"])
        
        # Get adverse events
        if drug_info.get("adverseEvents"):
            data["adverse_events"] = drug_info["adverseEvents"]
        
        return data
    
    def collect_association_evidence(self, target_id: str, disease_id: str,
                                    **kwargs) -> Dict[str, Any]:
        """Collect evidence for a specific target-disease association.
        
        Args:
            target_id: Ensembl gene ID
            disease_id: Disease ID
            **kwargs: Additional parameters
        
        Returns:
            Evidence data
        """
        logger.info(f"Collecting evidence for {target_id} - {disease_id}")
        
        # Get association score
        association = self.api_client.get_associations(target_id, disease_id)
        
        # Get detailed evidence
        evidence = self.api_client.get_evidence(
            target_id, disease_id,
            datasource=kwargs.get('datasource'),
            size=kwargs.get('evidence_limit', 100)
        )
        
        data = {
            "target_id": target_id,
            "disease_id": disease_id,
            "association": association,
            "evidence": evidence.get("rows", []) if evidence else [],
            "evidence_count": evidence.get("count", 0) if evidence else 0
        }
        
        # Group evidence by datasource
        if data["evidence"]:
            datasources = {}
            for ev in data["evidence"]:
                ds = ev.get("datasourceId", "unknown")
                if ds not in datasources:
                    datasources[ds] = []
                datasources[ds].append(ev)
            data["evidence_by_datasource"] = datasources
        
        return data
    
    def collect_batch(self, identifiers: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Collect data for multiple identifiers."""
        results = []
        for identifier in identifiers:
            try:
                data = self.collect_single(identifier, **kwargs)
                results.append(data)
            except Exception as e:
                logger.error(f"Failed to collect OpenTargets data for {identifier}: {e}")
        return results
    
    def save_target_data(self, data: Dict[str, Any]) -> Gene:
        """Save target data to database.
        
        Args:
            data: Collected target data
        
        Returns:
            Saved Gene instance
        """
        target_info = data["target_info"]
        target_id = target_info["id"]
        
        # Check if gene exists
        existing = self.db_session.query(Gene).filter_by(
            ensembl_id=target_id
        ).first()
        
        if existing:
            logger.info(f"Updating existing target {target_id}")
            gene = existing
        else:
            gene = Gene(
                id=self.generate_id("ot_target", target_id),
                gene_id=target_id,
                source="OpenTargets"
            )
        
        # Update fields
        gene.ensembl_id = target_id
        gene.symbol = target_info.get("approvedSymbol")
        gene.name = target_info.get("approvedName")
        gene.biotype = target_info.get("bioType")
        gene.hgnc_id = target_info.get("hgncId")
        
        # Set genomic location
        if target_info.get("genomicLocation"):
            loc = target_info["genomicLocation"]
            gene.chromosome = loc.get("chromosome")
            gene.start_position = loc.get("start")
            gene.end_position = loc.get("end")
            gene.strand = loc.get("strand")
        
        # Store UniProt IDs
        if target_info.get("uniprotIds"):
            gene.uniprot_ids = target_info["uniprotIds"]
        
        # Store OpenTargets-specific data as JSON
        gene.opentargets_data = {
            "disease_association_count": data.get("association_count", 0),
            "known_drug_count": data.get("drug_count", 0),
            "tractability": data.get("tractability"),
            "safety": data.get("safety"),
            "top_associations": [
                {
                    "disease": assoc["disease"]["name"],
                    "disease_id": assoc["disease"]["id"],
                    "score": assoc["score"]
                }
                for assoc in (data.get("disease_associations", [])[:5])
            ] if data.get("disease_associations") else []
        }
        
        if not existing:
            self.db_session.add(gene)
        
        self.db_session.commit()
        logger.info(f"Saved target {target_id} with {data.get('association_count', 0)} associations")
        
        return gene
    
    def save_disease_data(self, data: Dict[str, Any]) -> Disease:
        """Save disease data to database.
        
        Args:
            data: Collected disease data
        
        Returns:
            Saved Disease instance
        """
        disease_info = data["disease_info"]
        disease_id = disease_info["id"]
        
        # Check if disease exists
        existing = self.db_session.query(Disease).filter_by(
            disease_id=disease_id
        ).first()
        
        if existing:
            logger.info(f"Updating existing disease {disease_id}")
            disease = existing
        else:
            disease = Disease(
                id=self.generate_id("ot_disease", disease_id),
                disease_id=disease_id,
                source="OpenTargets"
            )
        
        # Update fields
        disease.name = disease_info.get("name")
        disease.description = disease_info.get("description")
        disease.synonyms = disease_info.get("synonyms", [])
        
        # Store therapeutic areas
        if data.get("therapeutic_areas"):
            disease.therapeutic_areas = data["therapeutic_areas"]
        
        # Store hierarchy
        if data.get("parent_diseases"):
            disease.parent_diseases = data["parent_diseases"]
        if data.get("child_diseases"):
            disease.child_diseases = data["child_diseases"]
        
        if not existing:
            self.db_session.add(disease)
        
        self.db_session.commit()
        logger.info(f"Saved disease {disease_id}")
        
        return disease
    
    def save_to_database(self, data: Dict[str, Any]) -> Any:
        """Save data to database based on data type."""
        if "target_info" in data:
            return self.save_target_data(data)
        elif "disease_info" in data:
            return self.save_disease_data(data)
        else:
            logger.warning("Unknown data type, cannot save to database")
            return None
    
    def search_targets(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Search for targets.
        
        Args:
            query: Search query
            limit: Maximum results
        
        Returns:
            Matching targets
        """
        logger.info(f"Searching OpenTargets for targets: '{query}'")
        return self.api_client.search_targets(query, size=limit)
    
    def search_diseases(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Search for diseases.
        
        Args:
            query: Search query
            limit: Maximum results
        
        Returns:
            Matching diseases
        """
        logger.info(f"Searching OpenTargets for diseases: '{query}'")
        return self.api_client.search_diseases(query, size=limit)
    
    def search_drugs(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Search for drugs.
        
        Args:
            query: Search query
            limit: Maximum results
        
        Returns:
            Matching drugs
        """
        logger.info(f"Searching OpenTargets for drugs: '{query}'")
        return self.api_client.search_drugs(query, size=limit)