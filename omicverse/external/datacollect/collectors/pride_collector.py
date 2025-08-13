"""PRIDE proteomics data collector."""

import logging
from typing import Any, Dict, List, Optional

from omicverse.external.datacollect.api.pride import PRIDEClient
from omicverse.external.datacollect.models.protein import Protein
from .base import BaseCollector
from ..config import settings


logger = logging.getLogger(__name__)


class PRIDECollector(BaseCollector):
    """Collector for PRIDE proteomics data."""
    
    def __init__(self, db_session=None):
        api_client = PRIDEClient()
        super().__init__(api_client, db_session)
    
    def collect_single(self, identifier: str, **kwargs) -> Dict[str, Any]:
        """Collect PRIDE data for a single identifier.
        
        Args:
            identifier: Protein accession, project accession, or peptide sequence
            **kwargs: Additional parameters (type)
        
        Returns:
            Collected PRIDE data
        """
        id_type = kwargs.get('type', 'protein')
        
        if id_type == 'project':
            return self.collect_project_data(identifier, **kwargs)
        elif id_type == 'protein':
            return self.collect_protein_data(identifier, **kwargs)
        elif id_type == 'peptide':
            return self.collect_peptide_data(identifier, **kwargs)
        else:
            raise ValueError(f"Unknown identifier type: {id_type}")
    
    def collect_project_data(self, project_accession: str, **kwargs) -> Dict[str, Any]:
        """Collect data for a PRIDE project.
        
        Args:
            project_accession: PRIDE project accession (e.g., PXD000001)
            **kwargs: Additional parameters
        
        Returns:
            Project data
        """
        logger.info(f"Collecting PRIDE project data for {project_accession}")
        
        # Get project details
        project = self.api_client.get_project(project_accession)
        
        data = {
            "project_accession": project.get("accession"),
            "title": project.get("title"),
            "description": project.get("projectDescription"),
            "sample_description": project.get("sampleProcessingProtocol"),
            "data_processing": project.get("dataProcessingProtocol"),
            "submission_date": project.get("submissionDate"),
            "publication_date": project.get("publicationDate"),
            "submission_type": project.get("submissionType"),
            "num_assays": project.get("numAssays", 0),
            "species": [],
            "tissues": [],
            "diseases": [],
            "instruments": [],
            "modifications": [],
            "quantification_methods": []
        }
        
        # Process species
        if project.get("species"):
            for species in project["species"]:
                data["species"].append({
                    "name": species.get("name"),
                    "taxonomy_id": species.get("accession")
                })
        
        # Process tissues
        if project.get("tissues"):
            for tissue in project["tissues"]:
                data["tissues"].append({
                    "name": tissue.get("name"),
                    "cv_param": tissue.get("cvParam")
                })
        
        # Process diseases
        if project.get("diseases"):
            for disease in project["diseases"]:
                data["diseases"].append({
                    "name": disease.get("name"),
                    "cv_param": disease.get("cvParam")
                })
        
        # Process instruments
        if project.get("instruments"):
            for instrument in project["instruments"]:
                data["instruments"].append({
                    "name": instrument.get("name"),
                    "cv_param": instrument.get("cvParam")
                })
        
        # Process PTMs
        if project.get("ptms"):
            for ptm in project["ptms"]:
                data["modifications"].append({
                    "name": ptm.get("name"),
                    "cv_param": ptm.get("cvParam")
                })
        
        # Process quantification methods
        if project.get("quantificationMethods"):
            for method in project["quantificationMethods"]:
                data["quantification_methods"].append({
                    "name": method.get("name"),
                    "cv_param": method.get("cvParam")
                })
        
        # Add references
        if project.get("references"):
            data["references"] = [
                {
                    "pubmed_id": ref.get("pubmedId"),
                    "doi": ref.get("doi"),
                    "reference_line": ref.get("referenceLine")
                }
                for ref in project["references"]
            ]
        
        # Add submitters
        if project.get("submitters"):
            data["submitters"] = [
                {
                    "name": sub.get("name"),
                    "email": sub.get("email"),
                    "affiliation": sub.get("affiliation")
                }
                for sub in project["submitters"]
            ]
        
        # Get project files if requested
        if kwargs.get('include_files'):
            try:
                files = self.api_client.get_project_files(project_accession)
                data["file_count"] = files.get("page", {}).get("totalElements", 0)
                data["files"] = self._process_project_files(files)
            except Exception as e:
                logger.warning(f"Could not get project files: {e}")
                data["file_count"] = 0
                data["files"] = []
        
        # Get modifications if requested
        if kwargs.get('include_modifications'):
            try:
                modifications = self.api_client.get_modifications(project_accession)
                data["detected_modifications"] = modifications
            except Exception as e:
                logger.warning(f"Could not get modifications: {e}")
                data["detected_modifications"] = []
        
        return data
    
    def collect_protein_data(self, protein_accession: str, **kwargs) -> Dict[str, Any]:
        """Collect proteomics data for a protein.
        
        Args:
            protein_accession: Protein accession (e.g., UniProt ID)
            **kwargs: Additional parameters
        
        Returns:
            Protein proteomics data
        """
        logger.info(f"Collecting PRIDE data for protein {protein_accession}")
        
        # Search for protein identifications
        protein_results = self.api_client.search_proteins(protein_accession=protein_accession)
        
        data = {
            "protein_accession": protein_accession,
            "identification_count": protein_results.get("page", {}).get("totalElements", 0),
            "projects": [],
            "peptides": [],
            "modifications": {},
            "tissues": {},
            "diseases": {}
        }
        
        # Process protein identifications
        if protein_results.get("_embedded", {}).get("proteinevidences"):
            for evidence in protein_results["_embedded"]["proteinevidences"][:50]:  # Top 50
                project_acc = evidence.get("projectAccession")
                if project_acc not in [p["accession"] for p in data["projects"]]:
                    data["projects"].append({
                        "accession": project_acc,
                        "assay_accession": evidence.get("assayAccession"),
                        "num_psms": evidence.get("numPSMs", 0),
                        "num_peptides": evidence.get("numPeptides", 0),
                        "sequence_coverage": evidence.get("sequenceCoverage"),
                        "best_search_engine_score": evidence.get("bestSearchEngineScore")
                    })
        
        # Search for peptides if requested
        if kwargs.get('include_peptides'):
            try:
                peptide_results = self.api_client.search_peptides(protein_accession=protein_accession)
                if peptide_results.get("_embedded", {}).get("peptideevidences"):
                    for peptide in peptide_results["_embedded"]["peptideevidences"][:100]:  # Top 100
                        peptide_info = {
                            "sequence": peptide.get("peptideSequence"),
                            "project": peptide.get("projectAccession"),
                            "charge": peptide.get("charge"),
                            "mz": peptide.get("mz"),
                            "retention_time": peptide.get("retentionTime"),
                            "modifications": []
                        }
                        
                        # Process PTMs
                        if peptide.get("ptmList"):
                            for ptm in peptide["ptmList"]:
                                mod_name = ptm.get("name")
                                peptide_info["modifications"].append({
                                    "name": mod_name,
                                    "position": ptm.get("position"),
                                    "mass_shift": ptm.get("massShift")
                                })
                                
                                # Count modifications
                                if mod_name not in data["modifications"]:
                                    data["modifications"][mod_name] = 0
                                data["modifications"][mod_name] += 1
                        
                        data["peptides"].append(peptide_info)
            except Exception as e:
                logger.warning(f"Could not get peptide data: {e}")
        
        # Aggregate project metadata
        for project in data["projects"][:10]:  # Get details for top 10 projects
            try:
                project_details = self.api_client.get_project(project["accession"])
                
                # Aggregate tissues
                if project_details.get("tissues"):
                    for tissue in project_details["tissues"]:
                        tissue_name = tissue.get("name")
                        if tissue_name not in data["tissues"]:
                            data["tissues"][tissue_name] = 0
                        data["tissues"][tissue_name] += 1
                
                # Aggregate diseases
                if project_details.get("diseases"):
                    for disease in project_details["diseases"]:
                        disease_name = disease.get("name")
                        if disease_name not in data["diseases"]:
                            data["diseases"][disease_name] = 0
                        data["diseases"][disease_name] += 1
            except Exception as e:
                logger.warning(f"Could not get project details for {project['accession']}: {e}")
        
        return data
    
    def collect_peptide_data(self, peptide_sequence: str, **kwargs) -> Dict[str, Any]:
        """Collect data for a peptide sequence.
        
        Args:
            peptide_sequence: Peptide sequence
            **kwargs: Additional parameters
        
        Returns:
            Peptide data
        """
        logger.info(f"Collecting PRIDE data for peptide {peptide_sequence}")
        
        # Search for peptide
        peptide_results = self.api_client.search_peptides(sequence=peptide_sequence)
        
        data = {
            "peptide_sequence": peptide_sequence,
            "identification_count": peptide_results.get("page", {}).get("totalElements", 0),
            "projects": [],
            "proteins": [],
            "modifications": [],
            "charge_states": {},
            "mass_values": []
        }
        
        # Process peptide identifications
        if peptide_results.get("_embedded", {}).get("peptideevidences"):
            for evidence in peptide_results["_embedded"]["peptideevidences"][:100]:  # Top 100
                # Add project
                project_acc = evidence.get("projectAccession")
                if project_acc not in [p["accession"] for p in data["projects"]]:
                    data["projects"].append({
                        "accession": project_acc,
                        "assay_accession": evidence.get("assayAccession")
                    })
                
                # Add protein
                protein_acc = evidence.get("proteinAccession")
                if protein_acc and protein_acc not in data["proteins"]:
                    data["proteins"].append(protein_acc)
                
                # Track charge states
                charge = evidence.get("charge")
                if charge:
                    if charge not in data["charge_states"]:
                        data["charge_states"][charge] = 0
                    data["charge_states"][charge] += 1
                
                # Track mass values
                mz = evidence.get("mz")
                if mz:
                    data["mass_values"].append(mz)
                
                # Process modifications
                if evidence.get("ptmList"):
                    for ptm in evidence["ptmList"]:
                        mod_info = {
                            "name": ptm.get("name"),
                            "position": ptm.get("position"),
                            "mass_shift": ptm.get("massShift"),
                            "project": project_acc
                        }
                        data["modifications"].append(mod_info)
        
        # Calculate statistics
        if data["mass_values"]:
            data["average_mz"] = sum(data["mass_values"]) / len(data["mass_values"])
            data["min_mz"] = min(data["mass_values"])
            data["max_mz"] = max(data["mass_values"])
        
        return data
    
    def _process_project_files(self, files_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process project files data."""
        processed = []
        
        if files_data.get("_embedded", {}).get("files"):
            for file in files_data["_embedded"]["files"][:20]:  # Top 20 files
                file_info = {
                    "name": file.get("fileName"),
                    "type": file.get("fileType"),
                    "size": file.get("fileSize"),
                    "checksum": file.get("checksum")
                }
                processed.append(file_info)
        
        return processed
    
    def collect_batch(self, identifiers: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Collect data for multiple identifiers."""
        results = []
        for identifier in identifiers:
            try:
                data = self.collect_single(identifier, **kwargs)
                results.append(data)
            except Exception as e:
                logger.error(f"Failed to collect PRIDE data for {identifier}: {e}")
                results.append({
                    "identifier": identifier,
                    "error": str(e)
                })
        return results
    
    def save_to_database(self, data: Dict[str, Any]) -> Protein:
        """Save PRIDE data to database.
        
        Args:
            data: PRIDE data
        
        Returns:
            Saved Protein instance
        """
        if "protein_accession" in data:
            protein_acc = data["protein_accession"]
        elif "peptide_sequence" in data and data.get("proteins"):
            protein_acc = data["proteins"][0]  # Use first protein
        else:
            logger.warning("Cannot determine protein accession to save")
            return None
        
        # Check if protein exists
        existing = self.db_session.query(Protein).filter_by(
            uniprot_accession=protein_acc
        ).first()
        
        if existing:
            logger.info(f"Updating existing protein {protein_acc}")
            protein = existing
        else:
            protein = Protein(
                id=self.generate_id("pride_protein", protein_acc),
                uniprot_accession=protein_acc,
                sequence="",  # Required field
                sequence_length=0,
                source="PRIDE"
            )
        
        # Store PRIDE-specific data
        if not protein.annotations:
            protein.annotations = {}
        
        if "protein_accession" in data:
            protein.annotations["pride_proteomics"] = {
                "identification_count": data.get("identification_count", 0),
                "projects": data.get("projects", [])[:10],  # Store top 10 projects
                "peptide_count": len(data.get("peptides", [])),
                "modifications": data.get("modifications", {}),
                "tissues": data.get("tissues", {}),
                "diseases": data.get("diseases", {})
            }
        elif "peptide_sequence" in data:
            if "pride_peptides" not in protein.annotations:
                protein.annotations["pride_peptides"] = []
            
            protein.annotations["pride_peptides"].append({
                "sequence": data["peptide_sequence"],
                "identification_count": data.get("identification_count", 0),
                "charge_states": data.get("charge_states", {}),
                "average_mz": data.get("average_mz"),
                "modifications": data.get("modifications", [])[:5]  # Top 5 modifications
            })
        
        if not existing:
            self.db_session.add(protein)
        
        self.db_session.commit()
        logger.info(f"Saved PRIDE data for protein {protein_acc}")
        
        return protein