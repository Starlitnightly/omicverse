"""AlphaFold data collector."""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from src.api.alphafold import AlphaFoldClient
from src.models.structure import Structure, Chain
from src.models.protein import Protein
from .base import BaseCollector
from ..config.config import settings


logger = logging.getLogger(__name__)


class AlphaFoldCollector(BaseCollector):
    """Collector for AlphaFold structure predictions."""
    
    def __init__(self, db_session=None):
        api_client = AlphaFoldClient()
        super().__init__(api_client, db_session)
    
    def collect_single(self, uniprot_accession: str, download_structure: bool = False, 
                      download_pae: bool = False) -> Dict[str, Any]:
        """Collect AlphaFold prediction for a UniProt accession.
        
        Args:
            uniprot_accession: UniProt accession
            download_structure: Whether to download structure file
            download_pae: Whether to download PAE data
        
        Returns:
            Collected prediction data
        """
        logger.info(f"Collecting AlphaFold prediction for {uniprot_accession}")
        
        # Get prediction metadata
        predictions = self.api_client.get_prediction_by_uniprot(uniprot_accession)
        
        if not predictions:
            raise ValueError(f"No AlphaFold prediction found for {uniprot_accession}")
        
        # AlphaFold returns a list, take the first (usually only) prediction
        prediction = predictions[0] if isinstance(predictions, list) else predictions
        
        # Extract data
        data = {
            "alphafold_id": prediction.get("entryId"),
            "uniprot_accession": uniprot_accession,
            "organism": prediction.get("organismScientificName"),
            "gene_name": prediction.get("gene"),
            "protein_name": prediction.get("uniprotDescription"),
            "sequence_length": prediction.get("sequenceLength"),
            "model_created_date": prediction.get("modelCreatedDate"),
            "confidence_version": prediction.get("confidenceVersion"),
            "mean_plddt": prediction.get("confidenceMean"),
            "coverage": prediction.get("coverage", 100),  # Usually 100% for AlphaFold
            "urls": {
                "pdb": prediction.get("pdbUrl"),
                "cif": prediction.get("cifUrl"),
                "bcif": prediction.get("bcifUrl"),
                "pae_image": prediction.get("paeImageUrl"),
                "pae_json": prediction.get("paeDocUrl"),
            }
        }
        
        # Download structure file if requested
        if download_structure:
            try:
                structure_content = self.api_client.get_structure_file(
                    uniprot_accession, format="pdb"
                )
                data["structure_file"] = self._save_structure_file(
                    data["alphafold_id"], structure_content, "pdb"
                )
            except Exception as e:
                logger.error(f"Failed to download structure: {e}")
        
        # Download PAE data if requested
        if download_pae:
            try:
                pae_data = self.api_client.get_pae_data(uniprot_accession)
                data["pae_data"] = pae_data
            except Exception as e:
                logger.error(f"Failed to download PAE data: {e}")
        
        return data
    
    def collect_batch(self, uniprot_accessions: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Collect predictions for multiple UniProt accessions."""
        logger.info(f"Collecting AlphaFold predictions for {len(uniprot_accessions)} proteins")
        
        results = []
        for accession in uniprot_accessions:
            try:
                data = self.collect_single(accession, **kwargs)
                results.append(data)
            except Exception as e:
                logger.error(f"Failed to collect {accession}: {e}")
                continue
        
        return results
    
    def save_to_database(self, data: Dict[str, Any]) -> Structure:
        """Save AlphaFold prediction to database."""
        # Check if structure already exists
        existing = self.db_session.query(Structure).filter_by(
            structure_id=data["alphafold_id"]
        ).first()
        
        if existing:
            logger.info(f"Updating existing AlphaFold structure {data['alphafold_id']}")
            structure = existing
        else:
            structure = Structure(
                id=self.generate_id("alphafold", data["alphafold_id"]),
                source="AlphaFold",
            )
        
        # Update fields
        structure.structure_id = data["alphafold_id"]
        structure.title = f"AlphaFold structure prediction of {data.get('protein_name', data['uniprot_accession'])}"
        structure.structure_type = "PREDICTED"
        structure.organism = data.get("organism")
        
        # AlphaFold doesn't have traditional resolution, use mean pLDDT as quality metric
        structure.resolution = None  # Not applicable
        structure.r_factor = data.get("mean_plddt")  # Store pLDDT in r_factor field
        
        # Handle dates
        if data.get("model_created_date"):
            try:
                structure.deposition_date = datetime.fromisoformat(
                    data["model_created_date"].replace("Z", "+00:00")
                ).date()
                structure.release_date = structure.deposition_date
            except:
                pass
        
        # Add file path if downloaded
        if data.get("structure_file"):
            structure.structure_file_path = str(data["structure_file"])
        
        # Add chain (AlphaFold predictions are single chain)
        if not structure.chains:
            chain = Chain(
                id=self.generate_id("chain", structure.id, "A"),
                source="AlphaFold",
                chain_id="A",
                molecule_type="protein",
                length=data.get("sequence_length"),
                uniprot_accession=data["uniprot_accession"],
            )
            structure.chains.append(chain)
        
        # Link to protein if exists
        protein = self.db_session.query(Protein).filter_by(
            accession=data["uniprot_accession"]
        ).first()
        
        if protein:
            # Update protein's structure reference
            if not protein.pdb_ids:
                protein.pdb_ids = data["alphafold_id"]
            elif data["alphafold_id"] not in protein.pdb_ids:
                protein.pdb_ids += f",{data['alphafold_id']}"
            protein.has_3d_structure = "Y"
        
        # Save to database
        if not existing:
            self.db_session.add(structure)
        
        self.db_session.commit()
        logger.info(f"Saved AlphaFold structure {structure.structure_id} to database")
        
        return structure
    
    def search_by_organism(self, organism: str, max_results: int = 100, 
                          download_structures: bool = False) -> List[Structure]:
        """Search and collect all AlphaFold predictions for an organism.
        
        Args:
            organism: Organism name or taxonomy ID
            max_results: Maximum number of results
            download_structures: Whether to download structure files
        
        Returns:
            List of saved Structure instances
        """
        logger.info(f"Searching AlphaFold for organism: {organism}")
        
        # Search for predictions
        predictions = self.api_client.search_by_organism(organism, limit=max_results)
        
        structures = []
        for pred in predictions[:max_results]:
            try:
                # Extract UniProt accession from the prediction
                uniprot_acc = pred.get("uniprotAccession")
                if not uniprot_acc:
                    continue
                
                # Collect full data
                data = self.collect_single(
                    uniprot_acc, 
                    download_structure=download_structures
                )
                
                # Save to database
                structure = self.save_to_database(data)
                structures.append(structure)
                
            except Exception as e:
                logger.error(f"Failed to process {pred.get('entryId')}: {e}")
                continue
        
        logger.info(f"Collected {len(structures)} AlphaFold structures for {organism}")
        return structures
    
    def collect_for_proteins(self, proteins: List[Protein], 
                            download_structures: bool = False) -> List[Structure]:
        """Collect AlphaFold predictions for a list of proteins.
        
        Args:
            proteins: List of Protein model instances
            download_structures: Whether to download structure files
        
        Returns:
            List of saved Structure instances
        """
        structures = []
        
        for protein in proteins:
            try:
                data = self.collect_single(
                    protein.accession,
                    download_structure=download_structures
                )
                structure = self.save_to_database(data)
                structures.append(structure)
            except Exception as e:
                logger.warning(f"No AlphaFold prediction for {protein.accession}: {e}")
                continue
        
        return structures
    
    def _save_structure_file(self, alphafold_id: str, content: str, format: str) -> str:
        """Save structure file to local storage."""
        filename = f"{alphafold_id}.{format}"
        file_path = settings.storage.raw_data_dir / "alphafold" / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, "w") as f:
            f.write(content)
        
        logger.info(f"Saved AlphaFold structure to {file_path}")
        return str(file_path)
