"""PDB data collector."""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from omicverse.external.datacollect.api.pdb_simple import SimplePDBClient
from omicverse.external.datacollect.models.structure import Structure, Chain, Ligand
from .base import BaseCollector
from config import settings


logger = logging.getLogger(__name__)


class PDBCollector(BaseCollector):
    """Collector for PDB structural data."""
    
    def __init__(self, db_session=None):
        api_client = SimplePDBClient()
        super().__init__(api_client, db_session)
    
    def collect_single(self, pdb_id: str, download_structure: bool = False) -> Dict[str, Any]:
        """Collect data for a single PDB entry.
        
        Args:
            pdb_id: 4-character PDB ID
            download_structure: Whether to download structure file
        
        Returns:
            Collected data dictionary
        """
        logger.info(f"Collecting PDB data for {pdb_id}")
        
        # Ensure uppercase
        pdb_id = pdb_id.upper()
        
        # Get entry data
        entry_data = self.api_client.get_entry(pdb_id)
        
        # Get simplified polymer info if available
        polymer_data = self.api_client.get_polymer_info(pdb_id)
        
        # Extract relevant fields
        struct_data = entry_data.get("struct", {})
        
        data = {
            "pdb_id": pdb_id,
            "title": struct_data.get("title"),
            "structure_type": self._determine_structure_type(entry_data),
            "resolution": self._extract_resolution(entry_data),
            "r_factor": self._extract_r_factor(entry_data),
            "deposition_date": entry_data.get("rcsb_accession_info", {}).get("deposit_date"),
            "release_date": entry_data.get("rcsb_accession_info", {}).get("initial_release_date"),
            "organism": self._extract_organism_simple(entry_data, polymer_data),
            "chains": self._process_chains_simple(entry_data),
            "ligands": [],  # Simplified - no ligand data for now
        }
        
        # Download structure file if requested
        if download_structure:
            file_path = self._download_structure_file(pdb_id)
            data["structure_file_path"] = str(file_path)
        
        return data
    
    def collect_batch(self, pdb_ids: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Collect data for multiple PDB entries."""
        logger.info(f"Collecting batch data for {len(pdb_ids)} structures")
        
        results = []
        for pdb_id in pdb_ids:
            try:
                data = self.collect_single(pdb_id, **kwargs)
                results.append(data)
            except Exception as e:
                logger.error(f"Failed to collect {pdb_id}: {e}")
                continue
        
        return results
    
    def save_to_database(self, data: Dict[str, Any]) -> Structure:
        """Save PDB data to database."""
        # Check if structure already exists
        existing = self.db_session.query(Structure).filter_by(
            structure_id=data["pdb_id"]
        ).first()
        
        if existing:
            logger.info(f"Updating existing structure {data['pdb_id']}")
            structure = existing
            # Clear existing chains and ligands
            structure.chains.clear()
            structure.ligands.clear()
        else:
            structure = Structure(
                id=self.generate_id("pdb", data["pdb_id"]),
                source="PDB",
            )
        
        # Update fields
        structure.structure_id = data["pdb_id"]
        structure.title = data.get("title")
        structure.structure_type = data.get("structure_type")
        structure.resolution = data.get("resolution")
        structure.r_factor = data.get("r_factor")
        structure.organism = data.get("organism")
        
        # Handle dates
        if data.get("deposition_date"):
            structure.deposition_date = datetime.fromisoformat(
                data["deposition_date"].replace("Z", "+00:00")
            ).date()
        
        if data.get("release_date"):
            structure.release_date = datetime.fromisoformat(
                data["release_date"].replace("Z", "+00:00")
            ).date()
        
        # Handle file path
        if data.get("structure_file_path"):
            structure.structure_file_path = data["structure_file_path"]
        
        # Add chains
        for chain_data in data.get("chains", []):
            chain = Chain(
                id=self.generate_id("chain", structure.id, chain_data["chain_id"]),
                source="PDB",
                chain_id=chain_data["chain_id"],
                sequence=chain_data.get("sequence"),
                molecule_type=chain_data.get("molecule_type"),
                length=chain_data.get("length"),
                uniprot_accession=chain_data.get("uniprot_accession"),
            )
            structure.chains.append(chain)
        
        # Add ligands
        for ligand_data in data.get("ligands", []):
            ligand = Ligand(
                id=self.generate_id("ligand", structure.id, ligand_data["ligand_id"]),
                source="PDB",
                ligand_id=ligand_data["ligand_id"],
                name=ligand_data.get("name"),
                formula=ligand_data.get("formula"),
                molecular_weight=ligand_data.get("molecular_weight"),
                smiles=ligand_data.get("smiles"),
                inchi=ligand_data.get("inchi"),
            )
            structure.ligands.append(ligand)
        
        # Save to database
        if not existing:
            self.db_session.add(structure)
        
        self.db_session.commit()
        logger.info(f"Saved structure {structure.structure_id} to database")
        
        return structure
    
    def _determine_structure_type(self, entry_data: Dict) -> str:
        """Determine structure determination method."""
        methods = entry_data.get("exptl", [])
        if methods:
            return methods[0].get("method", "").upper()
        return "UNKNOWN"
    
    def _extract_resolution(self, entry_data: Dict) -> Optional[float]:
        """Extract resolution value."""
        reflns = entry_data.get("reflns", [])
        if reflns:
            return reflns[0].get("d_resolution_high")
        
        # For EM structures
        em3d = entry_data.get("em_3d_reconstruction", [])
        if em3d:
            return em3d[0].get("resolution")
        
        return None
    
    def _extract_r_factor(self, entry_data: Dict) -> Optional[float]:
        """Extract R-factor value."""
        refine = entry_data.get("refine", [])
        if refine:
            return refine[0].get("ls_R_factor_R_work")
        return None
    
    def _extract_organism_simple(self, entry_data: Dict, polymer_data: Dict) -> Optional[str]:
        """Extract organism from available data."""
        # Try polymer data first
        if polymer_data:
            source_organism = polymer_data.get("rcsb_entity_source_organism", [])
            if source_organism:
                return source_organism[0].get("ncbi_scientific_name")
        
        # Fall back to entry data
        if entry_data:
            # Look for organism in entity_src_gen
            src_gen = entry_data.get("entity_src_gen", [])
            if src_gen:
                return src_gen[0].get("pdbx_gene_src_scientific_name", "")
        
        return None
    
    def _process_chains_simple(self, entry_data: Dict) -> List[Dict]:
        """Process chain data from entry data."""
        chains = []
        
        # Get entity information
        entities = entry_data.get("rcsb_entry_info", {}).get("polymer_entity_ids", [])
        
        # For simplicity, just create basic chain info
        # In a real implementation, we'd parse more detailed data
        for i, entity_id in enumerate(entities):
            chain_id = chr(65 + i)  # A, B, C, etc.
            chains.append({
                "chain_id": chain_id,
                "sequence": "",  # Would need additional API call to get sequence
                "molecule_type": "protein",
                "length": 0,
                "uniprot_accession": None,
            })
        
        # If no entity info, at least create one chain
        if not chains:
            chains.append({
                "chain_id": "A",
                "sequence": "",
                "molecule_type": "protein", 
                "length": 0,
                "uniprot_accession": None,
            })
        
        return chains
    
    def _download_structure_file(self, pdb_id: str, format: str = "pdb") -> str:
        """Download structure file to local storage."""
        logger.info(f"Downloading {pdb_id} structure file")
        
        # Get structure content
        content = self.api_client.get_structure(pdb_id, format=format)
        
        # Save to file
        filename = f"{pdb_id}.{format}"
        file_path = settings.storage.raw_data_dir / "structures" / filename
        file_path.parent.mkdir(exist_ok=True)
        
        with open(file_path, "w") as f:
            f.write(content)
        
        logger.info(f"Saved structure to {file_path}")
        return file_path
    
    def search_by_sequence(
        self,
        sequence: str,
        e_value: float = 0.1,
        max_results: int = 100,
        **kwargs
    ) -> List[Structure]:
        """Search PDB by sequence similarity and collect results."""
        logger.info("Searching PDB by sequence similarity")
        
        # Search
        search_results = self.api_client.sequence_search(
            sequence=sequence,
            e_value=e_value,
        )
        
        # Extract PDB IDs from results
        pdb_ids = []
        for result in search_results.get("result_set", []):
            pdb_id = result.get("identifier", "").split("_")[0]
            if pdb_id and len(pdb_ids) < max_results:
                pdb_ids.append(pdb_id)
        
        # Collect structures
        structures = []
        for pdb_id in pdb_ids:
            try:
                structure = self.process_and_save(pdb_id, **kwargs)
                if structure:
                    structures.append(structure)
            except Exception as e:
                logger.error(f"Failed to process {pdb_id}: {e}")
                continue
        
        logger.info(f"Collected {len(structures)} structures")
        return structures