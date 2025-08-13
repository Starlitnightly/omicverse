"""Data validation utilities."""

import re
import logging
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, ValidationError


logger = logging.getLogger(__name__)


class SequenceValidator:
    """Validator for biological sequences."""
    
    # Valid amino acid codes
    AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
    AMINO_ACIDS_WITH_AMBIGUOUS = set("ACDEFGHIKLMNPQRSTVWYBZX*")
    
    # Valid nucleotide codes
    DNA_BASES = set("ACGT")
    RNA_BASES = set("ACGU")
    NUCLEOTIDES_WITH_AMBIGUOUS = set("ACGTUNRYSWKMBDHV")
    
    @classmethod
    def is_valid_protein_sequence(cls, sequence: str, allow_ambiguous: bool = True) -> bool:
        """Validate protein sequence.
        
        Args:
            sequence: Amino acid sequence
            allow_ambiguous: Whether to allow ambiguous codes
        
        Returns:
            True if valid protein sequence
        """
        if not sequence:
            return False
        
        sequence = sequence.upper().strip()
        valid_chars = cls.AMINO_ACIDS_WITH_AMBIGUOUS if allow_ambiguous else cls.AMINO_ACIDS
        
        return all(aa in valid_chars for aa in sequence)
    
    @classmethod
    def is_valid_dna_sequence(cls, sequence: str, allow_ambiguous: bool = False) -> bool:
        """Validate DNA sequence."""
        if not sequence:
            return False
        
        sequence = sequence.upper().strip()
        valid_chars = cls.NUCLEOTIDES_WITH_AMBIGUOUS if allow_ambiguous else cls.DNA_BASES
        
        return all(base in valid_chars for base in sequence)
    
    @classmethod
    def is_valid_rna_sequence(cls, sequence: str, allow_ambiguous: bool = False) -> bool:
        """Validate RNA sequence."""
        if not sequence:
            return False
        
        sequence = sequence.upper().strip()
        valid_chars = cls.NUCLEOTIDES_WITH_AMBIGUOUS if allow_ambiguous else cls.RNA_BASES
        
        return all(base in valid_chars for base in sequence)


class IdentifierValidator:
    """Validator for biological identifiers."""
    
    # Regex patterns for common identifiers
    PATTERNS = {
        "uniprot": re.compile(r"^[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$"),
        "pdb": re.compile(r"^[0-9][A-Z0-9]{3}$", re.IGNORECASE),
        "ensembl_gene": re.compile(r"^ENS[A-Z]{0,3}G[0-9]{11}$"),
        "ensembl_transcript": re.compile(r"^ENS[A-Z]{0,3}T[0-9]{11}$"),
        "ensembl_protein": re.compile(r"^ENS[A-Z]{0,3}P[0-9]{11}$"),
        "refseq_protein": re.compile(r"^(NP|XP|YP|ZP)_[0-9]+(\.[0-9]+)?$"),
        "refseq_nucleotide": re.compile(r"^(NC|NM|NR|XM|XR)_[0-9]+(\.[0-9]+)?$"),
        "go": re.compile(r"^GO:[0-9]{7}$"),
        "pfam": re.compile(r"^PF[0-9]{5}$"),
        "interpro": re.compile(r"^IPR[0-9]{6}$"),
        "kegg": re.compile(r"^[a-z]{3,4}:[0-9]+$"),
        "ncbi_gene": re.compile(r"^[0-9]+$"),
        "dbsnp": re.compile(r"^rs[0-9]+$"),
    }
    
    @classmethod
    def validate(cls, identifier: str, id_type: str) -> bool:
        """Validate identifier format.
        
        Args:
            identifier: Identifier to validate
            id_type: Type of identifier
        
        Returns:
            True if valid format
        """
        if not identifier or id_type not in cls.PATTERNS:
            return False
        
        pattern = cls.PATTERNS[id_type]
        return bool(pattern.match(identifier.strip()))
    
    @classmethod
    def detect_type(cls, identifier: str) -> Optional[str]:
        """Detect identifier type.
        
        Args:
            identifier: Identifier to check
        
        Returns:
            Detected type or None
        """
        identifier = identifier.strip()
        
        for id_type, pattern in cls.PATTERNS.items():
            if pattern.match(identifier):
                return id_type
        
        return None


# Pydantic models for validation

class ProteinData(BaseModel):
    """Validation model for protein data."""
    
    accession: str
    sequence: str
    organism: Optional[str] = None
    gene_name: Optional[str] = None
    protein_name: Optional[str] = None
    
    @field_validator("accession")
    def validate_accession(cls, v):
        if not IdentifierValidator.validate(v, "uniprot"):
            raise ValueError(f"Invalid UniProt accession: {v}")
        return v
    
    @field_validator("sequence")
    def validate_sequence(cls, v):
        if not SequenceValidator.is_valid_protein_sequence(v):
            raise ValueError("Invalid protein sequence")
        return v.upper()
    
    model_config = {"extra": "allow"}


class StructureData(BaseModel):
    """Validation model for structure data."""
    
    pdb_id: str
    resolution: Optional[float] = Field(None, gt=0, le=100)
    structure_type: str
    chains: List[Dict[str, Any]] = []
    
    @field_validator("pdb_id")
    def validate_pdb_id(cls, v):
        if not IdentifierValidator.validate(v, "pdb"):
            raise ValueError(f"Invalid PDB ID: {v}")
        return v.upper()
    
    @field_validator("structure_type")
    def validate_structure_type(cls, v):
        valid_types = ["X-RAY DIFFRACTION", "SOLUTION NMR", "ELECTRON MICROSCOPY", 
                      "NEUTRON DIFFRACTION", "FIBER DIFFRACTION", "ELECTRON CRYSTALLOGRAPHY"]
        if v.upper() not in valid_types:
            raise ValueError(f"Invalid structure type: {v}")
        return v.upper()
    
    model_config = {"extra": "allow"}


class VariantData(BaseModel):
    """Validation model for variant data."""
    
    chromosome: str
    position: int = Field(gt=0)
    reference_allele: str
    alternate_allele: str
    rsid: Optional[str] = None
    
    @field_validator("chromosome")
    def validate_chromosome(cls, v):
        # Accept various chromosome formats
        v = str(v).upper().replace("CHR", "")
        if v not in [str(i) for i in range(1, 23)] + ["X", "Y", "M", "MT"]:
            raise ValueError(f"Invalid chromosome: {v}")
        return v
    
    @field_validator("reference_allele", "alternate_allele")
    def validate_alleles(cls, v):
        if not v or not SequenceValidator.is_valid_dna_sequence(v):
            raise ValueError(f"Invalid allele: {v}")
        return v.upper()
    
    @field_validator("rsid")
    def validate_rsid(cls, v):
        if v and not IdentifierValidator.validate(v, "dbsnp"):
            raise ValueError(f"Invalid dbSNP ID: {v}")
        return v
    
    model_config = {"extra": "allow"}


def validate_data(data: Dict[str, Any], data_type: str) -> Union[Dict[str, Any], List[str]]:
    """Validate data against schema.
    
    Args:
        data: Data to validate
        data_type: Type of data (protein, structure, variant)
    
    Returns:
        Validated data or list of validation errors
    """
    model_map = {
        "protein": ProteinData,
        "structure": StructureData,
        "variant": VariantData,
    }
    
    model_class = model_map.get(data_type)
    if not model_class:
        return ["Unknown data type"]
    
    try:
        validated = model_class(**data)
        return validated.dict()
    except ValidationError as e:
        errors = []
        for error in e.errors():
            field = ".".join(str(loc) for loc in error["loc"])
            errors.append(f"{field}: {error['msg']}")
        return errors


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file system usage.
    
    Args:
        filename: Original filename
    
    Returns:
        Sanitized filename
    """
    # Remove or replace invalid characters
    invalid_chars = r'<>:"/\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, "_")
    
    # Limit length
    max_length = 255
    if len(filename) > max_length:
        name, ext = filename.rsplit(".", 1) if "." in filename else (filename, "")
        filename = f"{name[:max_length-len(ext)-1]}.{ext}" if ext else name[:max_length]
    
    return filename