"""Tests for validation utilities."""

import pytest
from omicverse.external.datacollect.utils.validation import (
    SequenceValidator,
    IdentifierValidator,
    validate_data,
    sanitize_filename
)


class TestSequenceValidator:
    """Test sequence validation."""
    
    def test_valid_protein_sequence(self):
        """Test valid protein sequences."""
        assert SequenceValidator.is_valid_protein_sequence("ACDEFGHIKLMNPQRSTVWY")
        assert SequenceValidator.is_valid_protein_sequence("MVLSEGEWQLVLHVWAK")
        assert SequenceValidator.is_valid_protein_sequence("acdefg")  # Case insensitive
    
    def test_invalid_protein_sequence(self):
        """Test invalid protein sequences."""
        assert not SequenceValidator.is_valid_protein_sequence("")
        assert not SequenceValidator.is_valid_protein_sequence("ACDEFG123")
        assert not SequenceValidator.is_valid_protein_sequence("ACDEFG-HIJKL")
    
    def test_ambiguous_protein_sequence(self):
        """Test ambiguous amino acid codes."""
        assert SequenceValidator.is_valid_protein_sequence("ACDEFGXYZ", allow_ambiguous=True)
        assert not SequenceValidator.is_valid_protein_sequence("ACDEFGXYZ", allow_ambiguous=False)
    
    def test_valid_dna_sequence(self):
        """Test valid DNA sequences."""
        assert SequenceValidator.is_valid_dna_sequence("ACGT")
        assert SequenceValidator.is_valid_dna_sequence("acgtACGT")
        assert SequenceValidator.is_valid_dna_sequence("ACGTN", allow_ambiguous=True)
    
    def test_invalid_dna_sequence(self):
        """Test invalid DNA sequences."""
        assert not SequenceValidator.is_valid_dna_sequence("")
        assert not SequenceValidator.is_valid_dna_sequence("ACGTU")  # RNA base
        assert not SequenceValidator.is_valid_dna_sequence("ACGTN", allow_ambiguous=False)
    
    def test_valid_rna_sequence(self):
        """Test valid RNA sequences."""
        assert SequenceValidator.is_valid_rna_sequence("ACGU")
        assert SequenceValidator.is_valid_rna_sequence("acguACGU")
        assert not SequenceValidator.is_valid_rna_sequence("ACGT")  # DNA base


class TestIdentifierValidator:
    """Test identifier validation."""
    
    def test_uniprot_validation(self):
        """Test UniProt accession validation."""
        assert IdentifierValidator.validate("P12345", "uniprot")
        assert IdentifierValidator.validate("Q9Y6K9", "uniprot")
        assert IdentifierValidator.validate("A0A024R1X8", "uniprot")
        assert not IdentifierValidator.validate("P1234", "uniprot")  # Too short
        assert not IdentifierValidator.validate("12345", "uniprot")  # No letter prefix
    
    def test_pdb_validation(self):
        """Test PDB ID validation."""
        assert IdentifierValidator.validate("1ABC", "pdb")
        assert IdentifierValidator.validate("7xyz", "pdb")  # Case insensitive
        assert not IdentifierValidator.validate("ABC", "pdb")  # Too short
        assert not IdentifierValidator.validate("12345", "pdb")  # Too long
    
    def test_ensembl_validation(self):
        """Test Ensembl ID validation."""
        assert IdentifierValidator.validate("ENSG00000141510", "ensembl_gene")
        assert IdentifierValidator.validate("ENST00000269305", "ensembl_transcript")
        assert IdentifierValidator.validate("ENSP00000269305", "ensembl_protein")
        assert not IdentifierValidator.validate("ENS00000141510", "ensembl_gene")  # Missing G
    
    def test_go_validation(self):
        """Test GO term validation."""
        assert IdentifierValidator.validate("GO:0008270", "go")
        assert not IdentifierValidator.validate("GO:123456", "go")  # Too few digits
        assert not IdentifierValidator.validate("GO:12345678", "go")  # Too many digits
    
    def test_dbsnp_validation(self):
        """Test dbSNP ID validation."""
        assert IdentifierValidator.validate("rs12345", "dbsnp")
        assert IdentifierValidator.validate("rs123456789", "dbsnp")
        assert not IdentifierValidator.validate("rs", "dbsnp")  # No number
        assert not IdentifierValidator.validate("12345", "dbsnp")  # No rs prefix
    
    def test_detect_type(self):
        """Test identifier type detection."""
        assert IdentifierValidator.detect_type("P12345") == "uniprot"
        assert IdentifierValidator.detect_type("1ABC") == "pdb"
        assert IdentifierValidator.detect_type("GO:0008270") == "go"
        assert IdentifierValidator.detect_type("rs12345") == "dbsnp"
        assert IdentifierValidator.detect_type("unknown123") is None


class TestDataValidation:
    """Test data validation with Pydantic models."""
    
    def test_valid_protein_data(self, sample_protein_data):
        """Test valid protein data."""
        # Remove non-model fields
        data = {
            "accession": sample_protein_data["accession"],
            "sequence": sample_protein_data["sequence"],
            "organism": sample_protein_data["organism"],
            "gene_name": sample_protein_data["gene_name"],
            "protein_name": sample_protein_data["protein_name"]
        }
        
        result = validate_data(data, "protein")
        assert isinstance(result, dict)
        assert result["accession"] == "P12345"
        assert result["sequence"] == sample_protein_data["sequence"]
    
    def test_invalid_protein_data(self):
        """Test invalid protein data."""
        data = {
            "accession": "INVALID",
            "sequence": "ACGTU",  # Contains U which is only in RNA, not protein
        }
        
        result = validate_data(data, "protein")
        assert isinstance(result, list)  # List of errors
        assert any("accession" in error for error in result)
        assert any("sequence" in error for error in result)
    
    def test_valid_structure_data(self, sample_structure_data):
        """Test valid structure data."""
        data = {
            "pdb_id": sample_structure_data["pdb_id"],
            "resolution": sample_structure_data["resolution"],
            "structure_type": sample_structure_data["structure_type"],
            "chains": sample_structure_data["chains"]
        }
        
        result = validate_data(data, "structure")
        assert isinstance(result, dict)
        assert result["pdb_id"] == "1ABC"
        assert result["resolution"] == 2.5
    
    def test_valid_variant_data(self, sample_variant_data):
        """Test valid variant data."""
        data = {
            "chromosome": sample_variant_data["chromosome"],
            "position": sample_variant_data["position"],
            "reference_allele": sample_variant_data["reference_allele"],
            "alternate_allele": sample_variant_data["alternate_allele"],
            "rsid": sample_variant_data["rsid"]
        }
        
        result = validate_data(data, "variant")
        assert isinstance(result, dict)
        assert result["chromosome"] == "1"
        assert result["position"] == 100000


class TestUtilities:
    """Test utility functions."""
    
    def test_sanitize_filename(self):
        """Test filename sanitization."""
        assert sanitize_filename("test.txt") == "test.txt"
        assert sanitize_filename("test/file.txt") == "test_file.txt"
        assert sanitize_filename("test:file?.txt") == "test_file_.txt"
        assert sanitize_filename("test<>file|.txt") == "test__file_.txt"
        
        # Test length limiting
        long_name = "a" * 300 + ".txt"
        sanitized = sanitize_filename(long_name)
        assert len(sanitized) <= 255
        assert sanitized.endswith(".txt")