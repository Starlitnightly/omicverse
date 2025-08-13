"""Tests for data transformation utilities."""

import pytest
from omicverse.external.datacollect.utils.transformers import (
    SequenceTransformer,
    IdentifierMapper,
    DataNormalizer,
    FeatureExtractor
)


class TestSequenceTransformer:
    """Test sequence transformation utilities."""
    
    def test_translate_dna_to_protein(self):
        """Test DNA to protein translation."""
        # Standard genetic code
        dna = "ATGGCCGAA"  # Met-Ala-Glu
        protein = SequenceTransformer.translate_dna_to_protein(dna)
        assert protein == "MAE"
        
        # With stop codon
        dna = "ATGGCCTAA"  # Met-Ala-Stop
        protein = SequenceTransformer.translate_dna_to_protein(dna)
        assert protein == "MA*"
    
    def test_reverse_complement(self):
        """Test reverse complement."""
        assert SequenceTransformer.reverse_complement("ATCG") == "CGAT"
        assert SequenceTransformer.reverse_complement("AAAA") == "TTTT"
        assert SequenceTransformer.reverse_complement("GCTA") == "TAGC"
    
    def test_transcribe(self):
        """Test DNA to RNA transcription."""
        assert SequenceTransformer.transcribe("ATCG") == "AUCG"
        assert SequenceTransformer.transcribe("TTTT") == "UUUU"
        assert SequenceTransformer.transcribe("atcg") == "AUCG"  # Case handling
    
    def test_get_orf(self):
        """Test ORF finding."""
        # Simple ORF with start and stop
        sequence = "ATGGCCTAA"  # ATG(start)-GCC-TAA(stop)
        orfs = SequenceTransformer.get_orf(sequence, min_length=6)
        assert len(orfs) == 1
        assert orfs[0] == (0, 6, "MA")
        
        # No ORF (no start codon)
        sequence = "GCCTAAGCC"
        orfs = SequenceTransformer.get_orf(sequence, min_length=6)
        assert len(orfs) == 0
    
    def test_calculate_gc_content(self):
        """Test GC content calculation."""
        assert SequenceTransformer.calculate_gc_content("ATCG") == 50.0
        assert SequenceTransformer.calculate_gc_content("AAAA") == 0.0
        assert SequenceTransformer.calculate_gc_content("GGCC") == 100.0
        assert SequenceTransformer.calculate_gc_content("") == 0.0


class TestIdentifierMapper:
    """Test identifier mapping utilities."""
    
    def test_normalize_chromosome(self):
        """Test chromosome name normalization."""
        assert IdentifierMapper.normalize_chromosome("chr1") == "1"
        assert IdentifierMapper.normalize_chromosome("CHR1") == "1"
        assert IdentifierMapper.normalize_chromosome("1") == "1"
        assert IdentifierMapper.normalize_chromosome("chrX") == "X"
        assert IdentifierMapper.normalize_chromosome("chrM") == "MT"
        assert IdentifierMapper.normalize_chromosome("23") == "X"
        assert IdentifierMapper.normalize_chromosome("24") == "Y"
    
    def test_extract_gene_id_from_ensembl(self):
        """Test Ensembl gene ID extraction."""
        assert IdentifierMapper.extract_gene_id_from_ensembl("ENSG00000141510") == "ENSG00000141510"
        assert IdentifierMapper.extract_gene_id_from_ensembl("ENSG00000141510.11") == "ENSG00000141510"
        assert IdentifierMapper.extract_gene_id_from_ensembl("ENST00000269305") is None
        assert IdentifierMapper.extract_gene_id_from_ensembl("invalid") is None
    
    def test_uniprot_to_pdb_chains(self, test_db):
        """Test UniProt to PDB chain mapping."""
        from omicverse.external.datacollect.models.structure import Structure, Chain
        
        # Create test structures
        structure = Structure(
            id="test_struct_1",
            source="TEST",
            structure_id="1ABC"
        )
        
        chain1 = Chain(
            id="test_chain_1",
            source="TEST",
            chain_id="A",
            uniprot_accession="P12345"
        )
        
        chain2 = Chain(
            id="test_chain_2",
            source="TEST",
            chain_id="B",
            uniprot_accession="P67890"
        )
        
        structure.chains = [chain1, chain2]
        
        # Test mapping
        mapping = IdentifierMapper.uniprot_to_pdb_chains("P12345", [structure])
        assert mapping == {"1ABC": ["A"]}
        
        mapping = IdentifierMapper.uniprot_to_pdb_chains("P67890", [structure])
        assert mapping == {"1ABC": ["B"]}
        
        mapping = IdentifierMapper.uniprot_to_pdb_chains("P99999", [structure])
        assert mapping == {}


class TestDataNormalizer:
    """Test data normalization utilities."""
    
    def test_normalize_organism_name(self):
        """Test organism name normalization."""
        assert DataNormalizer.normalize_organism_name("H. sapiens") == "Homo sapiens"
        assert DataNormalizer.normalize_organism_name("Human") == "Homo sapiens"
        assert DataNormalizer.normalize_organism_name("M. musculus") == "Mus musculus"
        assert DataNormalizer.normalize_organism_name("Mouse") == "Mus musculus"
        assert DataNormalizer.normalize_organism_name("Unknown species") == "Unknown species"
        assert DataNormalizer.normalize_organism_name("") == ""
    
    def test_normalize_gene_name(self):
        """Test gene name normalization."""
        assert DataNormalizer.normalize_gene_name("TP53_HUMAN") == "TP53"
        assert DataNormalizer.normalize_gene_name("Tp53_MOUSE") == "TP53"
        assert DataNormalizer.normalize_gene_name("BRCA1.1") == "BRCA1"
        assert DataNormalizer.normalize_gene_name("gene-name") == "GENE-NAME"
        assert DataNormalizer.normalize_gene_name("") == ""
    
    def test_clean_sequence(self):
        """Test sequence cleaning."""
        # Protein sequence
        assert DataNormalizer.clean_sequence("A C D E F", "protein") == "ACDEF"
        assert DataNormalizer.clean_sequence("ACDEF123", "protein") == "ACDEF"
        assert DataNormalizer.clean_sequence("ACDEFXYZ", "protein") == "ACDEFY"  # X and Z invalid, Y is valid
        
        # DNA sequence
        assert DataNormalizer.clean_sequence("A T C G", "dna") == "ATCG"
        assert DataNormalizer.clean_sequence("ATCGU", "dna") == "ATCG"  # Remove RNA base
        
        # RNA sequence
        assert DataNormalizer.clean_sequence("A U C G", "rna") == "AUCG"
        assert DataNormalizer.clean_sequence("AUCGT", "rna") == "AUCG"  # Remove DNA base
    
    def test_standardize_variant_notation(self):
        """Test variant notation standardization."""
        # SNV
        assert DataNormalizer.standardize_variant_notation("A", "G", 100) == "g.100A>G"
        
        # Deletion
        assert DataNormalizer.standardize_variant_notation("ATG", "-", 100) == "g.100_102del"
        assert DataNormalizer.standardize_variant_notation("ATG", "", 100) == "g.100_102del"
        
        # Insertion
        assert DataNormalizer.standardize_variant_notation("A", "ATG", 100) == "g.100_101insTG"
        
        # Complex
        assert DataNormalizer.standardize_variant_notation("ATG", "CCC", 100) == "g.100_102delinsCCC"


class TestFeatureExtractor:
    """Test feature extraction utilities."""
    
    def test_extract_sequence_features(self):
        """Test protein sequence feature extraction."""
        sequence = "ACDEFGHIKLMNPQRSTVWY"  # All 20 amino acids
        features = FeatureExtractor.extract_sequence_features(sequence)
        
        # Check basic features
        assert features["length"] == 20
        assert features["aa_A"] == 1/20
        assert features["aa_C"] == 1/20
        
        # Check property ratios
        assert features["hydrophobic_ratio"] > 0
        assert features["charged_ratio"] > 0
        assert features["polar_ratio"] > 0
        assert features["molecular_weight"] > 0
        
        # Empty sequence
        features = FeatureExtractor.extract_sequence_features("")
        assert features == {}