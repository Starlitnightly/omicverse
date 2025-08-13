"""Tests for database models."""

import pytest
from datetime import datetime

from omicverse.external.datacollect.models.base import init_db, get_db
from omicverse.external.datacollect.models.protein import Protein, ProteinFeature, GOTerm
from omicverse.external.datacollect.models.structure import Structure, Chain, Ligand
from omicverse.external.datacollect.models.genomic import Gene, Variant, Expression


class TestProteinModel:
    """Test Protein model and relationships."""
    
    def test_create_protein(self, test_db):
        """Test creating a protein."""
        protein = Protein(
            id="test_protein_1",
            source="UniProt",
            accession="P12345",
            entry_name="TEST_HUMAN",
            protein_name="Test protein",
            gene_name="TEST",
            organism="Homo sapiens",
            organism_id=9606,
            sequence="MVLSEGEWQLVLHVWAK",
            sequence_length=17,
            molecular_weight=2000.0
        )
        
        test_db.add(protein)
        test_db.commit()
        
        # Query back
        saved = test_db.query(Protein).filter_by(accession="P12345").first()
        assert saved is not None
        assert saved.protein_name == "Test protein"
        assert saved.sequence_length == 17
    
    def test_protein_features(self, test_db):
        """Test protein features relationship."""
        protein = Protein(
            id="test_protein_2",
            source="UniProt",
            accession="P67890",
            sequence="ACDEFGHIKLMNPQRSTVWY"
        )
        
        feature1 = ProteinFeature(
            id="test_feature_1",
            source="UniProt",
            feature_type="domain",
            start_position=1,
            end_position=10,
            description="Test domain"
        )
        
        feature2 = ProteinFeature(
            id="test_feature_2",
            source="UniProt",
            feature_type="site",
            start_position=15,
            end_position=15,
            description="Active site"
        )
        
        protein.features = [feature1, feature2]
        
        test_db.add(protein)
        test_db.commit()
        
        # Query back
        saved = test_db.query(Protein).filter_by(accession="P67890").first()
        assert len(saved.features) == 2
        assert saved.features[0].feature_type == "domain"
        assert saved.features[1].feature_type == "site"
    
    def test_protein_go_terms(self, test_db):
        """Test protein-GO term many-to-many relationship."""
        protein1 = Protein(
            id="test_protein_3",
            source="UniProt",
            accession="Q12345",
            sequence="ACDEF"
        )
        
        protein2 = Protein(
            id="test_protein_4",
            source="UniProt",
            accession="Q67890",
            sequence="GHIKL"
        )
        
        go_term1 = GOTerm(
            id="test_go_1",
            source="UniProt",
            go_id="GO:0005737",
            name="cytoplasm",
            namespace="cellular_component"
        )
        
        go_term2 = GOTerm(
            id="test_go_2",
            source="UniProt",
            go_id="GO:0008270",
            name="zinc ion binding",
            namespace="molecular_function"
        )
        
        # Both proteins have cytoplasm annotation
        protein1.go_terms = [go_term1, go_term2]
        protein2.go_terms = [go_term1]
        
        test_db.add_all([protein1, protein2])
        test_db.commit()
        
        # Query relationships
        saved_go1 = test_db.query(GOTerm).filter_by(go_id="GO:0005737").first()
        assert len(saved_go1.proteins) == 2
        
        saved_protein1 = test_db.query(Protein).filter_by(accession="Q12345").first()
        assert len(saved_protein1.go_terms) == 2


class TestStructureModel:
    """Test Structure model and relationships."""
    
    def test_create_structure(self, test_db):
        """Test creating a structure."""
        structure = Structure(
            id="test_structure_1",
            source="PDB",
            structure_id="1ABC",
            title="Test structure of test protein",
            structure_type="X-RAY DIFFRACTION",
            resolution=2.5,
            r_factor=0.2,
            organism="Homo sapiens",
            deposition_date=datetime(2023, 1, 1).date(),
            release_date=datetime(2023, 1, 15).date()
        )
        
        test_db.add(structure)
        test_db.commit()
        
        # Query back
        saved = test_db.query(Structure).filter_by(structure_id="1ABC").first()
        assert saved is not None
        assert saved.title == "Test structure of test protein"
        assert saved.resolution == 2.5
    
    def test_structure_chains(self, test_db):
        """Test structure-chain relationship."""
        structure = Structure(
            id="test_structure_2",
            source="PDB",
            structure_id="2DEF"
        )
        
        chain_a = Chain(
            id="test_chain_1",
            source="PDB",
            chain_id="A",
            sequence="ACDEFGHIKL",
            molecule_type="protein",
            length=10,
            uniprot_accession="P12345"
        )
        
        chain_b = Chain(
            id="test_chain_2",
            source="PDB",
            chain_id="B",
            sequence="MNPQRSTVWY",
            molecule_type="protein",
            length=10,
            uniprot_accession="P67890"
        )
        
        structure.chains = [chain_a, chain_b]
        
        test_db.add(structure)
        test_db.commit()
        
        # Query back
        saved = test_db.query(Structure).filter_by(structure_id="2DEF").first()
        assert len(saved.chains) == 2
        chain_ids = [c.chain_id for c in saved.chains]
        assert "A" in chain_ids
        assert "B" in chain_ids
    
    def test_structure_ligands(self, test_db):
        """Test structure-ligand relationship."""
        structure = Structure(
            id="test_structure_3",
            source="PDB",
            structure_id="3GHI"
        )
        
        ligand1 = Ligand(
            id="test_ligand_1",
            source="PDB",
            ligand_id="ZN",
            name="ZINC ION",
            formula="Zn",
            molecular_weight=65.38
        )
        
        ligand2 = Ligand(
            id="test_ligand_2",
            source="PDB",
            ligand_id="ATP",
            name="ADENOSINE-5'-TRIPHOSPHATE",
            formula="C10 H16 N5 O13 P3",
            molecular_weight=507.18
        )
        
        structure.ligands = [ligand1, ligand2]
        
        test_db.add(structure)
        test_db.commit()
        
        # Query back
        saved = test_db.query(Structure).filter_by(structure_id="3GHI").first()
        assert len(saved.ligands) == 2
        ligand_ids = [l.ligand_id for l in saved.ligands]
        assert "ZN" in ligand_ids
        assert "ATP" in ligand_ids


class TestGenomicModels:
    """Test genomic data models."""
    
    def test_create_gene(self, test_db):
        """Test creating a gene."""
        gene = Gene(
            id="test_gene_1",
            source="Ensembl",
            gene_id="ENSG00000141510",
            symbol="TP53",
            ensembl_id="ENSG00000141510",
            entrez_id=7157,
            name="tumor protein p53",
            description="Tumor suppressor gene",
            gene_type="protein_coding",
            chromosome="17",
            start_position=7661779,
            end_position=7687550,
            strand="-"
        )
        
        test_db.add(gene)
        test_db.commit()
        
        # Query back
        saved = test_db.query(Gene).filter_by(symbol="TP53").first()
        assert saved is not None
        assert saved.chromosome == "17"
        assert saved.gene_type == "protein_coding"
    
    def test_create_variant(self, test_db):
        """Test creating a variant."""
        variant = Variant(
            id="test_variant_1",
            source="dbSNP",
            variant_id="rs1234567",
            rsid="rs1234567",
            chromosome="1",
            position=100000,
            reference_allele="A",
            alternate_allele="G",
            variant_type="SNP",
            consequence="missense",
            gene_symbol="TEST",
            protein_change="p.Val10Met",
            global_maf=0.05,
            clinical_significance="benign"
        )
        
        test_db.add(variant)
        test_db.commit()
        
        # Query back
        saved = test_db.query(Variant).filter_by(rsid="rs1234567").first()
        assert saved is not None
        assert saved.variant_type == "SNP"
        assert saved.global_maf == 0.05
    
    def test_create_expression(self, test_db):
        """Test creating expression data."""
        expression = Expression(
            id="test_expr_1",
            source="GEO",
            dataset_id="GSE12345",
            sample_id="GSM12345",
            gene_id="ENSG00000141510",
            expression_value=10.5,
            log2_fold_change=2.3,
            p_value=0.001,
            adjusted_p_value=0.01,
            tissue="liver",
            cell_type="hepatocyte",
            condition="treatment",
            platform="RNA-seq"
        )
        
        test_db.add(expression)
        test_db.commit()
        
        # Query back
        saved = test_db.query(Expression).filter_by(dataset_id="GSE12345").first()
        assert saved is not None
        assert saved.expression_value == 10.5
        assert saved.tissue == "liver"


class TestModelMethods:
    """Test model methods."""
    
    def test_to_dict(self, test_db):
        """Test model to_dict method."""
        protein = Protein(
            id="test_protein_dict",
            source="UniProt",
            accession="P99999",
            protein_name="Test protein",
            sequence="ACDEF"
        )
        
        test_db.add(protein)
        test_db.commit()
        
        # Convert to dict
        data = protein.to_dict()
        
        assert isinstance(data, dict)
        assert data["accession"] == "P99999"
        assert data["protein_name"] == "Test protein"
        assert data["sequence"] == "ACDEF"
        assert "created_at" in data
        assert "updated_at" in data