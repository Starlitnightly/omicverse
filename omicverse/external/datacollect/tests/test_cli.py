"""Tests for CLI functionality."""

import pytest
from click.testing import CliRunner
from unittest.mock import patch, Mock

from src.cli import cli, init, status, collect


class TestCLI:
    """Test CLI commands."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    @patch('src.cli.check_database_connection')
    @patch('src.cli.initialize_database')
    def test_init_command(self, mock_init_db, mock_check_conn):
        """Test database initialization command."""
        mock_check_conn.return_value = False  # Simulate no existing DB
        
        result = self.runner.invoke(init)
        
        assert result.exit_code == 0
        mock_init_db.assert_called_once()
        assert "Database initialized successfully" in result.output
    
    @patch('src.cli.check_database_connection')
    @patch('src.cli.get_table_stats')
    @patch('src.cli.get_db')
    def test_status_command(self, mock_get_db, mock_stats, mock_check_conn):
        """Test status command."""
        mock_check_conn.return_value = True
        mock_db = Mock()
        mock_get_db.return_value = iter([mock_db])
        
        mock_stats.return_value = {
            "proteins": 10,
            "structures": 5,
            "variants": 100
        }
        
        result = self.runner.invoke(status)
        
        assert result.exit_code == 0
        assert "Database Status" in result.output
        assert "proteins: 10 records" in result.output
        assert "structures: 5 records" in result.output
    
    @patch('src.cli.check_database_connection')
    @patch('src.cli.UniProtCollector')
    def test_collect_uniprot(self, mock_collector_class, mock_check_conn):
        """Test UniProt collection command."""
        mock_check_conn.return_value = True
        
        # Mock collector
        mock_collector = Mock()
        mock_collector_class.return_value = mock_collector
        
        mock_protein = Mock()
        mock_protein.protein_name = "Test protein"
        mock_protein.organism = "Homo sapiens"
        mock_protein.sequence_length = 350
        mock_protein.pdb_ids = "1ABC,2DEF"
        
        mock_collector.process_and_save.return_value = mock_protein
        
        result = self.runner.invoke(collect, ['uniprot', 'P12345'])
        
        assert result.exit_code == 0
        assert "Collecting UniProt data for P12345" in result.output
        assert "Successfully collected: Test protein" in result.output
        mock_collector.process_and_save.assert_called_once_with(
            'P12345',
            include_features=True,
            save_to_file=False
        )
    
    @patch('src.cli.check_database_connection')
    @patch('src.cli.UniProtCollector')
    def test_collect_uniprot_search(self, mock_collector_class, mock_check_conn):
        """Test UniProt search command."""
        mock_check_conn.return_value = True
        
        # Mock collector
        mock_collector = Mock()
        mock_collector_class.return_value = mock_collector
        
        mock_proteins = [
            Mock(accession="P12345", protein_name="Protein 1", organism="Human"),
            Mock(accession="P67890", protein_name="Protein 2", organism="Mouse")
        ]
        
        mock_collector.search_and_collect.return_value = mock_proteins
        
        result = self.runner.invoke(
            collect, 
            ['uniprot-search', 'kinase', '--limit', '10', '--organism', 'human']
        )
        
        assert result.exit_code == 0
        assert "Searching UniProt: kinase AND organism_name:human" in result.output
        assert "Collected 2 proteins" in result.output
        assert "P12345: Protein 1" in result.output
    
    @patch('src.cli.check_database_connection')
    @patch('src.cli.PDBCollector')
    def test_collect_pdb(self, mock_collector_class, mock_check_conn):
        """Test PDB collection command."""
        mock_check_conn.return_value = True
        
        # Mock collector
        mock_collector = Mock()
        mock_collector_class.return_value = mock_collector
        
        mock_structure = Mock()
        mock_structure.title = "Crystal structure of test protein"
        mock_structure.structure_type = "X-RAY DIFFRACTION"
        mock_structure.resolution = 2.5
        mock_structure.chains = [Mock(), Mock()]  # 2 chains
        mock_structure.ligands = [Mock()]  # 1 ligand
        
        mock_collector.process_and_save.return_value = mock_structure
        
        result = self.runner.invoke(
            collect, 
            ['pdb', '1ABC', '--download', '--save-file']
        )
        
        assert result.exit_code == 0
        assert "Collecting PDB data for 1ABC" in result.output
        assert "Successfully collected: Crystal structure of test protein" in result.output
        assert "Resolution: 2.5 Å" in result.output
        assert "Chains: 2" in result.output
        
        mock_collector.process_and_save.assert_called_once_with(
            '1ABC',
            download_structure=True,
            save_to_file=True
        )
    
    @patch('src.cli.check_database_connection')
    @patch('src.cli.PDBCollector')
    def test_collect_pdb_blast(self, mock_collector_class, mock_check_conn):
        """Test PDB sequence search command."""
        mock_check_conn.return_value = True
        
        # Mock collector
        mock_collector = Mock()
        mock_collector_class.return_value = mock_collector
        
        mock_structures = [
            Mock(structure_id="1ABC", title="Structure 1", resolution=2.0),
            Mock(structure_id="2DEF", title="Structure 2", resolution=2.5)
        ]
        
        mock_collector.search_by_sequence.return_value = mock_structures
        
        sequence = "MVLSEGEWQLVLHVWAK"
        result = self.runner.invoke(
            collect,
            ['pdb-blast', sequence, '--e-value', '0.01', '--limit', '5']
        )
        
        assert result.exit_code == 0
        assert "Searching PDB by sequence" in result.output
        assert "Found 2 similar structures" in result.output
        assert "1ABC: Structure 1" in result.output
        assert "Resolution: 2.0 Å" in result.output
    
    @patch('src.cli.UniProtCollector')
    def test_no_database_connection(self, mock_collector_class):
        """Test behavior when database is not connected."""
        # Make the collector initialization raise an exception
        mock_collector_class.side_effect = Exception("Database not initialized")
        
        result = self.runner.invoke(collect, ['uniprot', 'P12345'])
        
        # The command should fail with an exception
        assert result.exit_code != 0
        # Check that exception is raised (click's CliRunner shows it in result.exception)
        assert result.exception is not None
        assert "Database not initialized" in str(result.exception)