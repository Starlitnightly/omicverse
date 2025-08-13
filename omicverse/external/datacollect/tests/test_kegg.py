"""Tests for KEGG API client and collector."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from omicverse.external.datacollect.api.kegg import KEGGClient
from omicverse.external.datacollect.collectors.kegg_collector import KEGGCollector
from omicverse.external.datacollect.models.pathway import Pathway
from omicverse.external.datacollect.models.genomic import Gene


@pytest.fixture
def mock_kegg_pathway_text():
    """Sample KEGG pathway text response."""
    return """ENTRY       hsa04110                    Pathway
NAME        Cell cycle - Homo sapiens (human)
DESCRIPTION Cell cycle progression is regulated by cyclin-dependent kinases
CLASS       Cellular Processes; Cell growth and death
PATHWAY_MAP hsa04110  Cell cycle
ORGANISM    Homo sapiens (human) [GN:hsa]
GENE        595  CCND1; cyclin D1 [KO:K04503]
            894  CCND2; cyclin D2 [KO:K10151]
            896  CCND3; cyclin D3 [KO:K10152]
///"""


@pytest.fixture
def mock_kegg_gene_text():
    """Sample KEGG gene text response."""
    return """ENTRY       hsa:7157                    CDS       T01001
NAME        TP53, BCC7, BMFS5, LFS1, P53, TRP53
DEFINITION  tumor protein p53
ORGANISM    hsa  Homo sapiens (human)
POSITION    17p13.1
MOTIF       Pfam: P53 P53_tetramer P53_TAD DEC-1_N
DBLINKS     NCBI-GeneID: 7157
            UniProt: P04637
PATHWAY     hsa04110  Cell cycle
            hsa04115  p53 signaling pathway
///"""


@pytest.fixture
def mock_kegg_link_response():
    """Sample KEGG link response."""
    return """hsa04110\thsa:595
hsa04110\thsa:894
hsa04110\thsa:896"""


class TestKEGGClient:
    """Test KEGG API client."""
    
    def test_initialization(self):
        """Test client initialization."""
        client = KEGGClient()
        assert "rest.kegg.jp" in client.base_url
        assert client.rate_limit == 10
    
    def test_get_default_headers(self):
        """Test KEGG-specific headers."""
        client = KEGGClient()
        headers = client.get_default_headers()
        assert headers["Accept"] == "text/plain"
    
    @patch.object(KEGGClient, 'get')
    def test_info(self, mock_get):
        """Test getting database info."""
        mock_response = Mock()
        mock_response.text = "pathway\tKEGG Pathway Database\nrelease\t109.0"
        mock_get.return_value = mock_response
        
        client = KEGGClient()
        result = client.info("pathway")
        
        mock_get.assert_called_once_with("/info/pathway")
        assert result["pathway"] == "KEGG Pathway Database"
        assert result["release"] == "109.0"
    
    @patch.object(KEGGClient, 'get')
    def test_get_pathway(self, mock_get, mock_kegg_pathway_text):
        """Test getting pathway information."""
        mock_response = Mock()
        mock_response.text = mock_kegg_pathway_text
        mock_get.return_value = mock_response
        
        client = KEGGClient()
        result = client.get_pathway("hsa04110")
        
        mock_get.assert_called_once_with("/get/hsa04110")
        assert result["ENTRY"] == "hsa04110                    Pathway"
        assert result["NAME"] == "Cell cycle - Homo sapiens (human)"
        assert "Cell cycle progression" in result["DESCRIPTION"]
    
    @patch.object(KEGGClient, 'get')
    def test_get_gene(self, mock_get, mock_kegg_gene_text):
        """Test getting gene information."""
        mock_response = Mock()
        mock_response.text = mock_kegg_gene_text
        mock_get.return_value = mock_response
        
        client = KEGGClient()
        result = client.get_gene("hsa:7157")
        
        mock_get.assert_called_once_with("/get/hsa:7157")
        assert result["ENTRY"] == "hsa:7157                    CDS       T01001"
        assert result["NAME"] == "TP53, BCC7, BMFS5, LFS1, P53, TRP53"
        assert result["DEFINITION"] == "tumor protein p53"
    
    @patch.object(KEGGClient, 'get')
    def test_find(self, mock_get):
        """Test find functionality."""
        mock_response = Mock()
        mock_response.text = "hsa:7157\tTP53; tumor protein p53\nhsa:7158\tTP53BP1; tumor protein p53 binding protein 1"
        mock_get.return_value = mock_response
        
        client = KEGGClient()
        result = client.find("genes", "p53")
        
        mock_get.assert_called_once_with("/find/genes/p53")
        assert len(result) == 2
        assert result[0]["entry_id"] == "hsa:7157"
        assert "TP53" in result[0]["description"]
    
    @patch.object(KEGGClient, 'get')
    def test_link(self, mock_get, mock_kegg_link_response):
        """Test link functionality."""
        mock_response = Mock()
        mock_response.text = mock_kegg_link_response
        mock_get.return_value = mock_response
        
        client = KEGGClient()
        result = client.link("genes", "pathway", "hsa04110")
        
        mock_get.assert_called_once_with("/link/genes/hsa04110")
        assert len(result) == 3
        assert result[0]["source"] == "hsa04110"
        assert result[0]["target"] == "hsa:595"
    
    @patch.object(KEGGClient, 'get')
    def test_conv(self, mock_get):
        """Test ID conversion."""
        mock_response = Mock()
        mock_response.text = "up:P04637\thsa:7157"
        mock_get.return_value = mock_response
        
        client = KEGGClient()
        result = client.conv("hsa", "uniprot", ["up:P04637"])
        
        mock_get.assert_called_once_with("/conv/hsa/up:P04637")
        assert result["up:P04637"] == "hsa:7157"
    
    def test_parse_entry(self):
        """Test entry parsing."""
        client = KEGGClient()
        text = """ENTRY       test
NAME        Test entry
DESCRIPTION Multi-line
            description
///"""
        
        result = client._parse_entry(text)
        assert result["ENTRY"] == "test"
        assert result["NAME"] == "Test entry"
        assert result["DESCRIPTION"] == "Multi-line\ndescription"


class TestKEGGCollector:
    """Test KEGG collector."""
    
    @patch('src.collectors.kegg_collector.KEGGClient')
    def test_initialization(self, mock_client_class):
        """Test collector initialization."""
        collector = KEGGCollector()
        mock_client_class.assert_called_once()
    
    def test_collect_single(self, mock_kegg_pathway_text, mock_kegg_link_response):
        """Test collecting single pathway."""
        collector = KEGGCollector()
        
        with patch.object(collector.api_client, 'get_pathway') as mock_pathway:
            with patch.object(collector.api_client, 'get_genes_by_pathway') as mock_genes:
                with patch.object(collector.api_client, 'link') as mock_link:
                    mock_pathway.return_value = {
                        "NAME": "Cell cycle - Homo sapiens (human)",
                        "DESCRIPTION": "Cell cycle progression",
                        "ORGANISM": "Homo sapiens (human) [GN:hsa]",
                        "CLASS": "Cellular Processes; Cell growth and death"
                    }
                    mock_genes.return_value = [
                        {"source": "hsa04110", "target": "hsa:595"},
                        {"source": "hsa04110", "target": "hsa:894"}
                    ]
                    mock_link.return_value = []
                    
                    result = collector.collect_single("hsa04110")
                    
                    assert result["pathway_id"] == "hsa04110"
                    assert result["name"] == "Cell cycle - Homo sapiens (human)"
                    assert result["gene_count"] == 2
                    assert len(result["genes"]) == 2
    
    def test_collect_gene(self, mock_kegg_gene_text):
        """Test collecting gene data."""
        collector = KEGGCollector()
        
        with patch.object(collector.api_client, 'get_gene') as mock_gene:
            with patch.object(collector.api_client, 'get_pathways_by_gene') as mock_pathways:
                with patch.object(collector.api_client, 'conv') as mock_conv:
                    mock_gene.return_value = {
                        "NAME": "TP53, BCC7, BMFS5, LFS1, P53, TRP53",
                        "DEFINITION": "tumor protein p53",
                        "ORGANISM": "hsa  Homo sapiens (human)"
                    }
                    mock_pathways.return_value = [
                        {"source": "hsa:7157", "target": "hsa04110"},
                        {"source": "hsa:7157", "target": "hsa04115"}
                    ]
                    mock_conv.return_value = {"hsa:7157": "up:P04637"}
                    
                    result = collector.collect_gene("hsa:7157")
                    
                    assert result["kegg_id"] == "hsa:7157"
                    assert "TP53" in result["name"]
                    assert result["pathway_count"] == 2
                    assert "up:P04637" in result["uniprot_ids"]
    
    def test_save_to_database(self, test_db):
        """Test saving pathway to database."""
        collector = KEGGCollector(db_session=test_db)
        
        data = {
            "pathway_id": "hsa04110",
            "name": "Cell cycle",
            "description": "Cell cycle progression",
            "organism": "Homo sapiens",
            "category": "Cellular Processes",
            "genes": [{"target": "hsa:595"}, {"target": "hsa:894"}],
            "gene_count": 2
        }
        
        pathway = collector.save_to_database(data)
        
        assert isinstance(pathway, Pathway)
        assert pathway.pathway_id == "hsa04110"
        assert pathway.name == "Cell cycle"
        assert pathway.database == "KEGG"
        assert pathway.organism == "Homo sapiens"
    
    def test_save_gene_to_database(self, test_db):
        """Test saving gene to database."""
        collector = KEGGCollector(db_session=test_db)
        
        data = {
            "kegg_id": "hsa:7157",
            "name": "TP53",
            "definition": "tumor protein p53",
            "organism": "Homo sapiens",
            "pathways": [],
            "uniprot_ids": ["up:P04637"]
        }
        
        gene = collector.save_gene_to_database(data)
        
        assert isinstance(gene, Gene)
        assert gene.kegg_id == "hsa:7157"
        assert gene.symbol == "TP53"
        assert gene.description == "tumor protein p53"
        assert gene.organism == "Homo sapiens"
    
    @patch.object(KEGGCollector, 'collect_single')
    @patch.object(KEGGCollector, 'save_to_database')
    def test_search_pathways(self, mock_save, mock_collect):
        """Test searching pathways."""
        collector = KEGGCollector()
        
        with patch.object(collector.api_client, 'find') as mock_find:
            mock_find.return_value = [
                {"entry_id": "hsa04110", "description": "Cell cycle"},
                {"entry_id": "hsa04111", "description": "Cell cycle - yeast"}
            ]
            
            mock_collect.side_effect = [
                {"pathway_id": "hsa04110", "name": "Cell cycle"},
                Exception("Skip non-human")  # Should skip hsa04111
            ]
            
            mock_save.return_value = Mock()
            
            results = collector.search_pathways("cell cycle", "hsa")
            
            assert len(results) == 1  # Only human pathway
            assert mock_collect.call_count == 2
            assert mock_save.call_count == 1