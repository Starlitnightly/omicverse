"""Tests for BLAST API client."""

import pytest
from unittest.mock import patch, MagicMock, Mock
import xml.etree.ElementTree as ET

from omicverse.external.datacollect.api.blast import BLASTClient


class TestBLASTClient:
    def setup_method(self):
        self.client = BLASTClient()
        # Mock the session
        self.client.session = MagicMock()
    
    def test_blast_sequence(self):
        """Test BLAST sequence submission."""
        mock_response = MagicMock()
        mock_response.text = """
        RID = TEST123
        RTOE = 30
        """
        mock_response.raise_for_status = MagicMock()
        self.client.session.post.return_value = mock_response
        
        rid = self.client.blast_sequence("ATCGATCGATCG", program="blastn", database="nt")
        
        assert rid == "TEST123"
        self.client.session.post.assert_called_once()
    
    def test_check_status(self):
        """Test checking BLAST search status."""
        mock_response = MagicMock()
        mock_response.text = "Status=READY"
        mock_response.raise_for_status = MagicMock()
        self.client.session.get.return_value = mock_response
        
        status = self.client.check_status("TEST123")
        
        assert status == "READY"
    
    def test_get_results(self):
        """Test getting BLAST results."""
        xml_content = """
        <BlastOutput>
            <BlastOutput_program>blastn</BlastOutput_program>
            <BlastOutput_version>BLASTN 2.12.0+</BlastOutput_version>
            <BlastOutput_db>nt</BlastOutput_db>
            <BlastOutput_query-ID>Query_1</BlastOutput_query-ID>
            <BlastOutput_query-def>Test sequence</BlastOutput_query-def>
            <BlastOutput_query-len>100</BlastOutput_query-len>
            <BlastOutput_iterations>
                <Iteration>
                    <Iteration_hits>
                        <Hit>
                            <Hit_num>1</Hit_num>
                            <Hit_id>gi|123456</Hit_id>
                            <Hit_def>Test hit</Hit_def>
                            <Hit_accession>NM_001234</Hit_accession>
                            <Hit_len>500</Hit_len>
                            <Hit_hsps>
                                <Hsp>
                                    <Hsp_num>1</Hsp_num>
                                    <Hsp_bit-score>100.5</Hsp_bit-score>
                                    <Hsp_score>50</Hsp_score>
                                    <Hsp_evalue>1e-10</Hsp_evalue>
                                    <Hsp_query-from>1</Hsp_query-from>
                                    <Hsp_query-to>100</Hsp_query-to>
                                    <Hsp_hit-from>1</Hsp_hit-from>
                                    <Hsp_hit-to>100</Hsp_hit-to>
                                    <Hsp_identity>95</Hsp_identity>
                                    <Hsp_positive>98</Hsp_positive>
                                    <Hsp_gaps>0</Hsp_gaps>
                                    <Hsp_align-len>100</Hsp_align-len>
                                    <Hsp_qseq>ATCG</Hsp_qseq>
                                    <Hsp_hseq>ATCG</Hsp_hseq>
                                    <Hsp_midline>||||</Hsp_midline>
                                </Hsp>
                            </Hit_hsps>
                        </Hit>
                    </Iteration_hits>
                </Iteration>
            </BlastOutput_iterations>
        </BlastOutput>
        """
        
        mock_response = MagicMock()
        mock_response.text = xml_content
        mock_response.raise_for_status = MagicMock()
        self.client.session.get.return_value = mock_response
        
        results = self.client.get_results("TEST123")
        
        assert results["program"] == "blastn"
        assert results["database"] == "nt"
        assert len(results["hits"]) == 1
        assert results["hits"][0]["accession"] == "NM_001234"
        assert results["hits"][0]["hsps"][0]["percent_identity"] == 95.0
    
    @patch('src.api.blast.time.sleep')
    def test_blast_and_wait(self, mock_sleep):
        """Test BLAST search with waiting for results."""
        # Mock blast_sequence
        self.client.blast_sequence = Mock(return_value="TEST123")
        # Mock check_status
        self.client.check_status = Mock(side_effect=["WAITING", "WAITING", "READY"])
        # Mock get_results
        self.client.get_results = Mock(return_value={"hits": []})
        
        results = self.client.blast_and_wait("ATCGATCG")
        
        assert self.client.blast_sequence.called
        assert self.client.check_status.call_count == 3
        assert self.client.get_results.called
        assert results == {"hits": []}
    
    def test_search_by_accession(self):
        """Test searching by accession number."""
        mock_response = MagicMock()
        mock_response.text = ">NM_001234 Test sequence\nATCGATCGATCG"
        mock_response.raise_for_status = MagicMock()
        self.client.session.get.return_value = mock_response
        
        result = self.client.search_by_accession("NM_001234")
        
        assert result["accession"] == "NM_001234"
        assert ">NM_001234" in result["sequence"]