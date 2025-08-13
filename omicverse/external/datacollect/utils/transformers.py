"""Data transformation and normalization utilities."""

import re
import logging
from typing import Dict, List, Optional, Tuple, Union
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import pandas as pd

from ..models.protein import Protein
from ..models.structure import Structure
from ..models.genomic import Gene, Variant


logger = logging.getLogger(__name__)


class SequenceTransformer:
    """Transform and manipulate biological sequences."""
    
    @staticmethod
    def translate_dna_to_protein(dna_sequence: str, table: int = 1) -> str:
        """Translate DNA sequence to protein.
        
        Args:
            dna_sequence: DNA sequence
            table: Genetic code table (default: standard)
        
        Returns:
            Protein sequence
        """
        seq = Seq(dna_sequence.upper())
        return str(seq.translate(table=table))
    
    @staticmethod
    def reverse_complement(sequence: str) -> str:
        """Get reverse complement of DNA sequence."""
        seq = Seq(sequence.upper())
        return str(seq.reverse_complement())
    
    @staticmethod
    def transcribe(dna_sequence: str) -> str:
        """Transcribe DNA to RNA."""
        return dna_sequence.upper().replace('T', 'U')
    
    @staticmethod
    def get_orf(sequence: str, min_length: int = 100) -> List[Tuple[int, int, str]]:
        """Find open reading frames in DNA sequence.
        
        Args:
            sequence: DNA sequence
            min_length: Minimum ORF length in nucleotides
        
        Returns:
            List of (start, end, protein_sequence) tuples
        """
        orfs = []
        seq = Seq(sequence.upper())
        
        # Check all three reading frames
        for frame in range(3):
            # Forward strand
            for i in range(frame, len(seq) - 2, 3):
                if seq[i:i+3] == "ATG":  # Start codon
                    for j in range(i + 3, len(seq) - 2, 3):
                        codon = seq[j:j+3]
                        if codon in ["TAA", "TAG", "TGA"]:  # Stop codons
                            if j - i >= min_length:
                                protein = str(seq[i:j].translate())
                                orfs.append((i, j, protein))
                            break
        
        return orfs
    
    @staticmethod
    def calculate_gc_content(sequence: str) -> float:
        """Calculate GC content of nucleotide sequence."""
        sequence = sequence.upper()
        gc_count = sequence.count('G') + sequence.count('C')
        return (gc_count / len(sequence)) * 100 if sequence else 0


class IdentifierMapper:
    """Map between different database identifiers."""
    
    # Common ID mappings
    ID_PATTERNS = {
        'uniprot_to_gene': {
            'P04637': 'TP53',
            'P00533': 'EGFR',
            'P01308': 'INS',
            # Add more mappings as needed
        }
    }
    
    @classmethod
    def normalize_chromosome(cls, chrom: str) -> str:
        """Normalize chromosome names.
        
        Args:
            chrom: Chromosome identifier
        
        Returns:
            Normalized chromosome name
        """
        chrom = str(chrom).upper().strip()
        
        # Remove common prefixes
        chrom = chrom.replace('CHR', '').replace('CHROMOSOME', '')
        
        # Handle special cases
        if chrom == 'M':
            chrom = 'MT'
        elif chrom == '23':
            chrom = 'X'
        elif chrom == '24':
            chrom = 'Y'
        
        return chrom
    
    @classmethod
    def extract_gene_id_from_ensembl(cls, ensembl_id: str) -> Optional[str]:
        """Extract gene ID from Ensembl identifier."""
        match = re.match(r'(ENS[A-Z]{0,3}G[0-9]{11})', ensembl_id)
        return match.group(1) if match else None
    
    @classmethod
    def uniprot_to_pdb_chains(cls, uniprot_acc: str, structures: List[Structure]) -> Dict[str, List[str]]:
        """Map UniProt accession to PDB chains.
        
        Args:
            uniprot_acc: UniProt accession
            structures: List of Structure objects
        
        Returns:
            Dictionary mapping PDB IDs to chain IDs
        """
        pdb_chains = {}
        
        for structure in structures:
            for chain in structure.chains:
                if chain.uniprot_accession == uniprot_acc:
                    if structure.structure_id not in pdb_chains:
                        pdb_chains[structure.structure_id] = []
                    pdb_chains[structure.structure_id].append(chain.chain_id)
        
        return pdb_chains


class DataExporter:
    """Export data to various formats."""
    
    @staticmethod
    def proteins_to_fasta(proteins: List[Protein], output_file: str):
        """Export proteins to FASTA format.
        
        Args:
            proteins: List of Protein objects
            output_file: Output file path
        """
        records = []
        
        for protein in proteins:
            description = f"{protein.protein_name} | {protein.organism}"
            record = SeqRecord(
                Seq(protein.sequence),
                id=protein.accession,
                description=description
            )
            records.append(record)
        
        with open(output_file, 'w') as f:
            SeqIO.write(records, f, 'fasta')
        
        logger.info(f"Exported {len(records)} proteins to {output_file}")
    
    @staticmethod
    def variants_to_vcf(variants: List[Variant], output_file: str, reference: str = "GRCh38"):
        """Export variants to VCF format.
        
        Args:
            variants: List of Variant objects
            output_file: Output file path
            reference: Reference genome version
        """
        # VCF header
        header_lines = [
            "##fileformat=VCFv4.3",
            f"##reference={reference}",
            "##INFO=<ID=RS,Number=1,Type=Integer,Description=\"dbSNP ID\">",
            "##INFO=<ID=GENE,Number=1,Type=String,Description=\"Gene symbol\">",
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO"
        ]
        
        with open(output_file, 'w') as f:
            # Write header
            for line in header_lines:
                f.write(line + '\n')
            
            # Write variants
            for variant in variants:
                chrom = IdentifierMapper.normalize_chromosome(variant.chromosome)
                info_fields = []
                
                if variant.rsid:
                    info_fields.append(f"RS={variant.rsid.replace('rs', '')}")
                if variant.gene_symbol:
                    info_fields.append(f"GENE={variant.gene_symbol}")
                
                info = ";".join(info_fields) if info_fields else "."
                
                line = f"{chrom}\t{variant.position}\t{variant.rsid or '.'}\t"
                line += f"{variant.reference_allele}\t{variant.alternate_allele}\t.\t.\t{info}"
                
                f.write(line + '\n')
        
        logger.info(f"Exported {len(variants)} variants to {output_file}")
    
    @staticmethod
    def structures_to_csv(structures: List[Structure], output_file: str):
        """Export structures to CSV format.
        
        Args:
            structures: List of Structure objects
            output_file: Output file path
        """
        data = []
        
        for structure in structures:
            data.append({
                'pdb_id': structure.structure_id,
                'title': structure.title,
                'method': structure.structure_type,
                'resolution': structure.resolution,
                'organism': structure.organism,
                'deposition_date': structure.deposition_date,
                'chain_count': len(structure.chains),
                'ligand_count': len(structure.ligands),
                'uniprot_accessions': ','.join(set(
                    chain.uniprot_accession for chain in structure.chains 
                    if chain.uniprot_accession
                ))
            })
        
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        
        logger.info(f"Exported {len(structures)} structures to {output_file}")


class DataNormalizer:
    """Normalize and clean biological data."""
    
    @staticmethod
    def normalize_organism_name(organism: str) -> str:
        """Normalize organism names to standard format.
        
        Args:
            organism: Original organism name
        
        Returns:
            Normalized organism name
        """
        if not organism:
            return ""
        
        # Common normalizations
        replacements = {
            "H. sapiens": "Homo sapiens",
            "Human": "Homo sapiens",
            "M. musculus": "Mus musculus",
            "Mouse": "Mus musculus",
            "E. coli": "Escherichia coli",
            "S. cerevisiae": "Saccharomyces cerevisiae",
            "Yeast": "Saccharomyces cerevisiae",
        }
        
        organism = organism.strip()
        return replacements.get(organism, organism)
    
    @staticmethod
    def normalize_gene_name(gene_name: str) -> str:
        """Normalize gene names.
        
        Args:
            gene_name: Original gene name
        
        Returns:
            Normalized gene name
        """
        if not gene_name:
            return ""
        
        # Remove common suffixes
        gene_name = re.sub(r'[-_](HUMAN|MOUSE|RAT|YEAST)$', '', gene_name.upper())
        
        # Remove version numbers
        gene_name = re.sub(r'\.\d+$', '', gene_name)
        
        return gene_name
    
    @staticmethod
    def clean_sequence(sequence: str, sequence_type: str = "protein") -> str:
        """Clean biological sequence.
        
        Args:
            sequence: Raw sequence
            sequence_type: Type of sequence (protein, dna, rna)
        
        Returns:
            Cleaned sequence
        """
        # Remove whitespace and numbers
        sequence = re.sub(r'[\s\d]', '', sequence.upper())
        
        if sequence_type == "protein":
            # Remove invalid amino acids (standard 20 amino acids)
            valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
            sequence = ''.join(aa for aa in sequence if aa in valid_aa)
        elif sequence_type == "dna":
            valid_bases = set("ACGT")
            sequence = ''.join(base for base in sequence if base in valid_bases)
        elif sequence_type == "rna":
            valid_bases = set("ACGU")
            sequence = ''.join(base for base in sequence if base in valid_bases)
        
        return sequence
    
    @staticmethod
    def standardize_variant_notation(ref: str, alt: str, position: int) -> str:
        """Standardize variant notation to HGVS format.
        
        Args:
            ref: Reference allele
            alt: Alternate allele
            position: Genomic position
        
        Returns:
            HGVS notation
        """
        if len(ref) == 1 and len(alt) == 1:
            # SNV
            return f"g.{position}{ref}>{alt}"
        elif len(ref) > len(alt):
            # Deletion
            if alt == "-" or not alt:
                return f"g.{position}_{position + len(ref) - 1}del"
            else:
                return f"g.{position}_{position + len(ref) - 1}delins{alt}"
        elif len(ref) < len(alt):
            # Insertion
            return f"g.{position}_{position + 1}ins{alt[len(ref):]}"
        else:
            # Complex
            return f"g.{position}_{position + len(ref) - 1}delins{alt}"


class FeatureExtractor:
    """Extract features from biological data."""
    
    @staticmethod
    def extract_sequence_features(sequence: str) -> Dict[str, float]:
        """Extract features from protein sequence.
        
        Args:
            sequence: Protein sequence
        
        Returns:
            Dictionary of features
        """
        if not sequence:
            return {}
        
        # Amino acid composition
        aa_counts = {}
        for aa in "ACDEFGHIKLMNPQRSTVWY":
            aa_counts[f"aa_{aa}"] = sequence.count(aa) / len(sequence)
        
        # Physical properties
        hydrophobic = set("AILMFWYV")
        charged = set("DEKR")
        polar = set("STNQ")
        
        features = {
            **aa_counts,
            "length": len(sequence),
            "hydrophobic_ratio": sum(1 for aa in sequence if aa in hydrophobic) / len(sequence),
            "charged_ratio": sum(1 for aa in sequence if aa in charged) / len(sequence),
            "polar_ratio": sum(1 for aa in sequence if aa in polar) / len(sequence),
            "molecular_weight": sum(
                {"A": 89.1, "C": 121.2, "D": 133.1, "E": 147.1, "F": 165.2,
                 "G": 75.1, "H": 155.2, "I": 131.2, "K": 146.2, "L": 131.2,
                 "M": 149.2, "N": 132.1, "P": 115.1, "Q": 146.2, "R": 174.2,
                 "S": 105.1, "T": 119.1, "V": 117.1, "W": 204.2, "Y": 181.2
                }.get(aa, 0) for aa in sequence
            ),
        }
        
        return features