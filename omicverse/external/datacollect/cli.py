"""Command-line interface for bioinformatics data collection."""

import click
import sys
from typing import Optional, List

from src.utils.logging import setup_logging, console, get_logger
from src.utils.database import initialize_database, check_database_connection, get_table_stats
from src.collectors.uniprot_collector import UniProtCollector
from src.collectors.pdb_collector import PDBCollector
from src.models.base import get_db


logger = get_logger(__name__)


@click.group()
@click.option("--log-level", default="INFO", help="Logging level")
@click.option("--log-file", help="Log file path")
def cli(log_level: str, log_file: Optional[str]):
    """Bioinformatics Data Collection Tool."""
    setup_logging(log_level, log_file)
    
    # Check database connection
    if not check_database_connection():
        console.print("[red]Database connection failed. Run 'biocollect init' first.[/red]")
        sys.exit(1)


@cli.command()
def init():
    """Initialize database and create tables."""
    console.print("[blue]Initializing database...[/blue]")
    try:
        initialize_database()
        console.print("[green]Database initialized successfully![/green]")
    except Exception as e:
        console.print(f"[red]Initialization failed: {e}[/red]")
        sys.exit(1)


@cli.command()
def status():
    """Show database status and statistics."""
    console.print("[blue]Database Status[/blue]")
    
    db = next(get_db())
    stats = get_table_stats(db)
    
    console.print("\n[bold]Table Statistics:[/bold]")
    for table, count in stats.items():
        if count >= 0:
            console.print(f"  {table}: {count} records")
        else:
            console.print(f"  {table}: [red]error[/red]")
    
    db.close()


@cli.group()
def collect():
    """Data collection commands."""
    pass


@collect.command()
@click.argument("accession")
@click.option("--features/--no-features", default=True, help="Include protein features")
@click.option("--save-file", is_flag=True, help="Save to JSON file")
def uniprot(accession: str, features: bool, save_file: bool):
    """Collect data for a UniProt accession."""
    console.print(f"[blue]Collecting UniProt data for {accession}...[/blue]")
    
    collector = UniProtCollector()
    
    try:
        protein = collector.process_and_save(
            accession,
            include_features=features,
            save_to_file=save_file
        )
        
        if protein:
            console.print(f"[green]Successfully collected: {protein.protein_name}[/green]")
            console.print(f"  Organism: {protein.organism}")
            console.print(f"  Length: {protein.sequence_length} aa")
            if protein.pdb_ids:
                console.print(f"  PDB IDs: {protein.pdb_ids}")
    except Exception as e:
        console.print(f"[red]Collection failed: {e}[/red]")
        logger.error(f"UniProt collection failed for {accession}", exc_info=True)
    finally:
        collector.close()


@collect.command()
@click.argument("query")
@click.option("--limit", default=10, help="Maximum number of results")
@click.option("--organism", help="Filter by organism name")
def uniprot_search(query: str, limit: int, organism: Optional[str]):
    """Search UniProt and collect results."""
    # Build query
    full_query = query
    if organism:
        full_query += f" AND organism_name:{organism}"
    
    console.print(f"[blue]Searching UniProt: {full_query}[/blue]")
    
    collector = UniProtCollector()
    
    try:
        proteins = collector.search_and_collect(
            full_query,
            max_results=limit
        )
        
        console.print(f"\n[green]Collected {len(proteins)} proteins:[/green]")
        for protein in proteins:
            console.print(f"  • {protein.accession}: {protein.protein_name} ({protein.organism})")
    except Exception as e:
        console.print(f"[red]Search failed: {e}[/red]")
        logger.error(f"UniProt search failed", exc_info=True)
    finally:
        collector.close()


@collect.command()
@click.argument("pdb_id")
@click.option("--download", is_flag=True, help="Download structure file")
@click.option("--save-file", is_flag=True, help="Save to JSON file")
def pdb(pdb_id: str, download: bool, save_file: bool):
    """Collect data for a PDB structure."""
    console.print(f"[blue]Collecting PDB data for {pdb_id}...[/blue]")
    
    collector = PDBCollector()
    
    try:
        structure = collector.process_and_save(
            pdb_id,
            download_structure=download,
            save_to_file=save_file
        )
        
        if structure:
            console.print(f"[green]Successfully collected: {structure.title}[/green]")
            console.print(f"  Method: {structure.structure_type}")
            if structure.resolution:
                console.print(f"  Resolution: {structure.resolution} Å")
            console.print(f"  Chains: {len(structure.chains)}")
            console.print(f"  Ligands: {len(structure.ligands)}")
    except Exception as e:
        console.print(f"[red]Collection failed: {e}[/red]")
        logger.error(f"PDB collection failed for {pdb_id}", exc_info=True)
    finally:
        collector.close()


@collect.command()
@click.argument("sequence")
@click.option("--e-value", default=0.1, help="E-value cutoff")
@click.option("--limit", default=10, help="Maximum number of results")
def pdb_blast(sequence: str, e_value: float, limit: int):
    """Search PDB by sequence similarity."""
    console.print(f"[blue]Searching PDB by sequence (E-value: {e_value})...[/blue]")
    
    collector = PDBCollector()
    
    try:
        structures = collector.search_by_sequence(
            sequence,
            e_value=e_value,
            max_results=limit
        )
        
        console.print(f"\n[green]Found {len(structures)} similar structures:[/green]")
        for structure in structures:
            console.print(f"  • {structure.structure_id}: {structure.title}")
            if structure.resolution:
                console.print(f"    Resolution: {structure.resolution} Å")
    except Exception as e:
        console.print(f"[red]Search failed: {e}[/red]")
        logger.error(f"PDB sequence search failed", exc_info=True)


@collect.command()
@click.argument("uniprot_accession")
@click.option("--download", is_flag=True, help="Download structure file")
@click.option("--download-pae", is_flag=True, help="Download PAE data")
@click.option("--save-file", is_flag=True, help="Save metadata to JSON file")
def alphafold(uniprot_accession: str, download: bool, download_pae: bool, save_file: bool):
    """Collect AlphaFold structure prediction for a UniProt accession."""
    from src.collectors.alphafold_collector import AlphaFoldCollector
    
    console.print(f"[blue]Collecting AlphaFold prediction for {uniprot_accession}...[/blue]")
    
    collector = AlphaFoldCollector()
    
    try:
        structure = collector.process_and_save(
            uniprot_accession,
            download_structure=download,
            download_pae=download_pae,
            save_to_file=save_file
        )
        
        if structure:
            console.print(f"[green]Successfully collected: {structure.title}[/green]")
            console.print(f"  AlphaFold ID: {structure.structure_id}")
            console.print(f"  Organism: {structure.organism}")
            if structure.r_factor:  # Contains mean pLDDT
                console.print(f"  Mean pLDDT: {structure.r_factor:.1f}")
            if structure.chains:
                console.print(f"  Sequence length: {structure.chains[0].length} aa")
            if download:
                console.print(f"  Structure file: {structure.structure_file_path}")
    except Exception as e:
        console.print(f"[red]Collection failed: {e}[/red]")
        logger.error(f"AlphaFold collection failed for {uniprot_accession}", exc_info=True)


@collect.command()
@click.argument("uniprot_accession")
@click.option("--save-file", is_flag=True, help="Save to JSON file")
def interpro(uniprot_accession: str, save_file: bool):
    """Collect InterPro domain annotations for a protein."""
    from src.collectors.interpro_collector import InterProCollector
    
    console.print(f"[blue]Collecting InterPro data for {uniprot_accession}...[/blue]")
    
    collector = InterProCollector()
    
    try:
        data = collector.collect_single(uniprot_accession)
        domains = collector.save_to_database(data)
        
        console.print(f"[green]Found {len(domains)} InterPro entries[/green]")
        console.print(f"  Domains: {data['domain_count']}")
        console.print(f"  Families: {data['family_count']}")
        
        for domain in domains[:5]:  # Show first 5
            console.print(f"  • {domain.interpro_id}: {domain.name} ({domain.type})")
        
        if len(domains) > 5:
            console.print(f"  ... and {len(domains) - 5} more")
            
        if save_file:
            collector.save_to_file(data, f"interpro_{uniprot_accession}.json")
    except Exception as e:
        console.print(f"[red]Collection failed: {e}[/red]")
        logger.error(f"InterPro collection failed for {uniprot_accession}", exc_info=True)


@collect.command()
@click.argument("pathway_id")
@click.option("--save-file", is_flag=True, help="Save to JSON file")
def kegg(pathway_id: str, save_file: bool):
    """Collect KEGG pathway data."""
    from src.collectors.kegg_collector import KEGGCollector
    
    console.print(f"[blue]Collecting KEGG pathway {pathway_id}...[/blue]")
    
    collector = KEGGCollector()
    
    try:
        data = collector.collect_single(pathway_id)
        pathway = collector.save_to_database(data)
        
        console.print(f"[green]Successfully collected: {pathway.name}[/green]")
        console.print(f"  Organism: {pathway.organism}")
        console.print(f"  Category: {pathway.category}")
        console.print(f"  Genes: {data['gene_count']}")
        
        if save_file:
            collector.save_to_file(data, f"kegg_{pathway_id}.json")
    except Exception as e:
        console.print(f"[red]Collection failed: {e}[/red]")
        logger.error(f"KEGG collection failed for {pathway_id}", exc_info=True)


@collect.command()
@click.argument("identifier")
@click.option("--species", type=int, default=9606, help="NCBI taxonomy ID (default: 9606 for human)")
@click.option("--partners/--no-partners", default=True, help="Collect interaction partners")
@click.option("--partner-limit", default=20, help="Maximum number of partners to collect")
@click.option("--save-file", is_flag=True, help="Save to JSON file")
def string(identifier: str, species: int, partners: bool, partner_limit: int, save_file: bool):
    """Collect STRING protein interaction data."""
    from src.collectors.string_collector import STRINGCollector
    
    console.print(f"[blue]Collecting STRING data for {identifier} (species: {species})...[/blue]")
    
    collector = STRINGCollector()
    
    try:
        data = collector.collect_single(
            identifier, species, partners, partner_limit
        )
        interactions = collector.save_to_database(data)
        
        console.print(f"[green]Successfully collected STRING data[/green]")
        console.print(f"  STRING ID: {data['string_id']}")
        console.print(f"  Preferred name: {data['preferred_name']}")
        console.print(f"  Interactions: {data.get('interaction_count', 0)}")
        
        if data.get("enrichment"):
            pval = data["enrichment"].get("p_value", 1)
            console.print(f"  PPI enrichment p-value: {pval:.2e}")
        
        if save_file:
            collector.save_to_file(data, f"string_{identifier}.json")
    except Exception as e:
        console.print(f"[red]Collection failed: {e}[/red]")
        logger.error(f"STRING collection failed for {identifier}", exc_info=True)


@collect.command()
@click.argument("identifier")
@click.option("--id-type", type=click.Choice(["gene", "symbol"]), default="gene", 
              help="Type of identifier (gene ID or symbol)")
@click.option("--species", default="human", help="Species name (default: human)")
@click.option("--expand/--no-expand", default=True, help="Include transcripts and proteins")
@click.option("--save-file", is_flag=True, help="Save to JSON file")
def ensembl(identifier: str, id_type: str, species: str, expand: bool, save_file: bool):
    """Collect Ensembl genomic data."""
    from src.collectors.ensembl_collector import EnsemblCollector
    
    console.print(f"[blue]Collecting Ensembl data for {identifier} ({species})...[/blue]")
    
    collector = EnsemblCollector()
    
    try:
        if id_type == "symbol":
            data = collector.collect_gene_by_symbol(identifier, species)
        else:
            data = collector.collect_gene(identifier, expand)
        
        gene = collector.save_gene_to_database(data)
        
        console.print(f"[green]Successfully collected: {gene.symbol}[/green]")
        console.print(f"  Ensembl ID: {gene.ensembl_id}")
        console.print(f"  Description: {gene.description[:100]}..." if gene.description else "")
        console.print(f"  Location: {gene.chromosome}:{gene.start_position}-{gene.end_position}")
        console.print(f"  Transcripts: {gene.transcript_count}")
        console.print(f"  Proteins: {gene.protein_count}")
        
        if save_file:
            collector.save_to_file(data, f"ensembl_{identifier}.json")
    except Exception as e:
        console.print(f"[red]Collection failed: {e}[/red]")
        logger.error(f"Ensembl collection failed for {identifier}", exc_info=True)


@collect.command()
@click.argument("identifier")
@click.option("--id-type", type=click.Choice(["clinvar", "gene", "disease"]), default="clinvar",
              help="Type of identifier")
@click.option("--pathogenic-only", is_flag=True, help="Only collect pathogenic variants (for gene)")
@click.option("--limit", default=50, help="Maximum number of variants to collect")
@click.option("--save-file", is_flag=True, help="Save to JSON file")
def clinvar(identifier: str, id_type: str, pathogenic_only: bool, limit: int, save_file: bool):
    """Collect ClinVar clinical variant data."""
    from src.collectors.clinvar_collector import ClinVarCollector
    
    console.print(f"[blue]Collecting ClinVar data for {identifier} (type: {id_type})...[/blue]")
    
    collector = ClinVarCollector()
    
    try:
        if id_type == "clinvar":
            # Single variant by ClinVar ID
            data = collector.collect_single(identifier)
            variant = collector.save_to_database(data)
            
            console.print(f"[green]Successfully collected variant[/green]")
            console.print(f"  Gene: {variant.gene_symbol}")
            console.print(f"  Significance: {variant.clinical_significance}")
            console.print(f"  Type: {variant.variant_type}")
            if variant.disease_associations:
                console.print(f"  Conditions: {variant.disease_associations}")
                
            if save_file:
                collector.save_to_file(data, f"clinvar_{identifier}.json")
                
        elif id_type == "gene":
            # Multiple variants by gene
            variants_data = collector.collect_by_gene(identifier, pathogenic_only, limit)
            variants = collector.save_variants_to_database(variants_data)
            
            console.print(f"[green]Collected {len(variants)} variants for {identifier}[/green]")
            
            # Show summary
            sig_counts = {}
            for v in variants:
                sig = v.clinical_significance or "Unknown"
                sig_counts[sig] = sig_counts.get(sig, 0) + 1
            
            for sig, count in sig_counts.items():
                console.print(f"  {sig}: {count}")
                
            if save_file:
                collector.save_to_file(variants_data, f"clinvar_gene_{identifier}.json")
                
        elif id_type == "disease":
            # Variants by disease
            variants_data = collector.collect_by_disease(identifier, limit)
            variants = collector.save_variants_to_database(variants_data)
            
            console.print(f"[green]Collected {len(variants)} variants for disease '{identifier}'[/green]")
            
            # Show gene summary
            gene_counts = {}
            for v in variants:
                gene = v.gene_symbol or "Unknown"
                gene_counts[gene] = gene_counts.get(gene, 0) + 1
            
            console.print("  Top genes:")
            for gene, count in sorted(gene_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                console.print(f"    {gene}: {count}")
                
            if save_file:
                collector.save_to_file(variants_data, f"clinvar_disease_{identifier}.json")
                
    except Exception as e:
        console.print(f"[red]Collection failed: {e}[/red]")
        logger.error(f"ClinVar collection failed for {identifier}", exc_info=True)


@collect.command()
@click.argument("accession")
@click.option("--save-file", is_flag=True, help="Save to JSON file")
def geo(accession: str, save_file: bool):
    """Collect GEO gene expression data."""
    from src.collectors.geo_collector import GEOCollector
    
    console.print(f"[blue]Collecting GEO data for {accession}...[/blue]")
    
    collector = GEOCollector()
    
    try:
        data = collector.collect_single(accession)
        
        # Save to database if applicable
        saved = None
        if data["type"] in ["series", "sample"]:
            saved = collector.save_to_database(data)
        
        console.print(f"[green]Successfully collected {data['type']} {accession}[/green]")
        console.print(f"  Title: {data.get('title', 'N/A')}")
        
        if data["type"] == "series":
            console.print(f"  Platform: {data.get('platform', 'N/A')}")
            console.print(f"  Samples: {data.get('sample_count', 0)}")
        elif data["type"] == "sample":
            console.print(f"  Organism: {data.get('organism', 'N/A')}")
            console.print(f"  Platform: {data.get('platform', 'N/A')}")
        elif data["type"] == "dataset":
            console.print(f"  Summary: {data.get('summary', 'N/A')[:100]}...")
            console.print(f"  Samples: {data.get('sample_count', 0)}")
        
        if save_file:
            collector.save_to_file(data, f"geo_{accession}.json")
    except Exception as e:
        console.print(f"[red]Collection failed: {e}[/red]")
        logger.error(f"GEO collection failed for {accession}", exc_info=True)


@collect.command()
@click.argument("gene_symbol")
@click.option("--organism", default="Homo sapiens", help="Organism name")
@click.option("--limit", default=20, help="Maximum number of datasets")
@click.option("--save-file", is_flag=True, help="Save to JSON file")
def geo_search(gene_symbol: str, organism: str, limit: int, save_file: bool):
    """Search GEO for datasets containing a gene."""
    from src.collectors.geo_collector import GEOCollector
    
    console.print(f"[blue]Searching GEO for {gene_symbol} in {organism}...[/blue]")
    
    collector = GEOCollector()
    
    try:
        datasets = collector.search_by_gene(gene_symbol, organism, limit)
        
        console.print(f"[green]Found {len(datasets)} datasets[/green]")
        
        for dataset in datasets[:10]:  # Show first 10
            console.print(f"\n  {dataset.get('accession', 'Unknown')}:")
            console.print(f"    Title: {dataset.get('title', 'N/A')}")
            console.print(f"    Samples: {dataset.get('sample_count', 'N/A')}")
            
        if len(datasets) > 10:
            console.print(f"\n  ... and {len(datasets) - 10} more")
        
        if save_file:
            collector.save_to_file(datasets, f"geo_search_{gene_symbol}.json")
    except Exception as e:
        console.print(f"[red]Search failed: {e}[/red]")
        logger.error(f"GEO search failed for {gene_symbol}", exc_info=True)


@collect.command()
@click.argument("rsid")
@click.option("--save-file", is_flag=True, help="Save to JSON file")
def dbsnp(rsid: str, save_file: bool):
    """Collect dbSNP variant data."""
    from src.collectors.dbsnp_collector import dbSNPCollector
    
    console.print(f"[blue]Collecting dbSNP data for {rsid}...[/blue]")
    
    collector = dbSNPCollector()
    
    try:
        data = collector.collect_single(rsid)
        variant = collector.save_to_database(data)
        
        console.print(f"[green]Successfully collected variant {variant.rsid}[/green]")
        console.print(f"  Type: {variant.variant_type}")
        console.print(f"  Location: chr{variant.chromosome}:{variant.position}")
        console.print(f"  Alleles: {variant.reference_allele}/{variant.alternative_allele}")
        
        if variant.gene_symbol:
            console.print(f"  Gene: {variant.gene_symbol}")
        
        if variant.minor_allele_frequency:
            console.print(f"  Global MAF: {variant.minor_allele_frequency:.4f}")
        
        if variant.clinical_significance:
            console.print(f"  Clinical: {variant.clinical_significance}")
        
        if save_file:
            collector.save_to_file(data, f"dbsnp_{rsid}.json")
    except Exception as e:
        console.print(f"[red]Collection failed: {e}[/red]")
        logger.error(f"dbSNP collection failed for {rsid}", exc_info=True)


@collect.command()
@click.argument("gene_symbol")
@click.option("--organism", default="human", help="Organism name")
@click.option("--limit", default=50, help="Maximum number of variants")
@click.option("--save-file", is_flag=True, help="Save to JSON file")
def dbsnp_gene(gene_symbol: str, organism: str, limit: int, save_file: bool):
    """Collect dbSNP variants for a gene."""
    from src.collectors.dbsnp_collector import dbSNPCollector
    
    console.print(f"[blue]Collecting dbSNP variants for {gene_symbol}...[/blue]")
    
    collector = dbSNPCollector()
    
    try:
        variants_data = collector.collect_by_gene(gene_symbol, organism, limit)
        
        console.print(f"[green]Found {len(variants_data)} variants[/green]")
        
        # Save to database
        saved_count = 0
        for var_data in variants_data:
            try:
                collector.save_to_database(var_data)
                saved_count += 1
            except Exception as e:
                logger.error(f"Failed to save variant: {e}")
        
        console.print(f"  Saved {saved_count} variants to database")
        
        # Show summary
        var_types = {}
        for var in variants_data:
            vtype = var.get("variant_type", "Unknown")
            var_types[vtype] = var_types.get(vtype, 0) + 1
        
        console.print("\n  Variant types:")
        for vtype, count in var_types.items():
            console.print(f"    {vtype}: {count}")
        
        if save_file:
            collector.save_to_file(variants_data, f"dbsnp_gene_{gene_symbol}.json")
    except Exception as e:
        console.print(f"[red]Collection failed: {e}[/red]")
        logger.error(f"dbSNP gene collection failed for {gene_symbol}", exc_info=True)


@collect.command()
@click.argument("chromosome")
@click.argument("start", type=int)
@click.argument("end", type=int)
@click.option("--organism", default="human", help="Organism name")
@click.option("--save-file", is_flag=True, help="Save to JSON file")
def dbsnp_region(chromosome: str, start: int, end: int, organism: str, save_file: bool):
    """Collect dbSNP variants in a genomic region."""
    from src.collectors.dbsnp_collector import dbSNPCollector
    
    console.print(f"[blue]Collecting dbSNP variants for chr{chromosome}:{start}-{end}...[/blue]")
    
    collector = dbSNPCollector()
    
    try:
        variants_data = collector.collect_by_position(chromosome, start, end, organism)
        
        console.print(f"[green]Found {len(variants_data)} variants in region[/green]")
        
        # Save to database
        saved_count = 0
        for var_data in variants_data:
            try:
                collector.save_to_database(var_data)
                saved_count += 1
            except Exception as e:
                logger.error(f"Failed to save variant: {e}")
        
        console.print(f"  Saved {saved_count} variants to database")
        
        # Show variant distribution
        if variants_data:
            positions = [v.get("position", 0) for v in variants_data if v.get("position")]
            if positions:
                console.print(f"  Position range: {min(positions)} - {max(positions)}")
        
        if save_file:
            filename = f"dbsnp_region_chr{chromosome}_{start}_{end}.json"
            collector.save_to_file(variants_data, filename)
    except Exception as e:
        console.print(f"[red]Collection failed: {e}[/red]")
        logger.error(f"dbSNP region collection failed", exc_info=True)


@cli.group()
def export():
    """Data export commands."""
    pass


@export.command()
@click.argument("output_file")
@click.option("--format", type=click.Choice(["json", "csv", "fasta"]), default="json")
@click.option("--table", help="Table to export")
def database(output_file: str, format: str, table: Optional[str]):
    """Export database contents."""
    console.print(f"[blue]Exporting {table or 'all tables'} to {output_file}...[/blue]")
    
    # Implementation would go here
    console.print("[yellow]Export functionality not yet implemented[/yellow]")


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()