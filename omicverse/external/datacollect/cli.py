"""
CLI integration for datacollect module within OmicVerse.
"""

import click
from .collectors import *


@click.group()
def datacollect():
    """DataCollect commands for OmicVerse."""
    pass


@datacollect.command()
@click.argument("identifier")
@click.option("--format", default="dict", help="Output format (dict, pandas, anndata)")
@click.option("--save", is_flag=True, help="Save to database")
def collect_protein(identifier, format, save):
    """Collect protein data."""
    from .collectors.uniprot_collector import UniProtCollector
    
    collector = UniProtCollector()
    data = collector.collect_single(identifier)
    
    if format == "pandas":
        from .utils.omicverse_adapters import to_pandas
        data = to_pandas(data, "protein")
    elif format == "anndata":
        from .utils.omicverse_adapters import to_anndata
        data = to_anndata(data)
    
    if save:
        collector.save_to_database(data)
    
    click.echo(f"Collected data for {identifier}")
    return data


@datacollect.command()
@click.argument("accession")
@click.option("--format", default="dict", help="Output format (dict, pandas, anndata)")
@click.option("--save", is_flag=True, help="Save to database")
def collect_expression(accession, format, save):
    """Collect gene expression data."""
    from .collectors.geo_collector import GEOCollector
    
    collector = GEOCollector()
    data = collector.collect_single(accession)
    
    if format == "pandas":
        from .utils.omicverse_adapters import to_pandas
        data = to_pandas(data, "expression")
    elif format == "anndata":
        from .utils.omicverse_adapters import to_anndata
        data = to_anndata(data)
    
    if save:
        collector.save_to_database(data)
    
    click.echo(f"Collected expression data for {accession}")
    return data


if __name__ == "__main__":
    datacollect()
