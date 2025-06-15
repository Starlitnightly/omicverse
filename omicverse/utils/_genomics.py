r"""
Genomics operations
Copy from scglue: https://github.com/gao-lab/GLUE/
"""

import collections
import os
import re
from ast import literal_eval
from functools import reduce
from itertools import chain, product
from operator import add
from typing import Any, Callable, List, Mapping, Optional, Union

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse
import scipy.stats
from anndata import AnnData
from networkx.algorithms.bipartite import biadjacency_matrix
from statsmodels.stats.multitest import fdrcorrection
from tqdm.auto import tqdm

#from .utils import ConstrainedDataFrame, logged, get_rs


#--------------------------- Constrained data frame ----------------------------


class ConstrainedDataFrame(pd.DataFrame):

    r"""
    Data frame with certain format constraints

    Note
    ----
    Format constraints are checked and maintained automatically.
    """

    def __init__(self, *args, **kwargs) -> None:
        df = pd.DataFrame(*args, **kwargs)
        df = self.rectify(df)
        self.verify(df)
        super().__init__(df)

    def __setitem__(self, key, value) -> None:
        super().__setitem__(key, value)
        self.verify(self)

    @property
    def _constructor(self) -> type:
        return type(self)

    @classmethod
    def rectify(cls, df: pd.DataFrame) -> pd.DataFrame:
        r"""
        Rectify data frame for format integrity

        Parameters
        ----------
        df
            Data frame to be rectified

        Returns
        -------
        rectified_df
            Rectified data frame
        """
        return df

    @classmethod
    def verify(cls, df: pd.DataFrame) -> None:
        r"""
        Verify data frame for format integrity

        Parameters
        ----------
        df
            Data frame to be verified
        """

    @property
    def df(self) -> pd.DataFrame:
        r"""
        Convert to regular data frame
        """
        return pd.DataFrame(self)

    def __repr__(self) -> str:
        r"""
        Note
        ----
        We need to explicitly call :func:`repr` on the regular data frame
        to bypass integrity verification, because when the terminal is
        too narrow, :mod:`pandas` would split the data frame internally,
        causing format verification to fail.
        """
        return repr(self.df)


class Bed(ConstrainedDataFrame):

    r"""
    BED format data frame
    """

    COLUMNS = pd.Index([
        "chrom", "chromStart", "chromEnd", "name", "score",
        "strand", "thickStart", "thickEnd", "itemRgb",
        "blockCount", "blockSizes", "blockStarts"
    ])

    @classmethod
    def rectify(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = super(Bed, cls).rectify(df)
        COLUMNS = cls.COLUMNS.copy(deep=True)
        for item in COLUMNS:
            if item in df:
                if item in ("chromStart", "chromEnd"):
                    df[item] = df[item].astype(int)
                else:
                    df[item] = df[item].astype(str)
            elif item not in ("chrom", "chromStart", "chromEnd"):
                df[item] = "."
            else:
                raise ValueError(f"Required column {item} is missing!")
        return df.loc[:, COLUMNS]

    @classmethod
    def verify(cls, df: pd.DataFrame) -> None:
        super(Bed, cls).verify(df)
        if len(df.columns) != len(cls.COLUMNS) or np.any(df.columns != cls.COLUMNS):
            raise ValueError("Invalid BED format!")

    @classmethod
    def read_bed(cls, fname: os.PathLike) -> "Bed":
        r"""
        Read BED file

        Parameters
        ----------
        fname
            BED file

        Returns
        -------
        bed
            Loaded :class:`Bed` object
        """
        COLUMNS = cls.COLUMNS.copy(deep=True)
        loaded = pd.read_csv(fname, sep="\t", header=None, comment="#")
        loaded.columns = COLUMNS[:loaded.shape[1]]
        return cls(loaded)

    def write_bed(self, fname: os.PathLike, ncols: Optional[int] = None) -> None:
        r"""
        Write BED file

        Parameters
        ----------
        fname
            BED file
        ncols
            Number of columns to write (by default write all columns)
        """
        if ncols and ncols < 3:
            raise ValueError("`ncols` must be larger than 3!")
        df = self.df.iloc[:, :ncols] if ncols else self
        df.to_csv(fname, sep="\t", header=False, index=False)

    def to_bedtool(self):
        r"""
        Convert to a :class:`pybedtools.BedTool` object

        Returns
        -------
        bedtool
            Converted :class:`pybedtools.BedTool` object
        """
        from pybedtools import BedTool
        from pybedtools.cbedtools import Interval
        return BedTool(Interval(
            row["chrom"], row["chromStart"], row["chromEnd"],
            name=row["name"], score=row["score"], strand=row["strand"]
        ) for _, row in self.iterrows())

    def nucleotide_content(self, fasta: os.PathLike) -> pd.DataFrame:
        r"""
        Compute nucleotide content in the BED regions

        Parameters
        ----------
        fasta
            Genomic sequence file in FASTA format

        Returns
        -------
        nucleotide_stat
            Data frame containing nucleotide content statistics for each region
        """
        import pybedtools
        result = self.to_bedtool().nucleotide_content(fi=os.fspath(fasta), s=True)  # pylint: disable=unexpected-keyword-arg
        result = pd.DataFrame(
            np.stack([interval.fields[6:15] for interval in result]),
            columns=[
                r"%AT", r"%GC",
                r"#A", r"#C", r"#G", r"#T", r"#N",
                r"#other", r"length"
            ]
        ).astype({
            r"%AT": float, r"%GC": float,
            r"#A": int, r"#C": int, r"#G": int, r"#T": int, r"#N": int,
            r"#other": int, r"length": int
        })
        pybedtools.cleanup()
        return result

    def strand_specific_start_site(self) -> "Bed":
        r"""
        Convert to strand-specific start sites of genomic features

        Returns
        -------
        start_site_bed
            A new :class:`Bed` object, containing strand-specific start sites
            of the current :class:`Bed` object
        """
        if set(self["strand"]) != set(["+", "-"]):
            raise ValueError("Not all features are strand specific!")
        df = pd.DataFrame(self, copy=True)
        pos_strand = df.query("strand == '+'").index
        neg_strand = df.query("strand == '-'").index
        df.loc[pos_strand, "chromEnd"] = df.loc[pos_strand, "chromStart"] + 1
        df.loc[neg_strand, "chromStart"] = df.loc[neg_strand, "chromEnd"] - 1
        return type(self)(df)

    def strand_specific_end_site(self) -> "Bed":
        r"""
        Convert to strand-specific end sites of genomic features

        Returns
        -------
        end_site_bed
            A new :class:`Bed` object, containing strand-specific end sites
            of the current :class:`Bed` object
        """
        if set(self["strand"]) != set(["+", "-"]):
            raise ValueError("Not all features are strand specific!")
        df = pd.DataFrame(self, copy=True)
        pos_strand = df.query("strand == '+'").index
        neg_strand = df.query("strand == '-'").index
        df.loc[pos_strand, "chromStart"] = df.loc[pos_strand, "chromEnd"] - 1
        df.loc[neg_strand, "chromEnd"] = df.loc[neg_strand, "chromStart"] + 1
        return type(self)(df)

    def expand(
            self, upstream: int, downstream: int,
            chr_len: Optional[Mapping[str, int]] = None
    ) -> "Bed":
        r"""
        Expand genomic features towards upstream and downstream

        Parameters
        ----------
        upstream
            Number of bps to expand in the upstream direction
        downstream
            Number of bps to expand in the downstream direction
        chr_len
            Length of each chromosome

        Returns
        -------
        expanded_bed
            A new :class:`Bed` object, containing expanded features
            of the current :class:`Bed` object

        Note
        ----
        Starting position < 0 after expansion is always trimmed.
        Ending position exceeding chromosome length is trimed only if
        ``chr_len`` is specified.
        """
        if upstream == downstream == 0:
            return self
        df = pd.DataFrame(self, copy=True)
        if upstream == downstream:  # symmetric
            df["chromStart"] -= upstream
            df["chromEnd"] += downstream
        else:  # asymmetric
            if set(df["strand"]) != set(["+", "-"]):
                raise ValueError("Not all features are strand specific!")
            pos_strand = df.query("strand == '+'").index
            neg_strand = df.query("strand == '-'").index
            if upstream:
                df.loc[pos_strand, "chromStart"] -= upstream
                df.loc[neg_strand, "chromEnd"] += upstream
            if downstream:
                df.loc[pos_strand, "chromEnd"] += downstream
                df.loc[neg_strand, "chromStart"] -= downstream
        df["chromStart"] = np.maximum(df["chromStart"], 0)
        if chr_len:
            chr_len = df["chrom"].map(chr_len)
            df["chromEnd"] = np.minimum(df["chromEnd"], chr_len)
        return type(self)(df)


class Gtf(ConstrainedDataFrame):  # gffutils is too slow

    r"""
    GTF format data frame
    """

    COLUMNS = pd.Index([
        "seqname", "source", "feature", "start", "end",
        "score", "strand", "frame", "attribute"
    ])  # Additional columns after "attribute" is allowed

    @classmethod
    def rectify(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = super(Gtf, cls).rectify(df)
        COLUMNS = cls.COLUMNS.copy(deep=True)
        for item in COLUMNS:
            if item in df:
                if item in ("start", "end"):
                    df[item] = df[item].astype(int)
                else:
                    df[item] = df[item].astype(str)
            elif item not in ("seqname", "start", "end"):
                df[item] = "."
            else:
                raise ValueError(f"Required column {item} is missing!")
        return df.sort_index(axis=1, key=cls._column_key)

    @classmethod
    def _column_key(cls, x: pd.Index) -> np.ndarray:
        x = cls.COLUMNS.get_indexer(x)
        x[x < 0] = x.max() + 1  # Put additional columns after "attribute"
        return x

    @classmethod
    def verify(cls, df: pd.DataFrame) -> None:
        super(Gtf, cls).verify(df)
        if len(df.columns) < len(cls.COLUMNS) or \
                np.any(df.columns[:len(cls.COLUMNS)] != cls.COLUMNS):
            raise ValueError("Invalid GTF format!")

    @classmethod
    def read_gtf(cls, fname: os.PathLike) -> "Gtf":
        r"""
        Read GTF file

        Parameters
        ----------
        fname
            GTF file

        Returns
        -------
        gtf
            Loaded :class:`Gtf` object
        """
        COLUMNS = cls.COLUMNS.copy(deep=True)
        loaded = pd.read_csv(fname, sep="\t", header=None, comment="#")
        loaded.columns = COLUMNS[:loaded.shape[1]]
        return cls(loaded)

    def split_attribute(self) -> "Gtf":
        r"""
        Extract all attributes from the "attribute" column
        and append them to existing columns

        Returns
        -------
        splitted
            Gtf with splitted attribute columns appended
        """
        pattern = re.compile(r'([^\s]+) "([^"]+)";')
        splitted = pd.DataFrame.from_records(np.vectorize(lambda x: {
            key: val for key, val in pattern.findall(x)
        })(self["attribute"]), index=self.index)
        if set(self.COLUMNS).intersection(splitted.columns):
            self.logger.warning(
                "Splitted attribute names overlap standard GTF fields! "
                "The standard fields are overwritten!"
            )
        return self.assign(**splitted)

    def to_bed(self, name: Optional[str] = None) -> Bed:
        r"""
        Convert GTF to BED format

        Parameters
        ----------
        name
            Specify a column to be converted to the "name" column in bed format,
            otherwise the "name" column would be filled with "."

        Returns
        -------
        bed
            Converted :class:`Bed` object
        """
        bed_df = pd.DataFrame(self, copy=True).loc[
            :, ("seqname", "start", "end", "score", "strand")
        ]
        bed_df.insert(3, "name", np.repeat(
            ".", len(bed_df)
        ) if name is None else self[name])
        bed_df["start"] -= 1  # Convert to zero-based
        bed_df.columns = (
            "chrom", "chromStart", "chromEnd", "name", "score", "strand"
        )
        return Bed(bed_df)


def interval_dist(x, y) -> int:
    r"""Compute distance and relative position between two bed intervals.

    Arguments:
        x: First interval
        y: Second interval

    Returns:
        Signed distance between x and y
    """
    if x.chrom != y.chrom:
        return np.inf * (-1 if x.chrom < y.chrom else 1)
    if x.start < y.stop and y.start < x.stop:
        return 0
    if x.stop <= y.start:
        return x.stop - y.start - 1
    if y.stop <= x.start:
        return x.start - y.stop + 1


def window_graph(
        left: Union[Bed, str], right: Union[Bed, str], window_size: int,
        left_sorted: bool = False, right_sorted: bool = False,
        attr_fn= None
) -> nx.MultiDiGraph:
    r"""
    Construct a window graph between two sets of genomic features, where
    features pairs within a window size are connected.

    Parameters
    ----------
    left
        First feature set, either a :class:`Bed` object or path to a bed file
    right
        Second feature set, either a :class:`Bed` object or path to a bed file
    window_size
        Window size (in bp)
    left_sorted
        Whether ``left`` is already sorted
    right_sorted
        Whether ``right`` is already sorted
    attr_fn
        Function to compute edge attributes for connected features,
        should accept the following three positional arguments:

        - l: left interval
        - r: right interval
        - d: signed distance between the intervals

        By default no edge attribute is created.

    Returns
    -------
    graph
        Window graph
    """
    import pybedtools
    if isinstance(left, Bed):
        pbar_total = len(left)
        left = left.to_bedtool()
    else:
        pbar_total = None
        left = pybedtools.BedTool(left)
    if not left_sorted:
        left = left.sort(stream=True)
    left = iter(left)  # Resumable iterator
    if isinstance(right, Bed):
        right = right.to_bedtool()
    else:
        right = pybedtools.BedTool(right)
    if not right_sorted:
        right = right.sort(stream=True)
    right = iter(right)  # Resumable iterator

    attr_fn = attr_fn or (lambda l, r, d: {})
    if pbar_total is not None:
        left = tqdm(left, total=pbar_total, desc="window_graph")
    graph = nx.MultiDiGraph()
    window = collections.OrderedDict()  # Used as ordered set
    for l in left:
        for r in list(window.keys()):  # Allow remove during iteration
            d = interval_dist(l, r)
            if -window_size <= d <= window_size:
                graph.add_edge(l.name, r.name, **attr_fn(l, r, d))
            elif d > window_size:
                del window[r]
            else:  # dist < -window_size
                break  # No need to expand window
        else:
            for r in right:  # Resume from last break
                d = interval_dist(l, r)
                if -window_size <= d <= window_size:
                    graph.add_edge(l.name, r.name, **attr_fn(l, r, d))
                elif d > window_size:
                    continue
                window[r] = None  # Placeholder
                if d < -window_size:
                    break
    pybedtools.cleanup()
    return graph


def dist_power_decay(x: int) -> float:
    r"""Distance-based power decay weight.
    
    Computed as w = ((d + 1000) / 1000) ^ (-0.75)

    Arguments:
        x: Distance (in bp)

    Returns:
        Decaying weight
    """
    return ((x + 1000) / 1000) ** (-0.75)





def write_links(
    graph: nx.Graph, source: Bed, target: Bed, file: os.PathLike,
    keep_attrs: Optional[List[str]] = None
) -> None:
    r"""
    Export regulatory graph into a links file

    Parameters
    ----------
    graph
        Regulatory graph
    source
        Genomic coordinates of source nodes
    target
        Genomic coordinates of target nodes
    file
        Output file
    keep_attrs
        A list of attributes to keep for each link
    """
    nx.to_pandas_edgelist(
        graph
    ).merge(
        source.df.iloc[:, :4], how="left", left_on="source", right_on="name"
    ).merge(
        target.df.iloc[:, :4], how="left", left_on="target", right_on="name"
    ).loc[:, [
        "chrom_x", "chromStart_x", "chromEnd_x",
        "chrom_y", "chromStart_y", "chromEnd_y",
        *(keep_attrs or [])
    ]].to_csv(file, sep="\t", index=False, header=False)




def write_scenic_feather(
        gene2tf_rank: pd.DataFrame, feather: os.PathLike,
        version: int = 2
) -> None:
    r"""
    Write cis-regulatory ranking to a SCENIC-compatible feather file

    Parameters
    ----------
    gene2tf_rank
        Cis regulatory ranking between genes and transcription factors,
        as generated by :func:`cis_reg_ranking`
    feather
        Path to the output feather file
    version
        SCENIC feather version
    """
    if version not in {1, 2}:
        raise ValueError("Unrecognized SCENIC feather version!")
    if version == 2:
        suffix = ".genes_vs_tracks.rankings.feather"
        if not str(feather).endswith(suffix):
            raise ValueError(f"Feather file name must end with `{suffix}`!")
    tf2gene_rank = gene2tf_rank.T
    tf2gene_rank = tf2gene_rank.loc[
        np.unique(tf2gene_rank.index), np.unique(tf2gene_rank.columns)
    ].astype(np.int16)
    tf2gene_rank.index.name = "features" if version == 1 else "tracks"
    tf2gene_rank.columns.name = None
    columns = tf2gene_rank.columns.tolist()
    tf2gene_rank = tf2gene_rank.reset_index()
    if version == 2:
        tf2gene_rank = tf2gene_rank.loc[:, [*columns, "tracks"]]
    tf2gene_rank.to_feather(feather)


def read_ctx_grn(file: os.PathLike) -> nx.DiGraph:
    r"""Read pruned TF-target GRN as generated by pyscenic ctx.

    Arguments:
        file: Input file (.csv)

    Returns:
        Pruned TF-target GRN
        
    Note:
        Node attribute 'type' can be used to distinguish TFs and genes
    """
    df = pd.read_csv(
        file, header=None, skiprows=3,
        usecols=[0, 8], names=["TF", "targets"]
    )
    df["targets"] = df["targets"].map(lambda x: set(i[0] for i in literal_eval(x)))
    df = df.groupby("TF").aggregate({"targets": lambda x: reduce(set.union, x)})
    grn = nx.DiGraph([
        (tf, target)
        for tf, row in df.iterrows()
        for target in row["targets"]]
    )
    nx.set_node_attributes(grn, "target", name="type")
    for tf in df.index:
        grn.nodes[tf]["target"] = "TF"
    return grn


def get_chr_len_from_fai(fai: os.PathLike) -> Mapping[str, int]:
    r"""Get chromosome length information from fasta index file.

    Arguments:
        fai: Fasta index file

    Returns:
        Length of each chromosome
    """
    return pd.read_table(fai, header=None, index_col=0)[1].to_dict()


def ens_trim_version(x: str) -> str:
    r"""Trim version suffix from Ensembl ID.

    Arguments:
        x: Ensembl ID

    Returns:
        Ensembl ID with version suffix trimmed
    """
    return re.sub(r"\.[0-9_-]+$", "", x)


# Aliases
read_bed = Bed.read_bed
read_gtf = Gtf.read_gtf