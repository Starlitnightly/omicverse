"""Classes storing gene expression data."""

from abc import ABC
from typing import (Any, Callable, Collection, Dict, Iterable, Iterator, Mapping,
                    MutableSequence, Pattern, Sequence, cast)

import pandas as pd
from tqdm import tqdm
from .sparsetools import rescale_rows, filter_cols, iterate_sparse
from .constantlookuplist import ConstantLookupList
from . import Colors, EMOJI


def matching(values: Sequence[str],
             pattern: Pattern | Collection[str]
             ) -> Iterable[str]:
    """Given values and a pattern returns all values matching the pattern.

    Args:
        values: A sequence of strings.
        pattern: A regular expression or a collection of strings.

    Returns:
        All elements of values which match or are an element of pattern.

    Raises:
        TypeError: If pattern is neither a regular expression nor a collection.
    """
    if isinstance(pattern, Pattern):
        return filter(pattern.match, values)
    try:
        pattern_set = set(pattern)
        return [value for value in values if value in pattern_set]
    except TypeError as error:
        raise TypeError from error


def _apply_or_not(func: Callable[[str], int],
                  values: Iterable[str] | Iterable[int] | None,
                  default: Iterable[int],
                  ) -> Iterator[int]:
    match values:
        case None:
            yield from default
        case [] | [int(), *_]:
            yield from cast(Sequence[int], values)
        case [str(), *_]:
            yield from map(func, values)
        case _:
            raise ValueError(f"Can't apply {func} to {values}")
    yield from ()  # Otherwise type checkers don't realise this is a generator


class CountData(ABC):
    """A collection of gene expression readings.

    Attributes:
        genes: An ordered list of gene identifiers.
        samples: An ordered list of sample identifiers.
        num_genes: The number of genes in the domain, i.e. len(genes).
        num_samples: The number of samples in the domain, i.e. len(samples).
        metadata: A dataframe where each row corresponds to a sample.
    """

    def __init__(self,
                 genes: MutableSequence[str] | None = None,
                 samples: MutableSequence[str] | None = None,
                 num_genes: int | None = None,
                 num_samples: int | None = None
                 ):
        """Inits count metadata.

        If genes (resp. samples) is None then a sequence of identifiers is
        inferred from num_genes (resp. num_samples). Therefore, at least one
        of genes and num_genes (resp. samples and num_samples) must be given.

        Args:
            genes: An sequence of genes.
            samples: An sequence of samples.
            num_genes: The number of genes.
            num_samples: The number of samples.
        """

        def infer(values: MutableSequence[str] | None,
                  num_values: int | None, label: str
                  ) -> tuple[MutableSequence[str], int]:
            if values:
                return values, len(values)
            if num_values:
                return list(map(str, range(num_values))), num_values
            raise ValueError(f"Must provide list or number of {label}")

        self.genes, self.num_genes = infer(genes, num_genes, "genes")
        self.genes = ConstantLookupList(self.genes)
        self.samples, self.num_samples = infer(samples, num_samples, "samples")
        self.samples = ConstantLookupList(self.samples)

        self.metadata = pd.DataFrame(data={'sample': self.samples})

    def add_metadata(self,
                     header: str,
                     values: Mapping[str, Any] | Sequence[Any]
                     ):
        """Add values to metadata under a header.

        If values is a mapping then this is used to infer new entries.
        Otherwise, it is assumed that the metadata entry for sample i is
        simply given by value i.

        Args:
            header:
                A string denoting the column name for the new data.
            values:
                The new metadata. Either a mapping from samples or a sequence
                of the same length as samples.
        """
        if not isinstance(values, Mapping):
            values = dict(zip(self.samples, values))

        def meta_from_row(row):
            return values[row['sample']]

        self.metadata[header] = self.metadata.apply(meta_from_row, axis=1)

    def filter_genes(self, pattern: Pattern | Collection[str]):
        """Filters genes according to a pattern.

        Edits the count data so that only genes matching the pattern
        are included.

        Args:
            pattern:
                Either a regular expression or a collection identifying gene
                identifiers to be kept.
        """
        raise NotImplementedError(f"Cannot identify {pattern}")

    def match_by_metadata(self, header: str, pattern: str) -> Iterator[str]:
        """Returns all samples matching the given metadata value.

        Args:
            header: A header of the object's metadata.
            pattern: A string to be matched against.

        Returns:
            An iterable of all samples whose metadata value under the
            given header matches the pattern.
        """
        yield from self.metadata[self.metadata[header] == pattern]['sample']

    def group_by_metadata(self, header: str) -> Dict[str, Iterator[str]]:
        """Returns all samples organised by the given metadata value.

        Args:
            header: A header of the object's metadata.

        Returns:
            A dictionary whose keys are all values under the given header, each
            mapping to an iterable of all samples matching that header.
        """
        values = set(self.metadata[header])
        return {value: self.match_by_metadata(header, value)
                for value in values}


class CountMatrix(CountData):
    """A collection of gene expression readings recorded in a sparse matrix

    Attributes:
        matrix:
            A sparse matrix whose [i,j]th entry is the expression of
            gene[j] in sample[i].
    """

    def __init__(self, matrix, **kwargs):
        num_samples, num_genes = matrix.shape
        print(f"{Colors.CYAN}{EMOJI['gene']} Initializing CountMatrix: {num_samples} samples × {num_genes} genes{Colors.ENDC}")
        self.matrix = matrix
        super().__init__(num_genes=num_genes,
                         num_samples=num_samples,
                         **kwargs
                         )
        print(f"{Colors.GREEN}{EMOJI['done']} CountMatrix initialized!{Colors.ENDC}")

    def expression(self,
                   samples: Sequence[str] | Sequence[int] | None = None,
                   genes: Sequence[str] | Sequence[int] | None = None
                   ):
        """The expression sub-matrix for the given samples and genes.

        Args:
            samples:
                A sequence of either sample identifiers or sample indices.
            genes:
                A sequence of either gene identifiers or gene indices.

        Returns:
            A 2D sparse array containing the expression of the given genes in
            the given samples.
        """
        sample_indices = list(_apply_or_not(self.samples.index,
                                            samples,
                                            range(self.num_samples)
                                            ))
        gene_indices = list(_apply_or_not(self.genes.index,
                                          genes,
                                          range(self.num_genes)
                                          ))

        return self.matrix[sample_indices][:, gene_indices]

    def avg_expression(self,
                       samples: Sequence[str] | Sequence[int] | None = None,
                       genes: Sequence[str] | Sequence[int] | None = None,
                       ):
        """The average expression sub-matrix for the given samples and genes"""
        total_expression = self.expression(samples, genes)
        num_samples = self.num_samples if samples is None else len(samples)
        return total_expression / num_samples

    def expressed_genes(self,
                        samples: Sequence[str] | Sequence[int] | None = None,
                        output_type: str = "ident"
                        ) -> Iterator[int] | Iterator[str]:
        """Returns all genes expressed at least once in a list of samples"""
        expression = self.expression(samples)
        gene_indices = set(expression.tocoo().col)
        match output_type:
            case "index":
                yield from gene_indices
            case "ident":
                for gene_index in gene_indices:
                    yield self.genes[gene_index]
            case _:
                raise ValueError(f"output type must be ident or index, not {output_type}")
        yield from ()

    def rescale_genes(self, factors: Sequence[float]):
        """Rescales gene expression according to the given factors.

        Column j of the gene matrix is multipled by the jth value of factors.

        Args:
            factors: A sequence of factors by which columns are rescaled.
        """
        self.matrix = rescale_columns(self.matrix, factors)

    def filter_genes(self, pattern: Pattern | Collection[str]):
        print(f"{Colors.CYAN}{EMOJI['filter']} Filtering genes...{Colors.ENDC}")
        new_genes = ConstantLookupList(matching(self.genes, pattern))
        print(f"{Colors.BLUE}  → Keeping {len(new_genes)}/{self.num_genes} genes{Colors.ENDC}")
        keep_indices = list(map(self.genes.index, new_genes))
        new_matrix = filter_cols(self.matrix, keep_indices)
        self.matrix = new_matrix
        self.genes = new_genes
        self.num_genes = len(new_genes)
        print(f"{Colors.GREEN}{EMOJI['done']} Gene filtering completed!{Colors.ENDC}")

    def to_count_table(self, **kwargs):
        """Converts the CountMatrix into a CountTable.

        Args:
            gene_col: The name of the gene ID column.
            sample_col: The name of the sample ID column.
            count_col: The name of the count column.

        Returns:
            A CountTable holding the same expression data.
        """
        print(f"{Colors.CYAN}{EMOJI['process']} Converting CountMatrix to CountTable...{Colors.ENDC}")
        entries = []
        for i, j, count in tqdm(iterate_sparse(self.matrix), desc=f"{Colors.BLUE}Processing entries{Colors.ENDC}"):
            sample = self.samples[i]
            gene = self.genes[j]
            entries.append([gene, sample, count])
        table = pd.DataFrame(entries, columns=["gene", "sample", "count"])
        print(f"{Colors.GREEN}{EMOJI['done']} Conversion completed!{Colors.ENDC}")
        return CountTable(table,
                          genes=self.genes,
                          samples=self.samples,
                          num_genes=self.num_genes,
                          num_samples=self.num_samples,
                          metadata=self.metadata,
                          **kwargs
                          )


class CountTable(CountData):

    def __init__(self,
                 table,
                 genes: MutableSequence[str] | None = None,
                 samples: MutableSequence[str] | None = None,
                 gene_col: str = "gene",
                 sample_col: str = "sample",
                 count_col: str = "count",
                 **kwargs
                 ):

        genes = genes or list(set(table[gene_col]))
        samples = samples or list(set(table[sample_col]))
        self.gene_col = gene_col
        self.sample_col = sample_col
        self.count_col = count_col
        self.table = table[(table[gene_col].isin(genes)) & (table[sample_col].isin(samples))]
        super().__init__(genes=genes,
                         samples=samples,
                         **kwargs
                         )

    def filter_genes(self, pattern: Pattern | Collection[str]):
        new_genes = ConstantLookupList(matching(self.genes, pattern))
        new_table = self.table[self.table[self.gene_col] in new_genes]
        self.table = new_table
        self.genes = new_genes
        self.num_genes = len(new_genes)

    def toCountMatrix(self):
        raise NotImplementedError
