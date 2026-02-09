from typing import Any
import math

import numpy as np
from .countdata import CountMatrix


def log_fold_change(old, new, base=2):
    return math.log(new/old, base)


def expression_modulo_metadata(count_matrix: CountMatrix,
                               header: str
                               ) -> tuple[list[str], Any]:
    metadata_factors = list(set(count_matrix.metadata[header]))
    average_expressions = []
    for factor in metadata_factors:
        samples = list(count_matrix.match_by_metadata(header, factor))
        expression = np.array(count_matrix.expression(samples).sum(axis=0))
        average_expression = expression / len(samples)
        average_expressions.append(average_expression)
    total_expression = np.vstack(average_expressions)
    return metadata_factors, total_expression


def filter_genes(count_matrix: CountMatrix,
                 header: str,
                 expr_threshold: float,
                 diff_threshold: float
                 ) -> list[str]:
    _, expression = expression_modulo_metadata(count_matrix, header)
    highly_expressed = highly_expressed_columns(expression, expr_threshold)
    diff_expressed = differentially_expressed(expression, diff_threshold)
    to_keep = sorted(list(set(highly_expressed).intersection(set(diff_expressed))))
    genes_to_keep = [count_matrix.genes[i] for i in to_keep]
    count_matrix.filter_genes(genes_to_keep)
    return genes_to_keep


def highly_expressed_columns(array, threshold=0.0625/500):
    height, width = array.shape
    to_keep = []
    for j in range(width):
        i = 0
        done = False
        while i < height and not done:
            if array[i, j] >= threshold:
                to_keep.append(j)
                done = True
            i += 1
    return to_keep


def differentially_expressed(array, threshold=0.5):
    height, width = array.shape
    averages = array.mean(axis=0)
    to_keep = []
    for j in range(width):
        i = 0
        done = False
        while i < height and not done:
            if array[i, j] > 0 and log_fold_change(averages[j], array[i, j]) > threshold:
                to_keep.append(j)
                done = True
            i += 1
    return to_keep
