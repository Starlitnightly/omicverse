from typing import Any
import math

import numpy as np
from tqdm import tqdm
from .countdata import CountMatrix
from . import Colors, EMOJI


def log_fold_change(old, new, base=2):
    return math.log(new/old, base)


def expression_modulo_metadata(count_matrix: CountMatrix,
                               header: str
                               ) -> tuple[list[str], Any]:
    metadata_factors = list(set(count_matrix.metadata[header]))
    print(f"{Colors.CYAN}{EMOJI['process']} Computing expression for {len(metadata_factors)} metadata groups...{Colors.ENDC}")
    average_expressions = []
    for factor in tqdm(metadata_factors, desc=f"{Colors.BLUE}Processing groups{Colors.ENDC}"):
        samples = list(count_matrix.match_by_metadata(header, factor))
        expression = np.array(count_matrix.expression(samples).sum(axis=0))
        average_expression = expression / len(samples)
        average_expressions.append(average_expression)
    total_expression = np.vstack(average_expressions)
    print(f"{Colors.GREEN}{EMOJI['done']} Expression computation completed!{Colors.ENDC}")
    return metadata_factors, total_expression


def filter_genes(count_matrix: CountMatrix,
                 header: str,
                 expr_threshold: float,
                 diff_threshold: float
                 ) -> list[str]:
    print(f"{Colors.HEADER}{EMOJI['filter']} Filtering genes with expr_threshold={expr_threshold}, diff_threshold={diff_threshold}{Colors.ENDC}")
    _, expression = expression_modulo_metadata(count_matrix, header)
    print(f"{Colors.CYAN}  → Identifying highly expressed genes...{Colors.ENDC}")
    highly_expressed = highly_expressed_columns(expression, expr_threshold)
    print(f"{Colors.BLUE}    Found {len(highly_expressed)} highly expressed genes{Colors.ENDC}")
    print(f"{Colors.CYAN}  → Identifying differentially expressed genes...{Colors.ENDC}")
    diff_expressed = differentially_expressed(expression, diff_threshold)
    print(f"{Colors.BLUE}    Found {len(diff_expressed)} differentially expressed genes{Colors.ENDC}")
    to_keep = sorted(list(set(highly_expressed).intersection(set(diff_expressed))))
    genes_to_keep = [count_matrix.genes[i] for i in to_keep]
    count_matrix.filter_genes(genes_to_keep)
    print(f"{Colors.GREEN}{EMOJI['done']} Filtered to {len(genes_to_keep)} genes!{Colors.ENDC}")
    return genes_to_keep


def highly_expressed_columns(array, threshold=0.0625/500):
    height, width = array.shape
    to_keep = []
    for j in tqdm(range(width), desc=f"{Colors.BLUE}Checking expression{Colors.ENDC}", leave=False):
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
    for j in tqdm(range(width), desc=f"{Colors.BLUE}Checking diff. expr{Colors.ENDC}", leave=False):
        i = 0
        done = False
        while i < height and not done:
            if array[i, j] > 0 and log_fold_change(averages[j], array[i, j]) > threshold:
                to_keep.append(j)
                done = True
            i += 1
    return to_keep
