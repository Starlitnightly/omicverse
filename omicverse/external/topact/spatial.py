"""Classes and methods for array-based spatial transcriptomics analysis."""

import itertools
from typing import Iterable, Iterator, Sequence, cast
from multiprocessing import Process, Queue
from warnings import simplefilter

import numpy as np
import numpy.typing as npt
from scipy import sparse
from tqdm import tqdm
import pandas as pd
from .countdata import CountTable
from .classifier import Classifier
from .densetools import first_nonzero_1d, density_hull
from . import Colors, EMOJI


def combine_coords(coords: Iterable[int]) -> str:
    """Combines a tuple of ints into a unique string identifier."""
    return ','.join(map(str, coords))


def split_coords(ident: str) -> tuple[int, ...]:
    """Splits a unique identifier into its corresponding coordinates.

    Args:
        ident: A string of the form '{x1},{x2},...,{xn}'.

    Returns:
        A tuple of integers (x1, x2, ..., xn).
    """
    return tuple(map(int, ident.split(','))) if ident else ()


def first_coord(ident: str) -> int:
    """Obtains the first coordinate from a unique identifier.

    Args:
        ident: A string of the form '{x1},{x2},...,{xn}' where n>=1.

    Returns:
        The integer x1.
    """
    return split_coords(ident)[0]


def second_coord(ident: str) -> int:
    """Obtains the first coordinate from a unique identifier.

    Args:
        ident: A string of the form '{x1},{x2},...,{xn}' where n >= 2.

    Returns:
        The integer x2.
    """
    return split_coords(ident)[1]


def cartesian_product(x: npt.ArrayLike, y: npt.ArrayLike) -> npt.NDArray:
    """Computes the cartesian products of two 1-d vectors.

    Args:
        x: The first vector
        y: The second vector

    Returns:
        An array of shape (len(x) * len(y), 2) whose rows are precisely all
        possible tuples with first value in x and second value in y.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    return np.transpose([np.tile(x, len(y)),
                         np.repeat(y, len(x))])


def square_nbhd(point: tuple[int, int],
                scale: int,
                x_range: tuple[int, int],
                y_range: tuple[int, int]
                ) -> Iterator[tuple[int, int]]:
    """All coordinates in a square neighbourhood about a point.

    Args:
        point: the (x,y) coordinates of the point
        scale: the radius of the square
        x_range: the least and greatest acceptable x values
        y_range: the least and greated acceptable y values

    Yields:
        All tuples (i,j) satisfying the following:
            - x_range[0] <= i <= x_range[1]
            - y_range[0] <= j <= y_range[1]
            - d(x,i) <= scale
            - d(y,j) <= scale
    """
    x, y = point
    x_min, x_max = x_range
    y_min, y_max = y_range
    x_min = max(x_min, x - scale)
    x_max = min(x_max, x + scale)
    y_min = max(y_min, y - scale)
    y_max = min(y_max, y + scale)
    return itertools.product(range(x_min, x_max+1), range(y_min, y_max+1))


def square_nbhd_vec(point: tuple[int, int],
                    scale: int,
                    x_range: tuple[int, int],
                    y_range: tuple[int, int]
                    ) -> npt.NDArray:
    """Returns a 2D array of all coords in a square nbhd about a point

    Args:
        point: the (x,y) coordinates of the point
        scale: the radius of the square
        x_range: the least and greatest acceptable x values
        y_range: the least and greated acceptable y values

    Returns:
        An array A of shape (n*m, 2) where n = x_range[1] - x_range[0] + 1
        and m = y_range[1] - y_range[0] + 1, whose rows are precisely the
        elements of square_nbhd(point, scale, x_range, y_range).
    """
    x, y = point
    x_min, x_max = x_range
    y_min, y_max = y_range
    if x_max < x_min or y_max < y_min:
        return np.empty((0,2))
    x_min = max(x_min, x - scale)
    x_max = min(x_max, x + scale)
    y_min = max(y_min, y - scale)
    y_max = min(y_max, y + scale)
    return cartesian_product(np.arange(x_min, x_max + 1),
                             np.arange(y_min, y_max + 1))


def extract_classifications(confidence_matrix: npt.NDArray,
                            threshold: float
                            ) -> dict[tuple[int, int], int]:
    """Extracts a dictionary of all spot classifications given the threshold.

    Args:
        confidence_matrix:
            A matrix X such that X[i, j, s, c] is the confidence that the
            cell type of spot (i, j) is c at scale s.
        threshold:
            The confidence threshold.

    Returns:
        A dictionary d such that (i, j) is in d if and only if there is some
        scale s and cell type c so that confidence_matrix[i, j, s, c] >= threshold.
        Moreover, d[x,y] is the value of c corresponding to the lowest such
        value of s.
    """
    confident = zip(*np.where(confidence_matrix >= threshold))
    confident = cast(Iterator[tuple[int, int, int, int]], confident)

    classifications: dict[tuple[int, int], int] = {}

    for i, j, _, cell_type in confident:
        if (i, j) not in classifications:
            classifications[i, j] = cell_type

    return classifications


def extract_image(confidence_matrix: npt.NDArray,
                  threshold: float
                  ) -> npt.NDArray:
    classifications = extract_classifications(confidence_matrix, threshold)

    image = np.empty(confidence_matrix.shape[:2])
    image[:] = np.nan

    for (i, j), c in classifications.items():
        image[i, j] = c

    return image


class ExpressionGrid:
    """A spatial grid equipped with gene expressions.

    An ExpressionGrid encapsulates a 2D grid. For each coordinate (x,y) in the
    grid, we have a corresponding gene expression vector where each entry
    counts the number of reads of its corresponding gene.

    Attributes:
        x_min: The smallest x coordinate in the grid.
        y_min: The smallest y coordinate in the grid.
        x_max: The largest x coordinate in the grid.
        y_max: The largest y coordinate in the grid.
        height: The height of the grid.
        width: The width of the grid.
    """

    @classmethod
    def from_anndata(cls, adata, genes: Sequence[str]):
        """Create ExpressionGrid directly from AnnData (efficient).

        Args:
            adata: AnnData object with spatial coordinates
            genes: List of genes to use (from reference)

        Returns:
            ExpressionGrid object
        """
        print(f"{Colors.CYAN}{EMOJI['grid']} Creating ExpressionGrid from AnnData (efficient mode)...{Colors.ENDC}")

        # Step 1: Extract spatial coordinates
        print(f"{Colors.CYAN}  [1/5] Extracting spatial coordinates...{Colors.ENDC}")
        if 'spatial' in adata.obsm:
            spatial_coords = adata.obsm['spatial']
            x_coords = np.round(spatial_coords[:, 0]).astype(int)
            y_coords = np.round(spatial_coords[:, 1]).astype(int)
        elif 'x' in adata.obs.columns and 'y' in adata.obs.columns:
            x_coords = np.round(adata.obs['x'].values).astype(int)
            y_coords = np.round(adata.obs['y'].values).astype(int)
        else:
            raise ValueError("AnnData must have spatial coordinates")
        print(f"{Colors.GREEN}        ✓ Extracted coordinates for {len(x_coords)} spots{Colors.ENDC}")

        # Create instance without going through __init__
        grid = cls.__new__(cls)

        # Step 2: Set coordinate bounds
        print(f"{Colors.CYAN}  [2/5] Computing grid dimensions...{Colors.ENDC}")
        grid.x_min, grid.x_max = int(x_coords.min()), int(x_coords.max())
        grid.y_min, grid.y_max = int(y_coords.min()), int(y_coords.max())
        grid.height = grid.x_max - grid.x_min + 1
        grid.width = grid.y_max - grid.y_min + 1
        grid.num_genes = len(genes)

        print(f"{Colors.GREEN}        ✓ Grid size: {grid.height}×{grid.width}{Colors.ENDC}")
        print(f"{Colors.BLUE}        → Total grid positions: {grid.height * grid.width}{Colors.ENDC}")
        print(f"{Colors.BLUE}        → AnnData spots: {adata.n_obs}{Colors.ENDC}")
        print(f"{Colors.BLUE}        → Non-zero values: {adata.X.nnz:,}{Colors.ENDC}")

        # Step 3: Filter genes to match reference
        print(f"{Colors.CYAN}  [3/5] Matching genes with reference...{Colors.ENDC}")
        gene_indices = []
        gene_mapping = {gene: idx for idx, gene in enumerate(genes)}
        adata_genes = list(adata.var_names)

        for i in tqdm(range(len(adata_genes)), desc=f"{Colors.BLUE}        Matching genes{Colors.ENDC}",
                     leave=False, ncols=80):
            gene = adata_genes[i]
            if gene in gene_mapping:
                gene_indices.append((i, gene_mapping[gene]))

        print(f"{Colors.GREEN}        ✓ Matched {len(gene_indices)}/{len(adata_genes)} genes{Colors.ENDC}")

        # Step 4: Build sparse matrix directly using COO format (memory efficient)
        print(f"{Colors.CYAN}  [4/5] Building grid matrix (memory-efficient mode)...{Colors.ENDC}")
        n_grid_spots = grid.height * grid.width

        # Get matrix in efficient format
        print(f"{Colors.BLUE}        → Preparing expression matrix...{Colors.ENDC}")
        if not sparse.isspmatrix(adata.X):
            adata_matrix = sparse.csr_matrix(adata.X)
        else:
            adata_matrix = adata.X.tocsr()
        print(f"{Colors.GREEN}        ✓ Matrix ready for processing{Colors.ENDC}")

        # Build lists for COO format (memory efficient - only store non-zero values)
        print(f"{Colors.BLUE}        → Extracting non-zero values (this avoids creating huge matrix)...{Colors.ENDC}")
        row_indices = []
        col_indices = []
        data_values = []

        # Convert gene_indices to dict for faster lookup
        gene_idx_dict = dict(gene_indices)

        # Process in batches to show detailed progress
        batch_size = 10000  # Process 10k spots at a time
        n_batches = (adata.n_obs + batch_size - 1) // batch_size

        print(f"{Colors.BLUE}        → Processing {adata.n_obs:,} spots in {n_batches} batches...{Colors.ENDC}")

        # Overall progress bar with detailed stats
        with tqdm(total=adata.n_obs,
                 desc=f"{Colors.CYAN}        Processing spots{Colors.ENDC}",
                 ncols=120,
                 bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}',
                 unit='spot') as pbar:

            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, adata.n_obs)
                batch_spots = end_idx - start_idx

                # Update batch progress info
                pbar.set_postfix_str(
                    f"Batch {batch_idx+1}/{n_batches} | Values: {len(data_values):,}",
                    refresh=True
                )

                # Process spots in this batch
                for spot_idx in range(start_idx, end_idx):
                    x = x_coords[spot_idx]
                    y = y_coords[spot_idx]
                    grid_idx = grid.width * (x - grid.x_min) + (y - grid.y_min)

                    # Get non-zero entries for this spot (efficient)
                    spot_data = adata_matrix[spot_idx, :]
                    if sparse.isspmatrix(spot_data):
                        spot_data = spot_data.tocoo()
                        adata_gene_indices = spot_data.col
                        adata_gene_values = spot_data.data
                    else:
                        spot_data_arr = spot_data.toarray().flatten()
                        adata_gene_indices = np.nonzero(spot_data_arr)[0]
                        adata_gene_values = spot_data_arr[adata_gene_indices]

                    # Map to reference genes (only process non-zero values)
                    for i, adata_gene_idx in enumerate(adata_gene_indices):
                        if adata_gene_idx in gene_idx_dict:
                            ref_gene_idx = gene_idx_dict[adata_gene_idx]
                            row_indices.append(grid_idx)
                            col_indices.append(ref_gene_idx)
                            data_values.append(adata_gene_values[i])

                    # Update progress bar
                    pbar.update(1)

        print(f"{Colors.GREEN}        ✓ Collected {len(data_values):,} non-zero values{Colors.ENDC}")

        # Step 5: Create sparse matrix from COO format (most memory efficient)
        print(f"{Colors.CYAN}  [5/5] Creating sparse matrix from {len(data_values):,} non-zero values...{Colors.ENDC}")

        # Convert lists to numpy arrays for faster sparse matrix creation
        print(f"{Colors.BLUE}        → Converting to numpy arrays...{Colors.ENDC}")
        row_indices = np.array(row_indices, dtype=np.int32)
        col_indices = np.array(col_indices, dtype=np.int32)
        data_values = np.array(data_values, dtype=np.float32)
        print(f"{Colors.GREEN}        ✓ Arrays ready{Colors.ENDC}")

        # Create sparse matrix
        print(f"{Colors.BLUE}        → Building sparse matrix...{Colors.ENDC}")
        grid.matrix = sparse.coo_matrix(
            (data_values, (row_indices, col_indices)),
            shape=(n_grid_spots, len(genes))
        ).tocsc()

        sparsity = (1 - grid.matrix.nnz / (n_grid_spots * len(genes))) * 100
        print(f"{Colors.GREEN}        ✓ Matrix created! Non-zero: {grid.matrix.nnz:,}, Sparsity: {sparsity:.2f}%{Colors.ENDC}")

        print(f"{Colors.GREEN}{EMOJI['done']} ExpressionGrid created successfully!{Colors.ENDC}")
        return grid

    def __init__(self,
                 table,
                 genes: Sequence[str],
                 gene_col: str = "gene",
                 count_col: str = "count"
                 ):
        """Inits grid with expression readingsfrom a dataframe.

        Args:
            table: A dataframe of spot-level gene counts.
                Each row in the dataframe corresponds to a reading of one
                gene at one spot.
                Columns:
                    x: The x coordinate of the reading.
                    y: The y coordinate of the reading.
                    {gene_col}: The gene detected by the reading.
                    {count_col}: The number of transcripts measured.
            genes:
                A full list of all genes under consideration, in order. This
                should match the list of genes used for other CountData
                intended to be compared with this sample.
            gene_col:
                A string labelling the column containing gene names.
            count_col:
                A string labelling the column containing transcript counts.
        """
        print(f"{Colors.CYAN}{EMOJI['grid']} Initializing expression grid...{Colors.ENDC}")
        self.x_min, self.x_max = table.x.min(), table.x.max()
        self.y_min, self.y_max = table.y.min(), table.y.max()
        self.height: int = self.x_max - self.x_min + 1
        self.width: int = self.y_max - self.y_min + 1
        num_genes = len(genes)
        print(f"{Colors.BLUE}  → Grid size: {self.height}x{self.width}, Genes: {num_genes}{Colors.ENDC}")
        matrix = sparse.lil_matrix(((self.height) * (self.width),
                                   num_genes)
                                   )
        for row in tqdm(table.itertuples(), total=len(table), desc=f"{Colors.BLUE}Building matrix{Colors.ENDC}"):
            matrix[self._flatten_coords(row.x, row.y),
                   genes.index(getattr(row, gene_col))
                   ] = getattr(row, count_col)
        self.matrix = matrix.tocsc()
        self.num_genes = cast(int, num_genes)
        print(f"{Colors.GREEN}{EMOJI['done']} Expression grid initialized!{Colors.ENDC}")

    def rows(self) -> range:
        """Returns a range of all row indices in the grid."""
        return range(self.x_min, self.x_max+1)

    def cols(self) -> range:
        """Returns a range of all column indices in the grid."""
        return range(self.y_min, self.y_max+1)

    def _flatten_coords(self, i: int, j: int) -> int:
        return self.width * (i - self.x_min) + (j - self.y_min)

    def _flatten_coords_vec(self, coords) -> npt.ArrayLike:
        return ((self.width, 1) * (coords - (self.x_min, self.y_min))).sum(axis=1)

    def expression(self, *coords: tuple[(int, int)]) -> sparse.spmatrix:
        """The total expression at these coordinates in the grid"""
        return self.matrix[list(map(lambda p: self._flatten_coords(*p),
                                    coords
                                    ))].sum(axis=0)

    def expression_vec(self, coords: npt.NDArray) -> sparse.spmatrix:
        flattened = self._flatten_coords_vec(coords)
        return self.matrix[flattened].sum(axis=0)

    def square_nbhd(self,
                    i: int,
                    j: int,
                    scale: int
                    ) -> Iterator[tuple[int, int]]:
        """All coordinates (x,y) in the grid such that d(x,i), d(y-j) <= scale"""
        return square_nbhd((i, j), scale, (self.x_min, self.x_max),
                           (self.y_min, self.y_max))

    def square_nbhd_vec(self, i: int, j: int, scale: int) -> npt.NDArray:
        return square_nbhd_vec((i, j), scale, (self.x_min, self.x_max),
                               (self.y_min, self.y_max))


class Worker(Process):

    def __init__(self,
                 grid: ExpressionGrid,
                 min_scale: int,
                 max_scale: int,
                 classifier: Classifier,
                 job_queue: Queue,
                 res_queue: Queue,
                 procid: int,
                 verbose: bool
                 ):
        super().__init__()
        self.grid = grid
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.classifier = classifier
        self.job_queue = job_queue
        self.res_queue = res_queue
        self.procid = procid
        self.verbose = verbose

    def run(self):
        simplefilter(action='ignore', category=FutureWarning)
        if self.verbose:
            print(f'{Colors.CYAN}{EMOJI["process"]} Worker {self.procid} started{Colors.ENDC}')

        num_classes = len(self.classifier.classes)

        cols = list(self.grid.cols())
        num_scales = self.max_scale - self.min_scale + 1
        exprs = np.zeros((num_scales, self.grid.num_genes))
        for i, col_values in iter(self.job_queue.get, None):
            if self.verbose:
                print(f"{Colors.BLUE}  → Worker {self.procid} processing row {i}{Colors.ENDC}")
            for col_index in col_values:
                j = cols[col_index]
                for scale in range(self.min_scale, self.max_scale + 1):
                    nbhd = self.grid.square_nbhd_vec(i, j, scale)
                    expr = self.grid.expression_vec(nbhd)
                    exprs[scale - self.min_scale] = expr

                first_nonzero = first_nonzero_1d(exprs.sum(axis=1))

                probs = np.empty((num_scales, num_classes))
                probs[:] = -1

                if 0 <= first_nonzero < num_scales:
                    to_classify = np.vstack(exprs[first_nonzero:])  # pyright: ignore # noqa: E501

                    # Use silent=True to avoid excessive output in multiprocessing
                    all_confidences = self.classifier.classify(to_classify, silent=True)

                    probs[first_nonzero:] = all_confidences

                    # sample_confidences = np.max(all_confidences, axis=1)

                    # first_confident = utf1st.find_1st(sample_confidences,
                    #                                   self.threshold,
                    #                                   utf1st.cmp_larger_eq)

                    # if 0 <= first_confident < num_scales - first_nonzero:
                    #     sample_probs = all_confidences[first_confident]
                    #     index = np.argmax(sample_probs)
                    #     result = classifier.classes[index]
                    #     scale = first_confident + first_nonzero + self.min_scale
                    # else:
                    #     result = scale = None
                    # self.res_queue.put((i, j, result, scale))
                self.res_queue.put((i, j, probs.tolist()))
        self.res_queue.put(None)
        if self.verbose:
            print(f'{Colors.GREEN}{EMOJI["done"]} Worker {self.procid} finished{Colors.ENDC}')


class CountGrid(CountTable):
    """A spatial transcriptomics object with associated methods.

    Attributes:
        grid: An expression grid.
    """

    def __init__(self, *args, **kwargs):
        """Inits spatial data with values from a dataframe."""
        # Check if a pre-built grid is provided (for memory efficiency)
        _prebuilt_grid = kwargs.pop('_prebuilt_grid', None)

        print(f"{Colors.HEADER}{EMOJI['spatial']} Initializing spatial CountGrid...{Colors.ENDC}")
        super().__init__(*args, **kwargs)

        if _prebuilt_grid is not None:
            # Use the pre-built grid (efficient path from AnnData)
            print(f"{Colors.BLUE}  → Using pre-built expression grid (skipping rebuild){Colors.ENDC}")
            self.grid = _prebuilt_grid
            self.height, self.width = self.grid.height, self.grid.width
        else:
            # Build grid from table (standard path)
            self.generate_expression_grid()
            self.height, self.width = self.grid.height, self.grid.width

        print(f"{Colors.GREEN}{EMOJI['done']} CountGrid initialized! Grid dimensions: {self.height}×{self.width}{Colors.ENDC}")

    @classmethod
    def from_coord_table(cls, table, **kwargs):
        # Check if input is AnnData object
        is_anndata = False
        try:
            from anndata import AnnData
            is_anndata = isinstance(table, AnnData)
        except ImportError:
            pass

        if is_anndata:
            print(f"{Colors.HEADER}{EMOJI['spatial']} Creating CountGrid from AnnData spatial object (efficient mode)...{Colors.ENDC}")
            adata = table
            print(f"{Colors.BLUE}AnnData info: {adata.n_obs:,} spots × {adata.n_vars:,} genes{Colors.ENDC}")

            # Extract genes from kwargs
            if 'genes' not in kwargs:
                raise ValueError("Must provide 'genes' parameter when using AnnData input")

            genes = kwargs['genes']
            print(f"{Colors.BLUE}Reference genes: {len(genes):,}{Colors.ENDC}")

            # Extract spatial coordinates
            print(f"\n{Colors.CYAN}{'─'*60}{Colors.ENDC}")
            print(f"{Colors.CYAN}Step 1: Extracting spatial information{Colors.ENDC}")
            print(f"{Colors.CYAN}{'─'*60}{Colors.ENDC}")

            if 'spatial' in adata.obsm:
                print(f"{Colors.BLUE}  → Source: adata.obsm['spatial']{Colors.ENDC}")
                spatial_coords = adata.obsm['spatial']
                x_coords_array = np.round(spatial_coords[:, 0]).astype(int)
                y_coords_array = np.round(spatial_coords[:, 1]).astype(int)
            elif 'x' in adata.obs.columns and 'y' in adata.obs.columns:
                print(f"{Colors.BLUE}  → Source: adata.obs['x'] and adata.obs['y']{Colors.ENDC}")
                x_coords_array = np.round(adata.obs['x'].values).astype(int)
                y_coords_array = np.round(adata.obs['y'].values).astype(int)
            else:
                raise ValueError("AnnData object must have spatial coordinates in adata.obsm['spatial'] or adata.obs['x'/'y']")

            # Get coordinate bounds
            x_min, x_max = int(x_coords_array.min()), int(x_coords_array.max())
            y_min, y_max = int(y_coords_array.min()), int(y_coords_array.max())

            print(f"{Colors.GREEN}  ✓ Coordinate range: X[{x_min}, {x_max}], Y[{y_min}, {y_max}]{Colors.ENDC}")

            # Calculate grid dimensions (but don't create all positions!)
            print(f"\n{Colors.CYAN}Step 2: Calculating grid dimensions{Colors.ENDC}")
            grid_height = x_max - x_min + 1
            grid_width = y_max - y_min + 1
            total_grid_positions = grid_height * grid_width
            print(f"{Colors.GREEN}  ✓ Grid dimensions: {grid_height} × {grid_width} = {total_grid_positions:,} positions{Colors.ENDC}")
            print(f"{Colors.BLUE}  → Occupied spots: {adata.n_obs:,} ({adata.n_obs/total_grid_positions*100:.2f}%){Colors.ENDC}")
            print(f"{Colors.WARNING}  ⚠ Only creating sample IDs for {adata.n_obs:,} actual spots (not all {total_grid_positions:,} positions){Colors.ENDC}")

            # Create ExpressionGrid directly from AnnData (efficient!)
            print(f"\n{Colors.CYAN}{'─'*60}{Colors.ENDC}")
            print(f"{Colors.CYAN}Step 3: Building expression grid (this is the main step){Colors.ENDC}")
            print(f"{Colors.CYAN}{'─'*60}{Colors.ENDC}")
            expression_grid = ExpressionGrid.from_anndata(adata, genes)

            # Create sample IDs ONLY for actual spots (memory efficient!)
            print(f"\n{Colors.CYAN}Step 4: Creating sample IDs for actual spots{Colors.ENDC}")
            print(f"{Colors.BLUE}  → Generating {adata.n_obs:,} sample IDs...{Colors.ENDC}")
            samples = [f'{x},{y}' for x, y in zip(x_coords_array, y_coords_array)]
            print(f"{Colors.GREEN}  ✓ Created {len(samples):,} sample IDs{Colors.ENDC}")

            # Create table with actual data
            print(f"\n{Colors.CYAN}Step 5: Initializing CountGrid structure{Colors.ENDC}")
            print(f"{Colors.BLUE}  → Building data table...{Colors.ENDC}")

            # Build table from actual spots (not a minimal table)
            table_data = {
                'sample': samples,
                'gene': [genes[0]] * adata.n_obs,  # Placeholder gene
                'count': [0] * adata.n_obs,  # Placeholder counts
                'x': x_coords_array,
                'y': y_coords_array
            }
            full_table = pd.DataFrame(table_data)
            print(f"{Colors.GREEN}  ✓ Table created with {len(full_table):,} rows{Colors.ENDC}")

            # Set kwargs for table-based initialization
            kwargs['genes'] = genes
            kwargs['gene_col'] = 'gene'
            kwargs['sample_col'] = 'sample'
            kwargs['count_col'] = 'count'
            # CRITICAL: Pass the pre-built grid to avoid rebuilding (memory efficient!)
            kwargs['_prebuilt_grid'] = expression_grid

            # Create CountGrid using actual spots
            # The __init__ will use the pre-built grid instead of rebuilding
            print(f"{Colors.BLUE}  → Initializing CountGrid with pre-built grid...{Colors.ENDC}")
            count_grid = cls(full_table, samples=samples, **kwargs)
            print(f"{Colors.GREEN}  ✓ CountGrid initialized (using efficient pre-built grid){Colors.ENDC}")

            # Metadata already includes x, y from the table
            print(f"{Colors.GREEN}  ✓ Spatial metadata already included{Colors.ENDC}")

            print(f"\n{Colors.GREEN}{'='*60}{Colors.ENDC}")
            print(f"{Colors.GREEN}{EMOJI['done']} CountGrid created successfully!{Colors.ENDC}")
            print(f"{Colors.GREEN}{'='*60}{Colors.ENDC}")
            print(f"{Colors.BLUE}Final grid dimensions: {count_grid.height} × {count_grid.width}{Colors.ENDC}")
            print(f"{Colors.BLUE}Matrix shape: {count_grid.grid.matrix.shape}{Colors.ENDC}")
            print(f"{Colors.BLUE}Non-zero values: {count_grid.grid.matrix.nnz:,}{Colors.ENDC}")
            print()
            return count_grid

        # Original DataFrame path
        x_coords = range(table.x.min(), table.x.max() + 1)
        y_coords = range(table.y.min(), table.y.max() + 1)

        coords = itertools.product(x_coords, y_coords)
        samples = map(combine_coords, coords)

        new_table = table.copy()
        new_table['sample'] = new_table.apply(lambda row: f'{row.x},{row.y}',
                                              axis=1)

        count_grid = cls(new_table, samples=list(samples), **kwargs)
        samples = count_grid.samples
        x_coords = {sample: first_coord(sample) for sample in samples}
        y_coords = {sample: second_coord(sample) for sample in samples}
        count_grid.add_metadata('x', x_coords)
        count_grid.add_metadata('y', y_coords)
        return count_grid

    def pseudobulk(self) -> npt.NDArray:
        return np.array(self.grid.matrix.sum(axis=0))[0]

    def count_matrix(self) -> npt.NDArray:
        # It's tempting to try and do something clever with numpy or pandas
        # here. There be dragons.
        x_min = self.table.x.min()
        y_min = self.table.y.min()
        count_matrix = np.zeros((self.width, self.height))
        counts = self.table.groupby(['x', 'y']).sum().reset_index()
        count_index = self.table.columns.get_loc(self.count_col)
        for row in counts.itertuples():
            count_matrix[row.y-y_min, row.x-x_min] += row[count_index]
        return count_matrix

    def density_mask(self, radius: int, threshold: int) -> npt.NDArray:
        return density_hull(self.count_matrix(), radius, threshold)

    def generate_expression_grid(self):
        # Skip if grid already exists (e.g., created from AnnData)
        if hasattr(self, 'grid') and self.grid is not None:
            print(f"{Colors.BLUE}  → Using existing ExpressionGrid{Colors.ENDC}")
            return

        self.grid = ExpressionGrid(self.table,
                                   genes=self.genes,
                                   gene_col=self.gene_col,
                                   count_col=self.count_col
                                   )

    def classify_parallel(self,
                          classifier: Classifier,
                          min_scale: int,
                          max_scale: int,
                          outfile: str,
                          mask: npt.NDArray | None = None,
                          num_proc: int = 1,
                          verbose: bool = False
                          ):

        print(f"{Colors.HEADER}{EMOJI['spatial']} Starting parallel spatial classification...{Colors.ENDC}")
        print(f"{Colors.CYAN}  → Scales: {min_scale}-{max_scale}, Processes: {num_proc}{Colors.ENDC}")

        outfile += '' if outfile[-4:] == '.npy' else '.npy'
        shape = (self.grid.height,
                 self.grid.width,
                 max_scale - min_scale + 1,
                 len(classifier.classes)
                 )
        print(f"{Colors.BLUE}  → Output shape: {shape}, File: {outfile}{Colors.ENDC}")
        result = np.lib.format.open_memmap(outfile, dtype=np.float32,
                                           mode='w+', shape=shape)
        result[:] = np.nan
        result.flush()

        job_queue: Queue[tuple[int, list[int]] | None] = Queue()
        res_queue: Queue[tuple[int, int, list[list[int]]] | None] = Queue()

        if mask is not None:
            mask = mask.T
            if mask.shape != (self.height, self.width):
                raise ValueError(f'Mask has shape {mask.shape} but expected {(self.height, self.width)}')
            col_values = [[j for j in range(self.width) if mask[i, j] == 1] for i in range(self.height)]
            print(f"{Colors.WARNING}  → Using mask for spot selection{Colors.ENDC}")
        else:
            col_values = [list(range(self.width)) for _ in range(self.height)]

        print(f"{Colors.CYAN}{EMOJI['process']} Launching {num_proc} worker processes...{Colors.ENDC}")
        workers = []
        for i in range(num_proc):
            worker = Worker(self.grid,
                           min_scale,
                           max_scale,
                           classifier,
                           job_queue,
                           res_queue,
                           i,
                           verbose
                           )
            worker.start()
            workers.append(worker)
            print(f"{Colors.BLUE}  → Worker {i+1}/{num_proc} started{Colors.ENDC}")

        print(f"\n{Colors.CYAN}{EMOJI['grid']} Preparing classification jobs...{Colors.ENDC}")
        num_spots = 0
        num_rows = 0

        for i in self.grid.rows():
            index = i - self.grid.x_min
            if len(col_values[index]) > 0:
                job_queue.put((i, col_values[index]))
                num_spots += len(col_values[index])
                num_rows += 1

        for _ in range(num_proc):
            job_queue.put(None)

        print(f"{Colors.GREEN}  ✓ Queued {num_rows:,} rows with {num_spots:,} spots for classification{Colors.ENDC}")
        print(f"{Colors.BLUE}  → Scales to process: {max_scale - min_scale + 1}{Colors.ENDC}")
        print(f"{Colors.BLUE}  → Classes: {len(classifier.classes)}{Colors.ENDC}")
        print(f"{Colors.BLUE}  → Total classifications: {num_spots:,} × {max_scale - min_scale + 1} × {len(classifier.classes)} = {num_spots * (max_scale - min_scale + 1) * len(classifier.classes):,}{Colors.ENDC}")

        print(f"\n{Colors.CYAN}{EMOJI['classify']} Waiting for workers to process jobs...{Colors.ENDC}")
        print(f"{Colors.WARNING}  ⏳ This may take a few seconds to start (workers are initializing)...{Colors.ENDC}")

        # Wait for first result to confirm workers are running
        import time
        start_time = time.time()
        first_result_received = False

        pbar = tqdm(total=num_spots,
                   desc=f"{Colors.CYAN}Classifying spots{Colors.ENDC}",
                   ncols=120,
                   bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}',
                   unit='spot')

        processed_spots = 0
        flush_counter = 0

        for msg_index in range(num_spots + num_proc):
            res = res_queue.get()

            # Announce when first result arrives
            if not first_result_received:
                elapsed = time.time() - start_time
                print(f"\n{Colors.GREEN}  ✓ Workers started processing (initialization took {elapsed:.1f}s){Colors.ENDC}\n")
                first_result_received = True

            if res:
                i, j, probs = res
                result[i-self.grid.x_min, j-self.grid.y_min] = probs
                processed_spots += 1
                pbar.update(1)

                # Update postfix every 1000 spots
                if processed_spots % 1000 == 0:
                    pbar.set_postfix_str(f"Flushed: {flush_counter}x", refresh=True)

                # Flush to disk periodically
                if msg_index % 5000 == 0:
                    result.flush()
                    flush_counter += 1
                    if verbose:
                        print(f"{Colors.BLUE}  → Flushed to disk ({processed_spots:,}/{num_spots:,} spots completed){Colors.ENDC}")

        pbar.close()
        result.flush()

        print(f"\n{Colors.GREEN}{EMOJI['done']} Classification completed!{Colors.ENDC}")
        print(f"{Colors.BLUE}  → Processed: {processed_spots:,} spots{Colors.ENDC}")
        print(f"{Colors.BLUE}  → Results saved to: {outfile}{Colors.ENDC}")
        print(f"{Colors.BLUE}  → File size: ~{result.nbytes / (1024**2):.1f} MB{Colors.ENDC}")

    def annotate(self,
                 confidence_matrix: npt.NDArray,
                 threshold: float,
                 labels: tuple[str, ...],
                 column_label: str = "cell type"):

        print(f"{Colors.CYAN}{EMOJI['cell']} Annotating spots with threshold={threshold}...{Colors.ENDC}")
        classifications = extract_classifications(confidence_matrix, threshold)
        print(f"{Colors.BLUE}  → Found {len(classifications)} confident classifications{Colors.ENDC}")
        x_min, y_min = self.grid.x_min, self.grid.y_min
        to_add = {combine_coords((x+x_min, y+y_min)): labels[c]
                  for (x, y), c in classifications.items()
                  }

        self.add_metadata(column_label, to_add)
        print(f"{Colors.GREEN}{EMOJI['done']} Annotation completed! Column '{column_label}' added to metadata{Colors.ENDC}")
