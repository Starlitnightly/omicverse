"""Classes and methods for array-based spatial transcriptomics analysis."""

import itertools
from typing import Iterable, Iterator, Sequence, cast
from multiprocessing import Process, Queue
from warnings import simplefilter

import numpy as np
import numpy.typing as npt
from scipy import sparse
from tqdm import tqdm
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

                    all_confidences = self.classifier.classify(to_classify)

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
        print(f"{Colors.HEADER}{EMOJI['spatial']} Initializing spatial CountGrid...{Colors.ENDC}")
        super().__init__(*args, **kwargs)
        self.generate_expression_grid()
        self.height, self.width = self.grid.height, self.grid.width
        print(f"{Colors.GREEN}{EMOJI['done']} CountGrid initialized! Grid dimensions: {self.height}×{self.width}{Colors.ENDC}")

    @classmethod
    def from_coord_table(cls, table, **kwargs):
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
        for i in range(num_proc):
            Worker(self.grid,
                   min_scale,
                   max_scale,
                   classifier,
                   job_queue,
                   res_queue,
                   i,
                   verbose
                   ).start()

        num_spots = 0

        for i in self.grid.rows():
            index = i - self.grid.x_min
            if len(col_values[index]) > 0:
                job_queue.put((i, col_values[index]))
                num_spots += len(col_values[index])

        for _ in range(num_proc):
            job_queue.put(None)

        print(f"{Colors.CYAN}{EMOJI['classify']} Classifying {num_spots} spots...{Colors.ENDC}")
        pbar = tqdm(total=num_spots, desc=f"{Colors.BLUE}Processing spots{Colors.ENDC}")

        processed_spots = 0
        for msg_index in range(num_spots + num_proc):
            res = res_queue.get()
            if res:
                i, j, probs = res
                result[i-self.grid.x_min, j-self.grid.y_min] = probs
                processed_spots += 1
                pbar.update(1)
                if msg_index % 5000 == 0:
                    result.flush()
                    if verbose:
                        print(f"{Colors.BLUE}  → Flushed to disk ({processed_spots}/{num_spots} spots){Colors.ENDC}")

        pbar.close()
        result.flush()
        print(f"{Colors.GREEN}{EMOJI['done']} Classification completed! Results saved to {outfile}{Colors.ENDC}")

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
