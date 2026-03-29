from __future__ import annotations

import logging
from pathlib import Path

import scanpy as sc
import scvi
from rich.console import Console
from rich.logging import RichHandler

popv_logger = logging.getLogger("popv")


class Config:
    """
    Config manager for PopV.

    Examples
    --------
    To set the seed

    >>> popv.settings.seed = 1

    To set the verbosity

    >>> import logging
    >>> popv.settings.verbosity = logging.INFO

    To set the logging directory

    >>> popv.settings.logging_dir = "./popv_log/"

    To set the number of jobs to be used

    >>> popv.settings.n_jobs = 2

    To set the number of largest dense dataset to be used

    >>> popv.settings.shard_size = 200000

    To enable cuml for rapid GPU based methods

    >>> popv.settings.cuml = True

    To recompute embeddings instead of using NN lookup

    >>> popv.settings.recompute_embeddings = True

    To not return probabilities for all classifiers

    >>> popv.settings.return_probabilities = False

    To not compute UMAP embedding for integration methods

    >>> popv.settings.compute_umap_embedding = False

    """

    def __init__(
        self,
        verbosity: int = logging.WARNING,
        seed: int | None = None,
        logging_dir: str = "./popv_log/",
        n_jobs: int = 1,
        cuml: bool = False,
        accelerator: str = "auto",
        shard_size: int = 100000,
        recompute_embeddings: bool = False,
        return_probabilities: bool = True,
        compute_umap_embedding: bool = True,
    ):
        """Set up Config manager for PopV."""
        self.seed = seed
        self.logging_dir = logging_dir
        self.verbosity = verbosity
        self.n_jobs = n_jobs
        self.cuml = cuml
        self.accelerator = accelerator
        self.shard_size = shard_size
        self.recompute_embeddings = recompute_embeddings
        self.return_probabilities = return_probabilities
        self.compute_umap_embedding = compute_umap_embedding

    @property
    def logging_dir(self) -> Path:
        """Directory for training logs (default `'./popv_log/'`)."""
        return self._logging_dir

    @logging_dir.setter
    def logging_dir(self, logging_dir: str | Path):
        self._logging_dir = Path(logging_dir).resolve()

    @property
    def n_jobs(self) -> int:
        """Jobs used for multiprocessing."""
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, n_jobs: int):
        """Random seed for torch and numpy."""
        sc.settings.n_jobs = n_jobs
        self._n_jobs = n_jobs

    @property
    def cuml(self) -> int:
        """Use RAPIDS and cuml."""
        return self._cuml

    @cuml.setter
    def cuml(self, cuml: bool):
        """Use RAPIDS and cuml."""
        self._cuml = cuml

    @property
    def shard_size(self) -> int:
        """Maximum number of cells in dense arrays."""
        return self._shard_size

    @shard_size.setter
    def shard_size(self, shard_size: int):
        """Maximum number of cells in dense arrays."""
        self._shard_size = shard_size

    @property
    def seed(self) -> int:
        """Random seed for torch and numpy."""
        return self._seed

    @seed.setter
    def seed(self, seed: int):
        """Random seed for torch and numpy."""
        scvi.settings.seed = seed
        self._seed = seed

    @property
    def verbosity(self) -> int:
        """Verbosity level (default `logging.INFO`)."""
        return self._verbosity

    @verbosity.setter
    def verbosity(self, level: str | int):
        """
        Set logging configuration for popV based on chosen level of verbosity.

        Parameters
        ----------
        level
            Sets "popV" logging level to `level`
        force_terminal
            Rich logging option, set to False if piping to file output.
        """
        self._verbosity = level
        popv_logger.setLevel(level)
        if len(popv_logger.handlers) == 0:
            console = Console(force_terminal=True)
            if console.is_jupyter is True:
                console.is_jupyter = False
            ch = RichHandler(level=level, show_path=False, console=console, show_time=False)
            formatter = logging.Formatter("%(message)s")
            ch.setFormatter(formatter)
            popv_logger.addHandler(ch)
        else:
            popv_logger.setLevel(level)

    @property
    def accelerator(self) -> bool:
        """Accelerator for scvi-tools models."""
        return self._accelerator

    @accelerator.setter
    def accelerator(self, accelerator: str):
        self._accelerator = accelerator

    @property
    def recompute_embeddings(self) -> bool:
        """Recompute UMAPs and BBKNN and Harmony."""
        return self._recompute_embeddings

    @recompute_embeddings.setter
    def recompute_embeddings(self, recompute_embeddings: bool):
        self._recompute_embeddings = recompute_embeddings

    @property
    def compute_umap_embedding(self) -> bool:
        """Compute UMAP embedding for integration methods."""
        return self._compute_umap_embedding

    @compute_umap_embedding.setter
    def compute_umap_embedding(self, compute_umap_embedding: bool):
        self._compute_umap_embedding = compute_umap_embedding

    @property
    def return_probabilities(self) -> bool:
        """Return internal certainties for all classifiers."""
        return self._return_probabilities

    @return_probabilities.setter
    def return_probabilities(self, return_probabilities: bool):
        self._return_probabilities = return_probabilities


settings = Config()
