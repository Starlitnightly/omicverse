# -*- coding: utf-8 -*-

import logging
import os
import pickle
import re
import tempfile
from functools import partial
from math import ceil

# Using multiprocessing using dill package for pickling to avoid strange bugs.
from multiprocessing import cpu_count
from operator import concat
from typing import Callable, Sequence, Type, TypeVar

import pandas as pd
from boltons.iterutils import chunked_iter
#from ctxcore.genesig import GeneSignature, Regulon
#from ctxcore.rnkdb import MemoryDecorator, RankingDatabase
from multiprocess.connection import Pipe
from multiprocess.context import Process

from .log import create_logging_handler
from .transform import (
    get_df_meta_data,
    df2regulons,
    module2features_auc1st_impl,
    modules2df,
    modules2regulons,
)
from .utils import add_motif_url, load_motif_annotations

__all__ = ["prune2df", "find_features", "df2regulons"]


# Taken from: https://www.regular-expressions.info/ip.html
IP_PATTERN = re.compile(
    r"""(25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])\."""
    r"""(25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])\."""
    r"""(25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])\."""
    r"""(25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9]):\d+"""
)

LOGGER = logging.getLogger(__name__)


def _prepare_client(client_or_address, num_workers):
    """
    :param client_or_address: one of:
           * None
           * verbatim: 'local'
           * string address
           * a Client instance
    :return: a tuple: (Client instance, shutdown callback function).
    :raises: ValueError if no valid client input was provided.
    """
    # Credits to Thomas Moerman (arboreto package):
    # https://github.com/tmoerman/arboreto/blob/482ce8598da5385eb0e01a50362cb2b1e6f66a41/arboreto/algo.py#L145-L191

    from dask.distributed import Client, LocalCluster

    if client_or_address is None or str(client_or_address).lower() == "local":
        local_cluster = LocalCluster(n_workers=num_workers, threads_per_worker=1)
        client = Client(local_cluster)

        def close_client_and_local_cluster(verbose=False):
            #if verbose:
                #LOGGER.info("shutting down client and local cluster")

            client.close()
            local_cluster.close()

        return client, close_client_and_local_cluster

    elif isinstance(client_or_address, str) and client_or_address.lower() != "local":
        client = Client(client_or_address)

        def close_client(verbose=False):
            #if verbose:
            #    LOGGER.info("shutting down client")

            client.close()

        return client, close_client

    elif isinstance(client_or_address, Client):

        def close_dummy(verbose=False):
            #if verbose:
             #   LOGGER.info("not shutting down client, client was created externally")

            return None

        return client_or_address, close_dummy

    else:
        raise ValueError("Invalid client specified {}".format(str(client_or_address)))


class Worker(Process):
    def __init__(
        self,
        name: str,
        db,
        modules,
        motif_annotations_fname: str,
        sender,
        motif_similarity_fdr: float,
        orthologuous_identity_threshold: float,
        transformation_func,
    ):
        super().__init__(name=name)
        self.database = db
        self.modules = modules
        self.motif_annotations_fname = motif_annotations_fname
        self.motif_similarity_fdr = motif_similarity_fdr
        self.orthologuous_identity_threshold = orthologuous_identity_threshold
        self.transform_fnc = transformation_func
        self.sender = sender

    def run(self):
        # Load ranking database in memory.
        from ..ctxcore.rnkdb import MemoryDecorator, RankingDatabase
        rnkdb = MemoryDecorator(self.database)
        #LOGGER.info("Worker {}: database loaded in memory.".format(self.name))

        # Load motif annotations in memory.
        motif_annotations = load_motif_annotations(
            self.motif_annotations_fname,
            motif_similarity_fdr=self.motif_similarity_fdr,
            orthologous_identity_threshold=self.orthologuous_identity_threshold,
        )
        #LOGGER.info("Worker {}: motif annotations loaded in memory.".format(self.name))

        # Apply transformation on all modules.
        output = self.transform_fnc(
            rnkdb, self.modules, motif_annotations=motif_annotations
        )
        #LOGGER.info("Worker {}: All regulons derived.".format(self.name))

        # Sending information back to parent process: to avoid overhead of pickling the data, the output is first written
        # to disk in binary pickle format to a temporary file. The name of that file is shared with the parent process.
        output_fname = tempfile.mktemp()
        with open(output_fname, "wb") as f:
            pickle.dump(output, f)
        del output
        self.sender.send(output_fname)
        self.sender.close()
        #LOGGER.info("Worker {}: Done.".format(self.name))


T = TypeVar("T")


def _distributed_calc(
    rnkdbs,
    modules,
    motif_annotations_fname: str,
    transform_func,
    aggregate_func: Callable[[Sequence[T]], T],
    motif_similarity_fdr: float = 0.001,
    orthologuous_identity_threshold: float = 0.0,
    client_or_address="custom_multiprocessing",
    num_workers=None,
    module_chunksize=100,
) -> T:
    """
    Perform a parallelized or distributed calculation, either pruning targets or finding enriched motifs.

    :param rnkdbs: A sequence of ranking databases.
    :param modules: A sequence of gene signatures.
    :param motif_annotations_fname: The filename of the motif annotations to use.
    :param transform_func: A function having a signature (Type[RankingDatabase], Sequence[Type[GeneSignature]], str) and
        that returns Union[Sequence[Regulon]],pandas.DataFrame].
    :param aggregate_func: A function having a signature:
        - (Sequence[pandas.DataFrame]) => pandas.DataFrame
        - (Sequence[Sequence[Regulon]]) => Sequence[Regulon]
    :param motif_similarity_fdr: The maximum False Discovery Rate to find factor annotations for enriched motifs.
    :param orthologuous_identity_threshold: The minimum orthologuous identity to find factor annotations
        for enriched motifs.
    :param client_or_address: The client of IP address of the scheduler when working with dask. For local multi-core
        systems 'custom_multiprocessing' or 'dask_multiprocessing' can be supplied.
    :param num_workers: If not using a cluster, the number of workers to use for the calculation.
        None of all available CPUs need to be used.
    :param module_chunksize: The size of the chunk in signatures to use when using the dask framework with the
        multiprocessing scheduler.
    :return: A pandas dataframe or a sequence of regulons (depends on aggregate function supplied).
    """
    
    '''
    def is_valid(client_or_address):
        if isinstance(client_or_address, str) and (
            (
                client_or_address
                in {"custom_multiprocessing", "dask_multiprocessing", "local"}
            )
            or IP_PATTERN.fullmatch(client_or_address)
        ):
            return True
        elif isinstance(client_or_address, Client):
            return True
        return False

    assert is_valid(
        client_or_address
    ), '"{}"is not valid for parameter client_or_address.'.format(client_or_address)
    '''

    if client_or_address not in {"custom_multiprocessing", "dask_multiprocessing"}:
        module_chunksize = 1

    # Make sure warnings and info are being logged.
    if not len(LOGGER.handlers):
        LOGGER.addHandler(create_logging_handler(False))
        if LOGGER.getEffectiveLevel() > logging.INFO:
            LOGGER.setLevel(logging.INFO)

    if (
        client_or_address == "custom_multiprocessing"
    ):  # CUSTOM parallelized implementation.
        # This implementation overcomes the I/O-bounded performance. Each worker (subprocess) loads a dedicated ranking
        # database and motif annotation table into its own memory space before consuming module. The implementation of
        # each worker uses the AUC-first numba JIT based implementation of the algorithm.
        assert (
            len(rnkdbs) <= num_workers if num_workers else cpu_count()
        ), "The number of databases is larger than the number of cores."
        amplifier = int((num_workers if num_workers else cpu_count()) / len(rnkdbs))
        #("Using {} workers.".format(len(rnkdbs) * amplifier))
        receivers = []
        for db in rnkdbs:
            for idx, chunk in enumerate(
                chunked_iter(modules, ceil(len(modules) / float(amplifier)))
            ):
                sender, receiver = Pipe()
                receivers.append(receiver)
                Worker(
                    "{}({})".format(db.name, idx + 1),
                    db,
                    chunk,
                    motif_annotations_fname,
                    sender,
                    motif_similarity_fdr,
                    orthologuous_identity_threshold,
                    transform_func,
                ).start()
        # Retrieve the name of the temporary file to which the data is stored. This is a blocking operation.
        fnames = [recv.recv() for recv in receivers]
        # Load all data from disk and concatenate.
        def load(fname):
            with open(fname, "rb") as f:
                return pickle.load(f)

        try:
            return aggregate_func(list(map(load, fnames)))
        finally:
            # Remove temporary files.
            for fname in fnames:
                os.remove(fname)
    else:  # DASK framework.
        from dask import delayed
        from dask.distributed import Client

        
        # Load motif annotations.
        motif_annotations = load_motif_annotations(
            motif_annotations_fname,
            motif_similarity_fdr=motif_similarity_fdr,
            orthologous_identity_threshold=orthologuous_identity_threshold,
        )

        # Create dask graph.
        def create_graph(client=None):
            # NOTE ON CHUNKING SIGNATURES:
            # Chunking the gene signatures might not be necessary anymore because the overhead of the dask
            # scheduler is minimal (cf. blog http://matthewrocklin.com/blog/work/2016/05/05/performant-task-scheduling).
            # The original behind the decision to implement this was the refuted assumption that fast executing tasks
            # would greatly be impacted by scheduler overhead. The performance gain introduced by chunking of signatures
            # seemed to corroborate this assumption. However, the benefit was through less pickling and unpickling of
            # the motif annotations dataframe as this was not wrapped in a delayed() construct.
            # When using a distributed scheduler chunking even has a negative impact and is therefore overruled. The
            # negative impact is due to having these large chunks to be shipped to different workers across cluster nodes.

            # NOTE ON BROADCASTING DATASET:
            # There are three large pieces of data that need to be orchestrated between client/scheduler and workers:
            # 1. In a cluster the motif annotations need to be broadcasted to all nodes. Otherwise
            # the motif annotations need to wrapped in a delayed() construct to avoid needless pickling and
            # unpicking between processes.
            def wrap(data):
                return (
                    client.scatter(data, broadcast=True)
                    if client
                    else delayed(data, pure=True)
                )

            delayed_or_future_annotations = wrap(motif_annotations)
            # 2. The databases: these database objects are typically proxies to the data on disk. They only have
            # the name and location on shared storage as fields. For consistency reason we do broadcast these database
            # objects to the workers. If we decide to have all information of a database loaded into memory we can still
            # safely use clusters.
            # def memoize(db: Type[RankingDatabase]) -> Type[RankingDatabase]:
            #    return MemoryDecorator(db)
            # delayed_or_future_dbs = list(map(wrap, map(memoize, rnkdbs)))
            # Check also latest Stackoverflow message: https://stackoverflow.com/questions/50795901/dask-scatter-broadcast-a-list
            delayed_or_future_dbs = list(map(wrap, rnkdbs))
            # 3. The gene signatures: these signatures become large when chunking them, therefore chunking is overruled
            # when using dask.distributed.
            # See earlier.

            # NOTE ON SHARING RANKING DATABASES ACROSS NODES:
            # Because the frontnodes of the VSC share the staging disk, these databases can be accessed from all nodes
            # in the cluster and can all use the same path in the configuration file. The RankingDatabase objects shared
            # from scheduler to workers can therefore be just contain information on database file location.
            # There might be a need to be able to run on clusters that do not share a network drive. This can be
            # achieved via by loading all data in from the scheduler and use the broadcasting system to share data
            # across nodes. The only element that needs to be adapted to cater for this need is loading the databases
            # in memory on the scheduler via the already available MemoryDecorator for databases. But make sure the
            # adapt memory limits for workers to avoid "distributed.nanny - WARNING - Worker exceeded 95% memory budget.".

            # NOTE ON REMOVING I/O CONTENTION:
            # A potential improvement to reduce I/O contention for this shared drive (accessing the ranking
            # database) would be to load the database in memory (using the available decorator) for each task.
            # The penalty of loading the database in memory should be shared across multiple gene signature so
            # in this case chunking of gene signatures is mandatory to avoid severe performance penalties.
            # However, because of the memory need of a node running pyscenic is already high (i.e. pre-allocation
            # of recovery curves - 20K features (max. enriched) * rank_threshold * 8 bytes (float) * num_cores),
            # this might not be a sound idea to do.
            # Another approach to overcome the I/O bottleneck in a clustered infrastructure is to assign each cluster
            # to a different database which is achievable in the dask framework. This approach has of course many
            # limitations: for 6 database you need at least 6 cores and you cannot take advantage of more
            # (http://distributed.readthedocs.io/en/latest/locality.html)

            # NOTE ON REMAINING WARNINGS:
            # >> distributed.worker - WARNING - Memory use is high but worker has no data to store to disk.
            # >> Perhaps some other process is leaking memory?  Process memory: 1.51 GB -- Worker memory limit: 2.15 GB
            # My current idea is that this cannot be avoided processing a single module can sometimes required
            # substantial amount of memory because of pre-allocation of recovery curves (see code notes on how to
            # mitigate this problem). Setting module_chunksize=1 also limits this problem.
            #
            # >> distributed.utils_perf - WARNING - full garbage collections took 10% CPU time recently (threshold: 10%)
            # The current implementation of module2df removes substantial amounts of memory (i.e. the RCCs) so this might
            # again be unavoidable. TBI + See following stackoverflow question:
            # https://stackoverflow.com/questions/47776936/why-is-a-computation-much-slower-within-a-dask-distributed-worker

            return aggregate_func(
                (
                    delayed(transform_func)(db, gs_chunk, delayed_or_future_annotations)
                    for db in delayed_or_future_dbs
                    for gs_chunk in chunked_iter(modules, module_chunksize)
                )
            )

        # Compute dask graph ...
        if client_or_address == "dask_multiprocessing":
            # ... via multiprocessing.
            return create_graph().compute(
                scheduler="processes",
                num_workers=num_workers if num_workers else cpu_count(),
            )
        else:
            # ... via dask.distributed framework.
            client, shutdown_callback = _prepare_client(
                client_or_address,
                num_workers=num_workers if num_workers else cpu_count(),
            )
            try:
                return client.compute(create_graph(client), sync=True)
            finally:
                shutdown_callback(False)


def prune2df(
    rnkdbs,
    modules,
    motif_annotations_fname: str,
    rank_threshold: int = 1500,
    auc_threshold: float = 0.05,
    nes_threshold=3.0,
    motif_similarity_fdr: float = 0.001,
    orthologuous_identity_threshold: float = 0.0,
    weighted_recovery=False,
    client_or_address="dask_multiprocessing",
    num_workers=None,
    module_chunksize=100,
    filter_for_annotation=True,
) -> pd.DataFrame:
    """
    Calculate all regulons for a given sequence of ranking databases and a sequence of co-expression modules.
    The number of regulons derived from the supplied modules is usually much lower. In addition, the targets of the
    retained modules is reduced to only these ones for which a cis-regulatory footprint is present.

    :param rnkdbs: The sequence of databases.
    :param modules: The sequence of modules.
    :param motif_annotations_fname: The name of the file that contains the motif annotations to use.
    :param rank_threshold: The total number of ranked genes to take into account when creating a recovery curve.
    :param auc_threshold: The fraction of the ranked genome to take into account for the calculation of the
        Area Under the recovery Curve.
    :param nes_threshold: The Normalized Enrichment Score (NES) threshold to select enriched features.
    :param motif_similarity_fdr: The maximum False Discovery Rate to find factor annotations for enriched motifs.
    :param orthologuous_identity_threshold: The minimum orthologuous identity to find factor annotations
        for enriched motifs.
    :param weighted_recovery: Use weights of a gene signature when calculating recovery curves?
    :param num_workers: If not using a cluster, the number of workers to use for the calculation.
        None of all available CPUs need to be used.
    :param module_chunksize: The size of the chunk to use when using the dask framework.
    :param client_or_address: The client of IP address of the scheduler when working with dask. For local multi-core
        systems 'custom_multiprocessing' or 'dask_multiprocessing' can be supplied.
    :return: A dataframe.
    """
    # Always use module2features_auc1st_impl not only because of speed impact but also because of reduced memory footprint.
    module2features_func = partial(
        module2features_auc1st_impl,
        rank_threshold=rank_threshold,
        auc_threshold=auc_threshold,
        nes_threshold=nes_threshold,
        filter_for_annotation=filter_for_annotation,
    )
    transformation_func = partial(
        modules2df,
        module2features_func=module2features_func,
        weighted_recovery=weighted_recovery,
    )
    # Create a distributed dataframe from individual delayed objects to avoid out of memory problems.
    aggregation_func = (
        partial(_from_delayed_with_meta, meta=get_df_meta_data())
        if client_or_address != "custom_multiprocessing"
        else pd.concat
    )
    return _distributed_calc(
        rnkdbs,
        modules,
        motif_annotations_fname,
        transformation_func,
        aggregation_func,
        motif_similarity_fdr,
        orthologuous_identity_threshold,
        client_or_address,
        num_workers,
        module_chunksize,
    )

def _from_delayed_with_meta(delayed_objs, meta):
    """Helper function to import from_delayed when needed."""
    from dask.dataframe import from_delayed
    return from_delayed(delayed_objs, meta=meta)


def find_features(
    rnkdbs,
    signatures,
    motif_annotations_fname: str,
    motif_base_url: str = "http://motifcollections.aertslab.org/v9/logos/",
    **kwargs
) -> pd.DataFrame:
    """
    Find enriched features for gene signatures.

    :param rnkdbs: The sequence of databases.
    :param signatures: The sequence of gene signatures.
    :param motif_annotations_fname: The name of the file that contains the motif annotations to use.
    :param rank_threshold: The total number of ranked genes to take into account when creating a recovery curve.
    :param auc_threshold: The fraction of the ranked genome to take into account for the calculation of the
        Area Under the recovery Curve.
    :param nes_threshold: The Normalized Enrichment Score (NES) threshold to select enriched features.
    :param motif_similarity_fdr: The maximum False Discovery Rate to find factor annotations for enriched motifs.
    :param orthologuous_identity_threshold: The minimum orthologuous identity to find factor annotations
        for enriched motifs.
    :param weighted_recovery: Use weights of a gene signature when calculating recovery curves?
    :param client_or_address: The client of IP address of the scheduler when working with dask. For local multi-core
        systems 'custom_multiprocessing' or 'dask_multiprocessing' can be supplied.
    :param num_workers:  If not using a cluster, the number of workers to use for the calculation.
        None of all available CPUs need to be used.
    :param module_chunksize: The size of the chunk to use when using the dask framework.
    :param motif_base_url:
    :return: A dataframe with the enriched features.
    """
    return add_motif_url(
        prune2df(
            rnkdbs,
            signatures,
            motif_annotations_fname,
            filter_for_annotation=False,
            **kwargs
        ),
        base_url=motif_base_url,
    )
