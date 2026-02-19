from __future__ import annotations

import contextlib
import re
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Literal,
)

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.feather as pf

from .datatypes import (
    MotifOrTrackIDs,
    MotifsOrTracksType,
    RegionOrGeneIDs,
    RegionsOrGenesType,
    ScoresOrRankingsType,
)

if TYPE_CHECKING:
    import polars as pl


def is_feather_v1_or_v2(feather_filename: Path | str) -> int | None:
    """
    Check if the passed filename is a Feather v1 or v2 file.

    :param feather_filename: Feather v1 or v2 filename.
    :return: 1 (for Feather version 1), 2 (for Feather version 2) or None.
    """
    with open(feather_filename, "rb") as fh_feather:  # noqa: PTH123
        # Read first 6 and last 6 bytes to see if we have a Feather v2 file.
        fh_feather.seek(0, 0)
        feather_v2_magic_bytes_header = fh_feather.read(6)
        fh_feather.seek(-6, 2)
        feather_v2_magic_bytes_footer = fh_feather.read(6)

        if feather_v2_magic_bytes_header == feather_v2_magic_bytes_footer == b"ARROW1":
            # Feather v2 file.
            return 2

        # Read first 4 and last 4 bytes to see if we have a Feather v1 file.
        feather_v1_magic_bytes_header = feather_v2_magic_bytes_header[0:4]
        feather_v1_magic_bytes_footer = feather_v2_magic_bytes_footer[2:]

        if feather_v1_magic_bytes_header == feather_v1_magic_bytes_footer == b"FEA1":
            # Feather v1 file.
            return 1

    # Some other file format.
    return None


def get_ct_db_type_from_ct_db_filename(
    ct_db_filename: Path | str,
) -> tuple[str, str, str]:
    """
    Get cisTarget database type from cisTarget database filename.

    :param ct_db_filename:
        cisTarget database filename.
    :return:
        scores_or_rankings, column_kind, row_kind
    """
    if not isinstance(ct_db_filename, Path):
        ct_db_filename = Path(ct_db_filename)

    scores_or_rankings: str | None = None
    column_kind: str | None = None
    row_kind: str | None = None

    for suffix in ct_db_filename.suffixes:
        # Remove the leading ".".
        suffix = suffix[1:]

        if suffix in ("scores", "rankings"):
            scores_or_rankings = suffix
        else:
            m = re.match(
                "^(motifs|tracks)_vs_(regions|genes)$|^(regions|genes)_vs_(motifs|tracks)$",
                suffix,
            )
            if m:
                if m.group(1, 2) != (None, None):
                    column_kind, row_kind = m.group(1, 2)
                elif m.group(3, 4) != (None, None):
                    column_kind, row_kind = m.group(3, 4)

    if (
        isinstance(scores_or_rankings, str)
        and isinstance(column_kind, str)
        and isinstance(row_kind, str)
    ):
        return scores_or_rankings, column_kind, row_kind
    elif not scores_or_rankings and not column_kind and not row_kind:
        msg = (
            f'cisTarget database filename "{ct_db_filename}" does not end with '
            f'".((motifs|tracks)_vs_(regions|genes)|(regions|genes)_vs_(motifs|tracks)).(scores|rankings).feather".'
        )
        raise ValueError(msg)
    elif scores_or_rankings and not column_kind and not row_kind:
        msg = (
            f'cisTarget database filename "{ct_db_filename}" does not end with '
            f'".((motifs|tracks)_vs_(regions|genes)|(regions|genes)_vs_(motifs|tracks)).{scores_or_rankings}.feather".'
        )
        raise ValueError(msg)
    elif not scores_or_rankings and column_kind and row_kind:
        msg = (
            f'cisTarget database filename "{ct_db_filename}" does not end with '
            f'".{column_kind}_vs_{row_kind}.(scores|rankings).feather".'
        )
        raise ValueError(msg)
    else:
        msg = f'Unknown error while parsing cisTarget database filename "{ct_db_filename}".'
        raise ValueError(msg)


class CisTargetDatabase:
    """CisTargetDatabase class for reading rankings/scores for regions/genes from a cisTarget scores or rankings database."""

    @staticmethod
    def init_ct_db(
        ct_db_filename: Path | str,
        engine: Literal["polars", "polars_pyarrow", "pyarrow"] | str = "polars",
    ) -> CisTargetDatabase:
        """
        Create a CisTargetDatabase class by providing a cisTarget scores or rankings database file.

        This will:
          - get all gene IDs or region IDs available in the cisTarget database.
          - get all motif IDs or track IDs available in the cisTarget database.
          - get the cisTarget database kind (genes or regions, motif or tracks, scores
            or regions).
          - dtype used to store the scores or rankings.

        :param ct_db_filename:
            Path to cisTarget Feather scores or rankings database.
        :param engine:
            Engine to use when reading from cisTarget Feather database file:
              - `polars`: Use `pl.read_ipc(..., use_pyarrow=False)` to read to Polars
                dataframe.
              - `polars_pyarrow`: Use `pl.read_ipc(..., use_pyarrow=True)` to read to
                Polars dataframe.
              - `pyarrow`: Use `pyarrow.feather.read_table()` to read to pyarrow Table.

        :return: CisTargetDatabase object for regions or genes.
        """
        if not isinstance(ct_db_filename, Path):
            ct_db_filename = Path(ct_db_filename)

        if engine in ("polars", "polars_pyarrow"):
            use_pyarrow = False
        elif engine == "pyarrow":
            use_pyarrow = True
        else:
            msg = f'Unsupported engine "{engine}" for reading cisTarget database.'
            raise ValueError(msg)

        feather_v1_or_v2 = is_feather_v1_or_v2(ct_db_filename)

        if not feather_v1_or_v2:
            msg = f'"{ct_db_filename}" is not a cisTarget Feather database in Feather v1 or v2 format.'
            raise ValueError(msg)
        elif feather_v1_or_v2 == 1:
            msg = (
                f'"{ct_db_filename}" is a cisTarget Feather database in Feather v1 format, which is not supported '
                f'anymore. Convert them with "convert_cistarget_databases_v1_to_v2.py" '
                "(https://github.com/aertslab/create_cisTarget_databases/) to Feather v2 format."
            )
            raise ValueError(msg)

        # cisTarget Feather database is in v2 format.

        if use_pyarrow:
            # Get column names from cisTarget Feather file with pyarrow without loading
            # the whole database.
            schema = ds.dataset(ct_db_filename, format="feather").schema
            column_names = schema.names
            dtypes = schema.types
        else:
            import polars as pl

            # Get column names from cisTarget Feather file with polars without loading
            # the whole database.
            schema = pl.read_ipc_schema(file=ct_db_filename)
            column_names = list(schema.keys())
            dtypes = list(schema.values())

        index_column_idx: int | None = None
        index_column_name: str | None = None

        # Get database index column ("motifs", "tracks", "regions" or "genes" depending
        # on the database type). Start with the last column (as the index column
        # normally should be the latest).
        for column_idx, column_name in zip(
            range(len(column_names) - 1, -1, -1), column_names[::-1]
        ):
            if column_name in {"motifs", "tracks", "regions", "genes"}:
                index_column_idx = column_idx
                index_column_name = column_name

                if use_pyarrow:
                    row_names = (
                        pf.read_table(
                            source=ct_db_filename,
                            columns=[column_idx],
                            memory_map=False,
                            use_threads=False,
                        )
                        .column(0)
                        .to_pylist()
                    )
                else:
                    import polars as pl

                    row_names = (
                        pl.read_ipc(
                            file=ct_db_filename,
                            columns=[column_idx],
                            use_pyarrow=False,
                            rechunk=False,
                        )
                        .to_series()
                        .to_list()
                    )
                break

        if not index_column_name or not index_column_idx:
            msg = (
                '"{ct_db_filename}" is not a cisTarget database file as it does not contain a "motifs", "tracks", '
                '"regions" or "genes" column.'
            )
            raise ValueError(msg)

        # Get all column names without index column name.
        column_names = (
            column_names[0:index_column_idx] + column_names[index_column_idx + 1 :]
        )

        # Get dtype for those columns (should be the same for all of them).
        column_dtype = list(
            set(dtypes[0:index_column_idx] + dtypes[index_column_idx + 1 :])
        )

        if len(column_dtype) != 1:
            msg = f"Only one dtype is allowed for {column_names[0:10]} ...: {column_dtype}"
            raise ValueError(msg)

        column_dtype = column_dtype[0]
        dtype: type[np.int16 | np.int32 | np.float32]

        if use_pyarrow:
            if column_dtype == pa.int16():
                scores_or_rankings = "rankings"
                dtype = np.int16
            elif column_dtype == pa.int32():
                scores_or_rankings = "rankings"
                dtype = np.int32
            elif column_dtype == pa.float32():
                scores_or_rankings = "scores"
                dtype = np.float32
            else:
                msg = f'Unsupported dtype "{column_dtype}" for cisTarget database.'
                raise ValueError(msg)
        else:
            import polars as pl

            if column_dtype == pl.Int16:
                scores_or_rankings = "rankings"
                dtype = np.int16
            elif column_dtype == pl.Int32:
                scores_or_rankings = "rankings"
                dtype = np.int32
            elif column_dtype == pl.Float32:
                scores_or_rankings = "scores"
                dtype = np.float32
            else:
                msg = f'Unsupported dtype "{column_dtype}" for cisTarget database.'
                raise ValueError(msg)

        # Get cisTarget database type from cisTarget database filename extension.
        (
            scores_or_rankings_from_filename,
            column_kind_from_filename,
            row_kind_from_filename,
        ) = get_ct_db_type_from_ct_db_filename(ct_db_filename)

        row_kind = index_column_name

        if scores_or_rankings_from_filename != scores_or_rankings:
            msg = (
                f'cisTarget database "{ct_db_filename}" claims to contain {scores_or_rankings_from_filename} based on '
                f"the filename, but contains {scores_or_rankings}."
            )
            raise ValueError(msg)
        if row_kind_from_filename != row_kind:
            msg = (
                f'cisTarget database "{ct_db_filename}" claims to contain {row_kind_from_filename} based on the '
                f"filename, but contains {row_kind}."
            )
            raise ValueError(msg)

        # Assume column kind is correct if the other values were correct.
        column_kind = column_kind_from_filename

        if column_kind in ("regions", "genes"):
            # Create cisTarget database object if the correct database was provided.
            return CisTargetDatabase(
                ct_db_filename=ct_db_filename,
                region_or_gene_ids=RegionOrGeneIDs(
                    region_or_gene_ids=column_names,
                    regions_or_genes_type=RegionsOrGenesType.from_str(
                        regions_or_genes_type=column_kind
                    ),
                ),
                motif_or_track_ids=MotifOrTrackIDs(
                    motif_or_track_ids=row_names,
                    motifs_or_tracks_type=MotifsOrTracksType.from_str(
                        motifs_or_tracks_type=row_kind
                    ),
                ),
                scores_or_rankings=ScoresOrRankingsType.from_str(
                    scores_or_rankings_type=scores_or_rankings
                ),
                dtype=dtype,
                engine=engine,
            )
        else:
            msg = f'cisTarget database "{ct_db_filename}" has the wrong type. The transposed version is needed.'
            raise ValueError(msg)

    def __init__(
        self,
        ct_db_filename: Path,
        region_or_gene_ids: RegionOrGeneIDs,
        motif_or_track_ids: MotifOrTrackIDs,
        scores_or_rankings: ScoresOrRankingsType,
        dtype: type[np.int16 | np.int32 | np.float32],
        engine: Literal["polars", "polars_pyarrow", "pyarrow"] | str = "polars",
    ) -> None:
        """
        Create cisTargetDatabase object.

        Use cisTargetDatabase.init_ct_db() instead.
        """
        # cisTarget scores or rankings database file.
        self.ct_db_filename: Path = ct_db_filename

        self.all_region_or_gene_ids: RegionOrGeneIDs = region_or_gene_ids
        self.all_motif_or_track_ids: MotifOrTrackIDs = motif_or_track_ids
        self.scores_or_rankings: ScoresOrRankingsType = scores_or_rankings
        self.dtype: type[np.int16 | np.int32 | np.float32] = dtype
        self.engine = engine

        # Count number of region IDs or gene IDs.
        self._nbr_total_region_or_gene_ids = len(self.all_region_or_gene_ids)
        # Count number of motif IDs or track IDs.
        self._nbr_total_motif_or_track_ids = len(self.all_motif_or_track_ids)

        with contextlib.suppress(ImportError):
            pass

        # Polars dataframe or pyarrow Table with scores or rankings for those region IDs
        # or gene IDs that where loaded with cisTargetDatabase.prefetch().
        # This acts as a cache.
        self.df_cached: pl.DataFrame | pa.Table | None = None

        # Keep track for which region IDs or gene IDs, scores or rankings are loaded
        # with cisTargetDatabase.prefetch().
        self.region_or_gene_ids_loaded: RegionOrGeneIDs | None = None

    def __str__(self) -> str:
        all_regions_or_gene_ids_formatted = "\n    ".join(
            str(self.all_region_or_gene_ids).split("\n")
        )
        all_motif_or_track_ids_formatted = "\n    ".join(
            str(self.all_motif_or_track_ids).split("\n")
        )

        return (
            f"CisTargetDatabase(\n"
            f"    ct_db_filename={self.ct_db_filename},\n"
            f"    all_region_or_gene_ids={all_regions_or_gene_ids_formatted},\n"
            f"    all_motif_or_track_ids={all_motif_or_track_ids_formatted},\n"
            f"    scores_or_rankings={self.scores_or_rankings},\n"
            f"    dtype=np.{np.dtype(self.dtype)!s},\n"
            f"    engine={self.engine}\n"
            ")"
        )

    def __repr__(self) -> str:
        return (
            f"CisTargetDatabase(\n"
            f"    ct_db_filename={self.ct_db_filename!r},\n"
            f"    all_region_or_gene_ids={self.all_region_or_gene_ids!r},\n"
            f"    all_motif_or_track_ids={self.all_motif_or_track_ids!r},\n"
            f"    scores_or_rankings={self.scores_or_rankings!s},\n"
            f"    dtype=np.{np.dtype(self.dtype)!s},\n"
            f"    engine={self.engine}\n"
            ")"
        )

    @property
    def is_genes_db(self) -> bool:
        """Is cisTarget database a gene-based database?"""  # noqa: D400
        return self.all_region_or_gene_ids.type == RegionsOrGenesType.GENES

    @property
    def is_regions_db(self) -> bool:
        """Is cisTarget database a region-based database?"""  # noqa: D400
        return self.all_region_or_gene_ids.type == RegionsOrGenesType.REGIONS

    @property
    def is_motifs_db(self) -> bool:
        """Is cisTarget database a motif-based database?"""  # noqa: D400
        return self.all_motif_or_track_ids.type == MotifsOrTracksType.MOTIFS

    @property
    def is_tracks_db(self) -> bool:
        """Is cisTarget database a track-based database?"""  # noqa: D400
        return self.all_motif_or_track_ids.type == MotifsOrTracksType.TRACKS

    @property
    def is_scores_db(self) -> bool:
        """Does cisTarget database contain scores?"""  # noqa: D400
        return self.scores_or_rankings == ScoresOrRankingsType.SCORES

    @property
    def is_rankings_db(self) -> bool:
        """Does cisTarget database contain rankings?"""  # noqa: D400
        return self.scores_or_rankings == ScoresOrRankingsType.RANKINGS

    @property
    def nbr_total_region_or_gene_ids(self) -> int:
        """Total number or region IDs or gene IDs stored in the cisTarget database."""
        return self._nbr_total_region_or_gene_ids

    @property
    def nbr_total_motif_or_track_ids(self) -> int:
        """Total number or motif IDs or track IDs stored in the cisTarget database."""
        return self._nbr_total_motif_or_track_ids

    def has_all_region_or_gene_ids(
        self, region_or_gene_ids: RegionOrGeneIDs
    ) -> tuple[bool, RegionOrGeneIDs, RegionOrGeneIDs]:
        """
        Check if all input region IDs or gene IDs are found in the cisTarget database.

        :param region_or_gene_ids: RegionOrGeneIDs object
        :return all found, found region IDs or gene IDs, not found region IDs or gene IDs
        """
        if region_or_gene_ids.issubset(self.all_region_or_gene_ids):
            return (
                True,
                region_or_gene_ids,
                RegionOrGeneIDs([], region_or_gene_ids.type),
            )
        else:
            return (
                False,
                region_or_gene_ids.intersection(self.all_region_or_gene_ids),
                region_or_gene_ids.difference(self.all_region_or_gene_ids),
            )

    def clear_cache(self) -> None:
        """Remove prefetched scores or regions for region IDs or gene IDs from memory."""
        self.df_cached = None
        self.region_or_gene_ids_loaded = None

    def _prefetch_as_polars_dataframe(
        self,
        region_or_gene_ids: RegionOrGeneIDs,
        *,
        use_pyarrow: bool,
        sort: bool = False,
    ) -> None:
        """
        Fetch scores or rankings for input region IDs or gene IDs from cisTarget
        database file for region IDs or gene IDs which were not prefetched in previous
        calls.

        All prefetched scores or rankings are stored as a polars Dataframe in
        `self.df_cached`, so they do not need to be retrieved again from disk later if
        the same region IDs or gene IDs are requested.

        :param region_or_gene_ids:
            Input region IDs or gene IDs to load from the cisTarget database file.
        :param use_pyarrow:
            Use pyarrow to read from Feather file or use polars native reader:
            `pl.read_ipc(..., use_pyarrow=...)`.
        :param sort:
            Sort region IDs or gene IDs columns in self.df_cache.
        """
        (
            contains_all_input_gene_ids_or_regions_ids,
            found_region_or_gene_ids,
            not_found_region_or_gene_ids,
        ) = self.has_all_region_or_gene_ids(region_or_gene_ids)

        if contains_all_input_gene_ids_or_regions_ids is False:
            msg = f"Not all provided {self.all_region_or_gene_ids.type} are found: {not_found_region_or_gene_ids}"
            raise ValueError(msg)

        import polars as pl

        if self.df_cached and isinstance(self.df_cached, pa.Table):
            # Convert pyarrow Table to polars Dataframe (in case engine="pyarrow" was
            # used before).
            self.df_cached = pl.from_arrow(self.df_cached, rechunk=False)

        if not self.df_cached:
            # No region IDs or gene IDs scores/rankings where loaded before.

            # Get all found region IDs or gene IDs columns with scores/rankings and
            # "motifs" or "track" column from cisTarget Feather file as a pyarrow Table.
            self.df_cached = pl.read_ipc(
                file=self.ct_db_filename,
                columns=(
                    list(found_region_or_gene_ids.sort().ids)
                    if sort
                    else list(found_region_or_gene_ids.ids)
                )
                + [
                    self.all_motif_or_track_ids.type.value,
                ],
                use_pyarrow=use_pyarrow,
                memory_map=False,
                rechunk=False,
            )

            # Keep track of loaded region IDs or gene IDs scores/rankings.
            self.region_or_gene_ids_loaded = found_region_or_gene_ids
        else:
            if not self.region_or_gene_ids_loaded:
                msg = (
                    "CisTargetDatabase object is in an inconsistent state: "
                    '"region_or_gene_ids_loaded" attribute is None, but '
                    '"df_cached" is not.'
                )
                raise ValueError(msg)

            # Get region IDs or gene IDs subset for which no scores/rankings were loaded
            # before.
            region_or_gene_ids_to_load = found_region_or_gene_ids.difference(
                self.region_or_gene_ids_loaded
            )

            # Check if new region IDs or gene IDs need to be loaded.
            if len(region_or_gene_ids_to_load) != 0:
                # Get region IDs or gene IDs subset columns with scores/rankings from
                # cisTarget Feather file as a polars DataFrame.
                self.df_cached.hstack(
                    columns=pl.read_ipc(
                        file=self.ct_db_filename,
                        columns=list(region_or_gene_ids_to_load.ids),
                        use_pyarrow=use_pyarrow,
                        memory_map=False,
                        rechunk=False,
                    ),
                    in_place=True,
                )

                # Keep track of loaded region IDs or gene IDs scores/rankings.
                self.region_or_gene_ids_loaded = found_region_or_gene_ids.union(
                    self.region_or_gene_ids_loaded
                )

                # Store new pyarrow Table with previously and newly loaded region IDs or
                # gene IDs scores/rankings.
                self.df_cached = self.df_cached.select(
                    (
                        self.region_or_gene_ids_loaded.sort().ids
                        if sort
                        else self.region_or_gene_ids_loaded.ids
                    )
                    + (self.all_motif_or_track_ids.type.value,)
                )

    def _prefetch_as_pyarrow_table(
        self, region_or_gene_ids: RegionOrGeneIDs, *, sort: bool = False
    ) -> None:
        """
        Fetch scores or rankings for input region IDs or gene IDs from cisTarget
        database file for region IDs or gene IDs which were not prefetched in previous
        calls.

        All prefetched scores or rankings are stored as a pyarrow Table in
        `self.df_cached`, so they do not need to be retrieved again from disk later
        if the same region IDs or gene IDs are requested.

        :param region_or_gene_ids:
            Input region IDs or gene IDs to load from the cisTarget database file.
        :param sort:
            Sort region IDs or gene IDs columns in self.df_cache.
        """
        (
            contains_all_input_gene_ids_or_regions_ids,
            found_region_or_gene_ids,
            not_found_region_or_gene_ids,
        ) = self.has_all_region_or_gene_ids(region_or_gene_ids)

        if contains_all_input_gene_ids_or_regions_ids is False:
            msg = f"Not all provided {self.all_region_or_gene_ids.type} are found: {not_found_region_or_gene_ids}"
            raise ValueError(msg)

        if not self.df_cached or not isinstance(self.df_cached, pa.Table):
            # No region IDs or gene IDs scores/rankings where loaded before or cached
            # version was a polars DataFrame.

            # Get all found region IDs or gene IDs columns with scores/rankings and
            # "motifs" or "track" column from cisTarget Feather file as a pyarrow Table.
            self.df_cached = pf.read_table(
                source=self.ct_db_filename,
                columns=(
                    found_region_or_gene_ids.sort().ids
                    if sort
                    else found_region_or_gene_ids.ids
                )
                + (self.all_motif_or_track_ids.type.value,),
                memory_map=False,
                use_threads=True,
            )

            # Keep track of loaded region IDs or gene IDs scores/rankings.
            self.region_or_gene_ids_loaded = found_region_or_gene_ids
        else:
            if not self.region_or_gene_ids_loaded:
                msg = (
                    "CisTargetDatabase object is in an inconsistent state: "
                    '"region_or_gene_ids_loaded" attribute is None, but '
                    '"df_cached" is not.'
                )
                raise ValueError(msg)

            # Get region IDs or gene IDs subset for which no scores/rankings were loaded
            # before.
            region_or_gene_ids_to_load = found_region_or_gene_ids.difference(
                self.region_or_gene_ids_loaded
            )

            # Check if new region IDs or gene IDs need to be loaded.
            if len(region_or_gene_ids_to_load) != 0:
                # Get region IDs or gene IDs subset columns with scores/rankings from
                # cisTarget Feather file as a pyarrow Table.
                pa_table_subset = pf.read_table(
                    source=self.ct_db_filename,
                    columns=region_or_gene_ids_to_load.ids,
                    memory_map=False,
                    use_threads=True,
                )

                # Get current loaded pyarrow Table to which the pa_table_subset data
                # will be added.
                pa_table = self.df_cached

                for column in pa_table_subset.itercolumns():
                    # Append column with region IDs or gene IDs scores/rankings to
                    # existing pyarrow Table.
                    pa_table = pa_table.append_column(column._name, column)

                # Keep track of loaded region IDs or gene IDs scores/rankings.
                self.region_or_gene_ids_loaded = found_region_or_gene_ids.union(
                    self.region_or_gene_ids_loaded
                )

                # Store new pyarrow Table with previously and newly loaded region IDs or
                # gene IDs scores/rankings.
                self.df_cached = pa_table.select(
                    (
                        self.region_or_gene_ids_loaded.sort().ids
                        if sort
                        else self.region_or_gene_ids_loaded.ids
                    )
                    + (self.all_motif_or_track_ids.type.value,)
                )

    def prefetch(
        self,
        region_or_gene_ids: RegionOrGeneIDs,
        engine: Literal["polars", "polars_pyarrow", "pyarrow"] | str | None = None,
        *,
        sort: bool = False,
    ) -> None:
        """
        Fetch scores or rankings for input region IDs or gene IDs from cisTarget
        database file for region IDs or gene IDs which were not prefetched in previous
        calls.

        All prefetched scores or rankings are stored in self.df_cached (as a polars
        DataFrame or pyarrow Table, depending on the chosen engine), so they do not
        need to be retrieved again from disk later if the same region IDs or gene IDs
        are requested.

        :param region_or_gene_ids:
            Input region IDs or gene IDs to load from the cisTarget database file.
        :param engine:
            Engine to use when reading from cisTarget Feather database file:
              - `polars`: Use `pl.read_ipc(..., use_pyarrow=False)` to read to Polars
                dataframe.
              - `polars_pyarrow`: Use `pl.read_ipc(..., use_pyarrow=True)` to read to
                Polars dataframe.
              - `pyarrow`: Use `pyarrow.feather.read_table()` to read to pyarrow Table.
              - `None`: Use engine defined by `self.engine`.
        :param sort:
            Sort region IDs or gene IDs columns in self.df_cache.
        """
        (
            contains_all_input_gene_ids_or_regions_ids,
            found_region_or_gene_ids,
            not_found_region_or_gene_ids,
        ) = self.has_all_region_or_gene_ids(region_or_gene_ids)

        if contains_all_input_gene_ids_or_regions_ids is False:
            msg = f"Not all provided {self.all_region_or_gene_ids.type} are found: {not_found_region_or_gene_ids}"
            raise ValueError(msg)

        engine = engine if engine else self.engine

        if engine == "polars":
            # Store prefetched data as polars DataFrame (self.df_cached) and read data
            # with polars' native IPC reader.
            self._prefetch_as_polars_dataframe(
                region_or_gene_ids=region_or_gene_ids, use_pyarrow=False, sort=sort
            )
        elif engine == "polars_pyarrow":
            # Store prefetched data as polars DataFrame (self.df_cached) and read data
            # with pyarrow's native IPC reader.
            self._prefetch_as_polars_dataframe(
                region_or_gene_ids=region_or_gene_ids, use_pyarrow=True, sort=sort
            )
        elif engine == "pyarrow":
            # Store prefetched data as pyarrow Table (self.df_cached) and read data
            # with pyarrow's native IPC reader.
            self._prefetch_as_pyarrow_table(
                region_or_gene_ids=region_or_gene_ids, sort=sort
            )
        else:
            msg = f'Unsupported engine "{engine}" for reading cisTarget database.'
            raise ValueError(msg)

    def subset_to_pandas(
        self,
        region_or_gene_ids: RegionOrGeneIDs,
        engine: Literal["polars", "polars_pyarrow", "pyarrow"] | str | None = None,
    ) -> pd.DataFrame:
        """
        Create Pandas dataframe of scores or rankings for input region IDs or gene IDs
        from cisTarget database file.

        This calls `prefetch()` under the hood, which will cache previous retrieved
        scores or rankings for region IDs or gene IDs, so in case the query contains
        region IDs or gene IDs that were retrieved before, those will not be retrieved
        again from disk.

        All prefetched scores or rankings are stored in self.df_cached (as a polars
        DataFrame or pyarrow Table, depending on the chosen engine).

        :param region_or_gene_ids:
            Input region IDs or gene IDs to load from the cisTarget database file.
        :param engine:
            Engine to use when reading from cisTarget Feather database file:
              - `polars`: Use `pl.read_ipc(..., use_pyarrow=False)` to read to Polars
                dataframe.
              - `polars_pyarrow`: Use `pl.read_ipc(..., use_pyarrow=True)` to read to
                Polars dataframe.
              - `pyarrow`: Use `pyarrow.feather.read_table()` to read to pyarrow Table.
              - `None`: Use engine defined by `self.engine`.
        """
        (
            contains_all_input_gene_ids_or_regions_ids,
            found_region_or_gene_ids,
            not_found_region_or_gene_ids,
        ) = self.has_all_region_or_gene_ids(region_or_gene_ids)

        if contains_all_input_gene_ids_or_regions_ids is False:
            msg = f"Not all provided {self.all_region_or_gene_ids.type} are found: {not_found_region_or_gene_ids}"
            raise ValueError(msg)

        engine = engine if engine else self.engine

        # Fetch scores or rankings for input region IDs or gene IDs from cisTarget
        # database file for region IDs or gene IDs which were not prefetched in
        # previous calls.
        self.prefetch(region_or_gene_ids=region_or_gene_ids, engine=engine, sort=True)

        if not self.df_cached:
            msg = (
                f"Prefetch failed to retrieve {self.scores_or_rankings} for "
                f"{region_or_gene_ids} from cisTarget database "
                f'"{self.ct_db_filename}".'
            )
            raise RuntimeError(msg)

        if engine == "pyarrow":
            # Select input region IDs or gene IDs subset and motif or track column from
            # pyarrow Table and convert to pandas DataFrame.
            pd_df = pd.DataFrame(
                data=self.df_cached.select(
                    # Region IDs or gene IDs columns.
                    found_region_or_gene_ids.ids
                    # motifs or track column.
                    + (("motifs",) if self.is_motifs_db else ("tracks",))
                ).to_pandas()
            )

            # Set motifs or tracks column as index (inplace to avoid extra copy).
            pd_df.set_index("motifs" if self.is_motifs_db else "tracks", inplace=True)

            # Add "regions" or "genes" as column index name.
            pd_df.rename_axis(
                columns="regions" if self.is_regions_db else "genes", inplace=True
            )

            return pd_df
        else:
            # Convert input region IDs or gene IDs subset from polars Dataframe to numpy
            # and construct a pandas Dataframe from it.
            return pd.DataFrame(
                data=self.df_cached.select(found_region_or_gene_ids.ids).to_numpy(),
                columns=pd.Index(
                    found_region_or_gene_ids.ids,
                    name="regions" if self.is_regions_db else "genes",
                ),
                index=pd.Index(
                    self.all_motif_or_track_ids.ids,
                    name="motifs" if self.is_motifs_db else "tracks",
                ),
            )
