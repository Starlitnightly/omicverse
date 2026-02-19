from __future__ import annotations

import re
from enum import Enum, unique


@unique
class RegionsOrGenesType(Enum):
    """Enum describing all possible regions or genes types."""

    REGIONS = "regions"
    GENES = "genes"

    @classmethod
    def from_str(cls, regions_or_genes_type: str) -> RegionsOrGenesType:
        """
        Create RegionsOrGenesType Enum member from string.

        :param regions_or_genes_type: 'regions' or 'genes'.
        :return: RegionsOrGenesType Enum member.
        """
        regions_or_genes_type = regions_or_genes_type.upper()
        regions_or_genes_type_instance = cls.__members__.get(regions_or_genes_type)
        if regions_or_genes_type_instance:
            return regions_or_genes_type_instance
        else:
            msg = f'Unsupported RegionsOrGenesType "{regions_or_genes_type}".'
            raise ValueError(msg)


@unique
class MotifsOrTracksType(Enum):
    """Enum describing all possible motif or track types."""

    MOTIFS = "motifs"
    TRACKS = "tracks"

    @classmethod
    def from_str(cls, motifs_or_tracks_type: str) -> MotifsOrTracksType:
        """
        Create MotifsOrTracksType Enum member from string.

        :param motifs_or_tracks_type: 'motifs' or 'tracks'.
        :return: MotifsOrTracksType Enum member.
        """
        motifs_or_tracks_type = motifs_or_tracks_type.upper()
        motifs_or_tracks_type_instance = cls.__members__.get(motifs_or_tracks_type)
        if motifs_or_tracks_type_instance:
            return motifs_or_tracks_type_instance
        else:
            msg = f'Unsupported MotifsOrTracksType "{motifs_or_tracks_type}".'
            raise ValueError(msg)


@unique
class ScoresOrRankingsType(Enum):
    """Enum describing all possible scores or rankings types."""

    SCORES = "scores"
    RANKINGS = "rankings"

    @classmethod
    def from_str(cls, scores_or_rankings_type: str) -> ScoresOrRankingsType:
        """
        Create ScoresOrRankingsType Enum member from string.

        :param scores_or_rankings_type: 'scores' or 'rankings'.
        :return: ScoresOrRankingsType Enum member.
        """
        scores_or_rankings_type = scores_or_rankings_type.upper()
        scores_or_rankings_type_instance = cls.__members__.get(scores_or_rankings_type)
        if scores_or_rankings_type_instance:
            return scores_or_rankings_type_instance
        else:
            msg = f'Unsupported ScoresOrRankingsType "{scores_or_rankings_type}".'
            raise ValueError(msg)


class RegionOrGeneIDs:
    """
    RegionOrGeneIDs class represents a unique sorted tuple of region or gene IDs
    for constructing a Pandas dataframe index for a cisTarget database.
    """

    @staticmethod
    def get_region_or_gene_ids_from_bed(
        bed_filename: str,
        extract_gene_id_from_region_id_regex_replace: str | None = None,
    ) -> RegionOrGeneIDs:
        """
        Get all region or gene IDs (from column 4) from BED filename.

        Get all region or gene IDs (from column 4) from BED filename:
          - When extract_gene_id_from_region_id_regex_replace=None, region IDs are
            returned and each region ID is only allowed once in the BED file.
          - When extract_gene_id_from_region_id_regex_replace is set to a regex to
            remove the non gene ID part from the region IDs, gene IDs are returned
            and each gene is allowed to appear more than once in the BED file.

        :param bed_filename:
             BED filename with sequences for region or gene IDs.
        :param extract_gene_id_from_region_id_regex_replace:
             regex for removing unwanted parts from the region ID to extract the gene
             ID.
        :return: RegionOrGeneIDs object for regions or genes.
        """
        gene_ids = []
        region_ids = []
        gene_ids_set = set()
        region_ids_set = set()

        with open(bed_filename, encoding="utf-8") as fh:  # noqa: PTH123
            for line in fh:
                if line and not line.startswith("#"):
                    columns = line.strip().split("\t")

                    if len(columns) < 4:
                        msg = f'Error: BED file "{bed_filename:s}" has less than 4 columns.'
                        raise ValueError(msg)

                    # Get region ID from column 4 of the BED file.
                    region_id = columns[3]

                    if extract_gene_id_from_region_id_regex_replace:
                        # Extract gene ID from region ID.
                        gene_id = re.sub(
                            extract_gene_id_from_region_id_regex_replace, "", region_id
                        )

                        if gene_id not in gene_ids_set:
                            gene_ids.append(gene_id)
                            gene_ids_set.add(gene_id)
                    else:
                        # Check if all region IDs only appear once.
                        if region_id in region_ids_set:
                            msg = f'Error: region ID "{region_id:s}" is not unique in BED file "{bed_filename:s}".'
                            raise ValueError(msg)
                        else:
                            region_ids.append(region_id)
                            region_ids_set.add(region_id)

        if extract_gene_id_from_region_id_regex_replace:
            return RegionOrGeneIDs(
                region_or_gene_ids=gene_ids,
                regions_or_genes_type=RegionsOrGenesType.GENES,
            )
        else:
            return RegionOrGeneIDs(
                region_or_gene_ids=region_ids,
                regions_or_genes_type=RegionsOrGenesType.REGIONS,
            )

    @staticmethod
    def get_region_or_gene_ids_from_fasta(
        fasta_filename: str,
        extract_gene_id_from_region_id_regex_replace: str | None = None,
    ) -> RegionOrGeneIDs:
        """
        Get all region or gene IDs from FASTA filename.

        Get all region or gene IDs from FASTA filename:
          - When extract_gene_id_from_region_id_regex_replace=None, region IDs are
            returned and each region ID is only allowed once in the FASTA file.
          - When extract_gene_id_from_region_id_regex_replace is set to a regex to
            remove the non gene ID part from the region IDs, gene IDs are returned
            and each gene is allowed to appear more than once in the FASTA file.

        :param fasta_filename:
             FASTA filename with sequences for region or gene IDs.
        :param extract_gene_id_from_region_id_regex_replace:
             regex for removing unwanted parts from the region ID to extract the gene
             ID.
        :return: RegionOrGeneIDs object for regions or genes.
        """
        gene_ids = []
        region_ids = []
        gene_ids_set = set()
        region_ids_set = set()

        with open(fasta_filename, encoding="utf-8") as fh:  # noqa: PTH123
            for line in fh:
                if line.startswith(">"):
                    # Get region ID by getting everything after '>' up till the first
                    # whitespace.
                    region_id = line[1:].split(maxsplit=1)[0]

                    if extract_gene_id_from_region_id_regex_replace:
                        # Extract gene ID from region ID.
                        gene_id = re.sub(
                            extract_gene_id_from_region_id_regex_replace, "", region_id
                        )

                        if gene_id not in gene_ids_set:
                            gene_ids.append(gene_id)
                            gene_ids_set.add(gene_id)
                    else:
                        # Check if all region IDs only appear once.
                        if region_id in region_ids:
                            msg = f'Error: region ID "{region_id:s}" is not unique in FASTA file "{fasta_filename:s}".'
                            raise ValueError(msg)
                        else:
                            region_ids.append(region_id)
                            region_ids_set.add(region_id)

        if extract_gene_id_from_region_id_regex_replace:
            return RegionOrGeneIDs(
                region_or_gene_ids=gene_ids,
                regions_or_genes_type=RegionsOrGenesType.GENES,
            )
        else:
            return RegionOrGeneIDs(
                region_or_gene_ids=region_ids,
                regions_or_genes_type=RegionsOrGenesType.REGIONS,
            )

    def __init__(
        self,
        region_or_gene_ids: list[str] | set[str] | tuple[str, ...],
        regions_or_genes_type: RegionsOrGenesType | str,
    ) -> None:
        """
        Create unique region IDs or gene IDs.

        Create unique tuple of region or gene IDs from a list, set or tuple of region
        or gene IDs, annotated with RegionsOrGenesType Enum.

        :param region_or_gene_ids: list, set or tuple of region or gene IDs.
        :param regions_or_genes_type: RegionsOrGenesType.REGIONS ("regions") or
            RegionsOrGenesType.GENES ("genes").
        """
        if isinstance(regions_or_genes_type, str):
            regions_or_genes_type = RegionsOrGenesType.from_str(regions_or_genes_type)

        if isinstance(region_or_gene_ids, set):
            region_or_gene_ids = sorted(region_or_gene_ids)

        # Collapse duplicates, keep order, and add sort value.
        self.ids_dict = {rg_id: idx for idx, rg_id in enumerate(region_or_gene_ids)}

        if len(region_or_gene_ids) != len(self.ids_dict):
            # Recreate dict if region IDs or gene IDs contained duplicates.
            self.ids_dict = {rg_id: idx for idx, rg_id in enumerate(self.ids_dict)}

        self.ids = tuple(self.ids_dict)
        self.ids_set = set(self.ids_dict)
        self.type = regions_or_genes_type

    def __str__(self) -> str:
        return (
            f"RegionOrGeneIDs(\n"
            f"    region_or_gene_ids={self.ids if len(self.ids) <= 6 else self.ids[0:6] + ('...',)},\n"
            f"    regions_or_genes_type={self.type}\n"
            ")"
        )

    def __repr__(self) -> str:
        return f"RegionOrGeneIDs(\n    region_or_gene_ids={self.ids},\n    regions_or_genes_type={self.type}\n)"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RegionOrGeneIDs):
            return NotImplemented

        return self.type == other.type and self.ids_set == other.ids_set

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, items: int | slice) -> RegionOrGeneIDs:
        if isinstance(items, int):
            return RegionOrGeneIDs((self.ids[items],), self.type)

        return RegionOrGeneIDs(self.ids[items], self.type)

    def difference(self, other: RegionOrGeneIDs) -> RegionOrGeneIDs:
        """
        Get which region or gene IDs in the current RegionOrGeneIDs object are not
        present in the other RegionOrGeneIDs object.

        :param other: RegionOrGeneIDs object
        :return: RegionOrGeneIDs object
        """
        if not isinstance(other, RegionOrGeneIDs):
            return NotImplemented

        assert self.type == other.type, (
            "RegionOrGeneIDs objects are of a different type."
        )

        return RegionOrGeneIDs(
            region_or_gene_ids=sorted(
                self.ids_set.difference(other.ids_set), key=lambda x: self.ids_dict[x]
            ),
            regions_or_genes_type=self.type,
        )

    def intersection(self, other: RegionOrGeneIDs) -> RegionOrGeneIDs:
        """
        Get which region or gene IDs in the current RegionOrGeneIDs object are present
        in the other RegionOrGeneIDs object.

        :param other: RegionOrGeneIDs object
        :return: RegionOrGeneIDs object
        """
        if not isinstance(other, RegionOrGeneIDs):
            return NotImplemented

        assert self.type == other.type, (
            "RegionOrGeneIDs objects are of a different type."
        )

        return RegionOrGeneIDs(
            region_or_gene_ids=sorted(
                self.ids_set.intersection(other.ids_set), key=lambda x: self.ids_dict[x]
            ),
            regions_or_genes_type=self.type,
        )

    def issubset(self, other: RegionOrGeneIDs) -> bool:
        """
        Check if all region or gene IDs in the current RegionOrGeneIDs object are at
        least present in the other RegionOrGeneIDs object.

        :param other: RegionOrGeneIDs object
        :return: True or False
        """
        if not isinstance(other, RegionOrGeneIDs):
            return NotImplemented

        assert self.type == other.type, (
            "RegionOrGeneIDs objects are of a different type."
        )

        return self.ids_set.issubset(other.ids_set)

    def issuperset(self, other: RegionOrGeneIDs) -> bool:
        """
        Check if all region or gene IDs in the other RegionOrGeneIDs object are at
        least present in the current RegionOrGeneIDs object.

        :param other: RegionOrGeneIDs object
        :return: True or False
        """
        if not isinstance(other, RegionOrGeneIDs):
            return NotImplemented

        assert self.type == other.type, (
            "RegionOrGeneIDs objects are of a different type."
        )

        return self.ids_set.issuperset(other.ids_set)

    def sort(self) -> RegionOrGeneIDs:
        """Sort region IDs or gene IDs."""
        return RegionOrGeneIDs(sorted(self.ids), self.type)

    def union(self, other: RegionOrGeneIDs) -> RegionOrGeneIDs:
        """
        Get union of region or gene IDs in the current RegionOrGeneIDs object and in
        the other RegionOrGeneIDs object.

        :param other: RegionOrGeneIDs object
        :return: RegionOrGeneIDs object
        """
        if not isinstance(other, RegionOrGeneIDs):
            return NotImplemented

        assert self.type == other.type, (
            "RegionOrGeneIDs objects are of a different type."
        )

        return RegionOrGeneIDs(
            region_or_gene_ids=sorted(
                self.ids_set.union(other.ids_set),
                key=lambda x: (
                    self.ids_dict.get(x, len(self.ids) + 1),
                    other.ids_dict.get(x, 0),
                ),
            ),
            regions_or_genes_type=self.type,
        )

    def has_genes(self) -> bool:
        return self.type == RegionsOrGenesType.GENES

    def has_regions(self) -> bool:
        return self.type == RegionsOrGenesType.REGIONS


class MotifOrTrackIDs:
    """
    MotifOrTrackIDs class represents a unique sorted tuple of motif IDs or track IDs
    for constructing a Pandas dataframe index for a cisTarget database.
    """

    def __init__(
        self,
        motif_or_track_ids: list[str] | set[str] | tuple[str, ...],
        motifs_or_tracks_type: MotifsOrTracksType | str,
    ) -> None:
        """
        Create unique tuple of motif IDs or track IDs from a list, set or tuple of
        motif IDs or track IDs, annotated with MotifsOrTracksType Enum.

        :param motif_or_track_ids: list, set or tuple of motif IDs or track IDs.
        :param motifs_or_tracks_type: MotifsOrTracksType.MOTIFS ("motifs") or
            MotifsOrTracksType.TRACKS ("tracks").
        """
        if isinstance(motifs_or_tracks_type, str):
            motifs_or_tracks_type = MotifsOrTracksType.from_str(motifs_or_tracks_type)

        if isinstance(motif_or_track_ids, set):
            motif_or_track_ids = sorted(motif_or_track_ids)

        # Collapse duplicates, keep order, and add sort value.
        self.ids_dict = {rg_id: idx for idx, rg_id in enumerate(motif_or_track_ids)}

        if len(motif_or_track_ids) != len(self.ids_dict):
            # Recreate dict if motif IDs or track IDs contained duplicates.
            self.ids_dict = {mt_id: idx for idx, mt_id in enumerate(self.ids_dict)}

        self.ids = tuple(self.ids_dict)
        self.ids_set = set(self.ids_dict)
        self.type = motifs_or_tracks_type

    def __str__(self) -> str:
        return (
            f"MotifOrTrackIDs(\n"
            f"    motif_or_track_ids={self.ids if len(self.ids) <= 6 else self.ids[0:6] + ('...',)},\n"
            f"    motifs_or_tracks_type={self.type}\n"
            ")"
        )

    def __repr__(self) -> str:
        return f"MotifOrTrackIDs(\n    motif_or_track_ids={self.ids},\n    motifs_or_tracks_type={self.type}\n)"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MotifOrTrackIDs):
            return NotImplemented

        return self.type == other.type and self.ids_set == other.ids_set

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, items: int | slice) -> MotifOrTrackIDs:
        if isinstance(items, int):
            return MotifOrTrackIDs((self.ids[items],), self.type)

        return MotifOrTrackIDs(self.ids[items], self.type)

    def sort(self) -> MotifOrTrackIDs:
        """Sort motif IDs or track IDs."""
        return MotifOrTrackIDs(sorted(self.ids), self.type)

    def has_motifs(self) -> bool:
        return self.type == MotifsOrTracksType.MOTIFS

    def has_tracks(self) -> bool:
        return self.type == MotifsOrTracksType.TRACKS
