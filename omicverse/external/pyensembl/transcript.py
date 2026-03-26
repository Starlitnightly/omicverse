# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import cached_property

from .common import memoize
from .locus_with_genome import LocusWithGenome


class Transcript(LocusWithGenome):
    """
    Transcript encompasses the locus, exons, and sequence of a transcript.

    Lazily fetches sequence in case we"re constructing many Transcripts
    and not using the sequence, avoid the memory/performance overhead
    of fetching and storing sequences from a FASTA file.
    """

    def __init__(
        self,
        transcript_id,
        transcript_name,
        contig,
        start,
        end,
        strand,
        biotype,
        gene_id,
        genome,
        support_level=None,
    ):
        LocusWithGenome.__init__(
            self,
            contig=contig,
            start=start,
            end=end,
            strand=strand,
            biotype=biotype,
            genome=genome,
        )
        self.transcript_id = transcript_id
        self.transcript_name = transcript_name
        self.gene_id = gene_id
        self.support_level = support_level

    @property
    def id(self):
        """
        Alias for transcript_id necessary for backward compatibility.
        """
        return self.transcript_id

    @property
    def name(self):
        """
        Alias for transcript_name necessary for backward compatibility.
        """
        return self.transcript_name

    def __str__(self):
        return (
            "Transcript(transcript_id='%s',"
            " transcript_name='%s',"
            " gene_id='%s',"
            " biotype='%s',"
            " contig='%s',"
            " start=%d,"
            " end=%d, strand='%s', genome='%s')"
        ) % (
            self.transcript_id,
            self.name,
            self.gene_id,
            self.biotype,
            self.contig,
            self.start,
            self.end,
            self.strand,
            self.genome.reference_name,
        )

    def __len__(self):
        """
        Length of a transcript is the sum of its exon lengths
        """
        return sum(len(exon) for exon in self.exons)

    def __eq__(self, other):
        return (
            other.__class__ is Transcript
            and self.id == other.id
            and self.genome == other.genome
        )

    def __hash__(self):
        return hash(self.id)

    def to_dict(self):
        state_dict = LocusWithGenome.to_dict(self)
        state_dict["transcript_id"] = self.transcript_id
        state_dict["transcript_name"] = self.name
        state_dict["gene_id"] = self.gene_id
        state_dict["support_level"] = self.support_level
        return state_dict

    @property
    def gene(self):
        return self.genome.gene_by_id(self.gene_id)

    @property
    def gene_name(self):
        return self.gene.name

    @property
    def exons(self):
        # need to look up exon_number alongside ID since each exon may
        # appear in multiple transcripts and have a different exon number
        # in each transcript
        columns = ["exon_number", "exon_id"]
        exon_numbers_and_ids = self.db.query(
            columns, filter_column="transcript_id", filter_value=self.id, feature="exon"
        )

        # fill this list in its correct order (by exon_number) by using
        # the exon_number as a 1-based list offset
        exons = [None] * len(exon_numbers_and_ids)

        for exon_number, exon_id in exon_numbers_and_ids:
            exon = self.genome.exon_by_id(exon_id)
            if exon is None:
                raise ValueError(
                    "Missing exon %s for transcript %s" % (exon_number, self.id)
                )
            exon_number = int(exon_number)
            if exon_number < 1:
                raise ValueError("Invalid exon number: %s" % exon_number)
            elif exon_number > len(exons):
                raise ValueError(
                    "Invalid exon number: %s (max expected = %d)"
                    % (exon_number, len(exons))
                )

            # exon_number is 1-based, convert to list index by subtracting 1
            exon_idx = exon_number - 1
            exons[exon_idx] = exon
        return exons

    # possible annotations associated with transcripts
    _TRANSCRIPT_FEATURES = {"start_codon", "stop_codon", "UTR", "CDS"}

    @memoize
    def _transcript_feature_position_ranges(self, feature, required=True):
        """
        Find start/end chromosomal position range of features
        (such as start codon) for this transcript.
        """
        if feature not in self._TRANSCRIPT_FEATURES:
            raise ValueError("Invalid transcript feature: %s" % feature)

        results = self.db.query(
            select_column_names=["start", "end"],
            filter_column="transcript_id",
            filter_value=self.id,
            feature=feature,
        )

        if required and len(results) == 0:
            raise ValueError(
                "Transcript %s does not contain feature %s" % (self.id, feature)
            )
        return results

    @memoize
    def _transcript_feature_positions(self, feature):
        """
        Get unique positions for feature, raise an error if feature is absent.
        """
        ranges = self._transcript_feature_position_ranges(feature, required=True)
        results = []
        # a feature (such as a stop codon), maybe be split over multiple
        # contiguous ranges. Collect all the nucleotide positions into a
        # single list.
        for start, end in ranges:
            # since ranges are [inclusive, inclusive] and
            # Python ranges are [inclusive, exclusive) we have to increment
            # the end position
            for position in range(start, end + 1):
                if position in results:
                    raise ValueError(
                        "Repeated position %d for %s" % (position, feature)
                    )
                results.append(position)
        return results

    @memoize
    def _codon_positions(self, feature):
        """
        Parameters
        ----------
        feature : str
            Possible values are "start_codon" or "stop_codon"

        Returns list of three chromosomal positions.
        """
        results = self._transcript_feature_positions(feature)
        if len(results) != 3:
            raise ValueError(
                "Expected 3 positions for %s of %s but got %d"
                % (feature, self.id, len(results))
            )
        return results

    @cached_property
    def contains_start_codon(self):
        """
        Does this transcript have an annotated start_codon entry?
        """
        start_codons = self._transcript_feature_position_ranges(
            "start_codon", required=False
        )
        return len(start_codons) > 0

    @cached_property
    def contains_stop_codon(self):
        """
        Does this transcript have an annotated stop_codon entry?
        """
        stop_codons = self._transcript_feature_position_ranges(
            "stop_codon", required=False
        )
        return len(stop_codons) > 0

    @cached_property
    def start_codon_complete(self):
        """
        Does the start codon span 3 genomic positions?
        """
        try:
            self._codon_positions("start_codon")
        except ValueError:
            return False
        return True

    @cached_property
    def start_codon_positions(self):
        """
        Chromosomal positions of nucleotides in start codon.
        """
        return self._codon_positions("start_codon")

    @cached_property
    def stop_codon_positions(self):
        """
        Chromosomal positions of nucleotides in stop codon.
        """
        return self._codon_positions("stop_codon")

    @cached_property
    def exon_intervals(self):
        """List of (start,end) tuples for each exon of this transcript,
        in the order specified by the 'exon_number' column of the
        exon table.
        """
        results = self.db.query(
            select_column_names=["exon_number", "start", "end"],
            filter_column="transcript_id",
            filter_value=self.id,
            feature="exon",
        )
        sorted_intervals = [None] * len(results)
        for exon_number, start, end in results:
            sorted_intervals[int(exon_number) - 1] = (start, end)
        return sorted_intervals

    def spliced_offset(self, position):
        """
        Convert from an absolute chromosomal position to the offset into
        this transcript"s spliced mRNA.

        Position must be inside some exon (otherwise raise exception).
        """
        if type(position) is not int:
            raise TypeError(
                "Position argument must be an integer, got %s : %s"
                % (position, type(position))
            )

        if position < self.start or position > self.end:
            raise ValueError(
                "Invalid position: %d (must be between %d and %d)"
                % (position, self.start, self.end)
            )

        # offset from beginning of unspliced transcript (including introns)
        unspliced_offset = self.offset(position)
        total_spliced_offset = 0

        # traverse exons in order of their appearance on the strand
        # Since absolute positions may decrease if on the negative strand,
        # we instead use unspliced offsets to get always increasing indices.
        #
        # Example:
        #
        # Exon Name:                exon 1                exon 2
        # Spliced Offset:           123456                789...
        # Intron vs. Exon: ...iiiiiieeeeeeiiiiiiiiiiiiiiiieeeeeeiiiiiiiiiii...
        for exon in self.exons:
            exon_unspliced_start, exon_unspliced_end = self.offset_range(
                exon.start, exon.end
            )
            # If the relative position is not within this exon, keep a running
            # total of the total exonic length-so-far.
            #
            # Otherwise, if the relative position is within an exon, get its
            # offset into that exon by subtracting the exon"s relative start
            # position from the relative position. Add that to the total exonic
            # length-so-far.
            if exon_unspliced_start <= unspliced_offset <= exon_unspliced_end:
                # all offsets are base 0, can be used as indices into
                # sequence string
                exon_offset = unspliced_offset - exon_unspliced_start
                return total_spliced_offset + exon_offset
            else:
                exon_length = len(exon)  # exon_end_position - exon_start_position + 1
                total_spliced_offset += exon_length
        raise ValueError(
            "Couldn't find position %d on any exon of %s" % (position, self.id)
        )

    @cached_property
    def start_codon_unspliced_offsets(self):
        """
        Offsets from start of unspliced pre-mRNA transcript
        of nucleotides in start codon.
        """
        return [self.offset(position) for position in self.start_codon_positions]

    @cached_property
    def stop_codon_unspliced_offsets(self):
        """
        Offsets from start of unspliced pre-mRNA transcript
        of nucleotides in stop codon.
        """
        return [self.offset(position) for position in self.stop_codon_positions]

    def _contiguous_offsets(self, offsets):
        """
        Sorts the input list of integer offsets,
        ensures that values are contiguous.
        """
        offsets.sort()
        for i in range(len(offsets) - 1):
            if offsets[i] + 1 != offsets[i + 1]:
                raise ValueError("Offsets not contiguous: %s" % (offsets,))
        return offsets

    @cached_property
    def start_codon_spliced_offsets(self):
        """
        Offsets from start of spliced mRNA transcript
        of nucleotides in start codon.
        """
        offsets = [
            self.spliced_offset(position) for position in self.start_codon_positions
        ]
        return self._contiguous_offsets(offsets)

    @cached_property
    def stop_codon_spliced_offsets(self):
        """
        Offsets from start of spliced mRNA transcript
        of nucleotides in stop codon.
        """
        offsets = [
            self.spliced_offset(position) for position in self.stop_codon_positions
        ]
        return self._contiguous_offsets(offsets)

    @cached_property
    def coding_sequence_position_ranges(self):
        """
        Return absolute chromosome position ranges for CDS fragments
        of this transcript
        """
        return self._transcript_feature_position_ranges("CDS")

    @cached_property
    def complete(self):
        """
        Consider a transcript complete if it has start and stop codons and
        a coding sequence whose length is divisible by 3
        """
        return (
            self.contains_start_codon
            and self.start_codon_complete
            and self.contains_stop_codon
            and self.coding_sequence is not None
            and len(self.coding_sequence) % 3 == 0
        )

    @cached_property
    def sequence(self):
        """
        Spliced cDNA sequence of transcript
        (includes 5" UTR, coding sequence, and 3" UTR)
        """
        transcript_id = self.transcript_id
        if transcript_id.startswith("ENS"):
            transcript_id = transcript_id.rsplit(".", 1)[0]
        return self.genome.transcript_sequences.get(transcript_id)

    @cached_property
    def first_start_codon_spliced_offset(self):
        """
        Offset of first nucleotide in start codon into the spliced mRNA
        (excluding introns)
        """
        start_offsets = self.start_codon_spliced_offsets
        return min(start_offsets)

    @cached_property
    def last_stop_codon_spliced_offset(self):
        """
        Offset of last nucleotide in stop codon into the spliced mRNA
        (excluding introns)
        """
        stop_offsets = self.stop_codon_spliced_offsets
        return max(stop_offsets)

    @cached_property
    def coding_sequence(self):
        """
        cDNA coding sequence (from start codon to stop codon, without
        any introns)
        """
        if self.sequence is None:
            return None

        start = self.first_start_codon_spliced_offset
        end = self.last_stop_codon_spliced_offset

        # If start codon is the at nucleotide offsets [3,4,5] and
        # stop codon is at nucleotide offsets  [20,21,22]
        # then start = 3 and end = 22.
        #
        # Adding 1 to end since Python uses non-inclusive ends in slices/ranges.

        # pylint: disable=invalid-slice-index
        # TODO(tavi) Figure out pylint is not happy with this slice
        return self.sequence[start : end + 1]

    @cached_property
    def five_prime_utr_sequence(self):
        """
        cDNA sequence of 5' UTR
        (untranslated region at the beginning of the transcript)
        """
        # pylint: disable=invalid-slice-index
        # TODO(tavi) Figure out pylint is not happy with this slice
        return self.sequence[: self.first_start_codon_spliced_offset]

    @cached_property
    def three_prime_utr_sequence(self):
        """
        cDNA sequence of 3' UTR
        (untranslated region at the end of the transcript)
        """
        return self.sequence[self.last_stop_codon_spliced_offset + 1 :]

    @cached_property
    def protein_id(self):
        result_tuple = self.db.query_one(
            select_column_names=["protein_id"],
            filter_column="transcript_id",
            filter_value=self.id,
            feature="CDS",
            distinct=True,
            required=False,
        )
        if result_tuple:
            return result_tuple[0]
        else:
            return None

    @cached_property
    def protein_sequence(self):
        if self.protein_id:
            return self.genome.protein_sequences.get(self.protein_id)
        else:
            return None
