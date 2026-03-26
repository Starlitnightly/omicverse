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


from .locus_with_genome import LocusWithGenome


class Gene(LocusWithGenome):
    def __init__(self, gene_id, gene_name, contig, start, end, strand, biotype, genome):
        LocusWithGenome.__init__(
            self,
            contig=contig,
            start=start,
            end=end,
            strand=strand,
            biotype=biotype,
            genome=genome,
        )
        self.gene_id = gene_id
        self.gene_name = gene_name

    @property
    def id(self):
        """
        Alias for gene_id necessary for backwards compatibility.
        """
        return self.gene_id

    @property
    def name(self):
        """
        Alias for gene_name necessary for backwards compatibility.
        """
        return self.gene_name

    def __str__(self):
        return (
            "Gene(gene_id='%s',"
            " gene_name='%s',"
            " biotype='%s',"
            " contig='%s',"
            " start=%d,"
            " end=%d, strand='%s', genome='%s')"
        ) % (
            self.gene_id,
            self.gene_name,
            self.biotype,
            self.contig,
            self.start,
            self.end,
            self.strand,
            self.genome.reference_name,
        )

    def __eq__(self, other):
        return (
            other.__class__ is Gene
            and self.id == other.id
            and self.genome == other.genome
        )

    def __hash__(self):
        return hash(self.id)

    def to_dict(self):
        state_dict = LocusWithGenome.to_dict(self)
        state_dict["gene_id"] = self.gene_id
        state_dict["gene_name"] = self.gene_name
        return state_dict

    #@memoized_property
    def transcripts(self):
        """
        Property which dynamically construct transcript objects for all
        transcript IDs associated with this gene.
        """
        transcript_id_results = self.db.query(
            select_column_names=["transcript_id"],
            filter_column="gene_id",
            filter_value=self.id,
            feature="transcript",
            distinct=False,
            required=False,
        )

        # We're doing a SQL query for each transcript ID to fetch
        # its particular information, might be more efficient if we
        # just get all the columns here, but how do we keep that modular?
        return [
            self.genome.transcript_by_id(result[0]) for result in transcript_id_results
        ]

    #@memoized_property
    def exons(self):
        exon_set = set([])
        for transcript in self.transcripts:
            for exon in transcript.exons:
                exon_set.add(exon)
        return list(sorted(exon_set))
