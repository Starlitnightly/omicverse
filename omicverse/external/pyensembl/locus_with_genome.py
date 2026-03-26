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


from .locus import Locus


class LocusWithGenome(Locus):
    """
    Common base class for Gene and Transcript to avoid copying
    their shared logic.
    """

    def __init__(self, contig, start, end, strand, biotype, genome):
        Locus.__init__(self, contig, start, end, strand)
        self.genome = genome
        self.db = self.genome.db
        self.biotype = biotype

    def to_dict(self):
        return dict(
            contig=self.contig,
            start=self.start,
            end=self.end,
            strand=self.strand,
            biotype=self.biotype,
            genome=self.genome,
        )

    @property
    def is_protein_coding(self):
        """
        We're not counting immunoglobulin-like genes from the T-cell receptor or
        or antibodies since they occur in fragments that must be recombined.
        It might be worth consider counting non-sense mediated decay and
        non-stop decay since variants in these could potentially make a
        functional protein. To read more about the biotypes used in Ensembl:
            http://vega.sanger.ac.uk/info/about/gene_and_transcript_types.html
            http://www.gencodegenes.org/gencode_biotypes.html

        For now let's stick with the simple category of 'protein_coding', which
        means that there is an open reading frame in this gene/transcript
        whose successful transcription has been observed.
        """
        return self.biotype == "protein_coding"
