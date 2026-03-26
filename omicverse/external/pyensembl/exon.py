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


class Exon(Locus):
    def __init__(self, exon_id, contig, start, end, strand, gene_name, gene_id):
        Locus.__init__(self, contig, start, end, strand)
        self.exon_id = exon_id
        self.gene_name = gene_name
        self.gene_id = gene_id

    @property
    def id(self):
        """
        Alias for exon_id necessary for backward compatibility.
        """
        return self.exon_id

    def __str__(self):
        return (
            "Exon(exon_id='%s',"
            " gene_id='%s',"
            " gene_name='%s',"
            " contig='%s',"
            " start=%d,"
            " end=%s,"
            " strand='%s')"
        ) % (
            self.exon_id,
            self.gene_id,
            self.gene_name,
            self.contig,
            self.start,
            self.end,
            self.strand,
        )

    def __eq__(self, other):
        if not isinstance(other, Exon):
            raise TypeError(
                "Cannot compare %s and %s"
                % (self.__class__.__name__, other.__class.__name__)
            )
        return (
            self.contig == other.contig
            and self.start == other.start
            and self.end == other.end
            and self.strand == other.strand
            and self.id == other.id
        )

    def __hash__(self):
        return hash(self.id)

    def to_dict(self):
        state_dict = Locus.to_dict(self)
        state_dict["exon_id"] = self.id
        state_dict["gene_name"] = self.gene_name
        state_dict["gene_id"] = self.gene_id
        return state_dict
