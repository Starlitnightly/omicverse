# Copyright (c) 2015-2016. Mount Sinai School of Medicine
#
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

"""
The worse sin in bioinformatics is to write your own FASTA parser.
Unfortunately, small errors creep in to different FASTA files on the
Ensembl FTP server that no proper FASTA parser lets you skip over.
"""


from gzip import GzipFile
import logging


logger = logging.getLogger(__name__)


def _parse_header_id(line):
    """
    Pull the transcript or protein identifier from the header line
    which starts with '>'
    """
    if type(line) is not bytes:
        raise TypeError(
            "Expected header line to be of type %s but got %s" % (bytes, type(line))
        )

    if len(line) <= 1:
        raise ValueError("No identifier on FASTA line")

    # split line at first space to get the unique identifier for
    # this sequence
    space_index = line.find(b" ")
    if space_index >= 0:
        identifier = line[1:space_index]
    else:
        identifier = line[1:]

    # annoyingly Ensembl83 reformatted the transcript IDs of its
    # cDNA FASTA to include sequence version numbers
    # .e.g.
    # "ENST00000448914.1" instead of "ENST00000448914"
    # So now we have to parse out the identifier

    # only split name of ENSEMBL naming. In other database, such as TAIR,
    # the '.1' notation is the isoform not the version.
    if identifier.startswith(b"ENS"):
        dot_index = identifier.find(b".")
        if dot_index >= 0:
            identifier = identifier[:dot_index]

    return identifier.decode("ascii")


class FastaParser(object):
    """
    FastaParser object consumes lines of a FASTA file incrementally
    while building up a dictionary mapping sequence identifiers to sequences.
    """

    def __init__(self):
        self.current_id = None
        self.current_lines = []

    def read_file(self, fasta_path):
        """
        Read the contents of a FASTA file into a dictionary
        """
        fasta_dictionary = {}
        for identifier, sequence in self.iterate_over_file(fasta_path):
            fasta_dictionary[identifier] = sequence
        return fasta_dictionary

    def iterate_over_file(self, fasta_path):
        """
        Generator that yields identifiers paired with sequences.
        """
        with self._open(fasta_path) as f:
            for line in f:
                line = line.rstrip()

                if len(line) == 0:
                    continue

                # have to slice into a bytes object or else I get a single integer
                first_char = line[0:1]

                if first_char == b">":
                    id_and_seq = self._read_header(line)
                    if id_and_seq is not None:
                        yield id_and_seq

                elif first_char == b";":
                    # semicolon are comment characters
                    continue
                else:
                    self.current_lines.append(line)
        # the last sequence is still in the lines buffer after we're done with
        # the file so make sure to yield it
        id_and_seq = self._current_entry()
        if id_and_seq is not None:
            yield id_and_seq

    def _open(self, fasta_path):
        """
        Open either a text file or compressed gzip file as a stream of bytes.
        """
        if fasta_path.endswith("gz") or fasta_path.endswith("gzip"):
            return GzipFile(fasta_path, "rb")
        else:
            return open(fasta_path, "rb")

    def _current_entry(self):
        # when we hit a new entry, if this isn't the first
        # entry of the file then put the last one in the dictionary
        if self.current_id:
            if len(self.current_lines) == 0:
                logger.warn("No sequence data for '%s'", self.current_id)
            else:
                sequence = b"".join(self.current_lines)
                sequence = sequence.decode("ascii")
                return self.current_id, sequence

    def _read_header(self, line):
        previous_entry = self._current_entry()

        self.current_id = _parse_header_id(line)

        if len(self.current_id) == 0:
            logger.warn("Unable to parse ID from header line: %s", line)

        self.current_lines = []
        return previous_entry


def parse_fasta_dictionary(fasta_path):
    """
    Given a path to a FASTA (or compressed FASTA) file, returns a dictionary
    mapping its sequence identifiers to sequences.

    Parameters
    ----------
    fasta_path : str
        Path to the FASTA file.

    Returns dictionary from string identifiers to string sequences.
    """
    parser = FastaParser()
    return parser.read_file(fasta_path)
