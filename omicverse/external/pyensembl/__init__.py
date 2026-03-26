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

from .database import Database
from .download_cache import DownloadCache
from .ensembl_release import EnsemblRelease, cached_release
from .ensembl_versions import MAX_ENSEMBL_RELEASE
from .exon import Exon
from .genome import Genome
from .gene import Gene
from .locus import Locus
from .reference_name import (
    ensembl_grch36,
    ensembl_grch37,
    ensembl_grch38,
    normalize_reference_name,
    find_species_by_reference,
    which_reference,
    genome_for_reference_name,
)

from .search import find_nearest_locus
from .sequence_data import SequenceData
from .species import find_species_by_name, check_species_object, normalize_species_name
from .transcript import Transcript
from .version import __version__

__all__ = [
    "__version__",
    "DownloadCache",
    "Database",
    "EnsemblRelease",
    "cached_release",
    "MAX_ENSEMBL_RELEASE",
    "Gene",
    "Transcript",
    "Exon",
    "SequenceData",
    "find_nearest_locus",
    "find_species_by_name",
    "find_species_by_reference",
    "genome_for_reference_name",
    "which_reference",
    "check_species_object",
    "normalize_reference_name",
    "normalize_species_name",
    "Genome",
    "Locus",
    "Exon",
    "ensembl_grch36",
    "ensembl_grch37",
    "ensembl_grch38",
]
