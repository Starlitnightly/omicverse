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

from .ensembl_release import EnsemblRelease
from .species import Species, find_species_by_name


def normalize_reference_name(name):
    """
    Search the dictionary of species-specific references to find a reference
    name that matches aside from capitalization.

    If no matching reference is found, raise an exception.
    """
    lower_name = name.strip().lower()
    for reference in Species._reference_names_to_species.keys():
        if reference.lower() == lower_name:
            return reference
    raise ValueError("Reference genome '%s' not found" % name)


def find_species_by_reference(reference_name):
    return Species._reference_names_to_species[normalize_reference_name(reference_name)]


def which_reference(species_name, ensembl_release):
    return find_species_by_name(species_name).which_reference(ensembl_release)


def max_ensembl_release(reference_name):
    species = find_species_by_reference(reference_name)
    (_, max_release) = species.reference_assemblies[reference_name]
    return max_release


def genome_for_reference_name(reference_name, allow_older_downloaded_release=True):
    """
    Given a genome reference name, such as "GRCh38", returns the
    corresponding Ensembl Release object.

    If `allow_older_downloaded_release` is True, and some older releases have
    been downloaded, then return the most recent locally available release.

    Otherwise, return the newest release of Ensembl (even if its data hasn't
    already been downloaded).
    """
    reference_name = normalize_reference_name(reference_name)
    species = find_species_by_reference(reference_name)
    (min_ensembl_release, max_ensembl_release) = species.reference_assemblies[
        reference_name
    ]
    if allow_older_downloaded_release:
        # go through candidate releases in descending order
        for release in reversed(range(min_ensembl_release, max_ensembl_release + 1)):
            # check if release has been locally downloaded
            candidate = EnsemblRelease.cached(release=release, species=species)
            if candidate.required_local_files_exist():
                return candidate
        # see if any of the releases between [max, min] are already locally
        # available
    return EnsemblRelease.cached(release=max_ensembl_release, species=species)


ensembl_grch36 = genome_for_reference_name("ncbi36")
ensembl_grch37 = genome_for_reference_name("grch37")
ensembl_grch38 = genome_for_reference_name("grch38")
