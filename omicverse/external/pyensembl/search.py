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
Helper functions for searching over collections of PyEnsembl objects
"""


def find_nearest_locus(start, end, loci):
    """
    Finds nearest locus (object with method `distance_to_interval`) to the
    interval defined by the given `start` and `end` positions.
    Returns the distance to that locus, along with the locus object itself.
    """
    best_distance = float("inf")
    best_locus = None
    for locus in loci:
        distance = locus.distance_to_interval(start, end)
        if best_distance > distance:
            best_distance = distance
            best_locus = locus
    return best_distance, best_locus
