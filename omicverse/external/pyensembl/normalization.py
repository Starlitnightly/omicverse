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

from sys import intern
from typechecks import is_string, is_integer

# Manually memoizing here, since our simple common.memoize function has
# noticable overhead in this instance.
NORMALIZE_CHROMOSOME_CACHE = {}


def normalize_chromosome(c):
    try:
        return NORMALIZE_CHROMOSOME_CACHE[c]
    except KeyError:
        pass

    if not (is_string(c) or is_integer(c)):
        raise TypeError("Chromosome cannot be '%s' : %s" % (c, type(c)))

    result = str(c)

    if result == "0":
        raise ValueError("Chromosome name cannot be 0")
    elif result == "":
        raise ValueError("Chromosome name cannot be empty")

    if result.startswith("chr") and "_" not in result:
        # excluding "_" for names like "chrUn_gl000212"
        # capitalize "chrx" -> "chrX"
        result = "chr" + result[3:].upper()
    elif result.isalpha():
        # capitalize e.g. "x" -> "X"
        result = result.upper()

    # interning strings since the chromosome names probably get constructed
    # or parsed millions of times, can save memory in tight situations
    # (such as parsing GTF files)
    result = intern(result)

    NORMALIZE_CHROMOSOME_CACHE[c] = result

    return result


def normalize_strand(strand):
    if strand == "+" or strand == 1 or strand == "+1" or strand == "1":
        return "+"
    elif strand == "-" or strand == -1 or strand == "-1":
        return "-"
    raise ValueError("Invalid strand: %s" % (strand,))
