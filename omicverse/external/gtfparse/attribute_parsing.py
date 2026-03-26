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

import logging
from collections import OrderedDict
from sys import intern

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def expand_attribute_strings(
        attribute_strings,
        quote_char="'",
        missing_value="",
        usecols=None):
    """
    The last column of GTF has a variable number of key value pairs
    of the format: "key1 value1; key2 value2;"
    Parse these into a dictionary mapping each key onto a list of values,
    where the value is None for any row where the key was missing.

    Parameters
    ----------
    attribute_strings : list of str

    quote_char : str
        Quote character to remove from values

    missing_value : any
        If an attribute is missing from a row, give it this value.

    usecols : list of str or None
        If not None, then only expand columns included in this set,
        otherwise use all columns.

    Returns OrderedDict of column->value list mappings, in the order they
    appeared in the attribute strings.
    """
    n = len(attribute_strings)

    extra_columns = {}
    column_order = []

    #
    # SOME NOTES ABOUT THE BIZARRE STRING INTERNING GOING ON BELOW
    #
    # While parsing millions of repeated strings (e.g. "gene_id" and "TP53"),
    # we can save a lot of memory by making sure there's only one string
    # object per unique string. The canonical way to do this is using
    # the 'intern' function. One problem is that Py2 won't let you intern
    # unicode objects, so to get around this we call intern(str(...)).
    #
    # It also turns out to be faster to check interned strings ourselves
    # using a local dictionary, hence the two dictionaries below
    # and pair of try/except blocks in the loop.
    column_interned_strings = {}

    for (i, kv_strings) in enumerate(attribute_strings):
        if type(kv_strings) is str:
            kv_strings = kv_strings.split(";")
        for kv in kv_strings:
            # We're slicing the first two elements out of split() because
            # Ensembl release 79 added values like:
            #   transcript_support_level "1 (assigned to previous version 5)";
            # ...which gets mangled by splitting on spaces.
            parts = kv.strip().split(" ", 2)[:2]

            if len(parts) != 2:
                continue

            column_name, value = parts

            try:
                column_name = column_interned_strings[column_name]
            except KeyError:
                column_name = intern(str(column_name))
                column_interned_strings[column_name] = column_name

            if usecols is not None and column_name not in usecols:
                continue

            # Smart quote removal: detect and remove both single and double quotes
            # Different GTF sources use different quote styles:
            # - Older Ensembl releases or human GTFs may use double quotes
            # - Newer releases or mouse GTFs may use single quotes
            if len(value) >= 2:
                if (value[0] == '"' and value[-1] == '"') or \
                   (value[0] == "'" and value[-1] == "'"):
                    value = value[1:-1]
                elif value[0] == quote_char:
                    # Fallback to original logic for backwards compatibility
                    value = value.replace(quote_char, "")
                
            try:
                column = extra_columns[column_name]
                # if an attribute is used repeatedly then
                # keep track of all its values in a list
                old_value = column[i]
                if old_value is missing_value:
                    column[i] = value
                else:
                    column[i] = "%s,%s" % (old_value, value)
            except KeyError:
                column = [missing_value] * n
                column[i] = value
                extra_columns[column_name] = column
                column_order.append(column_name)



    logging.info("Extracted GTF attributes: %s" % column_order)
    return OrderedDict(
        (column_name, extra_columns[column_name])
        for column_name in column_order)
