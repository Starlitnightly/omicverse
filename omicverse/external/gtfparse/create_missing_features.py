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

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_missing_features(
        dataframe,
        unique_keys={},
        extra_columns={},
        missing_value=None):
    """
    Helper function used to construct a missing feature such as 'transcript'
    or 'gene'. Some GTF files only have 'exon' and 'CDS' entries, but have
    transcript_id and gene_id annotations which allow us to construct those
    missing features.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Should contain at least the core GTF columns, such as "seqname",
        "start", and "end"

    unique_keys : dict
        Mapping from feature names to the name of the column which should
        act as a unique key for that feature. Example: {"gene": "gene_id"}

    extra_columns : dict
        By default the constructed feature row will include only the 8
        core columns and its unique key. Any other columns that should
        be included should be associated with the feature name in this
        dict.

    missing_value : any
        Which value to fill in for columns that we don't infer values for.

    Returns original dataframe (converted to Pandas if necessary) along with all 
    extra rows created for missing features.
    """
    if hasattr(dataframe, "to_pandas"):
        dataframe = dataframe.to_pandas()
  
    extra_dataframes = []

    existing_features = set(dataframe["feature"])
    existing_columns = set(dataframe.columns)
 
    for (feature_name, groupby_key) in unique_keys.items():
        
        if feature_name in existing_features:
            logging.info(
                "Feature '%s' already exists in GTF data" % feature_name)
            continue
        logging.info("Creating rows for missing feature '%s'" % feature_name)

        # don't include rows where the groupby key was missing
        missing = pd.Series([
            x is None or x == ""
            for x in dataframe[groupby_key]])
        not_missing = ~missing
        row_groups = dataframe[not_missing].groupby(groupby_key)

        # Each group corresponds to a unique feature entry for which the
        # other columns may or may not be uniquely defined. Start off by
        # assuming the values for every column are missing and fill them in
        # where possible.
        feature_values = OrderedDict([
            (column_name, [missing_value] * row_groups.ngroups)
            for column_name in dataframe.keys()
        ])

        # User specifies which non-required columns should we try to infer
        # values for
        feature_columns = list(extra_columns.get(feature_name, []))

        for i, (feature_id, group) in enumerate(row_groups):
            # fill in the required columns by assuming that this feature
            # is the union of all intervals of other features that were
            # tagged with its unique ID (e.g. union of exons which had a
            # particular gene_id).
            feature_values["feature"][i] = feature_name
            feature_values[groupby_key][i] = feature_id
            # set the source to 'gtfparse' to indicate that we made this
            # entry up from other data
            feature_values["source"][i] = "gtfparse"
            feature_values["start"][i] = group["start"].min()
            feature_values["end"][i] = group["end"].max()

            # assume that seqname and strand are the same for all other
            # entries in the GTF which shared this unique ID
            feature_values["seqname"][i] = group["seqname"].iat[0]
            feature_values["strand"][i] = group["strand"].iat[0]

            # there's probably no rigorous way to set the values of
            # 'score' or 'frame' columns so leave them empty
            for column_name in feature_columns:
                if column_name not in existing_columns:
                    raise ValueError(
                        "Column '%s' does not exist in GTF, columns = %s" % (
                            column_name, existing_columns))

                # expect that all entries related to a reconstructed feature
                # are related and are thus within the same interval of
                # positions on the same chromosome
                unique_values = group[column_name].dropna().unique()
                if len(unique_values) == 1:
                    feature_values[column_name][i] = unique_values[0]
        extra_dataframes.append(pd.DataFrame(feature_values))
    return pd.concat([dataframe] + extra_dataframes, ignore_index=True)
