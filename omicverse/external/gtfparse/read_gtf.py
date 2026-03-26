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
import subprocess
import sys
from os.path import exists

# Auto-install polars if not available
try:
    import polars
except ImportError:
    logger = logging.getLogger(__name__)
    logger.info("Polars not found. Installing polars...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "polars"])
        import polars
        logger.info("Polars installed successfully!")
    except Exception as e:
        print(
            f"Failed to install polars automatically: {e}\n"
            "Please install manually: pip install polars"
        )
        polars = None

from .attribute_parsing import expand_attribute_strings
from .parsing_error import ParsingError


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


"""
Columns of a GTF file:

    seqname   - name of the chromosome or scaffold; chromosome names
                without a 'chr' in Ensembl (but sometimes with a 'chr'
                elsewhere)
    source    - name of the program that generated this feature, or
                the data source (database or project name)
    feature   - feature type name.
                Features currently in Ensembl GTFs:
                    gene
                    transcript
                    exon
                    CDS
                    Selenocysteine
                    start_codon
                    stop_codon
                    UTR
                Older Ensembl releases may be missing some of these features.
    start     - start position of the feature, with sequence numbering
                starting at 1.
    end       - end position of the feature, with sequence numbering
                starting at 1.
    score     - a floating point value indiciating the score of a feature
    strand    - defined as + (forward) or - (reverse).
    frame     - one of '0', '1' or '2'. Frame indicates the number of base pairs
                before you encounter a full codon. '0' indicates the feature
                begins with a whole codon. '1' indicates there is an extra
                base (the 3rd base of the prior codon) at the start of this feature.
                '2' indicates there are two extra bases (2nd and 3rd base of the
                prior exon) before the first codon. All values are given with
                relation to the 5' end.
    attribute - a semicolon-separated list of tag-value pairs (separated by a space),
                providing additional information about each feature. A key can be
                repeated multiple times.

(from ftp://ftp.ensembl.org/pub/release-75/gtf/homo_sapiens/README)
"""

REQUIRED_COLUMNS = [
    "seqname",
    "source",
    "feature",
    "start",
    "end",
    "score",
    "strand",
    "frame",
    "attribute",
]


DEFAULT_COLUMN_DTYPES = {
    "seqname": polars.Categorical, 
    "source": polars.Categorical, 
    
    "start": polars.Int64,
    "end": polars.Int64,
    "score": polars.Float32,

    "feature": polars.Categorical, 
    "strand": polars.Categorical, 
    "frame": polars.UInt32,
}

def parse_with_polars_lazy(
        filepath_or_buffer,
        split_attributes=True,
        features=None,
        fix_quotes_columns=["attribute"]):
    # use a global string cache so that all strings get intern'd into
    # a single numbering system
    polars.enable_string_cache()
    kwargs = dict(
        has_header=False,
        separator="\t",
        comment_prefix="#",
        null_values=".",
        schema_overrides=DEFAULT_COLUMN_DTYPES)
    try:
        df = polars.read_csv(
                filepath_or_buffer,
                new_columns=REQUIRED_COLUMNS,
                **kwargs).lazy()
    except polars.exceptions.ShapeError:
        raise ParsingError("Wrong number of columns")

    df = df.with_columns([
        polars.col("frame").fill_null(0),
        polars.col("attribute").str.replace_all('"', "'")
    ])
    
    for fix_quotes_column in fix_quotes_columns:
        # Catch mistaken semicolons by replacing "xyz;" with "xyz"
        # Required to do this since the Ensembl GTF for Ensembl
        # release 78 has mistakes such as:
        #   gene_name = "PRAMEF6;" transcript_name = "PRAMEF6;-201"
        df = df.with_columns([
            polars.col(fix_quotes_column).str.replace(';\"', '\"').str.replace(";-", "-")
        ])

    if features is not None:
        features = sorted(set(features))
        df = df.filter(polars.col("feature").is_in(features))


    if split_attributes:
        df = df.with_columns([
            polars.col("attribute").str.split(";").alias("attribute_split")
        ])
    return df

def parse_gtf(
        filepath_or_buffer, 
        split_attributes=True, 
        features=None,
        fix_quotes_columns=["attribute"]):
    df_lazy = parse_with_polars_lazy(
        filepath_or_buffer=filepath_or_buffer,
        split_attributes=split_attributes,
        features=features,
        fix_quotes_columns=fix_quotes_columns)
    return df_lazy.collect()

def parse_gtf_pandas(*args, **kwargs):
    return parse_gtf(*args, **kwargs).to_pandas()

    
def parse_gtf_and_expand_attributes(
        filepath_or_buffer,
        restrict_attribute_columns=None,
        features=None):
    """
    Parse lines into column->values dictionary and then expand
    the 'attribute' column into multiple columns. This expansion happens
    by replacing strings of semi-colon separated key-value values in the
    'attribute' column with one column per distinct key, with a list of
    values for each row (using None for rows where key didn't occur).

    Parameters
    ----------
    filepath_or_buffer : str or buffer object

    chunksize : int

    restrict_attribute_columns : list/set of str or None
        If given, then only use these attribute columns.

    features : set or None
        Ignore entries which don't correspond to one of the supplied features
    """
    df = parse_gtf(
        filepath_or_buffer=filepath_or_buffer, 
        features=features,
        split_attributes=True)
    if type(restrict_attribute_columns) is str:
        restrict_attribute_columns = {restrict_attribute_columns}
    elif restrict_attribute_columns:
        restrict_attribute_columns = set(restrict_attribute_columns)
    df.drop_in_place("attribute")
    attribute_pairs = df.drop_in_place("attribute_split")
    return df.with_columns([
        polars.Series(k, vs)
        for (k, vs) in 
        expand_attribute_strings(attribute_pairs).items()
        if restrict_attribute_columns is None or k in restrict_attribute_columns
    ])
    

def read_gtf(
        filepath_or_buffer,
        expand_attribute_column=True,
        infer_biotype_column=False,
        column_converters={},
        column_cast_types={},
        usecols=None,
        features=None,
        result_type='polars'):
    """
    Parse a GTF into a dictionary mapping column names to sequences of values.

    Parameters
    ----------
    filepath_or_buffer : str or buffer object
        Path to GTF file (may be gzip compressed) or buffer object
        such as StringIO

    expand_attribute_column : bool
        Replace strings of semi-colon separated key-value values in the
        'attribute' column with one column per distinct key, with a list of
        values for each row (using None for rows where key didn't occur).

    infer_biotype_column : bool
        Due to the annoying ambiguity of the second GTF column across multiple
        Ensembl releases, figure out if an older GTF's source column is actually
        the gene_biotype or transcript_biotype.

    column_converters : dict, optional
        Dictionary mapping column names to conversion functions. Will replace
        empty strings with None and otherwise passes them to given conversion
        function.

    column_cast_types : dict, optional
        Dictionary mapping column names to dtypes. Will cast columns to given
        Polars types.
    
    usecols : list of str or None
        Restrict which columns are loaded to the give set. If None, then
        load all columns.

    features : set of str or None
        Drop rows which aren't one of the features in the supplied set

    result_type : One of 'polars', 'pandas', or 'dict'
        Default behavior is to return a Polars DataFrame, but will convert to 
        Pandas DataFrame or dictionary if specified.
    """
    if type(filepath_or_buffer) is str and not exists(filepath_or_buffer):
        raise ValueError("GTF file does not exist: %s" % filepath_or_buffer)

    if expand_attribute_column:
        result_df = parse_gtf_and_expand_attributes(
            filepath_or_buffer,
            restrict_attribute_columns=usecols,
            features=features)
    else:
        result_df = parse_gtf(result_df, features=features)

    # converting back to pandas here because Polars bugs manifest
    # as `pyo3_runtime.PanicException: assertion `left == right` failed: impl error`
    # and are generally insane to chase down
    result_df = result_df.to_pandas()
    if column_converters or column_cast_types:
        def wrap_to_always_accept_none(f):
            def wrapped_fn(x):
                if x is None or x == "":
                    return None
                else:
                    return f(x)
            return wrapped_fn

        column_names = set(column_converters.keys()).union(column_cast_types.keys())
        # Only process columns that actually exist in the DataFrame
        existing_columns = set(result_df.columns)
        for column_name in column_names:
            if column_name not in existing_columns:
                logging.warning(f"Column '{column_name}' not found in DataFrame, skipping converter")
                continue

            if column_name in column_converters:
                # Convert categorical columns to object dtype first to avoid pandas
                # categorical behavior where functions are applied to all categories.
                # This prevents issues when Polars string cache causes categories
                # to be shared across columns (e.g., 'havana' from source appearing
                # in strand categories).
                if hasattr(result_df[column_name].dtype, 'categories'):
                    result_df[column_name] = result_df[column_name].astype(object)

                column_fn = wrap_to_always_accept_none(
                    column_converters[column_name])
                result_df[column_name] = result_df[column_name].apply(column_fn)

            if column_name in column_cast_types:
                column_type = column_cast_types[column_name]
                result_df[column_name] = result_df[column_name].astype(column_type)
            
    # Hackishly infer whether the values in the 'source' column of this GTF
    # are actually representing a biotype by checking for the most common
    # gene_biotype and transcript_biotype value 'protein_coding'
    if infer_biotype_column:
        unique_source_values = set(result_df["source"])
        if "protein_coding" in unique_source_values:
            column_names = set(result_df.columns)
            # Disambiguate between the two biotypes by checking if
            # gene_biotype is already present in another column. If it is,
            # the 2nd column is the transcript_biotype (otherwise, it's the
            # gene_biotype)
            if "gene_biotype" not in column_names:
                logging.info("Using column 'source' to replace missing 'gene_biotype'")
                result_df['gene_biotype'] = result_df['source']
            if "transcript_biotype" not in column_names:
                logging.info("Using column 'source' to replace missing 'transcript_biotype'")
                result_df['transcript_biotype'] = result_df['source']
                
    if usecols is not None:
        column_names = set(result_df.columns)
        valid_columns = [c for c in usecols if c in column_names]
        result_df = result_df[valid_columns]

    result = None
    if result_type == "pandas":
        result = result_df
    elif result_type == "polars":
        result = polars.from_pandas(result_df)
    elif result_type == "dict":
        result = result_df.to_dict()
    return result
