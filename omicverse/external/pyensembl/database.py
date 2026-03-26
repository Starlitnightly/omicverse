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
from os.path import split, join, exists, splitext
import sqlite3

import datacache
from typechecks import require_integer, require_string
from ..gtfparse import read_gtf, create_missing_features

from .common import memoize
from .normalization import normalize_chromosome, normalize_strand
from .locus import Locus

# any time we update the database schema, increment this version number
DATABASE_SCHEMA_VERSION = 3


logger = logging.getLogger(__name__)


class Database(object):
    """
    Wrapper around sqlite3 database so that the rest of the
    library doesn't have to worry about constructing the .db file or
    writing SQL queries directly.
    """

    def __init__(
        self,
        gtf_path,
        install_string=None,
        cache_directory_path=None,
        restrict_gtf_columns=None,
        restrict_gtf_features=None,
    ):
        """
        Parameters
        ----------
        gtf_path : str
            Path to GTF annotation file

        install_string : str
            Message to tell user if database connection is requested before
            database is created.

        cache_directory_path : str
            Path to directory where database should be written. If omitted
            then use path of GTF file.

        restrict_gtf_columns : list/set of str or None
            If provided then extract only these columns before creating
            a database.

        restrict_gtf_features : list/set of str or None
            If provided then only create tables for these features.

        """
        self.gtf_path = gtf_path
        self.restrict_gtf_columns = restrict_gtf_columns
        self.restrict_gtf_features = restrict_gtf_features
        self.gtf_directory_path, self.gtf_filename = split(self.gtf_path)
        self.gtf_base_filename = splitext(self.gtf_filename)[0]

        # if cache directory isn't given then put cached files
        # alongside the GTF
        if cache_directory_path:
            self.cache_directory_path = cache_directory_path
        else:
            self.cache_directory_path = self.gtf_directory_path

        self.install_string = install_string
        self._connection = None
        # dictionary mapping table names to sets of columns
        self._columns = {}
        self._query_cache = {}

    def __eq__(self, other):
        return other.__class__ is Database and self.gtf_path == other.gtf_path

    def __str__(self):
        return "Database(gtf_path=%s)" % (self.gtf_path,)

    def __hash__(self):
        return hash((self.gtf_path))

    @property
    def local_db_filename(self):
        return self.gtf_base_filename + ".db"

    @property
    def local_db_path(self):
        return join(self.cache_directory_path, self.local_db_filename)

    def _all_possible_indices(self, column_names):
        """
        Create list of tuples containing all possible index groups
        we might want to create over tables in this database.

        If a set of genome annotations is missing some column we want
        to index on, we have to drop any indices which use that column.

        A specific table may later drop some of these indices if they're
        missing  values for that feature or are the same as the table's primary key.
        """
        candidate_column_groups = [
            ["seqname", "start", "end"],
            ["gene_name"],
            ["gene_id"],
            ["transcript_id"],
            ["transcript_name"],
            ["exon_id"],
            ["protein_id"],
            ["ccds_id"],
        ]
        indices = []
        column_set = set(column_names)
        # Since queries are often restricted by feature type
        # we should include that column in combination with all
        # other indices we anticipate might improve performance
        for column_group in candidate_column_groups:
            skip = False
            for column_name in column_group:
                # some columns, such as 'exon_id',
                # are not available in all releases of Ensembl (or
                # other GTFs)
                if column_name not in column_set:
                    logger.info(
                        "Skipping database index for {%s}", ", ".join(column_group)
                    )
                    skip = True
            if skip:
                continue
            indices.append(column_group)
        return indices

    # mapping from database tables to their primary keys
    # sadly exon IDs *are* not unique, so can't be in this dict
    PRIMARY_KEY_COLUMNS = {"gene": "gene_id", "transcript": "transcript_id"}

    def _get_primary_key(self, feature_name, feature_df):
        """Name of primary key for a feature table (e.g. "gene" -> "gene_id")

        Since we're potentially going to run this code over unseen data,
        make sure that the primary is unique and never null.

        If a feature doesn't have a primary key, return None.
        """
        if feature_name not in self.PRIMARY_KEY_COLUMNS:
            return None
        primary_key = self.PRIMARY_KEY_COLUMNS[feature_name]
        primary_key_values = feature_df[primary_key]
        if primary_key_values.isnull().any():
            raise ValueError(
                "Column '%s' can't be primary key of table '%s'"
                " because it contains nulls values" % (primary_key, feature_name)
            )
        elif len(primary_key_values.unique()) < len(primary_key_values):
            raise ValueError(
                "Column '%s' can't be primary key of table '%s'"
                " because it contains repeated values" % (primary_key, feature_name)
            )
        else:
            return primary_key

    def _feature_indices(self, all_index_groups, primary_key, feature_df):
        """Choose subset of index group tuples from `all_index_groups` which are
        applicable to a particular feature (not same as its primary key, have
        non-null values).
        """
        # each feature only gets indices if they're *not* the
        # primary key and have non-null values in the feature's
        # subset of data
        result = []
        for index_group in all_index_groups:
            # is the index group just a primary key?
            if len(index_group) == 1 and index_group[0] == primary_key:
                continue
            index_column_values = feature_df[index_group]
            if len(index_column_values.dropna()) == 0:
                continue
            result.append(index_group)
        return result

    def create(self, overwrite=False):
        """
        Create the local database (including indexing) if it's not
        already set up. If `overwrite` is True, always re-create
        the database from scratch.

        Returns a connection to the database.
        """
        logger.info("Creating database: %s", self.local_db_path)
        datacache.ensure_dir(self.cache_directory_path)

        df = self._load_gtf_as_dataframe(
            usecols=self.restrict_gtf_columns, features=self.restrict_gtf_features
        )
        if hasattr(df, "to_pandas"):
            df = df.to_pandas()
        all_index_groups = self._all_possible_indices(df.columns)

        if self.restrict_gtf_features:
            feature_names = self.restrict_gtf_features
        else:
            # split single DataFrame into dictionary mapping each unique
            # feature name onto that subset of the data
            feature_names = df["feature"].unique()
        dataframes = {}
        # every table gets the same set of indices
        indices_dict = {}
        # if a feature has an ID then make it that table's primary key
        primary_keys = {}

        for feature in feature_names:
            df_subset = df[df["feature"] == feature]
            if len(df_subset) == 0:
                continue
            dataframes[feature] = df_subset

            primary_key = self._get_primary_key(feature, df_subset)
            if primary_key:
                primary_keys[feature] = primary_key

            indices_dict[feature] = self._feature_indices(
                all_index_groups, primary_key, df_subset
            )

        self._connection = datacache.db_from_dataframes_with_absolute_path(
            db_path=self.local_db_path,
            table_names_to_dataframes=dataframes,
            table_names_to_primary_keys=primary_keys,
            table_names_to_indices=indices_dict,
            overwrite=overwrite,
            version=DATABASE_SCHEMA_VERSION,
        )
        return self._connection

    def _get_connection(self):
        if self._connection is None:
            if exists(self.local_db_path):
                # since version metadata is filled in last in datacache,
                # checking for a version also implicitly checks for
                # having finishing fill in all the tables/rows.
                #
                # TODO: expose this more explicitly in datacache
                #
                self._connection = datacache.connect_if_correct_version(
                    self.local_db_path, DATABASE_SCHEMA_VERSION
                )
        return self._connection

    @property
    def connection(self):
        """
        Get a connection to the database or raise an exception
        """
        connection = self._get_connection()
        if connection:
            return connection
        else:
            message = "GTF database needs to be created"
            if self.install_string:
                message += ", run: %s" % self.install_string
            raise ValueError(message)

    def connect_or_create(self, overwrite=False):
        """
        Return a connection to the database if it exists, otherwise create it.
        Overwrite the existing database if `overwrite` is True.
        """
        connection = self._get_connection()
        if connection:
            return connection
        else:
            return self.create(overwrite=overwrite)

    def columns(self, table_name):
        if self._columns.get(table_name) is None:
            sql = "PRAGMA table_info(%s)" % table_name
            table_info = self.connection.execute(sql).fetchall()
            column_set = set([info[1] for info in table_info])
            self._columns[table_name] = column_set
        return self._columns[table_name]

    def column_exists(self, table_name, column_name):
        return column_name in self.columns(table_name)

    def column_values_at_locus(
        self,
        column_name,
        feature,
        contig,
        position,
        end=None,
        strand=None,
        distinct=False,
        sorted=False,
    ):
        """
        Get the non-null values of a column from the database
        at a particular range of loci
        """

        # TODO: combine with the query method, since they overlap
        # significantly
        require_string(column_name, "column_name", nonempty=True)

        contig = normalize_chromosome(contig)

        require_integer(position, "position")

        if end is None:
            end = position

        require_integer(end, "end")

        if not self.column_exists(feature, column_name):
            raise ValueError(
                "Table %s doesn't have column %s"
                % (
                    feature,
                    column_name,
                )
            )

        if distinct:
            distinct_string = "DISTINCT "
        else:
            distinct_string = ""

        query = """
            SELECT %s%s
            FROM %s
            WHERE seqname = ?
            AND start <= ?
            AND end >= ?

        """ % (
            distinct_string,
            column_name,
            feature,
        )

        query_params = [contig, end, position]

        if strand:
            query += " AND strand = ?"
            query_params.append(strand)

        tuples = self.connection.execute(query, query_params).fetchall()

        # each result is a tuple, so pull out its first element
        results = [t[0] for t in tuples if t[0] is not None]

        if sorted:
            results.sort()
        return results

    def distinct_column_values_at_locus(
        self, column, feature, contig, position, end=None, strand=None
    ):
        """
        Gather all the distinct values for a property/column at some specified
        locus.

        Parameters
        ----------
        column : str
            Which property are we getting the values of.

        feature : str
            Which type of entry (e.g. transcript, exon, gene) is the property
            associated with?

        contig : str
            Chromosome or unplaced contig name

        position : int
            Chromosomal position

        end : int, optional
            End position of a range, if unspecified assume we're only looking
            at the single given position.

        strand : str, optional
            Either the positive ('+') or negative strand ('-'). If unspecified
            then check for values on either strand.
        """
        return self.column_values_at_locus(
            column,
            feature,
            contig,
            position,
            end=end,
            strand=strand,
            distinct=True,
            sorted=True,
        )

    def run_sql_query(self, sql, required=False, query_params=[]):
        """
        Given an arbitrary SQL query, run it against the database
        and return the results.

        Parameters
        ----------
        sql : str
            SQL query

        required : bool
            Raise an error if no results found in the database

        query_params : list
            For each '?' in the query there must be a corresponding value in
            this list.
        """
        try:
            cursor = self.connection.execute(sql, query_params)
        except sqlite3.OperationalError as e:
            error_message = e.message if hasattr(e, "message") else str(e)
            logger.warn(
                'Encountered error "%s" from query "%s" with parameters %s',
                error_message,
                sql,
                query_params,
            )
            raise
        results = cursor.fetchall()
        if required and not results:
            raise ValueError(
                "No results found for query:\n%s\nwith parameters: %s"
                % (sql, query_params)
            )

        return results

    @memoize
    def query(
        self,
        select_column_names,
        filter_column,
        filter_value,
        feature,
        distinct=False,
        required=False,
    ):
        """
        Construct a SQL query and run against the sqlite3 database,
        filtered both by the feature type and a user-provided column/value.
        """
        sql = """
            SELECT %s%s
            FROM %s
            WHERE %s = ?
        """ % (
            "distinct " if distinct else "",
            ", ".join(select_column_names),
            feature,
            filter_column,
        )
        query_params = [filter_value]
        return self.run_sql_query(sql, required=required, query_params=query_params)

    def query_one(
        self,
        select_column_names,
        filter_column,
        filter_value,
        feature,
        distinct=False,
        required=False,
    ):
        results = self.query(
            select_column_names,
            filter_column,
            filter_value,
            feature,
            distinct=distinct,
            required=required,
        )

        if len(results) == 0:
            if required:
                raise ValueError("%s not found: %s" % (filter_column, filter_value))
            else:
                return None
        elif len(results) > 1:
            raise ValueError(
                "Found multiple entries with %s=%s (%s)"
                % (filter_column, filter_value, results)
            )
        return results[0]

    @memoize
    def query_feature_values(
        self, column, feature, distinct=True, contig=None, strand=None
    ):
        """
        Run a SQL query against the sqlite3 database, filtered
        only on the feature type.
        """
        query = """
            SELECT %s%s
            FROM %s
            WHERE 1=1
        """ % (
            "DISTINCT " if distinct else "",
            column,
            feature,
        )
        query_params = []

        if contig:
            contig = normalize_chromosome(contig)
            query += " AND seqname = ?"
            query_params.append(contig)

        if strand:
            strand = normalize_strand(strand)
            query += " AND strand = ?"
            query_params.append(strand)

        rows = self.run_sql_query(query, query_params=query_params)
        return [row[0] for row in rows if row is not None]

    def query_distinct_on_contig(self, column_name, feature, contig):
        return self.query_feature_values(
            column=column_name, feature=feature, contig=contig, distinct=True
        )

    def query_loci(self, filter_column, filter_value, feature):
        """
        Query for loci satisfying a given filter and feature type.


        Parameters
        ----------
        filter_column : str
            Name of column to filter results by.

        filter_value : str
            Only return loci which have this value in the their filter_column.

        feature : str
            Feature names such as 'transcript', 'gene', and 'exon'

        Returns list of Locus objects
        """
        # list of values containing (contig, start, stop, strand)
        result_tuples = self.query(
            select_column_names=["seqname", "start", "end", "strand"],
            filter_column=filter_column,
            filter_value=filter_value,
            feature=feature,
            distinct=True,
            required=True,
        )
        return [
            Locus(contig, start, end, strand)
            for (contig, start, end, strand) in result_tuples
        ]

    def query_locus(self, filter_column, filter_value, feature):
        """
        Query for unique locus, raises error if missing or more than
        one locus in the database.

        Parameters
        ----------
        filter_column : str
            Name of column to filter results by.

        filter_value : str
            Only return loci which have this value in the their filter_column.

        feature : str
            Feature names such as 'transcript', 'gene', and 'exon'

        Returns single Locus object.
        """
        loci = self.query_loci(
            filter_column=filter_column, filter_value=filter_value, feature=feature
        )

        if len(loci) == 0:
            raise ValueError(
                "Couldn't find locus for %s with %s = %s"
                % (feature, filter_column, filter_value)
            )
        elif len(loci) > 1:
            raise ValueError(
                "Too many loci for %s with %s = %s: %s"
                % (feature, filter_column, filter_value, loci)
            )
        return loci[0]

    def _load_gtf_as_dataframe(self, usecols=None, features=None):
        """
        Parse this genome source's GTF file and load it as a Pandas DataFrame
        """
        logger.info("Reading GTF from %s", self.gtf_path)
        df = read_gtf(
            self.gtf_path,
            column_converters={
                "seqname": normalize_chromosome,
                "strand": normalize_strand,
            },
            infer_biotype_column=True,
            usecols=usecols,
            features=features,
        )

        column_names = set(df.columns)
        expect_gene_feature = features is None or "gene" in features
        expect_transcript_feature = features is None or "transcript" in features
        observed_features = set(df["feature"])

        # older Ensembl releases don't have "gene" or "transcript"
        # features, so fill in those rows if they're missing
        if expect_gene_feature and "gene" not in observed_features:
            # if we have to reconstruct gene feature rows then
            # fill in values for 'gene_name' and 'gene_biotype'
            # but only if they're actually present in the GTF
            logger.info("Creating missing gene features...")

            df = create_missing_features(
                dataframe=df,
                unique_keys={"gene": "gene_id"},
                extra_columns={
                    "gene": {"gene_name", "gene_biotype"}.intersection(column_names),
                },
                missing_value="",
            )
            logger.info("Done.")

        if expect_transcript_feature and "transcript" not in observed_features:
            logger.info("Creating missing transcript features...")
            df = create_missing_features(
                dataframe=df,
                unique_keys={"transcript": "transcript_id"},
                extra_columns={
                    "transcript": {
                        "gene_id",
                        "gene_name",
                        "gene_biotype",
                        "transcript_name",
                        "transcript_biotype",
                        "protein_id",
                    }.intersection(column_names)
                },
                missing_value="",
            )
            logger.info("Done.")

        return df
