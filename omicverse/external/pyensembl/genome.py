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
Contains the Genome class, with its millions of accessors and wrappers
around an arbitrary genomic database.
"""


from os import remove
from os.path import exists, getsize

from .serializable import Serializable

from .download_cache import DownloadCache
from .database import Database
from .exon import Exon
from .gene import Gene
from .sequence_data import SequenceData
from .transcript import Transcript


class Genome(Serializable):
    """
    Bundles together the genomic annotation and sequence data associated with
    a particular genomic database source (e.g. a single Ensembl release) and
    provides a wide variety of helper methods for accessing this data.
    """

    def __init__(
        self,
        reference_name,
        annotation_name,
        annotation_version=None,
        gtf_path_or_url=None,
        transcript_fasta_paths_or_urls=None,
        protein_fasta_paths_or_urls=None,
        decompress_on_download=False,
        copy_local_files_to_cache=False,
        cache_directory_path=None,
    ):
        """
        Parameters
        ----------
        reference_name : str
            Name of genome assembly which annotations in GTF are aligned against
            (and from which sequence data is drawn)

        annotation_name : str
            Name of annotation source (e.g. "Ensembl)

        annotation_version : int or str
            Version of annotation database (e.g. 75)

        gtf_path_or_url : str
            Path or URL of GTF file

        transcript_fasta_paths_or_urls : list
            List of paths or URLs of FASTA files containing transcript sequences

        protein_fasta_paths_or_urls : list
            List of paths or URLs of FASTA files containing protein sequences

        decompress_on_download : bool
            If remote file is compressed, decompress the local copy?

        copy_local_files_to_cache : bool
            If genome data file is local use it directly or copy to cache first?

        cache_directory_path : None
            Where to place downloaded and cached files for this genome,
            by default inferred from reference name, annotation name,
            annotation version, and global cache dir for pyensembl.
        """
        if transcript_fasta_paths_or_urls is None:
            transcript_fasta_paths_or_urls = []
        elif isinstance(transcript_fasta_paths_or_urls, str):
            transcript_fasta_paths_or_urls = [transcript_fasta_paths_or_urls]

        if protein_fasta_paths_or_urls is None:
            protein_fasta_paths_or_urls = []
        elif isinstance(protein_fasta_paths_or_urls, str):
            protein_fasta_paths_or_urls = [protein_fasta_paths_or_urls]

        self.reference_name = reference_name
        self.annotation_name = annotation_name
        self.annotation_version = annotation_version
        self.decompress_on_download = decompress_on_download
        self.copy_local_files_to_cache = copy_local_files_to_cache
        self.cache_directory_path = cache_directory_path
        self._gtf_path_or_url = gtf_path_or_url
        self._transcript_fasta_paths_or_urls = transcript_fasta_paths_or_urls
        self._protein_fasta_paths_or_urls = protein_fasta_paths_or_urls

        self.download_cache = DownloadCache(
            reference_name=self.reference_name,
            annotation_name=self.annotation_name,
            annotation_version=self.annotation_version,
            decompress_on_download=self.decompress_on_download,
            copy_local_files_to_cache=self.copy_local_files_to_cache,
            install_string_function=self.install_string,
            cache_directory_path=cache_directory_path,
        )
        self._init_lazy_fields()

    @property
    def requires_gtf(self):
        return self._gtf_path_or_url is not None

    @property
    def requires_transcript_fasta(self):
        return (
            self._transcript_fasta_paths_or_urls is not None
            and len(self._transcript_fasta_paths_or_urls) > 0
        )

    @property
    def requires_protein_fasta(self):
        return (
            self._protein_fasta_paths_or_urls is not None
            and len(self._protein_fasta_paths_or_urls) > 0
        )

    def to_dict(self):
        """
        Returns a dictionary of the essential fields of this Genome.
        """
        return dict(
            reference_name=self.reference_name,
            annotation_name=self.annotation_name,
            annotation_version=self.annotation_version,
            gtf_path_or_url=self._gtf_path_or_url,
            transcript_fasta_paths_or_urls=self._transcript_fasta_paths_or_urls,
            protein_fasta_paths_or_urls=self._protein_fasta_paths_or_urls,
            decompress_on_download=self.decompress_on_download,
            copy_local_files_to_cache=self.copy_local_files_to_cache,
            cache_directory_path=self.cache_directory_path,
        )

    def _init_lazy_fields(self):
        """
        Member data that gets loaded or constructed on demand
        """
        self.gtf_path = None
        self._protein_sequences = None
        self._transcript_sequences = None
        self._db = None
        self.protein_fasta_paths = None
        self.transcript_fasta_paths = None

        # only memoizing the Gene, Transcript, and Exon objects
        self._genes = {}
        self._transcripts = {}
        self._exons = {}

    def _get_cached_path(
        self, field_name, path_or_url, download_if_missing=False, overwrite=False
    ):
        """
        Get the local path for a possibly remote file, invoking either
        a download or install error message if it's missing.
        """
        if len(field_name) == 0:
            raise ValueError("Expected non-empty field name")
        if len(path_or_url) == 0:
            raise ValueError("Expected non-empty path_or_url")
        return self.download_cache.local_path_or_install_error(
            field_name=field_name,
            path_or_url=path_or_url,
            download_if_missing=download_if_missing,
            overwrite=overwrite,
        )

    def _get_gtf_path(self, download_if_missing=False, overwrite=False):
        return self._get_cached_path(
            field_name="gtf",
            path_or_url=self._gtf_path_or_url,
            download_if_missing=download_if_missing,
            overwrite=overwrite,
        )

    def _get_transcript_fasta_paths(self, download_if_missing=False, overwrite=False):
        if not self.requires_transcript_fasta:
            raise ValueError("No transcript FASTA source for %s" % self)
        return [
            self._get_cached_path(
                field_name="transcript-fasta",
                path_or_url=path,
                download_if_missing=download_if_missing,
                overwrite=overwrite,
            )
            for path in self._transcript_fasta_paths_or_urls
        ]

    def _get_protein_fasta_paths(self, download_if_missing=False, overwrite=False):
        # get the path for peptide FASTA files containing
        # this genome's protein sequences
        if not self.requires_protein_fasta:
            raise ValueError("No protein FASTA source for %s" % self)
        return [
            self._get_cached_path(
                field_name="protein-fasta",
                path_or_url=path,
                download_if_missing=download_if_missing,
                overwrite=overwrite,
            )
            for path in self._protein_fasta_paths_or_urls
        ]

    def _set_local_paths(self, download_if_missing=True, overwrite=False):
        if self.requires_gtf:
            self.gtf_path = self._get_gtf_path(
                download_if_missing=download_if_missing, overwrite=overwrite
            )
        if self.requires_transcript_fasta:
            self.transcript_fasta_paths = self._get_transcript_fasta_paths(
                download_if_missing=download_if_missing, overwrite=overwrite
            )
        if self.requires_protein_fasta:
            self.protein_fasta_paths = self._get_protein_fasta_paths(
                download_if_missing=download_if_missing, overwrite=overwrite
            )

    def required_local_files(self):
        paths = []
        if self._gtf_path_or_url:
            paths.append(self.download_cache.cached_path(self._gtf_path_or_url))
        if self._transcript_fasta_paths_or_urls:
            paths.extend(
                [
                    self.download_cache.cached_path(path_or_url)
                    for path_or_url in self._transcript_fasta_paths_or_urls
                ]
            )
        if self._protein_fasta_paths_or_urls:
            paths.extend(
                [
                    self.download_cache.cached_path(path_or_url)
                    for path_or_url in self._protein_fasta_paths_or_urls
                ]
            )
        return paths

    def required_local_files_exist(self, empty_files_ok=False):
        for path in self.required_local_files():
            if not exists(path):
                return False
            if not empty_files_ok:
                if getsize(path) == 0:
                    return False
        return True

    def download(self, overwrite=False):
        """
        Download data files needed by this Genome instance.

        Parameters
        ----------
        overwrite : bool, optional
            Download files regardless whether local copy already exists.
        """
        self._set_local_paths(download_if_missing=True, overwrite=overwrite)

    def index(self, overwrite=False):
        """
        Assuming that all necessary data for this Genome has been downloaded,
        generate the GTF database and save efficient representation of
        FASTA sequence files.
        """
        if self.requires_gtf:
            self.db.connect_or_create(overwrite=overwrite)
        if self.requires_transcript_fasta:
            self.transcript_sequences.index(overwrite=overwrite)
        if self.requires_protein_fasta:
            self.protein_sequences.index(overwrite=overwrite)

    @property
    def db(self):
        if not self.requires_gtf:
            raise ValueError("Missing GTF source for %s" % self)
        if self._db is None:
            # make sure GTF file exists locally
            # and populate self.gtf_path
            self._set_local_paths(
                download_if_missing=True,  ## if set at False the files are not downloaded in interactive python, works anyways via command line though
                overwrite=False,
            )
            if self.gtf_path is None:
                raise ValueError("Property 'gtf_path' of %s cannot be None" % self)

            # Database object turns the GTF dataframes into sqlite3 tables
            # and wraps them with methods like `query_one`
            self._db = Database(
                gtf_path=self.gtf_path,
                install_string=self.install_string(),
                cache_directory_path=self.cache_directory_path,
                restrict_gtf_columns={
                    "seqname",
                    "source",
                    "feature",
                    "start",
                    "end",
                    "strand",
                    "gene_id",
                    "gene_version",
                    "gene_name",
                    "gene_biotype",
                    "transcript_id",
                    "transcript_version",
                    "transcript_name",
                    "transcript_biotype",
                    "transcript_support_level",
                    "exon_number",
                    "exon_id",
                    "exon_version",
                    "ccds_id",
                    "protein_id",
                    "protein_version",
                },
                # excluding 'UTR' and 'Selenocysteine'
                restrict_gtf_features={
                    "gene",
                    "transcript",
                    "exon",
                    "CDS",
                    "start_codon",
                    "stop_codon",
                },
            )
        return self._db

    @property
    def protein_sequences(self):
        if self._protein_sequences is None:
            if not self.requires_protein_fasta:
                raise ValueError("Missing protein FASTA source for %s" % self)
            # make sure protein FASTA file exists locally
            # and populate self.protein_fasta_paths
            self._set_local_paths(download_if_missing=False, overwrite=False)
            if self.protein_fasta_paths is None:
                raise ValueError(
                    "Property 'protein_fasta_paths' of %s cannot be None" % self
                )
            self._protein_sequences = SequenceData(
                fasta_paths=self.protein_fasta_paths,
                cache_directory_path=self.cache_directory_path,
            )
        return self._protein_sequences

    @property
    def transcript_sequences(self):
        if self._transcript_sequences is None:
            if not self.requires_transcript_fasta:
                raise ValueError("Missing transcript FASTA source for %s" % self)
            # make sure transcript FASTA file exists locally
            # and populate self.transcript_fasta_paths
            self._set_local_paths(download_if_missing=False, overwrite=False)
            if self.transcript_fasta_paths is None:
                raise ValueError(
                    "Property 'transcript_fasta_paths' of %s cannot be None" % (self,)
                )
            self._transcript_sequences = SequenceData(
                fasta_paths=self.transcript_fasta_paths,
                cache_directory_path=self.cache_directory_path,
            )
        return self._transcript_sequences

    def install_string(self):
        """
        Add every missing file to the install string shown to the user
        in an error message.
        """
        args = [
            "--reference-name",
            self.reference_name,
            "--annotation-name",
            self.annotation_name,
        ]
        if self.annotation_version:
            args.extend(["--annotation-version", str(self.annotation_version)])
        if self.requires_gtf:
            args.append("--gtf")
            args.append('"%s"' % self._gtf_path_or_url)
        if self.requires_protein_fasta:
            args += [
                '--protein-fasta "%s"' % path
                for path in self._protein_fasta_paths_or_urls
            ]
        if self.requires_transcript_fasta:
            args += [
                '--transcript-fasta "%s"' % path
                for path in self._transcript_fasta_paths_or_urls
            ]
        return "pyensembl install %s" % " ".join(args)

    def __str__(self):
        transcript_fasta_paths_or_urls = (
            ",".join(self._transcript_fasta_paths_or_urls)
            if self._transcript_fasta_paths_or_urls is not None
            else None
        )
        protein_fasta_paths_or_urls = (
            ",".join(self._protein_fasta_paths_or_urls)
            if self._protein_fasta_paths_or_urls is not None
            else None
        )
        return (
            "Genome(reference_name=%s, "
            "annotation_name=%s, "
            "annotation_version=%s, "
            "gtf_path_or_url=%s, "
            "transcript_fasta_paths_or_urls=%s, "
            "protein_fasta_paths_or_urls=%s)"
            % (
                self.reference_name,
                self.annotation_name,
                self.annotation_version,
                self._gtf_path_or_url,
                transcript_fasta_paths_or_urls,
                protein_fasta_paths_or_urls,
            )
        )

    def __repr__(self):
        return str(self)

    def _fields(self):
        return (
            self.reference_name,
            self.annotation_name,
            self.annotation_version,
            self._gtf_path_or_url,
            tuple(self._protein_fasta_paths_or_urls),
            tuple(self._transcript_fasta_paths_or_urls),
        )

    def __eq__(self, other):
        return other.__class__ is Genome and self._fields() == other._fields()

    def __hash__(self):
        return hash(self._fields())

    def clear_cache(self):
        """
        Clear any in-memory cached values
        """
        for maybe_fn in self.__dict__.values():
            # clear cache associated with all memoization decorators,
            # GTF and SequenceData objects
            if hasattr(maybe_fn, "clear_cache"):
                maybe_fn.clear_cache()

    def delete_index_files(self):
        """
        Delete all data aside from source GTF and FASTA files
        """
        self.clear_cache()
        db_path = self.db.local_db_path()
        if exists(db_path):
            remove(db_path)

    def _all_feature_values(
        self, column, feature, distinct=True, contig=None, strand=None
    ):
        """
        Cached lookup of all values for a particular feature property from
        the database, caches repeated queries in memory and
        stores them as a CSV.

        Parameters
        ----------

        column : str
            Name of property (e.g. exon_id)

        feature : str
            Type of entry (e.g. exon)

        distinct : bool, optional
            Keep only unique values

        contig : str, optional
            Restrict query to particular contig

        strand : str, optional
            Restrict results to "+" or "-" strands

        Returns a list constructed from query results.
        """
        return self.db.query_feature_values(
            column=column,
            feature=feature,
            distinct=distinct,
            contig=contig,
            strand=strand,
        )

    def transcript_sequence(self, transcript_id):
        """Return cDNA nucleotide sequence of transcript, or None if
        transcript doesn't have cDNA sequence.
        """
        if self.transcript_sequences is None:
            raise ValueError("No transcript FASTA supplied to this Genome: %s" % self)
        return self.transcript_sequences.get(transcript_id)

    def protein_sequence(self, protein_id):
        """Return cDNA nucleotide sequence of transcript, or None if
        transcript doesn't have cDNA sequence.
        """
        if self.protein_sequences is None:
            raise ValueError("No protein FASTA supplied to this Genome: %s" % self)
        return self.protein_sequences.get(protein_id)

    def genes_at_locus(self, contig, position, end=None, strand=None):
        gene_ids = self.gene_ids_at_locus(contig, position, end=end, strand=strand)
        return [self.gene_by_id(gene_id) for gene_id in gene_ids]

    def transcripts_at_locus(self, contig, position, end=None, strand=None):
        transcript_ids = self.transcript_ids_at_locus(
            contig, position, end=end, strand=strand
        )
        return [
            self.transcript_by_id(transcript_id) for transcript_id in transcript_ids
        ]

    def exons_at_locus(self, contig, position, end=None, strand=None):
        exon_ids = self.exon_ids_at_locus(contig, position, end=end, strand=strand)
        return [self.exon_by_id(exon_id) for exon_id in exon_ids]

    def gene_ids_at_locus(self, contig, position, end=None, strand=None):
        return self.db.distinct_column_values_at_locus(
            column="gene_id",
            feature="gene",
            contig=contig,
            position=position,
            end=end,
            strand=strand,
        )

    def gene_names_at_locus(self, contig, position, end=None, strand=None):
        return self.db.distinct_column_values_at_locus(
            column="gene_name",
            feature="gene",
            contig=contig,
            position=position,
            end=end,
            strand=strand,
        )

    def exon_ids_at_locus(self, contig, position, end=None, strand=None):
        return self.db.distinct_column_values_at_locus(
            column="exon_id",
            feature="exon",
            contig=contig,
            position=position,
            end=end,
            strand=strand,
        )

    def transcript_ids_at_locus(self, contig, position, end=None, strand=None):
        return self.db.distinct_column_values_at_locus(
            column="transcript_id",
            feature="transcript",
            contig=contig,
            position=position,
            end=end,
            strand=strand,
        )

    def transcript_names_at_locus(self, contig, position, end=None, strand=None):
        return self.db.distinct_column_values_at_locus(
            column="transcript_name",
            feature="transcript",
            contig=contig,
            position=position,
            end=end,
            strand=strand,
        )

    def protein_ids_at_locus(self, contig, position, end=None, strand=None):
        return self.db.distinct_column_values_at_locus(
            column="protein_id",
            feature="transcript",
            contig=contig,
            position=position,
            end=end,
            strand=strand,
        )

    ###################################################
    #
    #         Methods which return Locus objects
    #         containing (contig, start, stop, strand)
    #         of various genomic entities
    #
    ###################################################

    def locus_of_gene_id(self, gene_id):
        """
        Given a gene ID returns Locus with: chromosome, start, stop, strand
        """
        return self.db.query_locus(
            filter_column="gene_id", filter_value=gene_id, feature="gene"
        )

    def loci_of_gene_names(self, gene_name):
        """
        Given a gene name returns list of Locus objects with fields:
            chromosome, start, stop, strand
        You can get multiple results since a gene might have multiple copies
        in the genome.
        """
        return self.db.query_loci("gene_name", gene_name, "gene")

    def locus_of_transcript_id(self, transcript_id):
        return self.db.query_locus(
            filter_column="transcript_id",
            filter_value=transcript_id,
            feature="transcript",
        )

    def locus_of_exon_id(self, exon_id):
        """
        Given an exon ID returns Locus
        """
        return self.db.query_locus("exon_id", exon_id, feature="exon")

    ###################################################
    #
    #                  Contigs
    #
    ###################################################

    def contigs(self):
        """
        Returns all contig names for any gene in the genome
        (field called "seqname" in Ensembl GTF files)
        """
        return self.db.query_feature_values("seqname", "gene")

    ###################################################
    #
    #             Gene Info Objects
    #
    ###################################################

    def genes(self, contig=None, strand=None):
        """
        Returns all Gene objects in the database. Can be restricted to a
        particular contig/chromosome and strand by the following arguments:

        Parameters
        ----------
        contig : str
            Only return genes on the given contig.

        strand : str
            Only return genes on this strand.
        """
        gene_ids = self.gene_ids(contig=contig, strand=strand)
        return [self.gene_by_id(gene_id) for gene_id in gene_ids]

    def gene_by_id(self, gene_id):
        """
        Construct a Gene object for the given gene ID.
        """
        if gene_id not in self._genes:
            field_names = [
                "seqname",
                "start",
                "end",
                "strand",
            ]
            optional_field_names = [
                "gene_name",
                "gene_biotype",
            ]
            # Do not look for gene_name and gene_biotype if they are
            # not in the database.
            field_names.extend(
                [
                    name
                    for name in optional_field_names
                    if self.db.column_exists("gene", name)
                ]
            )
            result = self.db.query_one(
                field_names,
                filter_column="gene_id",
                filter_value=gene_id,
                feature="gene",
            )
            if not result:
                raise ValueError("Gene not found: %s" % (gene_id,))

            gene_name, gene_biotype = None, None
            if len(result) < 4 or len(result) > 6:
                raise ValueError("Result is not the expected length: %d" % len(result))
            contig, start, end, strand = result[:4]
            if len(result) == 5:
                if "gene_name" in field_names:
                    gene_name = result[4]
                else:
                    gene_biotype = result[4]
            elif len(result) == 6:
                gene_name, gene_biotype = result[4:]

            self._genes[gene_id] = Gene(
                gene_id=gene_id,
                gene_name=gene_name,
                contig=contig,
                start=start,
                end=end,
                strand=strand,
                biotype=gene_biotype,
                genome=self,
            )

        return self._genes[gene_id]

    def genes_by_name(self, gene_name):
        """
        Get all the unqiue genes with the given name (there might be multiple
        due to copies in the genome), return a list containing a Gene object
        for each distinct ID.
        """
        gene_ids = self.gene_ids_of_gene_name(gene_name)
        return [self.gene_by_id(gene_id) for gene_id in gene_ids]

    def gene_by_protein_id(self, protein_id):
        """
        Get the gene ID associated with the given protein ID,
        return its Gene object
        """
        gene_id = self.gene_id_of_protein_id(protein_id)
        return self.gene_by_id(gene_id)

    ###################################################
    #
    #             Gene Names
    #
    ###################################################

    def _query_gene_name(self, property_name, property_value, feature_type):
        results = self.db.query(
            select_column_names=["gene_name"],
            filter_column=property_name,
            filter_value=property_value,
            feature=feature_type,
            distinct=True,
            required=True,
        )
        return str(results[0][0])

    def gene_names(self, contig=None, strand=None):
        """
        Return all genes in the database,
        optionally restrict to a chromosome and/or strand.
        """
        return self._all_feature_values(
            column="gene_name", feature="gene", contig=contig, strand=strand
        )

    def gene_name_of_gene_id(self, gene_id):
        return self._query_gene_name("gene_id", gene_id, "gene")

    def gene_name_of_transcript_id(self, transcript_id):
        return self._query_gene_name("transcript_id", transcript_id, "transcript")

    def gene_name_of_transcript_name(self, transcript_name):
        return self._query_gene_name("transcript_name", transcript_name, "transcript")

    def gene_name_of_exon_id(self, exon_id):
        return self._query_gene_name("exon_id", exon_id, "exon")

    ###################################################
    #
    #             Gene IDs
    #
    ###################################################

    def _query_gene_ids(self, property_name, value, feature="gene"):
        results = self.db.query(
            select_column_names=["gene_id"],
            filter_column=property_name,
            filter_value=value,
            feature=feature,
            distinct=True,
            required=True,
        )
        return [str(result[0]) for result in results if result[0]]

    def gene_ids(self, contig=None, strand=None):
        """
        What are all the gene IDs
        (optionally restrict to a given chromosome/contig and/or strand)
        """
        return self._all_feature_values(
            column="gene_id", feature="gene", contig=contig, strand=strand
        )

    def gene_ids_of_gene_name(self, gene_name):
        """
        What are the gene IDs associated with a given gene name?
        (due to copy events, there might be multiple genes per name)
        """
        results = self._query_gene_ids("gene_name", gene_name)
        if len(results) == 0:
            raise ValueError("Gene name not found: %s" % gene_name)
        return results

    def gene_id_of_protein_id(self, protein_id):
        """
        What is the gene ID associated with a given protein ID?
        """
        results = self._query_gene_ids("protein_id", protein_id, feature="CDS")
        if len(results) == 0:
            raise ValueError("Protein ID not found: %s" % protein_id)
        elif len(results) > 1:
            raise ValueError(
                (
                    "Should have only one gene ID for a given protein ID, "
                    "but found %d: %s"
                )
                % (len(results), results)
            )
        return results[0]

    ###################################################
    #
    #             Transcript Info Objects
    #
    ###################################################

    def transcripts(self, contig=None, strand=None):
        """
        Construct Transcript object for every transcript entry in
        the database. Optionally restrict to a particular
        chromosome using the `contig` argument.
        """
        transcript_ids = self.transcript_ids(contig=contig, strand=strand)
        return [
            self.transcript_by_id(transcript_id) for transcript_id in transcript_ids
        ]

    def transcript_by_id(self, transcript_id):
        """Construct Transcript object with given transcript ID"""
        if transcript_id not in self._transcripts:
            optional_field_names = [
                "transcript_name",
                "transcript_biotype",
                "transcript_support_level",
            ]
            field_names = [
                "seqname",
                "start",
                "end",
                "strand",
                "gene_id",
            ]
            # Do not look for the optional fields if they are not in the database.
            field_names.extend(
                [
                    name
                    for name in optional_field_names
                    if self.db.column_exists("transcript", name)
                ]
            )
            result = self.db.query_one(
                select_column_names=field_names,
                filter_column="transcript_id",
                filter_value=transcript_id,
                feature="transcript",
                distinct=True,
            )
            if not result:
                raise ValueError("Transcript not found: %s" % (transcript_id,))

            transcript_name, transcript_biotype, tsl = None, None, None
            if len(result) < 5 or len(result) > (5 + len(optional_field_names)):
                raise ValueError("Result is not the expected length: %d" % len(result))
            contig, start, end, strand, gene_id = result[:5]
            if len(result) > 5:
                extra_field_names = [
                    f for f in optional_field_names if f in field_names
                ]
                extra_data = dict(zip(extra_field_names, result[5:]))
                transcript_name = extra_data.get("transcript_name")
                transcript_biotype = extra_data.get("transcript_biotype")
                tsl = extra_data.get("transcript_support_level")
                if not tsl or tsl == "NA":
                    tsl = None
                else:
                    tsl = int(tsl)

            self._transcripts[transcript_id] = Transcript(
                transcript_id=transcript_id,
                transcript_name=transcript_name,
                contig=contig,
                start=start,
                end=end,
                strand=strand,
                biotype=transcript_biotype,
                gene_id=gene_id,
                genome=self,
                support_level=tsl,
            )

        return self._transcripts[transcript_id]

    def transcripts_by_name(self, transcript_name):
        transcript_ids = self.transcript_ids_of_transcript_name(transcript_name)
        return [
            self.transcript_by_id(transcript_id) for transcript_id in transcript_ids
        ]

    def transcript_by_protein_id(self, protein_id):
        transcript_id = self.transcript_id_of_protein_id(protein_id)
        return self.transcript_by_id(transcript_id)

    ###################################################
    #
    #            Transcript Names
    #
    ###################################################

    def _query_transcript_names(self, property_name, value):
        results = self.db.query(
            select_column_names=["transcript_name"],
            filter_column=property_name,
            filter_value=value,
            feature="transcript",
            distinct=True,
            required=True,
        )
        return [result[0] for result in results]

    def transcript_names(self, contig=None, strand=None):
        """
        What are all the transcript names in the database
        (optionally, restrict to a given chromosome and/or strand)
        """
        return self._all_feature_values(
            column="transcript_name", feature="transcript", contig=contig, strand=strand
        )

    def transcript_names_of_gene_name(self, gene_name):
        return self._query_transcript_names("gene_name", gene_name)

    def transcript_name_of_transcript_id(self, transcript_id):
        transcript_names = self._query_transcript_names("transcript_id", transcript_id)
        if len(transcript_names) == 0:
            raise ValueError(
                "No transcript names for transcript ID = %s" % transcript_id
            )
        elif len(transcript_names) > 1:
            raise ValueError(
                "Multiple transcript names for transcript ID = %s" % (transcript_id,)
            )
        return transcript_names[0]

    ###################################################
    #
    #            Transcript IDs
    #
    ###################################################

    def _query_transcript_ids(self, property_name, value, feature="transcript"):
        results = self.db.query(
            select_column_names=["transcript_id"],
            filter_column=property_name,
            filter_value=value,
            feature=feature,
            distinct=True,
            required=True,
        )
        return [result[0] for result in results]

    def transcript_ids(self, contig=None, strand=None):
        return self._all_feature_values(
            column="transcript_id", feature="transcript", contig=contig, strand=strand
        )

    def transcript_ids_of_gene_id(self, gene_id):
        return self._query_transcript_ids("gene_id", gene_id)

    def transcript_ids_of_gene_name(self, gene_name):
        return self._query_transcript_ids("gene_name", gene_name)

    def transcript_ids_of_transcript_name(self, transcript_name):
        return self._query_transcript_ids("transcript_name", transcript_name)

    def transcript_ids_of_exon_id(self, exon_id):
        return self._query_transcript_ids("exon_id", exon_id)

    def transcript_id_of_protein_id(self, protein_id):
        """
        What is the transcript ID associated with a given protein ID?
        """
        results = self._query_transcript_ids("protein_id", protein_id, feature="CDS")
        if len(results) == 0:
            raise ValueError("Protein ID not found: %s" % protein_id)
        elif len(results) > 1:
            raise ValueError(
                (
                    "Should have only one transcript ID for a given protein ID, "
                    "but found %d: %s"
                )
                % (len(results), results)
            )
        return results[0]

    ###################################################
    #
    #             Exon Info Objects
    #
    ###################################################

    def exons(self, contig=None, strand=None):
        """
        Create exon object for all exons in the database, optionally
        restrict to a particular chromosome using the `contig` argument.
        """
        # DataFrame with single column called "exon_id"
        exon_ids = self.exon_ids(contig=contig, strand=strand)
        return [self.exon_by_id(exon_id) for exon_id in exon_ids]

    def exon_by_id(self, exon_id):
        """Construct an Exon object from its ID by looking up the exon"s
        properties in the given Database.
        """
        if exon_id not in self._exons:
            field_names = [
                "seqname",
                "start",
                "end",
                "strand",
                "gene_name",
                "gene_id",
            ]

            contig, start, end, strand, gene_name, gene_id = self.db.query_one(
                select_column_names=field_names,
                filter_column="exon_id",
                filter_value=exon_id,
                feature="exon",
                distinct=True,
            )

            self._exons[exon_id] = Exon(
                exon_id=exon_id,
                contig=contig,
                start=start,
                end=end,
                strand=strand,
                gene_name=gene_name,
                gene_id=gene_id,
            )

        return self._exons[exon_id]

    ###################################################
    #
    #                Exon IDs
    #
    ###################################################

    def _query_exon_ids(self, property_name, value):
        results = self.db.query(
            select_column_names=["exon_id"],
            filter_column=property_name,
            filter_value=value,
            feature="exon",
            distinct=True,
            required=True,
        )
        return [result[0] for result in results]

    def exon_ids(self, contig=None, strand=None):
        return self._all_feature_values(
            column="exon_id", feature="exon", contig=contig, strand=strand
        )

    def exon_ids_of_gene_id(self, gene_id):
        return self._query_exon_ids("gene_id", gene_id)

    def exon_ids_of_gene_name(self, gene_name):
        return self._query_exon_ids("gene_name", gene_name)

    def exon_ids_of_transcript_name(self, transcript_name):
        return self._query_exon_ids("transcript_name", transcript_name)

    def exon_ids_of_transcript_id(self, transcript_id):
        return self._query_exon_ids("transcript_id", transcript_id)

    ###################################################
    #
    #             Protein IDs
    #
    ###################################################

    def protein_ids(self, contig=None, strand=None):
        """
        What are all the protein IDs
        (optionally restrict to a given chromosome and/or strand)
        """
        protein_ids = self._all_feature_values(
            column="protein_id",
            feature="CDS",
            contig=contig,
            strand=strand,
            distinct=True,
        )
        # drop None values
        return [protein_id for protein_id in protein_ids if protein_id]
