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


from os import listdir, remove
from os.path import join, exists, split, abspath, isdir
from shutil import copy2, rmtree
import logging

import datacache


logger = logging.getLogger(__name__)

CACHE_BASE_SUBDIR = "pyensembl"
CACHE_DIR_ENV_KEY = "PYENSEMBL_CACHE_DIR"


def cache_subdirectory(
    reference_name=None, annotation_name=None, annotation_version=None
):
    """
    Which cache subdirectory to use for a given annotation database
    over a particular reference. All arguments can be omitted to just get
    the base subdirectory for all pyensembl cached datasets.
    """
    if reference_name is None:
        reference_name = ""
    if annotation_name is None:
        annotation_name = ""
    if annotation_version is None:
        annotation_version = ""
    reference_dir = join(CACHE_BASE_SUBDIR, reference_name)
    annotation_dir = "%s%s" % (annotation_name, annotation_version)
    return join(reference_dir, annotation_dir)


class MissingRemoteFile(Exception):
    def __init__(self, url):
        self.url = url


class MissingLocalFile(Exception):
    def __init__(self, path):
        self.path = path

    def __str__(self):
        return "MissingFile(%s)" % self.path


class DownloadCache(object):
    """
    Downloads remote files to cache, optionally copies local files into cache,
    raises custom message if data is missing.
    """

    def __init__(
        self,
        reference_name,
        annotation_name,
        annotation_version=None,
        decompress_on_download=False,
        copy_local_files_to_cache=False,
        install_string_function=None,
        cache_directory_path=None,
    ):
        """
        Parameters
        ----------
        reference_name : str
            Name of reference genome

        annotation_name : str
            Name of annotation database

        annotation_version : str or int, optional
            Version or release of annotation database

        decompress_on_download : bool, optional
            If downloading a .fa.gz file, should we automatically expand it
            into a decompressed FASTA file?

        copy_local_files_to_cache : bool, optional
            If file is on the local file system, should we still copy it
            into the cache?

        install_string_function : fn, optional
            Function which returns an error message with
            install instructions. If not provided then the error tells the
            user what data is missing without install instructions.

        cache_directory_path : str, optional
            Where to place downloaded and temporary files, by default
            inferred from reference name, annotation name, annotation version,
            and the global cache directory determined by datacache.
        """

        self.reference_name = reference_name
        self.annotation_name = annotation_name
        self.annotation_version = annotation_version

        # using hidden member variable _cache_directory path since access to
        # to the visible cache_directory_path (no underscore!) is combined
        # with ensuring that the directpry actually exists
        if cache_directory_path:
            self._cache_directory_path = cache_directory_path
        else:
            self.cache_subdirectory = cache_subdirectory(
                reference_name=reference_name,
                annotation_name=annotation_name,
                annotation_version=annotation_version,
            )

            # If `CACHE_DIR_ENV_KEY` is set, the cache will be saved there
            self._cache_directory_path = datacache.get_data_dir(
                subdir=self.cache_subdirectory, envkey=CACHE_DIR_ENV_KEY
            )

        self.decompress_on_download = decompress_on_download
        self.copy_local_files_to_cache = copy_local_files_to_cache
        self.install_string_function = install_string_function

    @property
    def cache_directory_path(self):
        return self._cache_directory_path

    def _fields(self):
        """
        Fields used for hashing, string representation, equality comparison
        """
        return (
            (
                "reference_name",
                self.reference_name,
            ),
            ("annotation_name", self.annotation_name),
            ("annotation_version", self.annotation_version),
            ("cache_directory_path", self.cache_directory_path),
            ("decompress_on_download", self.decompress_on_download),
            ("copy_local_files_to_cache", self.copy_local_files_to_cache),
        )

    def __eq__(self, other):
        return other.__class__ is DownloadCache and self._fields() == other._fields()

    def __hash__(self):
        return hash(self._fields())

    def __str__(self):
        fields_str = ", ".join("%s=%s" % (k, v) for (k, v) in self._fields())
        return "DownloadCache(%s)" % fields_str

    def __repr__(self):
        return str(self)

    def is_url_format(self, path_or_url):
        """
        Is the given string a URL?

        Parameters
        ----------
        path_or_url : str

        Returns
        -------
        bool
        """
        if path_or_url is None or path_or_url == "":
            raise ValueError("Expected non-empty string for path_or_url")
        return "://" in path_or_url

    def _remove_compression_suffix_if_present(self, filename):
        """
        If the given filename ends in one of the compression suffixes that
        datacache knows how to deal with, remove the suffix (since we expect
        the result of downloading to be a decompressed file)
        """
        for ext in [".gz", ".gzip", ".zip"]:
            if filename.endswith(ext):
                return filename[: -len(ext)]
        return filename

    def cached_path(self, path_or_url):
        """
        When downloading remote files, the default behavior is to name local
        files the same as their remote counterparts.
        """
        if path_or_url is None or path_or_url == "":
            raise ValueError("Expected non-empty string for path_or_url")
        remote_filename = split(path_or_url)[1]
        if self.is_url_format(path_or_url):
            # passing `decompress=False` since there is logic below
            # for stripping decompression extensions for both local
            # and remote files
            local_filename = datacache.build_local_filename(
                download_url=path_or_url, filename=remote_filename, decompress=False
            )
        else:
            local_filename = remote_filename

        # if we expect the download function to decompress this file then
        # we should use its name without the compression extension
        if self.decompress_on_download:
            local_filename = self._remove_compression_suffix_if_present(local_filename)

        if len(local_filename) == 0:
            raise ValueError("Can't determine local filename for %s" % (path_or_url,))

        return join(self.cache_directory_path, local_filename)

    def _download_if_necessary(self, url, download_if_missing, overwrite):
        """
        Return local cached path to a remote file, download it if necessary.
        """
        cached_path = self.cached_path(url)
        missing = not exists(cached_path)
        if (missing or overwrite) and download_if_missing:
            logger.info("Fetching %s from URL %s", cached_path, url)
            datacache.ensure_dir(self.cache_directory_path)
            datacache.download._download_and_decompress_if_necessary(
                full_path=cached_path, download_url=url, timeout=3600
            )
        elif missing:
            raise MissingRemoteFile(url)
        return cached_path

    def _copy_if_necessary(self, local_path, overwrite):
        """
        Return cached path to local file, copying it to the cache if necessary.
        """
        local_path = abspath(local_path)
        if not exists(local_path):
            raise MissingLocalFile(local_path)
        elif not self.copy_local_files_to_cache:
            return local_path
        else:
            cached_path = self.cached_path(local_path)
            if exists(cached_path) and not overwrite:
                return cached_path
            datacache.ensure_dir(self.cache_directory_path)
            copy2(local_path, cached_path)
            return cached_path

    def download_or_copy_if_necessary(
        self, path_or_url, download_if_missing=False, overwrite=False
    ):
        """
        Download a remote file or copy
        Get the local path to a possibly remote file.

        Download if file is missing from the cache directory and
        `download_if_missing` is True. Download even if local file exists if
        both `download_if_missing` and `overwrite` are True.

        If the file is on the local file system then return its path, unless
        self.copy_local_to_cache is True, and then copy it to the cache first.

        Parameters
        ----------
        path_or_url : str

        download_if_missing : bool, optional
            Download files if missing from local cache

        overwrite : bool, optional
            Overwrite existing copy if it exists
        """
        if path_or_url is None or path_or_url == "":
            raise ValueError("Expected non-empty string for path_or_url")
        if self.is_url_format(path_or_url):
            return self._download_if_necessary(
                path_or_url, download_if_missing, overwrite
            )
        else:
            return self._copy_if_necessary(path_or_url, overwrite)

    def _raise_missing_file_error(self, missing_urls_dict):
        missing_urls = list(missing_urls_dict.values())
        n_missing = len(missing_urls)
        error_message = "Missing genome data file%s from %s." % (
            ("s", missing_urls) if n_missing > 1 else ("", missing_urls[0])
        )
        if self.install_string_function:
            install_string = self.install_string_function()
            error_message += " Run %s" % install_string
        raise ValueError(error_message)

    def local_path_or_install_error(
        self, field_name, path_or_url, download_if_missing=False, overwrite=False
    ):
        try:
            return self.download_or_copy_if_necessary(
                path_or_url,
                download_if_missing=download_if_missing,
                overwrite=overwrite,
            )
        except MissingRemoteFile:
            self._raise_missing_file_error({field_name: path_or_url})

    def delete_cached_files(self, prefixes=[], suffixes=[]):
        """
        Deletes any cached files matching the prefixes or suffixes given
        """
        if isdir(self.cache_directory_path):
            for filename in listdir():
                delete = any([filename.endswith(ext) for ext in suffixes]) or any(
                    [filename.startswith(pre) for pre in prefixes]
                )
                if delete:
                    path = join(self.cache_directory_path, filename)
                    logger.info("Deleting %s", path)
                    remove(path)

    def delete_cache_directory(self):
        if isdir(self.cache_directory_path):
            rmtree(self.cache_directory_path)
