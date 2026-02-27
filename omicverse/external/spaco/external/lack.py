import functools
import logging
import sys
import time
from contextlib import contextmanager


def silence_logger(name):
    """Given a logger name, silence it completely.

    :param name: name of the logger
    :type name: str
    """
    package_logger = logging.getLogger(name)
    package_logger.setLevel(logging.CRITICAL + 100)
    package_logger.propagate = False


def set_logger_level(name, level):
    """Given a logger name, silence it completely.

    :param name: name of the logger
    :type name: str
    """
    package_logger = logging.getLogger(name)
    package_logger.setLevel(level)


def format_logging_message(msg, logging_level, indent_level=1, indent_space_num=6):
    indent_str = "-" * indent_space_num
    prefix = indent_str * indent_level
    prefix = "|" + prefix[1:]
    if logging_level == logging.INFO:
        prefix += ">"
    elif logging_level == logging.WARNING:
        prefix += "?"
    elif logging_level == logging.CRITICAL:
        prefix += "!!"
    elif logging_level == logging.DEBUG:
        prefix += ">>>"
    new_msg = prefix + " " + str(msg)
    return new_msg


class Logger:
    """Aristotle ecosystem-specific logger setting up logging for the package."""

    FORMAT = "%(message)s"

    def __init__(self, namespace="main", level=None):
        self.namespace = namespace
        self.logger = logging.getLogger(namespace)
        self.previous_timestamp = time.time()  # in seconds
        self.time_passed = 0
        self.report_hook_percent_state = None
        # TODO add file handler in future
        # e.g. logging.StreamHandler(None) if log_file_path is None else logging.FileHandler(name)

        # ensure only one stream handler exists in one logger instance
        if len(self.logger.handlers) == 0:
            self.logger_stream_handler = logging.StreamHandler(sys.stdout)
            self.logger_stream_handler.setFormatter(logging.Formatter(self.FORMAT))
            self.logger.addHandler(self.logger_stream_handler)
        else:
            self.logger_stream_handler = self.logger.handlers[0]

        self.logger.propagate = False
        self.log_time()

        # Other global initialization
        silence_logger("anndata")
        silence_logger("h5py")
        silence_logger("numba")
        silence_logger("pysam")
        silence_logger("pystan")

        if not (level is None):
            self.logger.setLevel(level)
        else:
            self.logger.setLevel(logging.INFO)

    def namespaced(self, namespace):
        """Function decorator to set the logging namespace for the duration of
        the function.

        :param namespace: the namespace
        :type namespace: str
        """

        def wrapper(func):
            @functools.wraps(func)
            def inner(*args, **kwargs):
                previous = self.namespace
                try:
                    self.namespace = namespace
                    return func(*args, **kwargs)
                finally:
                    self.namespace = previous

            return inner

        return wrapper

    @contextmanager
    def namespaced_context(self, namespace):
        """Context manager to set the logging namespace.

        :param namespace: the namespace
        :type namespace: str
        """
        previous = self.namespace
        self.namespace = namespace
        yield
        self.namespace = previous

    def namespace_message(self, message):
        """Add namespace information at the beginning of the logging message.

        :param message: the logging message
        :type message: str

        :return: namespaced message
        :rtype: string
        """
        return f"[{self.namespace}] {message}"

    def setLevel(self, *args, **kwargs):
        return self.logger.setLevel(*args, **kwargs)

    def debug(self, message, indent_level=1, *args, **kwargs):
        message = format_logging_message(message, logging.DEBUG, indent_level=indent_level)
        return self.logger.debug(message, *args, **kwargs)

    def info(self, message, indent_level=1, *args, **kwargs):
        message = format_logging_message(message, logging.INFO, indent_level=indent_level)
        return self.logger.info(message, *args, **kwargs)

    def warning(self, message, indent_level=1, *args, **kwargs):
        message = format_logging_message(message, logging.WARNING, indent_level=indent_level)
        return self.logger.warning(message, *args, **kwargs)

    def exception(self, message, indent_level=1, *args, **kwargs):
        message = format_logging_message(message, logging.ERROR, indent_level=indent_level)
        return self.logger.exception(message, *args, **kwargs)

    def critical(self, message, indent_level=1, *args, **kwargs):
        message = format_logging_message(message, logging.CRITICAL, indent_level=indent_level)
        return self.logger.critical(message, *args, **kwargs)

    def error(self, message, indent_level=1, *args, **kwargs):
        message = format_logging_message(message, logging.ERROR, indent_level=indent_level)
        return self.logger.error(message, *args, **kwargs)

    def info_insert_adata(self, key, adata_attr="obsm", indent_level=1, *args, **kwargs):
        message = "<insert> %s to %s in AnnData Object." % (key, adata_attr)
        message = format_logging_message(message, logging.INFO, indent_level=indent_level)
        return self.logger.info(message, *args, **kwargs)

    def info_insert_adata_var(self, key, indent_level=1, *args, **kwargs):
        return self.info_insert_adata(self, key, adata_attr="var", indent_level=1, *args, **kwargs)

    def info_insert_adata_obsm(self, key, indent_level=1, *args, **kwargs):
        return self.info_insert_adata(self, key, adata_attr="obsm", indent_level=1, *args, **kwargs)

    def info_insert_adata_uns(self, key, indent_level=1, *args, **kwargs):
        return self.info_insert_adata(self, key, adata_attr="uns", indent_level=1, *args, **kwargs)

    def log_time(self):
        now = time.time()
        self.time_passed = now - self.previous_timestamp
        self.previous_timestamp = now
        return self.time_passed

    def report_progress(self, percent=None, count=None, total=None, progress_name="", indent_level=1):
        if percent is None:
            assert (not count is None) and (not total is None)
            percent = count / total * 100
        saved_terminator = self.logger_stream_handler.terminator
        self.logger_stream_handler.terminator = ""
        if progress_name != "":
            progress_name = "[" + str(progress_name) + "] "
        message = "\r" + format_logging_message(
            "%sin progress: %.4f%%" % (progress_name, percent), logging_level=logging.INFO, indent_level=indent_level
        )
        self.logger.info(message)
        self.logger_stream_handler.flush()
        self.logger_stream_handler.terminator = saved_terminator

    def finish_progress(self, progress_name="", time_unit="s", indent_level=1):
        self.log_time()
        self.report_progress(percent=100, progress_name=progress_name)

        saved_terminator = self.logger_stream_handler.terminator
        self.logger_stream_handler.terminator = ""
        self.logger.info("\n")
        self.logger_stream_handler.flush()
        self.logger_stream_handler.terminator = saved_terminator

        if time_unit == "s":
            self.info("[%s] finished [%.4fs]" % (progress_name, self.time_passed), indent_level=indent_level)
        elif time_unit == "ms":
            self.info("[%s] finished [%.4fms]" % (progress_name, self.time_passed * 1e3), indent_level=indent_level)
        else:
            raise NotImplementedError
        # self.logger.info("|")
        self.logger_stream_handler.flush()

    def request_report_hook(self, bn: int, rs: int, ts: int) -> None:
        """A callback required by the request lib:
        The reporthook argument should be a callable that accepts a block number, a read size, and the
        total file size of the URL target. The data argument should be valid URL encoded data.

        Parameters
        ----------
        bs :
            block number
        rs :
            read size
        ts :
            total size
        """
        if self.report_hook_percent_state is None:
            self.report_hook_percent_state = 0
        cur_percent = rs * bn / ts
        if cur_percent - self.report_hook_percent_state > 0.01:
            self.report_progress(count=rs * bn, total=ts)
            self.report_hook_percent_state = cur_percent
        if rs * bn >= ts:
            self.report_hook_percent_state = None
            self.finish_progress(progress_name="download")


class LoggerManager:
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    CRITICAL = logging.CRITICAL
    EXCEPTION = logging.ERROR

    @staticmethod
    def gen_logger(namespace: str):
        return Logger(namespace)

    def __init__(self, namespace: str = "lack", temp_timer_logger: str = "lack-temp-timer-logger"):
        self.set_main_logger_namespace(namespace)
        self.temp_timer_logger = Logger(temp_timer_logger)

    def set_main_logger_namespace(self, namespace: str):
        self.main_logger = self.gen_logger(namespace)
        self.namespace = namespace

    def get_main_logger(self):
        return self.main_logger

    def get_temp_timer_logger(self):
        return self.temp_timer_logger

    def progress_logger(self, generator, logger=None, progress_name="", indent_level=1):
        if logger is None:
            logger = self.get_temp_timer_logger()
        iterator = iter(generator)
        logger.log_time()
        i = 0
        prev_progress_percent = 0
        while i < len(generator):
            i += 1
            new_progress_percent = i / len(generator) * 100
            # report every `interval` percent
            if new_progress_percent - prev_progress_percent > 1 or new_progress_percent >= 100:
                logger.report_progress(
                    count=i, total=len(generator), progress_name=progress_name, indent_level=indent_level
                )
                prev_progress_percent = new_progress_percent
            yield next(iterator)
        logger.finish_progress(progress_name=progress_name, indent_level=indent_level)

    def main_set_level(self, level):
        set_logger_level(self.namespace, level)

    def main_info(self, message, indent_level=1):
        self.main_logger.info(message, indent_level)

    def main_debug(self, message, indent_level=1):
        self.main_logger.debug(message, indent_level)

    def main_warning(self, message, indent_level=1):
        self.main_logger.warning(message, indent_level)

    def main_exception(self, message, indent_level=1):
        self.main_logger.exception(message, indent_level)

    def main_critical(self, message, indent_level=1):
        self.main_logger.critical(message, indent_level)

    def main_tqdm(self, generator, desc="", indent_level=1, logger=None):
        """a TQDM style wrapper for logging something like a loop.
        e.g.
        for item in main_tqdm(alist, desc=""):
            do something

        Parameters
        ----------
        generator : [type]
            same as what you put in tqdm
        desc : str, optional
            description of your progress
        """
        if logger is None:
            logger = self.main_logger
        return self.progress_logger(generator, logger=logger, progress_name=desc, indent_level=indent_level)

    def main_log_time(
        self,
    ):
        self.main_logger.log_time()

    def main_silence(
        self,
    ):
        self.main_logger.setLevel(logging.CRITICAL + 100)

    def main_finish_progress(self, progress_name=""):
        self.main_logger.finish_progress(progress_name=progress_name)

    def main_info_insert_adata(self, key, adata_attr="obsm", indent_level=1, *args, **kwargs):
        self.main_logger.info_insert_adata(key, adata_attr=adata_attr, indent_level=indent_level, *args, **kwargs)

    def main_info_insert_adata_var(self, key, indent_level=1, *args, **kwargs):
        self.main_info_insert_adata(self, key, "var", indent_level, *args, **kwargs)

    def main_info_insert_adata_uns(self, key, indent_level=1, *args, **kwargs):
        self.main_info_insert_adata(key, "uns", indent_level, *args, **kwargs)

    def main_info_insert_adata_obsm(self, key, indent_level=1, *args, **kwargs):
        self.main_info_insert_adata(key, "obsm", indent_level, *args, **kwargs)

    def main_info_insert_adata_obs(self, key, indent_level=1, *args, **kwargs):
        self.main_info_insert_adata(key, "obs", indent_level, *args, **kwargs)

    def main_info_insert_adata_layer(self, key, indent_level=1, *args, **kwargs):
        self.main_info_insert_adata(key, "layers", indent_level, *args, **kwargs)

    def main_info_verbose_timeit(self, msg):
        self.main_logger.info(msg)
