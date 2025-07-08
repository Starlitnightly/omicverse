# coding=utf-8

import logging
import sys


def create_logging_handler(debug: bool) -> logging.Handler:
    """
    Create a handler for logging purposes.

    :param debug: Does debug information need to be logged?
    :return: The logging handler.
    """
    # By default stderr is used as the stream for logging.
    ch = logging.StreamHandler(stream=sys.stderr)
    # Logging level DEBUG is less severe than INFO. Therefore when the logging level is set
    # to DEBUG, information will still be outputted. In addition, errors and warnings are more
    # severe than info and therefore will always be outputted to the log.
    ch.setLevel(logging.DEBUG if debug else logging.INFO)
    ch.setFormatter(
        logging.Formatter("\n%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    return ch
