"""
Utility functions.
"""


def load_tf_names(path):
    """
    :param path: the path of the transcription factor list file.
    :return: a list of transcription factor names read from the file.
    """

    with open(path) as file:
        tfs_in_file = [line.strip() for line in file.readlines()]

    return tfs_in_file