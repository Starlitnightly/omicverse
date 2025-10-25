from urllib.parse import urlparse


def stream_batch(batch_path: str, temp_dir: str = 'tmp') -> str:
    """Dry version of `count.stream_batch`.
    """
    with open(batch_path, 'r') as f:
        for line in f:
            if line.isspace() or line.startswith('#'):
                continue
            sep = '\t' if '\t' in line else ' '
            split = line.split(sep)
            if any(urlparse(fastq).scheme in ('http', 'https', 'ftp', 'ftps')
                   for fastq in split[1:]):
                raise Exception(
                    'Streaming remote FASTQs from a batch file is not dryable.'
                )
    return batch_path


def write_smartseq3_capture(capture_path: str) -> str:
    """Dry version of `count.write_smartseq3_capture`.
    """
    print(f'echo \"{"T" * 32}\" > {capture_path}')
    return capture_path
