import datetime as dt
import json
import os
import sys
from typing import Dict, List, Optional, Union

from . import __version__
from .config import (
    get_bustools_binary_path,
    get_kallisto_binary_path,
    is_dry,
)
from .dry import dummy_function
from .dry import dryable


class Stats:
    """Class used to collect kb run statistics.
    """

    def __init__(self):
        self.workdir = None
        self.kallisto = None
        self.bustools = None
        self.start_time = None
        self.call = None
        self.commands = []
        self.runtimes = []
        self.end_time = None
        self.elapsed = None
        self.version = __version__

    def start(self):
        """Start collecting statistics.

        Sets start time, the command line call, and the commands array to an empty list.
        Additionally, sets the kallisto and bustools paths and versions.
        """
        self.start_time = dt.datetime.now()
        self.call = ' '.join(sys.argv)
        self.commands = []
        self.workdir = os.getcwd()

        if not is_dry():
            # Import here to prevent circular imports
            from .utils import get_bustools_version, get_kallisto_version
            self.kallisto = {
                'path': get_kallisto_binary_path(),
                'version': '.'.join(str(i) for i in get_kallisto_version())
            }
            self.bustools = {
                'path': get_bustools_binary_path(),
                'version': '.'.join(str(i) for i in get_bustools_version())
            }

    def command(self, command: List[str], runtime: Optional[float] = None):
        """Report a shell command was run.

        Args:
            command: A shell command, represented as a list
            runtime: Command runtime
        """
        cmd = ' '.join(command)
        self.commands.append(cmd)
        self.runtimes.append(runtime or 'not measured')

    def end(self):
        """End collecting statistics.
        """
        self.end_time = dt.datetime.now()
        self.elapsed = (self.end_time - self.start_time).total_seconds()

    @dryable(dummy_function)
    def save(self, path: str) -> str:
        """Save statistics as JSON to path.

        Args:
            path: Path to JSON

        Returns:
            Path to saved JSON
        """
        if not is_dry():
            with open(path, 'w') as f:
                json.dump(self.to_dict(), f, indent=4)
        return path

    def to_dict(self) -> Dict[str, Union[str, float]]:
        """Convert statistics to dictionary, so that it is easily parsed
        by the report-rendering functions.

        Returns:
            Statistics dictionary
        """
        return {
            'workdir': self.workdir,
            'version': self.version,
            'kallisto': self.kallisto,
            'bustools': self.bustools,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'elapsed': self.elapsed,
            'call': self.call,
            'commands': self.commands,
            'runtimes': self.runtimes,
        }


STATS = Stats()
