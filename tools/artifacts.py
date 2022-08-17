# This file is adapted from https://git.corp.adobe.com/3di/python-scaffold
# This is for storing your large data outside of git.

import pathlib
import sys

from tiny.commands.main import main as tiny_main


def main(argvs=None):
    if argvs is None:
        argvs = sys.argv[1:]
        config_file = pathlib.Path(__file__).parent / "tiny.yaml"
        tiny_main(
            argvs
            + [
                "--num-retry",
                "10",  # try 10 times.
                "--retry-wait",
                "1",  # wait for 1 second after failure
                "--num-sim-jobs",
                "20",  # how many files to process at once
                "--config",
                str(config_file),
            ]
        )


if __name__ == "__main__":
    main()
