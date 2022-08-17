# This file is adapted from https://git.corp.adobe.com/3di/python-scaffold

import argparse
import io
import sys
import unittest
from pathlib import Path

import xmlrunner
import xmlrunner.extra.xunit_plugin
from tiny.commands.main import main as tiny_main

from multi_meta_ssd.log import Channel, logger, root_logger
from multi_meta_ssd.multi_meta_ssd_test_case import MultiMetaSSDTestCase

# Import the current folder so that every module is reachable.
root_dir = Path(__file__).parents[1]
sys.path.insert(0, str(root_dir))

# A shim to pass the log output from unittest to
# our logger instead of just print() them.
class UnitTestStream:
    def __init__(self):
        self.data_cache = []

    def write(self, text):
        if text == "\n":
            self.flush()
            return
        self.data_cache.append(text)

    def flush(self):
        logger.log(Channel.TESTRUNNER, "".join(self.data_cache))
        self.data_cache.clear()


def write_xml_test_report(path, content):
    path.parent.mkdir(exist_ok=True, parents=True)
    try:
        content = xmlrunner.extra.xunit_plugin.transform(content)
    except Exception as e:
        logger.warning("Failed creating xml report: %s, writing unparsed xml!", e)
    with path.open("wb") as f:
        f.write(content)


def main():
    parser = argparse.ArgumentParser("multi_meta_ssd test runner")
    parser.add_argument(
        "--no-discover",
        "-n",
        action="store_true",
        help="Specify to run only some tests",
    )
    parser.add_argument(
        "--no-tiny",
        action="store_true",
        help="Skip downloading test data",
    )
    parser.add_argument("--log", default=MultiMetaSSDTestCase.output_test_path() / "log_tests.txt", type=Path, help="Path to log file")
    parser.add_argument(
        "--xml",
        default=MultiMetaSSDTestCase.output_test_path() / "log_tests.xml",
        help="An optional file for test xml result to be written to.",
        type=Path,
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging on screen",
    )

    args, unittest_args = parser.parse_known_args(sys.argv[1:])

    # setup logging
    if args.log:
        # add the sink to root logger, this way it will also capture logging from
        # other loggers.
        root_logger.add_file_sink(args.log, Channel.DEBUG)

    logger.capture_warnings()
    if args.verbose:
        logger.console_handler.setLevel(Channel.DEBUG)
    else:
        logger.console_handler.setLevel(Channel.TESTRUNNER)

    # prepare xml output
    xml_out = io.BytesIO()

    class Runner(xmlrunner.XMLTestRunner):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs, output=xml_out, stream=UnitTestStream())

    unit_test_config = dict(
        failfast=False,
        buffer=False,
        catchbreak=False,
        exit=False,
        testRunner=Runner,
    )

    # download test artifacts
    if not args.no_tiny:
        config_file = MultiMetaSSDTestCase.root_dir / "tools" / "tiny.yaml"
        tiny_main(
            [
                "pull",
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

    # discover or run specific test
    if args.no_discover:
        logger.info("No discover mode")
        test_argvs = [
            sys.argv[0],
        ] + unittest_args
    else:
        test_argvs = [sys.argv[0], "discover", "-s", str(Path(__file__).parent), "-p", "test_*.py"] + unittest_args

    # run the tests
    try:
        test_result = unittest.main(
            verbosity=2,
            module=None,
            argv=test_argvs,
            **unit_test_config,
        )
    except KeyboardInterrupt:
        logger.warning("You pressed ^C, bye")
    except Exception as e:  # pylint: disable=broad-except
        logger.exception(e)
        sys.exit(100)
    finally:
        if args.xml:
            write_xml_test_report(args.xml, xml_out.getvalue())

    if len(test_result.result.errors) + len(test_result.result.failures) == 0:
        logger.info("Tests passed")
        sys.exit(0)
    else:
        logger.error("Tests failed")
        sys.exit(100)


if __name__ == "__main__":
    main()
