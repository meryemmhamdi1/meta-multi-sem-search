# This file is adapted from https://git.corp.adobe.com/3di/python-scaffold
import argparse
import sys
from pathlib import Path

from multi_meta_ssd.commands.data_utils.split_data import split_data_options
from multi_meta_ssd.commands.data_utils.get_statistics import get_statistics
from multi_meta_ssd.commands.data_utils.split_stsb_crossval import split_stsb_cross_val

from multi_meta_ssd.commands.asymsearch import create_a_sym_search_parser
from multi_meta_ssd.commands.asymsearch_kd import create_a_sym_search_kd_parser
from multi_meta_ssd.commands.symsearch import create_sym_search_parser

from multi_meta_ssd.commands.evaluation.performance_evaluation import perf_eval_options
from multi_meta_ssd.commands.evaluation.evaluate_mt_models import mt_perf_eval_options
from multi_meta_ssd.commands.evaluation.evaluate_stsb_mt import evaluate_stsb_mt_options
from multi_meta_ssd.commands.evaluation.evaluate_symsearch import symsearch_eval


def parse_args(argv):
    parser = argparse.ArgumentParser(description="MultiMetaSSD command line actions")

    parser.add_argument(
        "--verbose",
        action="store_true",
        dest="verbose",
    )
    parser.add_argument("--log-file", type=Path)

    # Actions
    subparser = parser.add_subparsers(help="Action to perform", dest="action", required=True)

    split_data_options(subparser)
    get_statistics(subparser)
    split_stsb_cross_val(subparser)

    create_a_sym_search_parser(subparser)
    create_a_sym_search_kd_parser(subparser)
    create_sym_search_parser(subparser)

    perf_eval_options(subparser)
    mt_perf_eval_options(subparser)
    symsearch_eval(subparser)
    evaluate_stsb_mt_options(subparser)

    # Parse arguments
    return parser.parse_args(argv)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)

    # # 0 handler is console
    # if args.verbose:
    #     logger.console_handler.setLevel("DEBUG")
    # else:
    #     logger.console_handler.setLevel("INFO")

    # if args.log_file:
    #     # Add the log file to the root logger so it captures both output
    #     # of your lib and any other lib that uses boa.
    #     root_logger.add_file_sink(args.log_file, "DEBUG")

    #     # If you add log file only to your logger, it captures
    #     # messages logged by your lib only
    #     # logger.add_log_file(args.log_file, "DEBUG")

    #     # you can do both (just make sure to use two different files)

    # Run the action.
    return args.func(args)


# Needed by setup.py console scripts
def main_cmd():
    main()


if __name__ == "__main__":
    main()
