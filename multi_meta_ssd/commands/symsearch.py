from multi_meta_ssd.log import logger


def create_action2_parser(subparser):
    parser = subparser.add_parser("action2")

    subparsers = parser.add_subparsers(metavar="action2_subaction", required=True)

    subaction1_parser = subparsers.add_parser("subaction1", help="Run multi_meta_ssd action2 subaction1")
    subaction1_parser.set_defaults(func=run_action2_subaction1)

    subaction2_parser = subparsers.add_parser("subaction2", help="Run multi_meta_ssd action2 subaction2")
    subaction2_parser.add_argument("--something", help="Pass something to me please", type=str, required=True)
    subaction2_parser.set_defaults(func=run_action2_subaction2)


def run_action2_subaction1(args):  # pylint: disable=unused-argument
    logger.info("You decided to run `multi_meta_ssd action2 subaction1`")


def run_action2_subaction2(args):
    logger.info("You decided to run `multi_meta_ssd action2 subaction2` with something=%s", args.something)
