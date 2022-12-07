# This file is adapted from https://git.corp.adobe.com/3di/python-scaffold

# import boa_toolkit.utils.log as base
import json
import logging

# root_logger = base.root_logger
# logger: base.Logger = root_logger.sub_logger("MultiMetaSSD")
# Channel = base.Channel
# ScopedLog = base.ScopedLog

# def logger2(log_file):
#     log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(funcName)s(%(lineno)d) %(message)s',
#                                       datefmt='%d/%m/%Y %H:%M:%S')


#     #Setup File handler
#     file_handler = logging.FileHandler(log_file)
#     file_handler.setFormatter(log_formatter)
#     file_handler.setLevel(logging.INFO)

#     #Setup Stream Handler (i.e. console)
#     stream_handler = logging.StreamHandler()
#     stream_handler.setFormatter(log_formatter)
#     stream_handler.setLevel(logging.INFO)

#     #Get our logger
#     app_log = logging.getLogger('root')
#     app_log.setLevel(logging.INFO)

#     #Add both Handlers
#     app_log.addHandler(file_handler)
#     app_log.addHandler(stream_handler)
#     return app_log

import logging as logger

logger = logger

def read_json(path):
    try:
        return json.load(open(path))
    except:
        raise Exception('Error reading JSON from %s' % path)

# Global statistics that we can output to monitor the run.


stats_path = None
STATS = {}


def logstats_init(path):
    global stats_path, STATS
    stats_path = path
    try:
        STATS = read_json(stats_path)
    except Exception:
        STATS = {}


def logstats_add(*args):
    # Example: add_stats('data', 'num_examples', 3)
    s = STATS
    prefix = args[:-2]
    for k in prefix:
        if k not in s and k != "func":
            s[k] = {}
        s = s[k]
    s[args[-2]] = args[-1]
    flush()


def logstats_add_args(key, args):
    args_var = {k: v for k, v in vars(args).items() if k not in ["func", "device"]}
    logstats_add(key, dict((arg, getattr(args, arg)) for arg in args_var))


def flush():
    if stats_path:
        out = open(stats_path, 'w')
        print(json.dumps(STATS))
        out.close()


def logstats_write_json(raw, path):
    with open(path, 'w') as out:
        json.dump(raw, out, indent=4, sort_keys=True)
