# This file is adapted from https://git.corp.adobe.com/3di/python-scaffold
import hashlib
import logging
import unittest
from pathlib import Path
from typing import Union

from multi_meta_ssd.log import logger


class MultiMetaSSDTestCase(unittest.TestCase):
    root_dir = Path(__file__).absolute().parents[1]

    @classmethod
    def setUpClass(cls):
        if len(logger.handlers) == 0:
            logger.setLevel("INFO")
            print_to_screen = logging.StreamHandler()
            print_to_screen.setLevel("INFO")
            logger.addHandler(print_to_screen)
        logger.info("setUp %r", cls.__name__)

    @classmethod
    def tearDownClass(cls):
        logger.info("tearDown %r", cls.__name__)

    @staticmethod
    def sha256(some_value):
        if isinstance(some_value, str):
            some_value = some_value.encode("utf-8")
        hash_sha256 = hashlib.sha256()
        hash_sha256.update(some_value)
        return hash_sha256.hexdigest()

    def read_file(self, path, asbytes=False) -> Union[bytes, str]:
        self.assertTrue(Path(path).is_file())
        mode = "rb" if asbytes else "r"
        with open(path, mode) as stream:
            contents = stream.read()
        return contents

    @staticmethod
    def file_md5(path) -> str:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    @classmethod
    def input_test_path(cls):
        return cls.root_dir / "data"

    @classmethod
    def output_test_path(cls):
        return cls.root_dir / ".tmp_test_out"
