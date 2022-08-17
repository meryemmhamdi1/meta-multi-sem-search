# This file is adapted from https://git.corp.adobe.com/3di/python-scaffold

import asyncio

from boa_toolkit.parallel.aioworker import execute_in_bg, resolve_coroutine_now

from multi_meta_ssd.log import Channel, ScopedLog, logger
from multi_meta_ssd.multi_meta_ssd_test_case import MultiMetaSSDTestCase


class TestDemo(MultiMetaSSDTestCase):
    def test_example_fixture(self):
        sample_file = self.input_test_path() / "__fixtures__" / "sample-file.txt"
        content = self.read_file(sample_file).strip()
        self.assertEqual(content, "Hello I am a fixture for testing.")

    def test_logger(self):
        with ScopedLog(Channel.INFO, logger) as test_log:
            logger.info("howdy")
        self.assertIn("howdy", test_log.get())

    def test_run_bg_command(self):
        # Here we show how to call external commands from python

        # Wrap tasks that need bg calls in an async function
        async def task_that_needs_running_external_commands():
            await execute_in_bg(["python", "--help"])

            task1 = execute_in_bg(["python", "-c", 'print("A")'])
            task2 = execute_in_bg(["python", "-c", 'print("B")'])
            await asyncio.gather(task1, task2)

        # Then run the async function in boa_toolkit's event loop
        with ScopedLog(Channel.DEBUG) as test_log:
            resolve_coroutine_now(task_that_needs_running_external_commands())

        logs = test_log.get()
        self.assertIn("python", logs)
        self.assertIn("A", logs)
        self.assertIn("B", logs)
