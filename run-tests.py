#!/usr/bin/env python
"""
run all of the unit tests with a single command:   python run-tests.py
(logger, model / ingestion and API tests; exit code 0 only if every test passes)
Production models and logs are never touched - see unittests/__init__.py
"""
import sys
import unittest

import unittests  # noqa  (isolates models / logs)
from unittests.LoggerTests import LoggerTest
from unittests.ModelTests import IngestionTest, FeatureTest, ModelTest
from unittests.ApiTests import ApiTest

if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    for case in (LoggerTest, IngestionTest, FeatureTest, ModelTest, ApiTest):
        suite.addTests(loader.loadTestsFromTestCase(case))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
