"""Run all tests"""

import unittest
from .comparator_test import ComparatorTest
from .engine_test import VisualSearchEngineTest, BagOfVisualWordsTest
from .utils_test import UtilityTest


if __name__ == '__main__':
    unittest.main()
