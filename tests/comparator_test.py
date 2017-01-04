import cv2
import unittest
from unittest.mock import Mock, patch, call
from vse.comparator import *
from vse.comparator import cosine_angle


class ComparatorTest(unittest.TestCase):
    @patch('vse.comparator.cv2.compareHist')
    def test_should_call_correlation(self, compare_hist_mock):
        comparator = Correlation()
        hist1 = Mock()
        hist2 = Mock()

        comparator.compare(hist1, hist2)

        compare_hist_mock.assert_called_with(hist1, hist2, cv2.HISTCMP_CORREL)

    @patch('vse.comparator.cv2.compareHist')
    def test_should_call_chi_squared(self, compare_hist_mock):
        comparator = ChiSquared()
        hist1 = Mock()
        hist2 = Mock()

        comparator.compare(hist1, hist2)

        compare_hist_mock.assert_called_with(hist1, hist2, cv2.HISTCMP_CHISQR)

    @patch('vse.comparator.cv2.compareHist')
    def test_should_call_intersection(self, compare_hist_mock):
        comparator = Intersection()
        hist1 = Mock()
        hist2 = Mock()

        comparator.compare(hist1, hist2)

        compare_hist_mock.assert_called_with(hist1, hist2, cv2.HISTCMP_INTERSECT)

    @patch('vse.comparator.cv2.compareHist')
    def test_should_call_hellinger(self, compare_hist_mock):
        comparator = Hellinger()
        hist1 = Mock()
        hist2 = Mock()

        comparator.compare(hist1, hist2)

        compare_hist_mock.assert_called_with(hist1, hist2, cv2.HISTCMP_HELLINGER)

    @patch('vse.comparator.cv2.compareHist')
    def test_should_call_bhattacharyya(self, compare_hist_mock):
        comparator = Bhattacharyya()
        hist1 = Mock()
        hist2 = Mock()

        comparator.compare(hist1, hist2)

        compare_hist_mock.assert_called_with(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)

    @patch('vse.comparator.cv2.compareHist')
    def test_should_call_chi_squared_alt(self, compare_hist_mock):
        comparator = ChiSquaredAlt()
        hist1 = Mock()
        hist2 = Mock()

        comparator.compare(hist1, hist2)

        compare_hist_mock.assert_called_with(hist1, hist2, cv2.HISTCMP_CHISQR_ALT)

    @patch('vse.comparator.cv2.compareHist')
    def test_should_call_kullback_leibler(self, compare_hist_mock):
        comparator = KullbackLeibler()
        hist1 = Mock()
        hist2 = Mock()

        comparator.compare(hist1, hist2)

        compare_hist_mock.assert_called_with(hist1, hist2, cv2.HISTCMP_KL_DIV)

    @patch('vse.comparator.numpy.linalg.norm')
    def test_should_call_euclidean(self, norm_mock):
        comparator = Euclidean()
        hist1 = Mock()
        hist2 = Mock()
        hist = Mock()
        hist1.__sub__ = Mock(return_value=hist)

        comparator.compare(hist1, hist2)

        norm_mock.assert_called_with(hist)

    @patch('vse.comparator.cosine_angle')
    def test_should_call_cosine_angle(self, cosine_angle_mock):
        hist1 = Mock()
        hist2 = Mock()
        comparator = CosineAngle()

        comparator.compare(hist1, hist2)

        cosine_angle_mock.assert_called_with(hist1, hist2)

    @patch('vse.comparator.numpy.dot')
    @patch('vse.comparator.unit_vector')
    def test_should_calculate_cosine_angle(self, unit_vector_mock, dot_mock):
        unit_v_mock = Mock()
        unit_vector_mock.return_value = unit_v_mock
        hist1 = Mock()
        hist2 = Mock()

        cosine_angle(hist1, hist2)

        self.assertEqual(unit_vector_mock.mock_calls, [call(hist1), call(hist2)])
        dot_mock.assert_called_with(unit_v_mock, unit_v_mock)
