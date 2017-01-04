import unittest
from unittest.mock import Mock, patch

from vse import VisualSearchEngine, BagOfVisualWords


class VisualSearchEngineTest(unittest.TestCase):
    def setUp(self):
        self.image_index = Mock()
        self.bovw = Mock()
        self.hist = Mock()
        self.bovw.generate_hist = Mock(return_value=self.hist)
        self.engine = VisualSearchEngine(self.image_index, self.bovw)

    def test_should_add_image_to_index(self):
        image_id = 'image_id'
        image = Mock()
        self.image_index.__setitem__ = Mock()

        self.engine.add_to_index(image_id, image)

        self.bovw.generate_hist.assert_called_with(image)
        self.image_index.__setitem__.assert_called_with(image_id, self.hist)

    def test_should_remove_image_from_index(self):
        image_id = 'image_id'
        self.image_index.__delitem__ = Mock()

        self.engine.remove_from_index(image_id)

        self.image_index.__delitem__.assert_called_with(image_id)

    def test_should_find_similar(self):
        image = Mock()
        result = Mock()
        self.image_index.find = Mock(return_value=result)

        similar_image = self.engine.find_similar(image)

        self.bovw.generate_hist.assert_called_with(image)
        self.image_index.find.assert_called_with(self.hist, 1)
        self.assertEqual(similar_image, result)

    def test_should_find_similar_n(self):
        n = 5
        image = Mock()
        result = Mock()
        self.image_index.find = Mock(return_value=result)

        similar_image = self.engine.find_similar(image, n)

        self.bovw.generate_hist.assert_called_with(image)
        self.image_index.find.assert_called_with(self.hist, n)
        self.assertEqual(similar_image, result)


class BagOfVisualWordsTest(unittest.TestCase):
    def setUp(self):
        self.extractor = Mock()
        self.key_points = Mock()
        self.extractor.detect = Mock(return_value=self.key_points)
        self.matcher = Mock()
        self.vocabulary = Mock()

    @patch('vse.engine.cv2.BOWImgDescriptorExtractor')
    def test_should_init_bowv(self, descriptor_extractor):
        extract_bow = Mock()
        extract_bow.setVocabulary = Mock()
        descriptor_extractor.return_value = extract_bow

        bovw = BagOfVisualWords(self.extractor, self.matcher, self.vocabulary)

        descriptor_extractor.assert_called_with(self.extractor, self.matcher)
        extract_bow.setVocabulary.assert_called_with(self.vocabulary)
        self.assertEqual(bovw.extractor, self.extractor)
        self.assertEqual(bovw.extract_bow, extract_bow)

    @patch('vse.engine.cv2.BOWImgDescriptorExtractor')
    def test_should_generate_hist(self, descriptor_extractor):
        extract_bow = Mock()
        extract_bow.setVocabulary = Mock()
        hist = Mock()
        extract_bow.compute = Mock(return_value=(hist, Mock()))
        descriptor_extractor.return_value = extract_bow
        image = Mock()

        bovw = BagOfVisualWords(self.extractor, self.matcher, self.vocabulary)
        result = bovw.generate_hist(image)

        self.extractor.detect.assert_called_with(image)
        extract_bow.compute.assert_called_with(image, self.key_points)
        self.assertEqual(result, hist)
