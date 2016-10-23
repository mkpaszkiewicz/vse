"""vse - visual search engine

This module contains configurable visual search engine based on OpenCV.
Engine consists of bag of visual words and image index. Visual words are computed
on the basis of image descriptors and then clustered. Image index is a dictionary
of image path and histogram of particular visual words occurrences. Image
comparison consist in their histograms comparison.

"""

from vse.utils import *

__version__ = '0.1.0'


class VisualSearchEngine:
    def __init__(self, image_index, ranker, bovw):
        self.image_index = image_index
        self.ranker = ranker
        self.bovw = bovw

    def add_image_to_index(self, image_path, image):
        """Adds image and its histogram to index. Argument 'image' contains binary image."""
        hist = self.bovw.generate_hist(image)
        self.image_index.add(image_path, hist, image)
        self.ranker.update(self.image_index)

    def remove_image_from_index(self, image_path):
        self.image_index.remove(image_path)
        self.ranker.update(self.image_index)

    def find_similar(self, image, n=1):
        image = load_image_from_buf(image)
        query_hist = self.bovw.generate_hist(image)
        results = self.ranker.rank(query_hist, self.image_index, n)
        return results


class BagOfVisualWords:
    def __init__(self, extractor, matcher, vocabulary):
        self.extractor = extractor
        self.extract_bow = cv2.BOWImgDescriptorExtractor(self.extractor, matcher)
        self.extract_bow.setVocabulary(vocabulary)

    def generate_hist(self, image):
        """Generates image visual words histogram."""
        key_points = self.extractor.detect(image)
        hist = self.extract_bow.compute(image, key_points)[0]
        return hist


def cluster_voc_from_img(images, extractor, cluster_count=100, filename=''):
    """Generates visual words vocabulary from images. Saves to file if filename given."""
    desc = []
    for image in images:
        desc.append(extractor.detectAndCompute(image, None)[1])
    return cluster_voc_from_desc(desc, cluster_count, filename)


def cluster_voc_from_desc(descriptors, cluster_count=100, filename=''):
    """Generates visual words vocabulary from images descriptors. Saves to file if filename given."""
    bow_kmeans_trainer = cv2.BOWKMeansTrainer(cluster_count)
    for desc in descriptors:
        bow_kmeans_trainer.add(desc)
    vocabulary = bow_kmeans_trainer.cluster()
    if filename:
        save(filename, vocabulary)
    return vocabulary
