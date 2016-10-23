import heapq
import math
from abc import ABCMeta, abstractmethod

__all__ = ['Ranker',
           'SimpleRanker',
           'TFIDFRanker'
           ]


class Ranker(metaclass=ABCMeta):
    def __init__(self, hist_comparator):
        self.hist_comparator = hist_comparator

    @abstractmethod
    def rank(self, query_hist, image_index, n):
        """Ranks images in index by similarity to query_hist. Returns list of tuples:
        (ratio, filename) and also boolean value of ratio direction. If direction is true
        then the higher ratio the more accuracy match."""
        pass

    def update(self, image_index):
        """Invoke after image index has changed. Ranker has to update its own parameters"""
        pass


class SimpleRanker(Ranker):
    def __init__(self, hist_comparator):
        Ranker.__init__(self, hist_comparator)

    def rank(self, query_hist, image_index, n):
        results = []
        for filename, hist in image_index.get(query_hist):
            diff_ratio = self.hist_comparator.compare(hist, query_hist)
            results.append((diff_ratio, filename))
        if self.hist_comparator.REVERSED:
            return heapq.nlargest(n, results, key=lambda tup: tup[0])
        else:
            return heapq.nsmallest(n, results, key=lambda tup: tup[0])


class TFIDFRanker(Ranker):
    def __init__(self, hist_comparator):
        Ranker.__init__(self, hist_comparator)
        self.tfidf_vector = []

    def update(self, image_index):
        self.tfidf_vector = [sum(x) for x in zip(*image_index.values())]
        self.tfidf_vector = self._normalize(self.tfidf_vector)

    def rank(self, query_hist, image_index, n):
        results = []
        for filename, hist in image_index.get():
            diff_ratio = self.hist_comparator(self._weight(hist), query_hist)
            results.append((diff_ratio, filename))
        if self.hist_comparator.REVERSED:
            return heapq.nlargest(n, results, key=lambda tup: tup[0])
        else:
            return heapq.nsmallest(n, results, key=lambda tup: tup[0])

    def _weight(self, hist):
        weighted_hist = hist.copy()
        for i, val in enumerate(hist):
            if self.tfidf_vector[i] != 0:
                weighted_hist[i] = -val * math.log(self.tfidf_vector[i], 10)

        return self._normalize(weighted_hist)

    def _normalize(self, hist):
        hist_sum = sum(hist)
        norm_hist = hist.copy()
        for i, val in enumerate(hist):
            norm_hist[i] = hist[i] / hist_sum
        return norm_hist
