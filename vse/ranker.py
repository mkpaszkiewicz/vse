import math
import heapq
import operator
from abc import ABCMeta, abstractmethod
from vse.utils import normalize

__all__ = ['Ranker',
           'SimpleRanker',
           'WeightingRanker'
           ]


def tfidf(hist, freq_hist):
    """Term frequency - inverse document frequency scoring."""
    return [n * -log(n_freq) for n, n_freq in zip(hist, freq_hist)]


def log(n):
    if n == 0:
        return 0
    else:
        return math.log(n)


class Ranker(metaclass=ABCMeta):
    def __init__(self, hist_comparator):
        self.hist_comparator = hist_comparator

    @abstractmethod
    def rank(self, query_hist, items, n, freq_vector):
        """Ranks index items by similarity to query_hist. Returns list of tuples: (image_id, diff_ratio)."""
        pass

    def _n_best_results(self, results, n):
        if self.hist_comparator.REVERSED:
            function = heapq.nlargest
        else:
            function = heapq.nsmallest
        return function(n, results, key=operator.itemgetter(1))


class SimpleRanker(Ranker):
    def __init__(self, hist_comparator):
        Ranker.__init__(self, hist_comparator)

    def rank(self, query_hist, items, n, freq_vector=None):
        results = [(image_id, self.hist_comparator.compare(hist, query_hist)) for image_id, hist in items]
        return self._n_best_results(results, n)


class WeightingRanker(Ranker):
    def __init__(self, hist_comparator, query_weight=tfidf, item_weight=tfidf):
        Ranker.__init__(self, hist_comparator)
        self.query_weight = query_weight
        self.item_weight = item_weight

    def rank(self, query_hist, items, n, freq_vector):
        results = []
        weighted_query_hist = normalize(self.query_weight(query_hist, freq_vector))
        for image_id, hist in items:
            weighted_item_hist = normalize(self.item_weight(hist, freq_vector))
            diff_ratio = self.hist_comparator.compare(weighted_item_hist, weighted_query_hist)
            results.append((image_id, diff_ratio))
        return self._n_best_results(results, n)
