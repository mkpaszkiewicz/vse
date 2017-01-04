import math
import heapq
import operator
from abc import ABCMeta, abstractmethod
from vse.utils import normalize

__all__ = ['Ranker',
           'SimpleRanker',
           'WeighingRanker'
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

    def _rank_best_results(self, items, n, diff_ratio_function):
        results = [(image_id, diff_ratio_function(hist)) for image_id, hist in items]
        return self._n_best_results(results, n)

    def _n_best_results(self, results, n):
        if self.hist_comparator.reversed:
            function = heapq.nlargest
        else:
            function = heapq.nsmallest
        return function(n, results, key=operator.itemgetter(1))


class SimpleRanker(Ranker):
    def __init__(self, hist_comparator):
        Ranker.__init__(self, hist_comparator)

    def rank(self, query_hist, items, n, freq_vector=None):

        def diff_ratio_function(hist):
            return self.hist_comparator.compare(hist, query_hist)

        return self._rank_best_results(items, n, diff_ratio_function)


class WeighingRanker(Ranker):
    def __init__(self, hist_comparator, query_weigh_function=tfidf, item_weigh_function=tfidf):
        Ranker.__init__(self, hist_comparator)
        self.query_weigh_function = query_weigh_function
        self.item_weigh_function = item_weigh_function

    def rank(self, query_hist, items, n, freq_vector):
        weighted_query_hist = normalize(self.query_weigh_function(query_hist, freq_vector))

        def diff_ratio_function(hist):
            weighted_item_hist = normalize(self.item_weigh_function(hist, freq_vector))
            return self.hist_comparator.compare(weighted_item_hist, weighted_query_hist)

        return self._rank_best_results(items, n, diff_ratio_function)
